r"""Run an isolated, temporary Localive brain/arbiter comparison using recorded speech.

    .venv\Scripts\python.exe bench/compare_live.py --brain qwen3.5 --arbiter qwen3 --repeat 2
    .venv\Scripts\python.exe bench/compare_live.py --brain qwen3 --arbiter qwen3.5 --interruptions

Stop the normal app first. This launcher refuses occupied ports and never saves settings.
All tool results are local fixtures; no web-search tool or external actions are enabled.
--client-only uses an already-running, matching local app and does not stop its servers.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import copy
import hashlib
import importlib.util
import json
import logging
import os
import re
import socket
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlencode

import aiohttp
import numpy as np
from aiohttp import web

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from localive import live, server  # noqa: E402
from localive.config import AltBrainConfig, Config, load_config  # noqa: E402
from localive.decision import SYSTEM as DECISION_PROMPT  # noqa: E402
from localive.decision import Arbiter  # noqa: E402
from localive.engines import Brain  # noqa: E402
from scripts import live_client  # noqa: E402

MODELS = {
    "qwen3": ("qwen", "models/Qwen3-4B-Instruct-2507/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
    "qwen3.5": ("qwen3.5", "models/Qwen3.5-4B/Qwen3.5-4B-Q4_K_M.gguf"),
    "qwen3.5-9b": ("qwen3.5-9b", "models/Qwen3.5-9B/Qwen3.5-9B-Q4_K_M.gguf"),
}
FIXTURES = [ROOT / "tests/fixtures" / name for name in
            ("wake_clock.wav", "wake_calculation.wav", "active_followup.wav")]
INTERRUPTIONS = ("acquiescement", "stop", "correction (réflexion)")


class Tee:
    def __init__(self, console, file):
        self.console, self.file = console, file

    def write(self, value):
        self.console.write(value)
        self.file.write(value)
        self.flush()
        return len(value)

    def flush(self):
        self.console.flush()
        self.file.flush()


class OwnedBrain(Brain):
    async def start(self, http):
        if await self.healthy(http):
            raise RuntimeError(f"Refusing to reuse an existing brain at {self.url}")
        await super().start(http)


class ComparisonApp(server.App):
    """Separate aliases without changing production prompts or decision logic."""

    def __init__(self, config, arbiter_alias):
        super().__init__(config)
        self.arbiter_config = copy.deepcopy(config)
        self.arbiter_config.llm.model = arbiter_alias

    async def boot(self):
        def factory(http, _config):
            return Arbiter(http, self.arbiter_config)

        # This standalone benchmark is the only app in this process. Restore the factory
        # after startup; the created object remains the unchanged production Arbiter.
        with patch.object(server, "Arbiter", factory):
            await super().boot()


def require_free_ports(ports):
    with contextlib.ExitStack() as stack:
        for port in ports:
            sock = stack.enter_context(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError as exc:
                raise RuntimeError(f"Local port {port} is occupied or unavailable") from exc


def fixture_tool_result(name, arguments):
    if name == "clock__read":
        result = {"time": "17:42", "date": "vendredi 6 novembre 2026"}
    elif name == "weather__read":
        result = {"city": arguments.get("city") or "Paris", "temperature_c": 9,
                  "conditions": "averses (fixture locale)"}
    elif name == "home_assistant__control":
        result = {"entity": arguments.get("entity"), "state": "done", "fixture": True}
    else:
        return {"schemaVersion": 1, "status": "failed",
                "error": "Unknown fixture tool; no action was performed."}
    return {"schemaVersion": 1, "status": "succeeded", "result": result, "summary": None}


def make_configuration(args):
    file = ROOT / "localive.toml"
    config = copy.deepcopy(load_config(file) if file.is_file() else Config(root=ROOT))
    config.file, config.file_values = None, {}
    config.root = ROOT
    config.server.host, config.server.port = "127.0.0.1", args.port
    if not args.client_only:
        config.server.api_key = ""  # local temporary listener only; source config is untouched
    config.search.engine = "off"
    config.huggingface.offline = True
    config.brain.managed = True
    config.brain.model = MODELS[args.brain][1]
    config.brain.port, config.brain.context = args.brain_port, 24576
    config.brain.llama_dir = args.brain_llama_dir or config.brain.llama_dir
    config.brain.extra_args = f"--alias {MODELS[args.brain][0]}"
    config.llm.model = MODELS[args.brain][0]
    config.llm.url = f"http://127.0.0.1:{args.brain_port}/v1"
    config.llm.api_key = ""
    config.decision.enabled = config.decision.dedicated = True
    config.decision.port, config.decision.context = args.arbiter_port, 8192
    config.decision.timeout_ms = 600
    return config


async def wait_ready(app, startup_limit):
    deadline = time.monotonic() + startup_limit
    while not app.ready.is_set():
        if app.failure:
            raise RuntimeError(f"Localive startup failed: {app.failure}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Localive did not become ready in {startup_limit:g} seconds")
        await asyncio.sleep(0.1)


async def observed_servers(config, args):
    observed = {}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as http:
        for role, port, selected in (("brain", args.brain_port, args.brain),
                                      ("arbiter", args.arbiter_port, args.arbiter)):
            async with http.get(f"http://127.0.0.1:{port}/props") as response:
                response.raise_for_status()
                props = await response.json()
            expected = Path(MODELS[selected][1]).name
            actual = Path(props.get("model_path", "")).name
            if actual != expected:
                raise RuntimeError(f"{role} loaded {actual!r}, expected {expected!r}")
            observed[role] = {key: value for key, value in props.items() if "template" not in key}
        async with http.get(f"http://127.0.0.1:{args.port}/health") as response:
            response.raise_for_status()
            health = await response.json()
        if not health.get("ready") or health.get("brain") != config.llm.model:
            raise RuntimeError("Localive is not ready with the requested main model alias")
        observed["app"] = health
    return observed


async def smoke(config, output, repeat):
    report_path = Path(f"{output}-run{repeat}.json")
    audio_path = Path(f"{output}-run{repeat}.wav")
    console_path = Path(f"{output}-run{repeat}-console.txt")
    session = copy.deepcopy(live_client.SESSION)
    session["delegation"]["responses"]["model"] = config.llm.model
    session["delegation"]["responses"]["tools"] = copy.deepcopy(live_client.TOOLS)
    calls = []

    def tool_result(name, arguments):
        result = fixture_tool_result(name, arguments)
        calls.append({"name": name, "arguments": arguments, "fixture_result": result})
        return result

    argv = [str(ROOT / "scripts/live_client.py"), "--profile", "cornelius", *map(str, FIXTURES),
            "--url", f"ws://127.0.0.1:{config.server.port}/v1/live/sessions",
            "--token", config.server.api_key, "--save", str(audio_path), "--report", str(report_path)]
    result = {"repeat": repeat, "audio": str(audio_path), "report": str(report_path),
              "console": str(console_path), "fixture_calls": calls}
    handle = await asyncio.to_thread(console_path.open, "w", encoding="utf-8")
    with handle, \
            contextlib.redirect_stdout(Tee(sys.stdout, handle)), \
            patch.object(live_client, "SESSION", session), \
            patch.object(live_client, "tool_result", tool_result), patch.object(sys, "argv", argv):
        try:
            await live_client.main()
            result["status"] = "completed"
        except (Exception, SystemExit) as exc:
            result["status"], result["error"] = "failed", f"{type(exc).__name__}: {exc}"
            print(result["error"], flush=True)
    if await asyncio.to_thread(report_path.exists):
        result["metrics"] = json.loads(await asyncio.to_thread(report_path.read_text, encoding="utf-8"))
    transcript = await asyncio.to_thread(console_path.read_text, encoding="utf-8")
    result["transcript_lines"] = [line for line in transcript.splitlines()
                                  if "user:" in line or "assistant:" in line or "tool " in line]
    return result


async def interruption_runs(config, output, repeat):
    # Import only at runtime: this legacy module adjusts cwd/RAYON_NUM_THREADS on import.
    previous_cwd, previous_threads = Path.cwd(), os.environ.get("RAYON_NUM_THREADS")
    try:
        spec = importlib.util.spec_from_file_location(
            "live_scenarios_comparison", ROOT / "bench/decision/live_scenarios.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        os.chdir(previous_cwd)
        if previous_threads is None:
            os.environ.pop("RAYON_NUM_THREADS", None)
        else:
            os.environ["RAYON_NUM_THREADS"] = previous_threads
    import ptts

    selected = [scenario for scenario in module.SCENARIOS if scenario[0] in INTERRUPTIONS]
    module.URL = f"ws://127.0.0.1:{config.server.port}/v1/live/sessions"
    if config.server.api_key:
        module.URL += "?" + urlencode({"api_key": config.server.api_key})
    module.ECHO.update(gain=0.0, brain=config.llm.model)

    cache = ROOT / ".runtime/bench/qwen-comparison-interruption-clips.npz"
    texts = sorted({s[1] for s in selected} | {s[2] for s in selected})
    metadata = {"version": 1, "model": config.voice.model, "voice": "Arnaud", "rate": 16000,
                "text_keys": {text: f"clip_{index}" for index, text in enumerate(texts)}}

    def make_clips():
        if cache.is_file():
            with np.load(cache, allow_pickle=False) as archive:
                if json.loads(archive["metadata"].item()) != metadata:
                    raise ValueError("Interruption clip cache differs from the requested fixed inputs")
                clips = {text: archive[key].copy() for text, key in metadata["text_keys"].items()}
                if any(audio.ndim != 1 or audio.dtype != np.float32 or not audio.size
                       for audio in clips.values()):
                    raise ValueError("Invalid cached interruption audio")
                return clips
        tts = ptts.TTS(config.voice.model, device="cpu", quant="q8", voice="Arnaud", lang="fr")
        clips = {text: module.synth(tts, text, "Arnaud") for text in texts}
        temporary = cache.with_suffix(".partial.npz")
        np.savez_compressed(temporary, metadata=np.array(json.dumps(metadata, ensure_ascii=False)),
                            **{key: clips[text] for text, key in metadata["text_keys"].items()})
        temporary.replace(cache)
        return clips

    clips = await asyncio.to_thread(make_clips)  # once, reused across every repetition
    result = {"scenarios": list(INTERRUPTIONS), "voice": "Arnaud", "echo_gain": 0,
              "clips_cache": str(cache), "clips_metadata": metadata,
              "clips_sha256": {text: hashlib.sha256(audio.tobytes()).hexdigest()
                               for text, audio in clips.items()}, "rows": []}
    destination = Path(f"{output}-interruptions.json")
    handle = await asyncio.to_thread(Path(f"{output}-interruptions-console.txt").open,
                                     "w", encoding="utf-8")
    with handle, \
            contextlib.redirect_stdout(Tee(sys.stdout, handle)):
        for number in range(1, repeat + 1):
            for scenario in selected:
                try:
                    row = await asyncio.wait_for(module.run(*scenario, clips), 60)
                except Exception as exc:
                    row = {"name": scenario[0], "expected": scenario[4], "expect_cut": scenario[5],
                           "decision": None, "interrupted": None,
                           "error": f"{type(exc).__name__}: {exc}"}
                row["repeat"] = number
                row["ok"] = row["decision"] == row["expected"] and row["interrupted"] == row["expect_cut"]
                result["rows"].append(row)
                await asyncio.to_thread(destination.write_text,
                                        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                print(json.dumps(row, ensure_ascii=False), flush=True)
    result["report"] = str(destination)
    return result


async def run(args):
    config = make_configuration(args)
    output = ROOT / ".runtime/bench" / args.prefix
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(f"{output}-manifest.json")
    manifest = {"started_utc": datetime.now(UTC).isoformat(), "brain": args.brain,
                "arbiter": args.arbiter, "repeat": args.repeat, "client_only": args.client_only,
                "smoke_skipped": args.skip_smoke,
                "status": "preparing", "main_prompt": live.LOCAL_INSTRUCTIONS,
                "decision_prompt": DECISION_PROMPT, "sampling": {
                    "temperature": config.llm.temperature, "max_tokens": config.llm.max_tokens},
                "audio_configuration": {"asr": asdict(config.asr), "voice": asdict(config.voice),
                                        "turn": asdict(config.turn), "echo": asdict(config.echo)},
                "scope": "Temporary in-memory settings only; loopback; local fixture tools; "
                         "web_search absent; no real household actions. Repeats are not independent tasks.",
                "fixtures": [str(path) for path in FIXTURES], "smoke": []}

    def save():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    runner = None
    app = None
    log_handler = logging.FileHandler(f"{output}-app.log", encoding="utf-8")
    log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.getLogger().addHandler(log_handler)
    try:
        if not args.client_only:
            await asyncio.to_thread(require_free_ports, (args.port, args.brain_port, args.arbiter_port))
            app = ComparisonApp(config, MODELS[args.arbiter][0])
            app.brain = OwnedBrain(config, name=f"{args.prefix}-brain")
            arbiter_spec = AltBrainConfig(
                name="arbiter", model=MODELS[args.arbiter][1], port=args.arbiter_port,
                context=8192, gpu_layers=config.brain.gpu_layers,
                llama_dir=args.arbiter_llama_dir or config.brain.llama_dir,
                extra_args=f"--alias {MODELS[args.arbiter][0]}")
            arbiter = OwnedBrain(config, arbiter_spec, "arbiter")
            arbiter.name = f"{args.prefix}-arbiter"
            app.alts = [arbiter]
            manifest["commands"] = {"brain": app.brain.command(), "arbiter": arbiter.command()}
            runner = web.AppRunner(app.build(), access_log=None)
            save()
            await runner.setup()
            await web.TCPSite(runner, "127.0.0.1", args.port).start()
            await wait_ready(app, args.startup_timeout)
        else:
            manifest["commands"] = None
        manifest["observed_servers"] = await observed_servers(config, args)
        manifest["status"] = "running"
        save()
        if not args.skip_smoke:
            for number in range(1, args.repeat + 1):
                manifest["smoke"].append(await smoke(config, output, number))
                save()
        if args.interruptions:
            manifest["interruptions"] = await interruption_runs(config, output, args.repeat)
        manifest["status"] = "completed" if all(r["status"] == "completed"
                                                for r in manifest["smoke"]) else "failed"
    except BaseException as exc:
        manifest["status"], manifest["error"] = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            if runner:
                await runner.cleanup()
        finally:
            # Also covers runner.setup failing before the normal cleanup hooks run.
            if app:
                app.brain.stop()
                for brain in app.alts:
                    brain.stop()
            logging.getLogger().removeHandler(log_handler)
            log_handler.close()
            save()
            print(f"Saved {manifest_path}", flush=True)
    return int(manifest["status"] != "completed")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain", choices=MODELS, required=True)
    parser.add_argument("--arbiter", choices=MODELS, required=True)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--prefix", help="Output basename below .runtime/bench")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--brain-port", type=int, default=8081)
    parser.add_argument("--arbiter-port", type=int, default=8084)
    parser.add_argument("--brain-llama-dir", help="Override main llama.cpp runtime directory")
    parser.add_argument("--arbiter-llama-dir", help="Override arbiter llama.cpp runtime directory")
    parser.add_argument("--startup-timeout", type=float, default=240)
    parser.add_argument("--client-only", action="store_true")
    parser.add_argument("--interruptions", action="store_true",
                        help="Also run three frozen live barge-in cases; no hesitation suite")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Run only interruptions; requires --interruptions")
    args = parser.parse_args()
    args.prefix = args.prefix or f"live-{args.brain}-{args.arbiter}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.prefix):
        parser.error("--prefix must be a plain filename prefix, without directories")
    ports = (args.port, args.brain_port, args.arbiter_port)
    if len(set(ports)) != 3 or any(not 1 <= port <= 65535 for port in ports):
        parser.error("The three ports must be distinct values between1 and65535")
    if args.repeat < 1 or args.startup_timeout <= 0:
        parser.error("--repeat and --startup-timeout must be positive")
    if args.skip_smoke and not args.interruptions:
        parser.error("--skip-smoke requires --interruptions")
    return args


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(asyncio.run(run(arguments())))
