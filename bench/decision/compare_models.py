r"""Compare production interruption decisions and plans on local llama.cpp models.

    .venv\Scripts\python.exe bench/decision/compare_models.py --models qwen3 qwen3.5 gemma-e4b
    .venv\Scripts\python.exe bench/decision/compare_models.py --models gemma-e4b \
        --url http://127.0.0.1:8083/v1 --repeat 2 --output .runtime/bench/gemma-arbiter.json

Managed models run sequentially with the production two-slot arbiter configuration.
An existing --url starts no process and keeps that server's actual settings. Avoid
other inference during a comparison. Reports include every case and server metadata.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import logging
import math
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from localive.config import AltBrainConfig, Config, load_config  # noqa: E402
from localive.decision import CONFIRM_BELOW, SYSTEM, Arbiter, Situation, warm_up_arbiter  # noqa: E402
from localive.engines import Brain  # noqa: E402
from localive.live import is_echo, is_stop  # noqa: E402

MODELS = {
    "qwen3": ("models/Qwen3-4B-Instruct-2507/Qwen3-4B-Instruct-2507-Q4_K_M.gguf", "qwen", ""),
    "qwen3.5": ("models/Qwen3.5-4B/Qwen3.5-4B-Q4_K_M.gguf", "qwen3.5", ""),
    "qwen3.5-9b": ("models/Qwen3.5-9B/Qwen3.5-9B-Q4_K_M.gguf", "qwen3.5-9b", ""),
    "gemma-e4b": ("models/gemma-4-E4B-it/gemma-4-E4B-it-Q4_K_M.gguf", "gemma", "--swa-full"),
    "minicpm5-1b": ("models/MiniCPM5-1B/MiniCPM5-1B-Q4_K_M.gguf", "minicpm5-1b", "--min-p 0"),
    "minicpm5-2b": ("models/MiniCPM5-2B/MiniCPM5-2B-Q4_K_M.gguf", "minicpm5-2b", "--min-p 0"),
}


class Warnings(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def load_cases(path):
    corpus = path.read_bytes()
    data = json.loads(corpus)
    cases = data.get("cases") if isinstance(data, dict) else data
    if not isinstance(cases, list) or not cases:
        raise ValueError("Decision corpus must be a nonempty list or an object with a cases list")
    labels = {"IGNORE", "ADD", "STOP", "NEW"}
    for index, case in enumerate(cases, 1):
        if not isinstance(case, dict):
            raise ValueError(f"Decision case {index} must be an object")
        if case.get("p") not in ("s", "t"):
            raise ValueError(f"Decision case {index} phase must be 's' or 't'")
        accepted = case.get("y")
        accepted = accepted if isinstance(accepted, list) else [accepted]
        if not accepted or any(not isinstance(label, str) or label not in labels
                               for label in accepted):
            raise ValueError(f"Decision case {index} labels must be IGNORE, ADD, STOP or NEW")
        if any(not isinstance(case.get(field), str) for field in ("r", "h")):
            raise ValueError(f"Decision case {index} request and heard text must be strings")
    return cases, hashlib.sha256(corpus).hexdigest()


def latency(rows, timeout_ms):
    values = sorted(row["wall_ms"] for row in rows)
    return {
        "median_ms": round(statistics.median(values), 2),
        "p95_ms": round(values[math.ceil(0.95 * len(values)) - 1], 2),
        "max_ms": round(max(values), 2),
        "over_budget": sum(value > timeout_ms for value in values),
    }


def summarize(rows, timeout_ms, *, plans=False):
    result = {"cases": len(rows), "fallbacks": sum(r["fallback"] for r in rows),
              **latency(rows, timeout_ms)}
    for quality in ("raw", "effective") if plans else ("raw", "effective", "echo_filtered"):
        result[f"{quality}_correct"] = sum(r[f"{quality}_ok"] for r in rows)
        result[f"{quality}_errors"] = [r["case"] for r in rows if not r[f"{quality}_ok"]]
    if not plans:
        result["would_ask"] = sum(r["would_ask"] for r in rows)
        result["would_ask_when_raw_correct"] = sum(r["would_ask"] and r["raw_ok"] for r in rows)
        result["confident_errors"] = [r["case"] for r in rows if not r["raw_ok"]
                                      and not r["fallback"] and r["confidence"] >= CONFIRM_BELOW]
    return {**result, "rows": rows}


async def server_metadata(http, url, headers):
    """Observed metadata, not an assumption that requested flags apply to --url."""
    root = url.removesuffix("/v1")
    out = {}
    for endpoint in ("/props", "/v1/models"):
        try:
            async with http.get(root + endpoint, headers=headers,
                                timeout=aiohttp.ClientTimeout(total=5)) as response:
                response.raise_for_status()
                data = await response.json()
            if endpoint == "/props":
                # Omit sample prompts/templates: record only relevant server settings.
                out["props"] = {key: data[key] for key in (
                    "total_slots", "n_slots", "model_path", "build_info", "chat_template_kwargs",
                ) if key in data}
                defaults = data.get("default_generation_settings", {})
                out["props"]["generation_settings"] = {
                    key: defaults[key] for key in ("n_ctx", "n_predict", "params") if key in defaults
                }
            else:
                out["models"] = data
        except (aiohttp.ClientError, TimeoutError, ValueError, AttributeError) as exc:
            out[endpoint + "_error"] = f"{type(exc).__name__}: {exc}"
    return out


async def decisions(arbiter, cases, warnings, label):
    rows = []
    for index, case in enumerate(cases, 1):
        warnings.messages.clear()
        situation = Situation(request=case["r"], heard=case["h"], speaking=case["p"] == "s",
                              said=case.get("s", ""), tool=case.get("tool", ""),
                              history=case.get("history", []), ignored=case.get("ignored", []))
        started = time.perf_counter()
        decision = await arbiter.decide(situation)
        wall_ms = (time.perf_counter() - started) * 1000
        accepted = case["y"] if isinstance(case["y"], list) else [case["y"]]
        fallback = "STOP" if is_stop(case["h"]) else ("NEW" if situation.speaking else "ADD")
        effective = decision.label if decision.label is not None else fallback
        echo = situation.speaking and is_echo(case["h"], case.get("s", ""))
        rows.append({**case, "case": index, "pred": decision.label,
                     "raw_ok": decision.label in accepted, "effective_pred": effective,
                     "effective_ok": effective in accepted, "echo_filtered": echo,
                     "echo_filtered_ok": ("IGNORE" if echo else effective) in accepted,
                     "fallback": decision.label is None, "confidence": decision.confidence,
                     "probabilities": decision.probabilities, "would_ask": decision.unsure,
                     "model_ms": decision.ms, "wall_ms": wall_ms,
                     "warnings": warnings.messages.copy()})
        if index % 20 == 0:
            print(f"{label}: decisions {index}/{len(cases)}", flush=True)
    return rows


async def plans(arbiter, cases, warnings):
    rows = []
    for index, (group, case) in enumerate(cases, 1):
        request, said, remaining, heard, accepted = case
        warnings.messages.clear()
        started = time.perf_counter()
        label, model_ms = await arbiter.plan(
            Situation(request=request, heard=heard, speaking=True, said=said), remaining, False)
        wall_ms = (time.perf_counter() - started) * 1000
        fallback = any(message.startswith("Plan unavailable") for message in warnings.messages)
        rows.append({"case": index, "group": group, "request": request, "said": said,
                     "remaining": remaining, "heard": heard, "expected": sorted(accepted),
                     "pred": label, "raw_ok": label in accepted and not fallback,
                     "effective_ok": label in accepted, "fallback": fallback,
                     "wall_ms": wall_ms, "model_ms": model_ms,
                     "warnings": warnings.messages.copy()})
    return rows


async def check(name, args, cases, plan_cases, result, save):
    model, alias, extra = MODELS[name]
    config_path = ROOT / "localive.toml"
    config = load_config(config_path) if config_path.exists() else Config(root=ROOT)
    config.brain.managed = not bool(args.url)
    config.brain.model = model
    config.llm.model = alias
    config.decision.enabled = config.decision.dedicated = True
    config.decision.port = args.port
    config.decision.context = args.context
    config.decision.timeout_ms = args.timeout_ms
    if args.llama_dir:
        config.brain.llama_dir = args.llama_dir
    brain = None
    if args.url:
        config.llm.url = args.url.rstrip("/")
        if not config.llm.url.endswith("/v1"):
            config.llm.url += "/v1"
        url = config.llm.url
    else:
        spec = AltBrainConfig(name="arbiter", model=model, port=args.port, context=args.context,
                              gpu_layers=args.gpu_layers, extra_args=f"{extra} {args.extra_args}")
        brain = Brain(config, spec, "arbiter")
        brain.name = f"arbiter-compare-{name}"  # retain two slots; preserve live arbiter logfile
        result["command"] = brain.command()
        url = brain.url + "/v1"
    result.update({"mode": "existing_server" if args.url else "managed", "url": url,
                   "requested_model_alias": alias, "model_file": model, "repetitions": []})
    collector = Warnings()
    logger = logging.getLogger("localive.decision")
    logger.addHandler(collector)
    try:
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(force_close=True)) as http:
            if brain:
                if await brain.healthy(http):
                    raise RuntimeError(f"Port {args.port} occupied; refusing an unverified model")
                print(f"Starting {name}", flush=True)
                await brain.start(http)
            headers = {"Authorization": f"Bearer {config.llm.api_key}"} if config.llm.api_key else {}
            result["observed_server"] = await server_metadata(http, url, headers)
            arbiter = Arbiter(http, config)
            await warm_up_arbiter(arbiter)
            result["warmup_warnings"] = collector.messages.copy()
            save()
            for repetition in range(1, args.repeat + 1):
                decision_rows = await decisions(arbiter, cases, collector, f"{name} #{repetition}")
                plan_rows = await plans(arbiter, plan_cases, collector)
                item = {"repeat": repetition,
                        "decision": summarize(decision_rows, args.timeout_ms),
                        "plan": summarize(plan_rows, args.timeout_ms, plans=True)}
                result["repetitions"].append(item)
                save()
                print(f"{name} #{repetition}: decision {item['decision']['raw_correct']}/{len(cases)} "
                      f"raw, {item['decision']['fallbacks']} fallbacks, "
                      f"p95 {item['decision']['p95_ms']:.0f} ms; plan "
                      f"{item['plan']['raw_correct']}/{len(plan_cases)} raw, "
                      f"{item['plan']['fallbacks']} fallbacks", flush=True)
    finally:
        if brain:
            brain.stop()
        logger.removeHandler(collector)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS,
                        default=["qwen3", "qwen3.5", "gemma-e4b"])
    parser.add_argument("--url", help="Existing llama.cpp base URL; requires exactly one --models value")
    parser.add_argument("--port", type=int, default=8096)
    parser.add_argument("--context", type=int, default=8192, help="Managed-server context only")
    parser.add_argument("--gpu-layers", type=int, default=99, help="Managed-server GPU layers only")
    parser.add_argument("--llama-dir", help="Override managed llama.cpp binary directory")
    parser.add_argument("--extra-args", default="", help="Additional managed llama-server flags")
    parser.add_argument("--timeout-ms", type=int, default=600)
    parser.add_argument("--repeat", type=int, default=1, help="Repeat corpus after one production warm-up")
    parser.add_argument("--cases", type=Path, default=ROOT / "bench/decision/cases.json",
                        help="Decision corpus JSON: a list or an object with a cases list")
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime/bench/arbiter-models.json")
    args = parser.parse_args()
    if args.repeat < 1 or args.timeout_ms < 1 or args.context < 1 or not 1 <= args.port <= 65535:
        parser.error("repeat, timeout and context must be positive; port must be 1..65535")
    if args.url:
        parts = urlsplit(args.url)
        if len(args.models) != 1 or parts.scheme not in {"http", "https"} or not parts.hostname:
            parser.error("--url needs a valid HTTP(S) URL and exactly one --models value")
        if parts.username or parts.password or parts.query or parts.fragment:
            parser.error("--url must not contain credentials, a query or a fragment")
    return args


async def main(args):
    corpus_path = args.cases.resolve()
    cases, corpus_sha256 = load_cases(corpus_path)
    spec = importlib.util.spec_from_file_location("plan_cases", ROOT / "bench/decision/plan.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan_cases = [("prompt_cases", case) for case in module.CASES]
    plan_cases += [("held_out", case) for case in module.HELD_OUT]
    report = {
        "started_utc": datetime.now(UTC).isoformat(), "timeout_ms": args.timeout_ms,
        "repeat": args.repeat, "corpus_path": str(corpus_path), "corpus_sha256": corpus_sha256,
        "classification_prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
        "confidence_threshold": CONFIRM_BELOW,
        "method": "All cases call production Arbiter. Raw accuracy counts fallback as wrong; "
                  "effective accuracy substitutes production timeout rules. Echo-filtered accuracy "
                  "also applies the upstream echo filter. Confidence-driven ASK is reported "
                  "separately, not counted as a correct action. One production warm-up precedes "
                  "all repetitions; no cache flush occurs between repetitions. Plans use the "
                  "server's actual slots; --url does not modify server settings.",
        "confidence_caveat": "Probabilities are the production first-token label renormalization, "
                             "not calibrated confidence. Tokenization and missing logprobs can "
                             "change them across models; calibrate each model before deployment.",
        "engines": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")

    for name in args.models:
        result = report["engines"][name] = {}
        try:
            await check(name, args, cases, plan_cases, result, save)
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            print(f"{name}: {result['error']}", file=sys.stderr, flush=True)
        finally:
            save()
    print(f"Saved {args.output}", flush=True)
    return int(any("error" in result for result in report["engines"].values()))


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(asyncio.run(main(arguments())))
