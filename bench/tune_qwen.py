"""Measure Qwen server tuning without executing tools or changing Localive settings.

Stop other model servers first: GPU memory pressure can dominate these measurements.
The two rounds are deterministic repeats, not independent evidence of general quality.

    .venv\\Scripts\\python.exe bench\\tune_qwen.py --variants baseline,threads4,kvq8
    .venv\\Scripts\\python.exe bench\\tune_qwen.py --llama-dir .runtime/llama/b11960-cuda
    .venv\\Scripts\\python.exe bench\\tune_qwen.py --sampling baseline,official
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import statistics
import subprocess
import sys
import time
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from brains import RESULTS, SYSTEM, TOOLS  # noqa: E402

from localive.config import AltBrainConfig, load_config  # noqa: E402
from localive.engines import Brain  # noqa: E402

VARIANTS = {
    "baseline": "",
    "threads4": "--threads 4 --threads-batch 4",
    "batch1024": "--batch-size 1024 --ubatch-size 1024",
    "kvq8": "--flash-attn on --cache-type-k q8_0 --cache-type-v q8_0",
    "checkpoints1024": "--checkpoint-min-step 1024",
    "combined": "--threads 4 --threads-batch 4 --flash-attn on",
    "combinedq8": "--threads 4 --threads-batch 4 --flash-attn on "
                  "--cache-type-k q8_0 --cache-type-v q8_0",
}
SAMPLING = {
    "baseline": {"temperature": 0.2},
    # Qwen's general non-thinking recommendation; evaluated separately from
    # the production low-temperature voice profile and never used for the arbiter.
    "official": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
                 "presence_penalty": 1.5, "repeat_penalty": 1.0},
}
CASES = [
    ("Quelle heure est-il ?", "clock__read", {}, None),
    ("Quel temps fera-t-il demain à Lyon ?", "weather__read", {"city": "lyon"}, None),
    ("Allume la lumière de la cuisine.", "lights__set", {"room": "cuisine", "on": True}, None),
    ("Mets un minuteur de douze minutes pour les pâtes.", "timer__start", {"seconds": 720}, None),
    ("Quelle est la capitale de l'Australie ?", None, {}, "canberra"),
    ("Combien font 37 fois 12 ?", None, {}, "444"),
]


def gpu_snapshot():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        return result.stdout.strip() or result.stderr.strip()
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"


async def complete(http, url, messages, sampling, *, seed, tools=None, cache=True, limit=180,
                   model="qwen3.5"):
    body = {"model": model, "messages": messages, "stream": True,
            "stream_options": {"include_usage": True}, "max_tokens": limit,
            "cache_prompt": cache, "seed": seed, **sampling}
    if tools:
        body.update(tools=tools, tool_choice="auto")
    started = time.perf_counter()
    first = first_text = None
    calls, text, reasoning = {}, "", ""
    usage, timings, finish, done = {}, {}, None, False
    async with http.post(url + "/v1/chat/completions", json=body) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {(await response.text())[:500]}")
        async for raw in response.content:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            if line[5:].strip() == "[DONE]":
                done = True
                break
            chunk = json.loads(line[5:])
            if chunk.get("error"):
                raise RuntimeError(str(chunk["error"]))
            usage = chunk.get("usage") or usage
            timings = chunk.get("timings") or timings
            for choice in chunk.get("choices") or []:
                finish = choice.get("finish_reason") or finish
                delta = choice.get("delta") or {}
                content = delta.get("content") or ""
                meaningful = content or any(
                    item.get("function", {}).get("name")
                    or item.get("function", {}).get("arguments")
                    for item in delta.get("tool_calls") or []
                )
                elapsed = (time.perf_counter() - started) * 1000
                if meaningful and first is None:
                    first = elapsed
                if content and first_text is None:
                    first_text = elapsed
                text += content
                reasoning += delta.get("reasoning_content") or delta.get("reasoning") or ""
                for item in delta.get("tool_calls") or []:
                    call = calls.setdefault(item.get("index", 0),
                                            {"id": "", "name": "", "arguments": ""})
                    call["id"] = item.get("id") or call["id"]
                    function = item.get("function") or {}
                    call["name"] += function.get("name") or ""
                    call["arguments"] += function.get("arguments") or ""
    if not done and finish is None:
        raise RuntimeError("Incomplete SSE response")
    return {"text": text, "reasoning": reasoning, "calls": list(calls.values()),
            "finish_reason": finish, "first_event_ms": first, "first_text_ms": first_text,
            "total_ms": (time.perf_counter() - started) * 1000,
            "usage": usage, "timings": timings}


def normalized(text):
    return "".join(c for c in unicodedata.normalize("NFKD", text.casefold())
                   if not unicodedata.combining(c))


def fixture_reply_checks(tool, text):
    """Conservative lexical checks, not a general semantic correctness judge.

    A generic 'done' may be correct but is scored as unverifiable. Keep the raw
    response so a reviewer can distinguish omissions from factual contradictions.
    """
    plain = normalized(text)
    checks = {}
    if tool == "clock__read":
        checks["time_17_42"] = bool(re.search(r"\b17\s*(?::|h(?:eures?)?)\s*42\b", plain))
        checks["no_other_numeric_time"] = all(
            hour == "17" and minute == "42"
            for hour, minute in re.findall(r"\b(\d{1,2})\s*(?::|h(?:eures?)?)\s*(\d{2})\b", plain)
        )
    elif tool == "weather__read":
        checks["temperature_11"] = bool(re.search(r"\b(?:11|onze)\b", plain))
        checks["cloudy"] = "nuage" in plain and not re.search(
            r"\b(?:sans|aucun|pas de)\s+nuage", plain
        )
        checks["no_other_numeric_temperature"] = all(
            value == "11" for value in re.findall(r"\b(\d+)\s*(?:°|degre)", plain)
        )
    elif tool == "lights__set":
        checks["kitchen"] = "cuisine" in plain
        checks["light_on"] = "allum" in plain and "etein" not in plain
    elif tool == "timer__start":
        checks["timer"] = "minuteur" in plain
        checks["started"] = any(word in plain for word in ("lance", "active", "demarre", "en route"))
        durations = re.findall(r"\b(\d+)\s*min", plain)
        checks["no_wrong_duration"] = all(value == "12" for value in durations)
    if tool in {"lights__set", "timer__start"}:
        checks["not_denied_or_deferred"] = not re.search(
            r"\b(?:impossible|echec|echoue|pas|ne|vais|voudrais|pourrais)\b|n['’]", plain
        )
    return {key: bool(value) for key, value in checks.items()}


async def quality_case(http, url, sampling, case, seed):
    question, expected_tool, expected_args, expected_text = case
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    first = await complete(http, url, messages, sampling, seed=seed, tools=TOOLS)
    calls = first["calls"]
    ok = not calls if expected_tool is None else (
        len(calls) == 1 and calls[0]["name"] == expected_tool
    )
    errors = []
    if not ok:
        errors.append("unexpected tool selection")
    try:
        parsed = [json.loads(c["arguments"] or "{}") for c in calls]
        if any(not isinstance(arguments, dict) for arguments in parsed):
            raise ValueError("tool arguments must be JSON objects")
        if calls and expected_tool:
            actual = parsed[0]
            for key, value in expected_args.items():
                found = actual.get(key)
                found = found.casefold() if isinstance(found, str) else found
                if found != value or (isinstance(value, bool) and type(found) is not bool):
                    errors.append(f"wrong argument {key}: {found!r}")
    except (ValueError, AttributeError) as exc:
        errors.append(f"invalid arguments: {exc}")
    if expected_text and expected_text not in first["text"].casefold():
        errors.append("incorrect factual answer")
    requests = [first]
    reply_checks = {}
    # Return fixture results only for a valid, completed expected call. No real
    # clock, light, weather service or timer is ever invoked by this benchmark.
    if calls and not errors and first["finish_reason"] in {"tool_calls", "stop"}:
        messages.append({"role": "assistant", "content": first["text"], "tool_calls": [
            {"id": call["id"], "type": "function", "function": {
                "name": call["name"], "arguments": call["arguments"]}} for call in calls]})
        for call in calls:
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": json.dumps(RESULTS[call["name"]])})
        follow = await complete(http, url, messages, sampling, seed=seed, tools=TOOLS)
        requests.append(follow)
        if not follow["text"].strip() or follow["calls"]:
            errors.append("missing spoken answer after tool result")
        reply_checks = fixture_reply_checks(expected_tool, follow["text"])
        for key, passed in reply_checks.items():
            if not passed:
                errors.append(f"fixture reply unverified or contradictory: {key}")
    for request in requests:
        if request["finish_reason"] not in {"stop", "tool_calls"}:
            errors.append("truncated response")
        if request["reasoning"] or "<think>" in request["text"]:
            errors.append("reasoning was enabled")
    first_text_ms = first["first_text_ms"]
    if first_text_ms is None and len(requests) == 2 and requests[1]["first_text_ms"] is not None:
        first_text_ms = first["total_ms"] + requests[1]["first_text_ms"]
    return {"question": question, "ok": not errors, "errors": errors,
            "fixture_reply_checks": reply_checks,
            "first_event_ms": first["first_event_ms"], "first_spoken_text_ms": first_text_ms,
            "total_ms": sum(r["total_ms"] for r in requests), "requests": requests}


def median(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    return round(statistics.median(values), 2) if values else None


async def long_context(http, url, sampling, seed):
    # Synthetic household facts approximate a large session instruction payload.
    lines = [f"Dossier {i}: la salle numéro {i} est bleue, son code est {1000 + i}, "
             "et les volets restent fermés pendant la nuit." for i in range(1, 101)]
    system = SYSTEM + "\nInformations de la maison :\n" + "\n".join(lines)
    async with http.post(url + "/tokenize", json={"content": system}) as response:
        response.raise_for_status()
        token_count = len((await response.json())["tokens"])
    rows = []
    # The first large prefill can compile new GPU kernels despite the tiny
    # greeting warmup. Keep it separate from steady uncached prefill timings.
    for label, number, cache in [("first_use_cold_prefix", 17, False),
                                ("steady_cold_prefix_1", 18, False),
                                ("steady_cold_prefix_2", 19, False),
                                ("reused_prefix_1", 20, True),
                                ("reused_prefix_2", 21, True)]:
        messages = [{"role": "system", "content": system}, {"role": "user", "content":
                     f"Quel est le code de la salle numéro {number} ? Réponds seulement le code."}]
        row = await complete(http, url, messages, sampling, seed=seed, cache=cache, limit=32)
        answer = re.sub(r"\s+", "", row["text"]).strip('`\"\'«»').rstrip(".")
        ok = (answer == str(1000 + number) and row["finish_reason"] == "stop"
              and not row["reasoning"] and not row["calls"])
        rows.append({"case": label, "ok": ok, **row})
    return {"system_tokens": token_count, "rows": rows}


async def variant(args, name):
    config = load_config(ROOT / "localive.toml")
    if args.llama_dir:
        config.brain.llama_dir = args.llama_dir
    spec = AltBrainConfig(name="qwen3.5", model=args.model, port=args.port,
                          context=args.context, gpu_layers=99,
                          extra_args=" ".join([VARIANTS[name], args.extra_args]).strip())
    log_name = f"tune-qwen-{Path(config.brain.llama_dir).name}-{name}"
    brain = Brain(config, spec, log_name)
    result = {"variant": name, "command": brain.command(), "gpu_before": gpu_snapshot(),
              "sampling": {}}
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(force_close=True),
                                    timeout=aiohttp.ClientTimeout(total=120, sock_read=60)) as http:
        if await brain.healthy(http):
            raise RuntimeError(f"Port {args.port} is occupied; refusing to reuse another server")
        try:
            await brain.start(http)
            result["gpu_loaded"] = gpu_snapshot()
            async with http.get(brain.url + "/props") as response:
                response.raise_for_status()
                props = await response.json()
                # Do not store the multi-KB Jinja template: parameters are the useful evidence.
                result["props"] = {key: value for key, value in props.items()
                                   if "template" not in key}
            for sampling_name in args.sampling.split(","):
                sampling = SAMPLING[sampling_name]
                warmup = await complete(http, brain.url,
                                        [{"role": "user", "content": "Dis bonjour."}],
                                        sampling, seed=args.seed, limit=32)
                context = await long_context(http, brain.url, sampling, args.seed)
                rows = []
                for round_id in range(1, args.rounds + 1):
                    # Warm the actual voice/tool prefix after the different long prompt.
                    await complete(http, brain.url,
                                   [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": "Bonjour"}],
                                   sampling, seed=args.seed, tools=TOOLS, limit=32)
                    for case in CASES:
                        row = await quality_case(http, brain.url, sampling, case, args.seed)
                        rows.append({"round": round_id, **row})
                summary = {"correct": sum(r["ok"] for r in rows), "cases": len(rows),
                           "median_first_event_ms": median(rows, "first_event_ms"),
                           "median_first_spoken_text_ms": median(rows, "first_spoken_text_ms"),
                           "median_total_ms": median(rows, "total_ms"),
                           "long_context_tokens": context["system_tokens"],
                           "long_context_first_ms": [r["first_event_ms"] for r in context["rows"]],
                           "long_context_first_use_ms": context["rows"][0]["first_event_ms"],
                           "long_context_steady_cold_median_ms": median(context["rows"][1:3], "first_event_ms"),
                           "long_context_reused_median_ms": median(context["rows"][3:], "first_event_ms"),
                           "long_context_correct": sum(r["ok"] for r in context["rows"])}
                result["sampling"][sampling_name] = {
                    "parameters": sampling, "warmup": warmup, "summary": summary,
                    "long_context": context, "rows": rows,
                }
                print(json.dumps({"variant": name, "sampling": sampling_name, **summary}), flush=True)
        finally:
            brain.stop()
    return result


async def main(args):
    report = {"date_utc": datetime.now(UTC).isoformat(), "seed": args.seed,
              "rounds": args.rounds, "context": args.context,
              "method": "Sequential isolated servers; simulated tools; fixed seed repeats. "
                        "Long first-use/steady cold/reused prefix timings are separate. "
                        "Tool reply quality uses conservative lexical fixture checks; generic "
                        "confirmations may be correct but unverifiable and are scored as failures. "
                        "No ASR, TTS, network tool latency, or full duplex concurrency included.",
              "variants": {}}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for name in args.variants.split(","):
        try:
            report["variants"][name] = await variant(args, name)
        except Exception as exc:
            report["variants"][name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"{name}: {exc}", flush=True)
        await asyncio.to_thread(output.write_text,
                                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {output}", flush=True)
    return 1 if any("error" in result for result in report["variants"].values()) else 0


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-dir")
    parser.add_argument("--model", default="models/Qwen3.5-4B/Qwen3.5-4B-Q4_K_M.gguf")
    parser.add_argument("--variants", default="baseline")
    parser.add_argument("--extra-args", default="")
    parser.add_argument("--sampling", default="baseline")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument("--context", type=int, default=24576)
    parser.add_argument("--output", default=".runtime/bench/qwen-tuning.json")
    args = parser.parse_args()
    for name in args.variants.split(","):
        if name not in VARIANTS:
            parser.error(f"unknown variant {name!r}; choose {', '.join(VARIANTS)}")
    for name in args.sampling.split(","):
        if name not in SAMPLING:
            parser.error(f"unknown sampling profile {name!r}")
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(asyncio.run(main(arguments())))
