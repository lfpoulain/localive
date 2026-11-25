"""Compare frozen Qwen3.5-9B runtime settings without changing Localive settings.

    .venv\\Scripts\\python.exe bench/optimize_qwen35_9b.py --variants vulkan_24k,cuda_24k
    .venv\\Scripts\\python.exe bench/optimize_qwen35_9b.py --variants cuda_q8_24k,cuda_q8_16k

Stop other model servers first for controlled measurements. This script starts only its
own sequential server on port 8098. All tools are simulated; manual quality review remains
required. Repeating a variant in --variants reruns a control with a separate result/log.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
import socket
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_chat import ADAPTED, CASES, MODELS, ComparisonBrain, med, run_case  # noqa: E402
from tune_qwen import TOOLS, complete, gpu_snapshot  # noqa: E402

from localive.config import AltBrainConfig, load_config  # noqa: E402

MODEL = "qwen3.5-9b"
SAMPLING = {"temperature": 0.2}
KVQ8 = "--flash-attn on --cache-type-k q8_0 --cache-type-v q8_0"
VARIANTS = {
    "vulkan_24k": {"runtime": "vulkan", "context": 24576, "extra_args": ""},
    "cuda_24k": {"runtime": "cuda12", "context": 24576, "extra_args": ""},
    "cuda_fa_8k": {"runtime": "cuda12", "context": 8192, "extra_args": "--flash-attn on"},
    "cuda_q8_8k": {"runtime": "cuda12", "context": 8192, "extra_args": KVQ8},
    "cuda_q8_24k": {"runtime": "cuda12", "context": 24576, "extra_args": KVQ8},
    "cuda_q8_16k": {"runtime": "cuda12", "context": 16384, "extra_args": KVQ8},
    "cuda_threads4_24k": {"runtime": "cuda12", "context": 24576,
                          "extra_args": "--threads 4 --threads-batch 4 --flash-attn on"},
    "vulkan_q8_8k": {"runtime": "vulkan", "context": 8192, "extra_args": KVQ8},
}
DEFAULT_VARIANTS = ",".join(name for name in VARIANTS if name != "vulkan_q8_8k")


def require_free_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"Port {port} is occupied or unavailable; refusing reuse") from exc


class OwnedBrain(ComparisonBrain):
    async def start(self, http):
        require_free_port(self.spec.port)
        await super().start(http)
        if self.process is None:
            raise RuntimeError("Server ownership was not established; refusing reuse")


def gpu_memory(snapshot):
    """Keep the original snapshot; parse only the first GPU's total/used MiB."""
    first = snapshot.splitlines()[0] if snapshot else ""
    match = re.search(r",\s*(\d+) MiB,\s*(\d+) MiB,", first)
    return {"total_mib": int(match[1]), "used_mib": int(match[2])} if match else None


async def post_json(http, url, body):
    async with http.post(url, json=body) as response:
        response.raise_for_status()
        return await response.json()


async def prompt_tokens(http, url, messages):
    # Render the server's own chat template before tokenizing, rather than counting
    # just the system prose. The completion usage below remains the actual truth.
    rendered = await post_json(http, url + "/apply-template", {"messages": messages})
    prompt = rendered["prompt"]
    tokens = await post_json(http, url + "/tokenize", {"content": prompt, "add_special": False})
    return {"count": len(tokens["tokens"]),
            "rendered_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}


def room_code(number):
    # Stable but non-arithmetic mapping: answering cannot be done by adding 1000
    # to the requested room number, as in the older minimal context probe.
    digest = hashlib.sha256(f"localive-9b-context-v1:{number}".encode()).hexdigest()
    return str(100000 + int(digest[:12], 16) % 900000)


def context_messages(count, number):
    facts = [f"Dossier {i}: la salle numéro {i} est bleue, son code est {room_code(i)}, "
             "et ses volets restent fermés pendant la nuit."
             for i in range(1, count + 1)]
    system = ADAPTED + "\nDonnées fournies pour cette conversation :\n" + "\n".join(facts)
    return [{"role": "system", "content": system}, {"role": "user", "content":
             f"Selon les données fournies, quel est le code de la salle numéro {number} ? "
             "Réponds seulement avec les six chiffres du code."}]


async def long_context(http, url, context, target_tokens, seed, result, save):
    # A deterministic binary search chooses a whole number of household facts. This
    # tokenization-only setup performs no generation and is outside measured latency.
    low, high = 1, 400
    while low < high:
        middle = (low + high + 1) // 2
        count = (await prompt_tokens(http, url, context_messages(middle, 1)))["count"]
        if count <= target_tokens:
            low = middle
        else:
            high = middle - 1
    fact_count = low
    if fact_count < 10:
        raise RuntimeError("Long context token target leaves too few facts")
    cases = [("first_use_cold_prefix", 0.15, False),
             ("steady_cold_prefix_1", 0.50, False),
             ("steady_cold_prefix_2", 0.85, False),
             ("reused_prefix_1", 0.32, True),
             ("reused_prefix_2", 0.68, True)]
    result.update(target_rendered_tokens=target_tokens, fact_count=fact_count,
                  max_output_tokens=64, rows=[],
                  token_count_method="Server /apply-template prompt then /tokenize "
                                     "with add_special=false; actual completion usage also recorded.",
                  cache_method="First three requests use cache_prompt=false; last two true. "
                               "Different questions share exactly the same long system prefix. "
                               "Inspect server timings/usage to confirm actual reused token counts.")
    for label, position, cache in cases:
        number = max(1, round(fact_count * position))
        messages = context_messages(fact_count, number)
        tokenization = await prompt_tokens(http, url, messages)
        if tokenization["count"] + 64 > context:
            raise RuntimeError("Long retrieval prompt plus output does not fit selected context")
        reply = await complete(http, url, messages, SAMPLING, seed=seed, model=MODEL,
                               cache=cache, limit=64)
        expected = room_code(number)
        # Keep semantic retrieval separate from strict six-digit-only formatting.
        codes = re.findall(r"(?<!\d)\d{6}(?!\d)", reply["text"])
        factual = bool(codes) and set(codes) == {expected}
        finished = reply["finish_reason"] == "stop" and not reply["reasoning"] and not reply["calls"]
        actual_tokens = reply["usage"].get("prompt_tokens")
        if actual_tokens is not None and actual_tokens + 64 > context:
            raise RuntimeError("Actual prompt usage plus reserved output exceeds context")
        row = {"case": label, "seed": seed, "room": number, "expected_code": expected,
               "cache_prompt": cache, "tokenization": tokenization,
               "actual_prompt_tokens": actual_tokens,
               "factual_code_correct": factual, "six_digits_only": reply["text"].strip() == expected,
               "clean_completion": finished, "objective_ok": factual and finished,
               "manual_review": {"status": "pending", "note":
                   "Check factual reading, contradictions and wording separately from exact formatting."},
               **reply}
        result["rows"].append(row)
        await save()
    rows = result["rows"]
    result["summary"] = {"correct_facts_and_clean_completion": sum(r["objective_ok"] for r in rows),
                         "strict_format_passed": sum(r["six_digits_only"] for r in rows),
                         "cases": len(rows), "first_use_cold_first_event_ms": rows[0]["first_event_ms"],
                         "first_use_cold_total_ms": rows[0]["total_ms"],
                         "steady_cold_median_first_event_ms": med(rows[1:3], "first_event_ms"),
                         "steady_cold_median_total_ms": med(rows[1:3], "total_ms"),
                         "reused_median_first_event_ms": med(rows[3:], "first_event_ms"),
                         "reused_median_total_ms": med(rows[3:], "total_ms")}
    # Store the complete shared fact payload once for later manual audit/reproduction.
    result["system"] = context_messages(fact_count, 1)[0]["content"]


async def run_variant(args, name, run_id, result, save):
    settings = VARIANTS[name]
    config = load_config(ROOT / "localive.toml")
    runtime_dir = args.vulkan_dir if settings["runtime"] == "vulkan" else args.cuda_dir
    spec = AltBrainConfig(name=MODEL, model=MODELS[MODEL], port=args.port,
                          context=settings["context"], gpu_layers=99, llama_dir=runtime_dir,
                          extra_args=settings["extra_args"])
    log_name = f"optimize-qwen35-9b-{run_id}"
    brain = OwnedBrain(config, spec, log_name, reasoning="off")
    result.update(variant=name, settings={**settings, "runtime_dir": runtime_dir}, model=MODEL,
                  gguf=MODELS[MODEL], profile="adapted", system=ADAPTED, sampling=SAMPLING,
                  seeds=args.seeds, max_tokens=400, reasoning="off", url=brain.url,
                  server_log=str(ROOT / ".runtime/logs" / f"brain-{log_name}.log"),
                  rows=[], long_context={}, status="starting")
    try:
        result["command"] = brain.command()
        result["gpu_before"] = await asyncio.to_thread(gpu_snapshot)
        await save()
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(force_close=True),
                                        timeout=aiohttp.ClientTimeout(total=240, sock_read=180)) as http:
            started = time.perf_counter()
            await brain.start(http)
            result["startup_ms"] = (time.perf_counter() - started) * 1000
            async with http.get(brain.url + "/props") as response:
                response.raise_for_status()
                props = await response.json()
            result["props"] = {key: value for key, value in props.items() if "template" not in key}
            if Path(props.get("model_path", "")).name != Path(MODELS[MODEL]).name:
                raise RuntimeError("Loaded model does not match Qwen3.5-9B Q4_K_M")
            if props.get("total_slots") != 1 or props["default_generation_settings"]["n_ctx"] != spec.context:
                raise RuntimeError("Observed slot count/context differs from the requested variant")
            result["gpu_loaded"] = await asyncio.to_thread(gpu_snapshot)
            result["status"] = "running"
            result["warmup"] = await complete(http, brain.url,
                [{"role": "system", "content": ADAPTED}, {"role": "user", "content": "Bonjour"}],
                SAMPLING, seed=args.seeds[0], model=MODEL, tools=TOOLS, limit=64)
            result["gpu_after_warmup"] = await asyncio.to_thread(gpu_snapshot)
            await save()
            for seed in args.seeds:
                for test_case in CASES:
                    row = await run_case(http, brain.url, MODEL, ADAPTED, SAMPLING,
                                         test_case, seed, 400, "off")
                    call_errors = [error for error in row["objective_errors"] if error.startswith(
                        ("invalid tool arguments", "missing expected call", "unexpected or repeated",
                         "tool written as text", "tool loop exceeded"))]
                    row["native_call_contract_ok"] = not call_errors
                    row["native_call_contract_errors"] = call_errors
                    result["rows"].append(row)
                    await save()
                    print(json.dumps({"run": run_id, "case": row["case"], "seed": seed,
                                      "objective_ok": row["objective_ok"],
                                      "native_calls_ok": row["native_call_contract_ok"],
                                      "total_ms": round(row["total_ms"], 1)}, ensure_ascii=False), flush=True)
            result["gpu_after_short_cases"] = await asyncio.to_thread(gpu_snapshot)
            await long_context(http, brain.url, spec.context, args.long_tokens, args.seeds[0],
                               result["long_context"], save)
            result["gpu_after_long_context"] = await asyncio.to_thread(gpu_snapshot)
            rows = result["rows"]
            result["summary"] = {"objective_passed": sum(r["objective_ok"] for r in rows),
                "native_call_contract_passed": sum(r["native_call_contract_ok"] for r in rows),
                "cases": len(rows), "unique_scenarios": len(CASES),
                "fully_reviewed_correct": None, "manual_reviews_pending": len(rows),
                "median_first_event_ms": med(rows, "first_event_ms"),
                "median_first_spoken_text_ms": med(rows, "first_spoken_text_ms"),
                "median_total_ms": med(rows, "total_ms")}
            result["status"] = "completed"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"{run_id}: {result['error']}", flush=True)
    finally:
        brain.stop()
        result["gpu_after_stop"] = await asyncio.to_thread(gpu_snapshot)
        result["gpu_memory_mib"] = {key: gpu_memory(value) for key, value in result.items()
                                    if key.startswith("gpu_") and isinstance(value, str)}
        before = result["gpu_memory_mib"].get("gpu_before")
        loaded = result["gpu_memory_mib"].get("gpu_loaded")
        if before and loaded:
            result["loaded_minus_before_mib"] = loaded["used_mib"] - before["used_mib"]
        await save()


async def main(args):
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    report = {"started_utc": datetime.now(UTC).isoformat(), "model": MODEL,
              "method": "Sequential isolated runtime variants; identical adapted prompt, T=0.2, "
                        "max_tokens=400, seeds 42/123 by default, 18 fixture scenarios. "
                        "Native-call contracts and lexical checks are not full quality scores: "
                        "manual review is pending. Long-context first-use, steady cold and reused "
                        "prefix measurements are separate; retrieval facts and exact format "
                        "are scored separately. Repeated variant names provide later controls.",
              "memory_scope": "nvidia-smi snapshots include all GPU processes, not just this server; "
                              "loaded minus before is approximate incremental use, not peak VRAM.",
              "context_tradeoff": "24576 retains the existing session capacity. 16384 and 8192 "
                                  "reserve less history/tool/instruction space. A successful 6000-token "
                                  "retrieval does not establish equivalent long conversations, "
                                  "20-tool sessions, or production truncation behavior.",
              "scope": "No Localive configuration changes, ASR, TTS, real external tools or "
                       "full-duplex arbiter concurrency. Runtime settings change only owned children.",
              "variants": {}}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    async def save():
        # A complete JSON snapshot after each row; temporary avoids half-written reads.
        temporary = output.with_suffix(output.suffix + ".partial")
        await asyncio.to_thread(temporary.write_text, json.dumps(report, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        temporary.replace(output)

    for ordinal, name in enumerate(args.variants.split(","), 1):
        run_id = f"{stamp}-{ordinal}-{name}"
        key = f"{ordinal:02d}-{name}"
        result = report["variants"][key] = {}
        await run_variant(args, name, run_id, result, save)
    report["finished_utc"] = datetime.now(UTC).isoformat()
    await save()
    print(f"Saved {output}", flush=True)
    return int(any(result["status"] != "completed" for result in report["variants"].values()))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", default=DEFAULT_VARIANTS,
                        help="Comma-separated ordered variants; duplicates rerun a control. Choices: "
                             + ", ".join(VARIANTS))
    parser.add_argument("--vulkan-dir", default=".runtime/llama/b11960-vulkan")
    parser.add_argument("--cuda-dir", default=".runtime/llama/b11960-cuda12")
    parser.add_argument("--port", type=int, default=8098)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123])
    parser.add_argument("--long-tokens", type=int, default=6000,
                        help="Approximate rendered long prompt size, always checked against context")
    parser.add_argument("--output", default=".runtime/bench/qwen35-9b-optimization.json")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if not 1000 <= args.long_tokens <= 7000:
        parser.error("--long-tokens must be between 1000 and 7000 to leave an 8K output margin")
    for name in args.variants.split(","):
        if name not in VARIANTS:
            parser.error(f"unknown variant: {name}")
    return args


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(asyncio.run(main(arguments())))
