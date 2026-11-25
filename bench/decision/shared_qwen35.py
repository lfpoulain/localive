"""Compare one shared 9B server with a dedicated 9B + Qwen3 arbiter pair.

    .venv\\Scripts\\python.exe bench/decision/shared_qwen35.py --output .runtime/bench/shared-9b.json

Owns sequential local servers only; stop the normal app before running. No real tools,
ASR or TTS. Streaming timestamps distinguish requested from observed load overlap.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import statistics
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_models import Warnings, load_cases  # noqa: E402
from optimize_qwen35_9b import prompt_tokens, require_free_port, room_code  # noqa: E402
from plan import CASES as PLAN_CASES  # noqa: E402
from plan import HELD_OUT  # noqa: E402
from tune_qwen import complete, gpu_snapshot  # noqa: E402

from localive.config import AltBrainConfig, load_config  # noqa: E402
from localive.decision import Arbiter, Situation, linking_cue, warm_up_arbiter  # noqa: E402
from localive.engines import Brain  # noqa: E402
from localive.live import is_stop  # noqa: E402

MODEL = "qwen3.5-9b"
MODEL_FILE = "models/Qwen3.5-9B/Qwen3.5-9B-Q4_K_M.gguf"
ARBITER_FILE = "models/Qwen3-4B-Instruct-2507/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
CASE_IDS = (1, 10, 27, 50, 56, 70)  # IGNORE×2, ADD, STOP×2, NEW; no ambiguous labels.
PLANS = (("light_now", PLAN_CASES[4]), ("fahrenheit_fix", HELD_OUT[3]))


class OwnedBrain(Brain):
    async def start(self, http):
        require_free_port(self.spec.port)
        await super().start(http)
        if self.process is None:
            raise RuntimeError("Refusing a server not started by this benchmark")


def load_messages(count):
    facts = "\n".join(f"Salle {i}: code {room_code(i)}, murs bleus, volets fermés la nuit, "
                       "table près de la fenêtre, chaise en bois et tapis au sol."
                       for i in range(1, count + 1))
    return [{"role": "system", "content": "Tu es un écrivain français. Produis un long récit "
             "détaillé, sans résumer ni conclure rapidement. Données de décor :\n" + facts},
            {"role": "user", "content": "Raconte en français une histoire de voyage dans cette "
             "maison, avec vingt scènes détaillées et des dialogues. Écris au moins mille mots."}]


async def make_load(http, url):
    lo, hi = 1, 400
    while lo < hi:
        middle = (lo + hi + 1) // 2
        if (await prompt_tokens(http, url, load_messages(middle)))["count"] <= 6000:
            lo = middle
        else:
            hi = middle - 1
    messages = load_messages(lo)
    return messages, await prompt_tokens(http, url, messages)


async def stream_load(http, url, messages, seed, limit, clock, evidence, first_token):
    evidence["request_start_ms"] = clock()
    body = {"model": MODEL, "messages": messages, "temperature": 0.2, "seed": seed,
            "max_tokens": limit, "cache_prompt": False, "stream": True,
            "stream_options": {"include_usage": True}}
    text, reasoning = "", ""
    try:
        async with http.post(url + "/v1/chat/completions", json=body) as response:
            evidence["response_headers_ms"] = clock()
            response.raise_for_status()
            async for raw in response.content:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                evidence["usage"] = chunk.get("usage") or evidence.get("usage", {})
                evidence["timings"] = chunk.get("timings") or evidence.get("timings", {})
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    content = delta.get("content") or ""
                    reasoning += delta.get("reasoning_content") or delta.get("reasoning") or ""
                    text += content
                    if content.strip() and "first_content_ms" not in evidence:
                        evidence["first_content_ms"] = clock()
                        first_token.set()
                    if choice.get("finish_reason"):
                        evidence["finish_reason"] = choice["finish_reason"]
        if evidence.get("finish_reason") not in ("stop", "length"):
            raise RuntimeError("Main load stream did not finish normally")
    finally:
        evidence.update(completed_ms=clock(), text=text, reasoning=reasoning)
        evidence["total_ms"] = evidence["completed_ms"] - evidence["request_start_ms"]
        evidence["first_content_latency_ms"] = (
            evidence["first_content_ms"] - evidence["request_start_ms"]
            if "first_content_ms" in evidence else None)


def summarize(rows):
    if not rows:
        return {"n": 0}
    values = sorted(row["wall_ms"] for row in rows)
    return {"n": len(rows), "correct_without_fallback": sum(row["correct"] for row in rows),
            "fallbacks": sum(row["fallback"] for row in rows),
            "median_ms": round(statistics.median(values), 2),
            "p95_ms": round(values[math.ceil(.95 * len(values)) - 1], 2),
            "max_ms": round(max(values), 2), "over_600ms": sum(value > 600 for value in values)}


async def trial(main_http, arbiter, warnings, trace_state, url, messages, case, repetition,
                phase, limit, delay_ms, *, plan=False):
    epoch = time.perf_counter()
    clock = lambda: (time.perf_counter() - epoch) * 1000  # noqa: E731
    first_token, main_evidence = asyncio.Event(), {}
    main_task = None
    if phase != "idle":
        main_task = asyncio.create_task(stream_load(main_http, url, messages,
            (42, 123)[repetition - 1], limit, clock, main_evidence, first_token))
    try:
        if phase == "prefill":
            await asyncio.sleep(delay_ms / 1000)
        elif phase == "decode":
            waiter = asyncio.create_task(first_token.wait())
            done, _ = await asyncio.wait((waiter, main_task), timeout=30,
                                         return_when=asyncio.FIRST_COMPLETED)
            if waiter not in done:
                waiter.cancel()
                await asyncio.gather(waiter, return_exceptions=True)
                if main_task.done():
                    await main_task
                raise RuntimeError("No meaningful main content available for decode overlap")
        warnings.messages.clear()
        requests = []
        trace_state.update(events=requests, clock=clock)
        started = clock()
        if plan:
            name, (request, said, remaining, heard, accepted) = case
            if linking_cue(heard) is not None:
                raise ValueError("Plan case would bypass the two-request path")
            situation = Situation(request=request, said=said, heard=heard, speaking=True)
            label, model_ms = await arbiter.plan(situation, remaining, True)
            fallback = any(message.startswith("Plan unavailable") for message in warnings.messages)
            decision = {"label": label, "ms": model_ms}
        else:
            name, spec = case
            situation = Situation(request=spec["r"], heard=spec["h"], speaking=spec["p"] == "s",
                                  said=spec.get("s", ""), tool=spec.get("tool", ""))
            accepted = {spec["y"]}
            value = await arbiter.decide(situation)
            decision, label, fallback = asdict(value), value.label, value.label is None
        ended = clock()
        trace_state["events"] = None
        row = {"kind": "plan" if plan else "decision", "case": name,
               "repetition": repetition, "phase_requested": phase,
               "situation": asdict(situation), "expected": sorted(accepted),
               "decision": decision, "correct": label in accepted and not fallback,
               "fallback": fallback, "wall_ms": ended - started,
               "call_started_ms": started, "call_completed_ms": ended,
               "http_requests": requests, "warnings": warnings.messages.copy()}
        if plan:
            row["remaining"] = remaining
            row["two_requests_started"] = sum(e["event"] == "start" for e in requests) == 2
        else:
            effective = label if label is not None else (
                "STOP" if is_stop(situation.heard) else "NEW" if situation.speaking else "ADD")
            row.update(effective_label=effective, effective_correct=effective in accepted)
        if main_task:
            await asyncio.wait_for(main_task, 60)
            first, last = main_evidence.get("first_content_ms"), main_evidence["completed_ms"]
            prefill = first is not None and main_evidence["request_start_ms"] <= started < first
            decode = first is not None and first <= started < last
            row.update(main_load=main_evidence,
                       phase_at_call_start="prefill_window" if prefill else "decode" if decode else "finished_or_unverified",
                       requested_phase_verified=prefill if phase == "prefill" else decode,
                       decision_entirely_during_generation=first is not None and first <= started <= ended <= last,
                       decision_entirely_before_first_content=first is not None and ended <= first,
                       response_headers_before_call=main_evidence.get("response_headers_ms", float("inf")) <= started,
                       observed_prefill_overlap_ms=max(0, min(ended, first or started) - started) if prefill else 0,
                       observed_decode_overlap_ms=max(0, min(ended, last) - max(started, first or last)))
        else:
            row["requested_phase_verified"] = True
        return row
    finally:
        trace_state["events"] = None
        if main_task and not main_task.done():
            main_task.cancel()
        if main_task:
            await asyncio.gather(main_task, return_exceptions=True)


async def architecture(args, mode, cases, result, save):
    config = load_config(ROOT / "localive.toml")
    config.brain.managed, config.brain.model = True, MODEL_FILE
    config.brain.llama_dir, config.brain.port = ".runtime/llama/b11960-cuda12", 8096
    config.brain.context, config.brain.gpu_layers = 24576, 99
    config.brain.extra_args = "--flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --no-mmproj"
    config.llm.model, config.llm.url, config.llm.api_key = MODEL, "http://127.0.0.1:8096/v1", ""
    config.decision.enabled, config.decision.dedicated = True, mode == "separate"
    config.decision.port, config.decision.context, config.decision.timeout_ms = 8097, 8192, 600
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    main = OwnedBrain(config, name=f"shared9b-{mode}-{stamp}")
    brains = [main]
    if mode == "separate":
        spec = AltBrainConfig(name="arbiter", model=ARBITER_FILE, port=8097, context=8192,
                              gpu_layers=99, llama_dir=".runtime/llama/b11960-vulkan")
        dedicated = OwnedBrain(config, spec, "arbiter")
        dedicated.name = f"shared9b-dedicated-{stamp}"
        brains.append(dedicated)
    warnings, trace_state = Warnings(), {"events": None}
    logger = logging.getLogger("localive.decision")
    logger.addHandler(warnings)
    trace = aiohttp.TraceConfig()

    async def event(kind, session, context, params):
        if trace_state["events"] is not None:
            trace_state["events"].append({"event": kind, "at_ms": trace_state["clock"](),
                                           "url": str(params.url)})

    trace.on_request_start.append(lambda *a: event("start", *a))
    trace.on_request_end.append(lambda *a: event("response_headers", *a))
    trace.on_request_exception.append(lambda *a: event("exception_or_cancel", *a))
    result.update(status="preparing", mode=mode, dedicated=config.decision.dedicated,
                  rows=[], commands=[brain.command() for brain in brains],
                  ports=[8096, 8097], timeout_ms=600, gpu_before=await asyncio.to_thread(gpu_snapshot))
    await save()
    print(f"Starting {mode} architecture", flush=True)
    try:
        for port in (8096, 8097):
            require_free_port(port)
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(force_close=True),
                                         timeout=aiohttp.ClientTimeout(total=90)) as main_http, \
                aiohttp.ClientSession(trace_configs=[trace], connector=aiohttp.TCPConnector(force_close=True)) as decision_http:
            for brain in brains:
                await brain.start(main_http)
            props = []
            for brain in brains:
                async with main_http.get(brain.url + "/props") as response:
                    response.raise_for_status()
                    data = await response.json()
                if Path(data["model_path"]).name != Path(brain.spec.model).name:
                    raise RuntimeError("Loaded model mismatch")
                expected_slots = 2 if brain.serves_arbiter else 1
                if data["total_slots"] != expected_slots or data["default_generation_settings"]["n_ctx"] != brain.spec.context:
                    raise RuntimeError("Loaded slots/context mismatch")
                props.append({key: value for key, value in data.items() if "template" not in key})
            result["props"] = props
            arbiter = Arbiter(decision_http, config)
            await warm_up_arbiter(arbiter)
            result["arbiter_warmup_warnings"] = warnings.messages.copy()
            result["main_warmup"] = await complete(main_http, main.url,
                [{"role": "user", "content": "Dis bonjour."}], {"temperature": 0.2},
                seed=42, model=MODEL, limit=32)
            messages, tokenization = await make_load(main_http, main.url)
            result.update(load_messages=messages, load_tokenization=tokenization,
                          load_sampling={"temperature": 0.2, "seeds": [42, 123],
                                         "max_tokens": args.load_max_tokens, "cache_prompt": False},
                          gpu_warm=await asyncio.to_thread(gpu_snapshot), status="running")
            await save()
            for phase in ("idle", "prefill", "decode"):
                for repetition in (1, 2):
                    for case in cases:
                        row = await trial(main_http, arbiter, warnings, trace_state, main.url,
                            messages, case, repetition, phase, args.load_max_tokens, args.prefill_delay_ms)
                        result["rows"].append(row)
                        await save()
            for repetition in (1, 2):
                for case in PLANS:
                    row = await trial(main_http, arbiter, warnings, trace_state, main.url,
                        messages, case, repetition, "decode", args.load_max_tokens,
                        args.prefill_delay_ms, plan=True)
                    result["rows"].append(row)
                    await save()
            result["gpu_after_load"] = await asyncio.to_thread(gpu_snapshot)
            result["summary"] = {}
            for kind, phase in (("decision", "idle"), ("decision", "prefill"),
                                ("decision", "decode"), ("plan", "decode")):
                selected = [row for row in result["rows"] if row["kind"] == kind and row["phase_requested"] == phase]
                result["summary"][f"{kind}_{phase}"] = {
                    "requested": summarize(selected),
                    "verified_phase_at_dispatch": summarize([r for r in selected if r["requested_phase_verified"]]),
                    "entire_call_during_generation": sum(r.get("decision_entirely_during_generation", False) for r in selected)}
            result["status"] = "completed"
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        for brain in reversed(brains):
            brain.stop()
        logger.removeHandler(warnings)
        result["gpu_after_stop"] = await asyncio.to_thread(gpu_snapshot)
        await save()


async def main(args):
    corpus, sha = load_cases(Path(__file__).with_name("cases.json"))
    cases = [(index, corpus[index - 1]) for index in CASE_IDS]
    for _, plan_case in PLANS:
        assert linking_cue(plan_case[3]) is None
    report = {"started_utc": datetime.now(UTC).isoformat(), "corpus_sha256": sha,
              "decision_case_ids": CASE_IDS, "repetitions": 2,
              "method": "Identical fixed classification inputs and identical cache_prompt=false "
                        "main load across shared/separate architectures. Real Brain and Arbiter; "
                        "dedicated selects the production URL path. Main load is an unrelated "
                        "synthetic writer task, not a coherent Live conversation. No decisions "
                        "are acted upon: the main load continues to measure contention.",
              "limits": ["Six simple classification cases cover all four labels but are not a broad quality test.",
                         "Prefill dispatch is requested 20 ms after scheduling the HTTP call by default. "
                         "Observed timestamps must confirm dispatch before first nonempty content. "
                         "This interval includes HTTP/queue time; it is not a GPU kernel trace.",
                         "Decode dispatch waits for actual non-whitespace content, not a role event. "
                         "Full-call overlap is separately reported; phase-at-dispatch alone is insufficient.",
                         "Two plan cases without linking cues exercise NOW/STILL requests concurrently; "
                         "main plus plan may require three requests with only two shared slots.",
                         "nvidia-smi values are background-inclusive snapshots, not peak VRAM. "
                         "No acoustic, ASR/TTS, true spoken interruption or production cancellation test.",
                         "Main output hitting its token cap is intentional workload, not a quality failure."],
              "architectures": {}}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    async def save():
        temporary = output.with_suffix(output.suffix + ".partial")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output)

    await save()
    for mode in args.architectures.split(","):
        result = report["architectures"][mode] = {}
        await architecture(args, mode, cases, result, save)
        print(json.dumps({"mode": mode, "status": result["status"], "summary": result.get("summary"),
                          "error": result.get("error")}, ensure_ascii=False), flush=True)
    report["finished_utc"] = datetime.now(UTC).isoformat()
    await save()
    print(f"Saved {output}", flush=True)
    return int(any(item["status"] != "completed" for item in report["architectures"].values()))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=".runtime/bench/shared-qwen35-9b.json")
    parser.add_argument("--architectures", default="shared,separate",
                        help="Ordered shared,separate architectures; permits a focused retry")
    parser.add_argument("--load-max-tokens", type=int, default=256)
    parser.add_argument("--prefill-delay-ms", type=float, default=20)
    args = parser.parse_args()
    selected = args.architectures.split(",")
    if any(mode not in ("shared", "separate") for mode in selected) or len(set(selected)) != len(selected):
        parser.error("--architectures must contain shared and/or separate, without duplicates")
    if not 64 <= args.load_max_tokens <= 1024:
        parser.error("--load-max-tokens must be between 64 and 1024")
    if not 0 <= args.prefill_delay_ms <= 1000:
        parser.error("--prefill-delay-ms must be between 0 and 1000")
    return args


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(asyncio.run(main(arguments())))
