"""Time a supplied French transcript until Live delivers a complete function_call.

Compare a cold Qwen prompt on CPU with and without local tool selection.
The in-memory client intercepts calls and never sends them to Cornelius.

    python bench/prompts/live_tool_latency.py
    python bench/prompts/live_tool_latency.py --router-device cpu
    python bench/prompts/live_tool_latency.py --profile full
    python bench/prompts/live_tool_latency.py --profile routed --router-model models/decision/d1-2B-Q8_0.gguf
    python bench/prompts/live_tool_latency.py --profile routed --existing-router aplomb-2 --router-port 18857
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import aiohttp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from localive.config import DEFAULT_BRAIN, AltBrainConfig, Config  # noqa: E402
from localive.engines import Brain  # noqa: E402
from localive.install import LLAMA_TAG  # noqa: E402
from localive.live import LiveSession, Services  # noqa: E402
from localive.llm import ChatClient, Completion  # noqa: E402
from localive.routing import ToolRouter  # noqa: E402

QUESTION = "Quelle heure est-il ?"
FIXTURE = Path(__file__).with_name("cornelius_session.json")


async def turn(http, config, router, fixture, question=QUESTION, details=False, wait_for_idle=False):
    chat = ChatClient(http, config.llm)
    async for _ in chat.stream([{"role": "user", "content": "Bonjour."}], None, Completion(), max_tokens=3):
        pass  # warm weights and kernels, without the Cornelius prefix

    received = asyncio.get_running_loop().create_future()

    async def capture(event):
        if received.done():
            return
        inner = event.get("event", {})
        item = inner.get("item", {})
        if inner.get("type") == "response.output_item.done" and item.get("type") == "function_call":
            received.set_result((time.perf_counter(), item["name"]))
        elif event["type"] == "voice.provider_warning":
            received.set_exception(RuntimeError(str(event["error"])))

    async def finish_without_audio(_sentences):
        # An ordinary spoken answer reaches synthesis only after the LLM has finished.
        if not received.done():
            if details and session._reply_turn:
                diagnostic["assistant_text"] = session._reply_turn.messages[-1].get("content", "")
            received.set_result((time.perf_counter(), None))

    diagnostic = {}
    if details:
        stream = chat.stream

        def observe(messages, tools, *args, **kwargs):
            diagnostic["offered_tools"] = [t["function"]["name"] for t in tools or []]
            diagnostic["system_date_lines"] = [
                line for line in messages[0]["content"].splitlines()
                if line.startswith(("Date du jour,", "Nous sommes le "))
            ]
            return stream(messages, tools, *args, **kwargs)

        chat.stream = observe

    services = Services(config=config, http=http, transcriber=None, chat=chat,
                        tts=None, make_segmenter=None, router=router)
    session = LiveSession(services, capture)
    try:
        session.configure(fixture)
        if router:
            await router.select(session.tools, "Raconte une histoire.")
            if wait_for_idle:
                # The synchronous Aplomb server can keep computing after a client timeout.
                async with http.get(f"http://127.0.0.1:{config.routing.port}/health") as reply:
                    reply.raise_for_status()
        # The clock result is never submitted. Disable the hosted web tool as well,
        # while keeping its schema available to the models.
        with (
            patch("localive.live.search.searxng", side_effect=RuntimeError("Unexpected web search")),
            patch.object(session, "_synthesis_done", finish_without_audio),
        ):
            started = time.perf_counter()
            await session.on_user_text(question)
            ended, name = await asyncio.wait_for(received, 180)
        elapsed_ms = round((ended - started) * 1000)
        return {"routed": router is not None, "question": question, "call": name,
                "selected_tools": session._turn["selected_tools"], "available_tools": len(session.tools),
                "router_ms": round(session._turn.get("tool_routing_ms", 0)),
                "complete_call_ms": elapsed_ms if name else None,
                "completed_without_call_ms": elapsed_ms if name is None else None, **diagnostic}
    finally:
        await session.close("benchmark")


async def compare(config, router_spec, fixture, profile="both", question=QUESTION, details=False,
                  existing_router=None, report=None):
    classifier = None if existing_router else Brain(config, router_spec, "tool-router")
    unexpected_calls = []
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(force_close=True)) as http:
        try:
            routers = []
            if profile != "full":
                if classifier:
                    await classifier.start(http, reuse=False)
                else:
                    async with http.get(f"http://127.0.0.1:{config.routing.port}/health") as reply:
                        reply.raise_for_status()
                routers.append(ToolRouter(http, config.routing))
            if profile != "routed":
                routers.append(None)
            for router in routers:
                brain = Brain(config)
                try:
                    await brain.start(http, reuse=False)  # fresh KV cache for each leg
                    row = await turn(http, config, router, fixture, question, details,
                                     wait_for_idle=bool(existing_router))
                    row["router_model"] = (existing_router or Path(router_spec.model).name) if router else None
                    line = json.dumps(row, ensure_ascii=False)
                    print(line, flush=True)
                    if report is not None:
                        print(line, file=report, flush=True)
                    if row["call"] != "clock__read":
                        unexpected_calls.append(row["call"])
                finally:
                    brain.stop()
        finally:
            if classifier:
                classifier.stop()
    if unexpected_calls:
        raise RuntimeError(f"Expected clock__read, received {unexpected_calls}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--router-device", choices=("cpu", "vulkan"), default="vulkan")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--router-model", type=Path, help="SystemOne GGUF path (defaults to Localive's tool router)")
    source.add_argument("--existing-router", metavar="NAME",
                        help="Use a running SystemOne server on --router-port and record this model name")
    parser.add_argument("--profile", choices=("both", "routed", "full"), default="both",
                        help="Select tool routing, the full catalogue, or both (default)")
    parser.add_argument("--question", default=QUESTION,
                        help="Clock or date question expected to result in clock__read")
    parser.add_argument("--details", action="store_true",
                        help="Record offered schemas' names, system date and any answer made without a tool")
    parser.add_argument("--output", type=Path, help="Write measured turns to a UTF-8 JSONL file")
    parser.add_argument("--timeout-ms", type=int, default=0, help="Router budget (0 uses Localive's adaptive deadline)")
    parser.add_argument("--llama-root", type=Path, default=ROOT / ".runtime/llama")
    parser.add_argument("--port", type=int, default=18091, help="Private Qwen port")
    parser.add_argument("--router-port", type=int, default=18092, help="Private tool selection port")
    args = parser.parse_args()
    ports = (args.port,) if args.profile == "full" else (args.port, args.router_port)
    if len(set(ports)) != len(ports):
        parser.error("Qwen and the router need distinct ports")
    for port in ports:
        if not 1 <= port <= 65535:
            parser.error("Ports must be between 1 and 65535")
        if args.existing_router and port == args.router_port and args.profile != "full":
            continue
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError as error:
                parser.error(f"Local port {port} is unavailable: {error}")
    if args.timeout_ms < 0:
        parser.error("--timeout-ms must be zero (automatic) or positive")
    fixture = json.loads(FIXTURE.read_text("utf-8"))
    binaries = args.llama_root.resolve()
    cpu_args = "--threads 6 --threads-batch 6"
    router_args = "-b 2048 -ub 2048 -fa auto -ctk f16 -ctv f16"
    if args.router_device == "cpu":
        router_args += " " + cpu_args
        router_layers = 0
    else:
        router_layers = 99
    with tempfile.TemporaryDirectory(prefix="localive-live-tools-") as folder:
        config = Config(root=Path(folder))
        config.brain.model = str(ROOT / DEFAULT_BRAIN)
        config.brain.llama_dir = str(binaries / f"{LLAMA_TAG}-cpu")
        config.brain.port, config.brain.context = args.port, 8192
        config.brain.gpu_layers, config.brain.extra_args = 0, cpu_args
        config.llm.url, config.llm.temperature = f"http://127.0.0.1:{args.port}/v1", 0.0
        config.routing.port = args.router_port
        config.routing.timeout_ms = args.timeout_ms
        model = args.router_model.resolve() if args.router_model else ROOT / config.routing.model
        spec = AltBrainConfig(name="tool-router", model=str(model),
                              llama_dir=str(binaries / f"{LLAMA_TAG}-{args.router_device}"),
                              port=args.router_port, context=4096, gpu_layers=router_layers,
                              extra_args=router_args)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") if args.output else nullcontext() as report:
            asyncio.run(compare(config, spec, fixture, args.profile, args.question, args.details,
                                args.existing_router, report))


if __name__ == "__main__":
    main()
