"""Measure the effect of tool schemas on Qwen's first text or tool event.

Runs an isolated llama-server with the Cornelius session fixture. The selected sets
are supplied by the benchmark, so the timings exclude tool selection, audio and tool
execution. Each profile starts with a cold prompt and warm model weights, then allows
prefix reuse for a paraphrase about the clock followed by two changes of topic.
Requests contain the same system instructions and only the current user message,
so changes in conversation history cannot affect the comparison.

    python bench/prompts/tool_prefill.py --llama-dir .runtime/llama/b11960-cpu --gpu-layers 0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import sys
import tempfile
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT / "src"))

from tune_qwen import complete  # noqa: E402

from localive.config import DEFAULT_BRAIN, Config  # noqa: E402
from localive.engines import Brain  # noqa: E402
from localive.live import LiveSession, Services, add_note  # noqa: E402

CASES = [
    ("froid", "Quelle heure est-il ?", "clock__read"),
    ("chaud", "Peux-tu me donner l'heure ?", "clock__read"),
    ("changement", "Quel temps fera-t-il demain à Lyon ?", "weather__read"),
    ("changement", "Merci, bonne journée !", None),
]


async def measure(args):
    with tempfile.TemporaryDirectory(prefix="localive-prefill-") as folder:
        config = Config(root=Path(folder))
        config.brain.model = str(args.model)
        config.brain.llama_dir = str(args.llama_dir)
        config.brain.port = args.port
        config.brain.context = 8192
        config.brain.gpu_layers = args.gpu_layers
        config.brain.extra_args = f"--threads {args.threads} --threads-batch {args.threads}"
        brain = Brain(config, name="prefill")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.port))
        connector = aiohttp.TCPConnector(force_close=True)
        async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=300)) as http:
            try:
                await brain.start(http, reuse=False)
                session = LiveSession(Services(config, http, None, None, None, None), None)
                fixture = Path(__file__).with_name("cornelius_session.json")
                session.configure(json.loads(fixture.read_text(encoding="utf-8")))
                # Compile kernels without populating the Cornelius prefix.
                await complete(http, brain.url, [{"role": "user", "content": "Bonjour."}],
                               {"temperature": 0}, seed=42, limit=3)
                cases = CASES[:2] if args.clock_only else CASES
                profiles = ["catalogue", "sélection"]
                if args.selected_first:
                    profiles.reverse()
                for profile in profiles:
                    for phase, question, expected in cases:
                        tools = session.tools
                        if profile == "sélection":
                            tools = [t for t in tools if t["function"]["name"] == expected]
                        messages = [{"role": "system", "content": session.system_prompt()},
                                    {"role": "user", "content": question}]
                        if note := session.context_note():
                            add_note(messages, note, label="Contexte")
                        result = await complete(http, brain.url, messages, {"temperature": 0},
                                                seed=42, tools=tools, limit=80, cache=phase != "froid",
                                                model=config.llm.model)
                        first = result["first_event_ms"]
                        print(json.dumps({
                            "profile": profile, "phase": phase, "question": question,
                            "tool_count": len(tools), "tokens": result["usage"].get("prompt_tokens"),
                            "prefilled": result["timings"].get("prompt_n"),
                            "first_event_ms": round(first) if first is not None else None,
                            "calls": [c["name"] for c in result["calls"]], "text": result["text"],
                        }, ensure_ascii=False), flush=True)
            finally:
                brain.stop()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(ROOT / DEFAULT_BRAIN))
    parser.add_argument("--llama-dir", default=str(ROOT / ".runtime/llama/b11960-vulkan"))
    parser.add_argument("--port", type=int, default=18091)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--clock-only", action="store_true", help="Cold clock and warm paraphrase only.")
    parser.add_argument("--selected-first", action="store_true",
                        help="Measure the selected schemas first to check order effects.")
    args = parser.parse_args()
    args.model = Path(args.model).resolve()
    args.llama_dir = Path(args.llama_dir).resolve()
    asyncio.run(measure(args))
