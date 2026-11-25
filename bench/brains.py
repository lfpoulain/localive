"""Compare the brains served by Localive (Qwen vs Gemma…) on what matters for the voice:
tool use, answer length, French, time to first token / first sentence.

Uses Localive's real system prompt with Cornelius-like tools, straight against each
brain's llama-server (no audio). Needs Localive running.

    .venv\\Scripts\\python.exe bench\\brains.py
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from localive.config import load_config  # noqa: E402
from localive.live import LOCAL_INSTRUCTIONS, chat_tools  # noqa: E402
from localive.llm import text_tool_call  # noqa: E402

TOOLS = chat_tools([
    {"type": "function", "name": "clock__read", "description": "Lire l'heure et la date.",
     "parameters": {"type": "object", "properties": {}}},
    {"type": "function", "name": "weather__read", "description": "Météo d'une ville.",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                    "day": {"type": "string"}}, "required": ["city"]}},
    {"type": "function", "name": "lights__set", "description": "Allumer/éteindre une lumière.",
     "parameters": {"type": "object", "properties": {"room": {"type": "string"},
                    "on": {"type": "boolean"}}, "required": ["room", "on"]}},
    {"type": "function", "name": "timer__start", "description": "Lancer un minuteur.",
     "parameters": {"type": "object", "properties": {"seconds": {"type": "integer"},
                    "label": {"type": "string"}}, "required": ["seconds"]}},
])  # fmt: skip
SYSTEM = "Tu es Cornelius, le miroir de la maison.\n\n" + LOCAL_INSTRUCTIONS
CASES = [
    # (question, expected tool or None, check on the final text)
    ("Quelle heure est-il ?", "clock__read", None),
    ("Quel temps fera-t-il demain à Lyon ?", "weather__read", "lyon"),
    ("Allume la lumière de la cuisine.", "lights__set", None),
    ("Mets un minuteur de douze minutes pour les pâtes.", "timer__start", None),
    ("Quelle est la capitale de l'Australie ?", None, "canberra"),
    ("Combien font 37 fois 12 ?", None, "444"),
    ("Raconte-moi une blague courte.", None, None),
    ("Explique-moi en deux phrases pourquoi le ciel est bleu.", None, None),
    ("Tu peux me faire une liste de cinq fruits ?", None, None),
    ("Merci, c'est tout.", None, None),
]
RESULTS = {
    "clock__read": {"time": "17:42", "date": "jeudi 5 novembre 2026"},
    "weather__read": {"temperature_c": 11, "conditions": "nuageux, averses le soir"},
    "lights__set": {"ok": True},
    "timer__start": {"ok": True, "seconds": 720},
}


async def ask(http, url, model, question):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    tool, text, first_token = None, "", None
    started = time.perf_counter()
    for _ in range(3):
        body = {"model": model, "messages": messages, "tools": TOOLS, "temperature": 0.2,
                "max_tokens": 300, "stream": True, "cache_prompt": True}  # fmt: skip
        calls, content = {}, ""
        async with http.post(url + "/chat/completions", json=body) as response:
            async for raw in response.content:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line.endswith("[DONE]"):
                    continue
                delta = json.loads(line[5:])["choices"][0].get("delta") or {}
                if (delta.get("content") or delta.get("tool_calls")) and first_token is None:
                    first_token = time.perf_counter() - started
                content += delta.get("content") or ""
                for item in delta.get("tool_calls") or []:
                    call = calls.setdefault(item.get("index", 0), {"id": item.get("id") or "c",
                                                                   "name": "", "args": ""})  # fmt: skip
                    fn = item.get("function") or {}
                    call["name"] += fn.get("name") or ""
                    call["args"] += fn.get("arguments") or ""
        names = {t["function"]["name"] for t in TOOLS}
        if not calls and (recovered := text_tool_call(content, names)):  # as Localive does
            calls = {0: {"id": "c", "name": recovered.name, "args": recovered.arguments}}
            content = ""
        text += content
        if not calls:
            break
        call = calls[0]
        tool = tool or call["name"]
        messages.append({"role": "assistant", "content": content, "tool_calls": [
            {"id": call["id"], "type": "function",
             "function": {"name": call["name"], "arguments": call["args"] or "{}"}}]})  # fmt: skip
        messages.append({"role": "tool", "tool_call_id": call["id"],
                         "content": json.dumps(RESULTS.get(call["name"], {"ok": True}))})  # fmt: skip
    return tool, text.strip(), first_token or 0.0, time.perf_counter() - started


async def main() -> None:
    brains = load_config().brains()
    connector = aiohttp.TCPConnector(force_close=True)  # llama-server drops idle sockets
    async with aiohttp.ClientSession(connector=connector) as http:
        for name, url in brains.items():
            await ask(http, url, name, "Bonjour")  # warm-up (prompt cache)
            ok, firsts, totals, lengths = 0, [], [], []
            print(f"\n===== {name}")
            for question, want_tool, must in CASES:
                tool, text, first, total = await ask(http, url, name, question)
                good = tool == want_tool and (must is None or must in text.lower())
                markdown = any(m in text for m in ("**", "#", "- ", "\n\n"))
                ok += good
                firsts.append(first)
                totals.append(total)
                lengths.append(len(text))
                flags = " [markdown]" if markdown else ""
                print(f"{'OK' if good else '!!'} {first * 1000:5.0f} ms 1er mot, {total:4.1f} s "
                      f"total, outil={tool or '-':14s} {question}\n      → {text[:160]}{flags}")  # fmt: skip
            print(f"{name}: {ok}/{len(CASES)} justes · 1er mot médiane "
                  f"{statistics.median(firsts) * 1000:.0f} ms · réponse médiane "
                  f"{statistics.median(lengths):.0f} caractères")  # fmt: skip


if __name__ == "__main__":
    asyncio.run(main())
