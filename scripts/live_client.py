"""Send speech from WAV files to a Live WebSocket and save the spoken reply.

The default session speaks with the supplied audio and declares no tools. The
Cornelius profile provides demonstration tools for the project's benchmark.

    python scripts/live_client.py --save reply.wav tests/fixtures/wake_calculation.wav
    python scripts/live_client.py --profile cornelius tests/fixtures/wake_clock.wav
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import time
import wave
from datetime import datetime
from pathlib import Path

import aiohttp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

TOOLS = [
    {
        "type": "function",
        "name": "clock__read",
        "description": "Lire l'heure et la date actuelles.",
        "strict": True,
        "parameters": {"type": "object", "properties": {}, "required": [],
                       "additionalProperties": False},
    },
    {
        "type": "function",
        "name": "weather__read",
        "description": "Météo actuelle et prévisions d'une ville.",
        "strict": True,
        "parameters": {"type": "object", "properties": {"city": {"type": ["string", "null"]}},
                       "required": ["city"], "additionalProperties": False},
    },
    {
        "type": "function",
        "name": "home_assistant__control",
        "description": "Piloter un appareil de la maison (lumière, chauffage, volets).",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"entity": {"type": "string"}, "action": {"type": "string"},
                           "value": {"type": ["number", "null"]}},
            "required": ["entity", "action", "value"], "additionalProperties": False,
        },
    },
]  # fmt: skip
SESSION = {
    "model": "gpt-live-1",
    "store": False,
    "audio": {"output": {"voice": "marin"}, "format": {"type": "audio/pcm", "rate": 16000}},
    "instructions": "Tu es Cornelius, le miroir de la maison. Parle français, avec chaleur et "
    "concision.",
    "delegation": {
        "type": "responses",
        "responses": {
            "instructions": "Tu aides Cornelius. Utilise les outils pour l'heure, la météo et "
            "la maison.",
            "tools": [{"type": "web_search"}, *TOOLS],
        },
    },
}


def load_16k(path: Path) -> np.ndarray:
    """Read a PCM16 WAV, mixing channels and resampling to the Live wire rate."""
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
            raise ValueError(f"{path}: expected a 16-bit PCM WAV")
        channels, rate = wav.getnchannels(), wav.getframerate()
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    if not samples.size:
        raise ValueError(f"{path}: empty WAV")
    mono = samples.reshape(-1, channels).mean(axis=1, dtype=np.float32) / 32768.0
    if rate == 16000:
        return mono
    positions = np.arange(0, len(mono), rate / 16000)
    return np.interp(positions, np.arange(len(mono)), mono).astype(np.float32)


def tool_result(name: str, args: dict) -> dict:
    now = datetime.now()
    if name == "clock__read":
        result = {"time": now.strftime("%H:%M"), "date": now.strftime("%A %d %B %Y")}
    elif name == "weather__read":
        result = {"city": args.get("city") or "Paris", "temperature_c": 9, "conditions": "averses"}
    elif name == "home_assistant__control":
        result = {"entity": args.get("entity"), "state": "done"}
    else:
        return {"error": "Unknown demonstration tool"}
    return {"schemaVersion": 1, "status": "succeeded", "result": result, "summary": None}


async def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wavs", nargs="+", help="16-bit PCM WAV files, one request per file")
    parser.add_argument("--url", default="ws://127.0.0.1:8787/v1/live/sessions")
    parser.add_argument("--token", default=os.environ.get("LOCALIVE_API_KEY", ""),
                        help="Localive bearer token (default: LOCALIVE_API_KEY)")
    parser.add_argument("--profile", choices=("generic", "cornelius"), default="generic",
                        help="Use a plain voice session or Cornelius demonstration tools")
    parser.add_argument("--save", default=str(ROOT / ".runtime" / "live_client_reply.wav"))
    parser.add_argument("--pause", type=float, default=1.0, help="silence after each question")
    parser.add_argument("--frame-ms", type=int, choices=(20, 40, 80), default=40)
    parser.add_argument("--report", type=Path, help="write actual server phase timings as JSON")
    args = parser.parse_args(argv)
    frame = args.frame_ms * 16

    received: list[np.ndarray] = []
    marks: dict[str, float] = {}
    turn_done = asyncio.Event()
    metrics: list[dict] = []
    client_audio: list[dict] = []
    errors: list[dict] = []
    headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}

    async with aiohttp.ClientSession() as http, http.ws_connect(args.url, headers=headers) as ws:
        t0 = time.monotonic()
        cornelius = args.profile == "cornelius"
        want_metrics = bool(args.report) or cornelius
        session = dict(SESSION) if cornelius else {}
        if want_metrics:
            session["localive"] = {"metrics": True}
        await ws.send_json({"type": "session.start", "session": session})
        started = asyncio.Event()
        closing = False

        async def reader() -> None:
            async for message in ws:
                if message.type == aiohttp.WSMsgType.ERROR:
                    raise ConnectionError(str(ws.exception()))
                if message.type != aiohttp.WSMsgType.TEXT:
                    continue
                event = json.loads(message.data)
                kind = event["type"]
                now = time.monotonic()
                if kind == "session.started":
                    print(f"[{now - t0:6.2f}s] session.started {event['session']['id']}")
                    started.set()
                elif kind == "session.output_audio.delta":
                    pcm = np.frombuffer(base64.b64decode(event["delta"]), "<i2")
                    received.append(pcm)
                    if (pcm.size and np.abs(pcm.astype(np.int32)).max() > 300
                            and "first_audio" not in marks):
                        marks["first_audio"] = now
                elif kind == "response.event":
                    inner = event["event"]
                    item = inner.get("item") or {}
                    if (
                        inner["type"] == "response.output_item.done"
                        and item.get("type") == "function_call"
                    ):
                        call_args = json.loads(item.get("arguments") or "{}")
                        print(f"[{now - t0:6.2f}s]   tool {item['name']}({call_args})")
                        result = (tool_result(item["name"], call_args) if cornelius
                                  else {"error": "No tool was declared by this client"})
                        output = json.dumps(result)
                        reply = {"type": "function_call_output", "call_id": item["call_id"],
                                 "output": output}  # fmt: skip
                        await ws.send_json({"type": "response.item.create", "item": reply})
                    elif inner["type"] == "response.web_search_call.searching":
                        print(f"[{now - t0:6.2f}s]   web_search…")
                    elif item.get("type") == "message":
                        notes = [a for part in item.get("content") or []
                                 for a in part.get("annotations") or []]  # fmt: skip
                        print(f"[{now - t0:6.2f}s]   sources: "
                              + ", ".join(a["url"] for a in notes[:3]))  # fmt: skip
                    elif cornelius and inner["type"] == "response.completed":
                        await ws.send_json({"type": "response.create"})
                elif kind == "session.input_transcript.delta":
                    marks["heard"] = now
                    print(f"[{now - t0:6.2f}s] user: {event['delta']}  "
                          f"({event['start_ms']}-{event['end_ms']} ms)")  # fmt: skip
                elif kind == "session.output_transcript.delta":
                    print(f"[{now - t0:6.2f}s]   assistant: {event['delta'].strip()}")
                elif kind == "voice.turn_completed":
                    turn_done.set()
                elif kind == "localive.turn":
                    metrics.append(event)
                    print(f"  metrics: {json.dumps(event, ensure_ascii=False)}")
                elif kind == "error":
                    error = event.get("error", {})
                    raise RuntimeError(f"{error.get('code', 'error')}: {error.get('message')}")
                elif kind == "voice.provider_warning":
                    errors.append(event)
                    print(f"[{now - t0:6.2f}s] {kind}: {event.get('error')}")
                elif kind == "session.closed":
                    if not closing:
                        raise ConnectionError(f"Session closed: {event.get('reason', 'unknown')}")
                    print(f"[{now - t0:6.2f}s] session.closed")
                    return
            if not closing:
                raise ConnectionError("Live WebSocket closed before the session finished")

        async def stream(samples: np.ndarray, last_voiced_sample: int | None = None) -> None:
            pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
            begin = time.monotonic()
            for index, offset in enumerate(range(0, len(pcm), frame * 2)):
                if task.done():
                    task.result()
                chunk = base64.b64encode(pcm[offset : offset + frame * 2]).decode()
                await ws.send_json({"type": "session.input_audio.append", "audio": chunk})
                if last_voiced_sample is not None and index == last_voiced_sample // frame:
                    marks["last_input"] = time.monotonic()
                await asyncio.sleep(max(0.0, begin + (index + 1) * frame / 16000
                                        - time.monotonic()))

        async def send_wav(path: str) -> None:
            speech = load_16k(Path(path))
            marks.clear()
            turn_done.clear()
            silence = np.zeros(int(16000 * args.pause), dtype=np.float32)
            samples = np.concatenate([np.zeros(4000, dtype=np.float32), speech])
            voiced = np.flatnonzero(np.abs(samples) > 300 / 32767)
            await stream(samples, int(voiced[-1]) if voiced.size else None)
            speech_end = time.monotonic()
            await stream(silence)
            # Keep the microphone "open" with silence until the turn completes.
            while not turn_done.is_set():
                await stream(np.zeros(1280 * 5, dtype=np.float32))
                if time.monotonic() - speech_end > 60:
                    print("  timeout waiting for the reply")
                    errors.append({"type": "timeout", "wav": path})
                    break
            # Client-side wire time, from sending the last audible input frame to
            # receiving the first audible output frame. WAV tail silence is excluded.
            delay = (round((marks["first_audio"] - marks["last_input"]) * 1000)
                     if {"first_audio", "last_input"} <= marks.keys() else None)
            client_audio.append({"file": path, "first_sound_ms": delay})
            if "first_audio" in marks and "heard" in marks:
                heard = marks["heard"] - speech_end
                answer = marks["first_audio"] - marks["heard"]
                print(
                    f"  -> transcript {heard:+.2f} s relative to WAV end "
                    f"(may include silence), first received audio {answer:.2f} s later"
                )

        task = asyncio.create_task(reader())
        task.add_done_callback(lambda _: started.set())  # wake the sender on handshake failure
        try:
            try:
                await asyncio.wait_for(started.wait(), 5)
            except TimeoutError as exc:
                raise RuntimeError(
                    "No session.started within 5 s. Check /health on the server."
                ) from exc
            if task.done():
                task.result()
            print(f"session.started after {(time.monotonic() - t0) * 1000:.0f} ms")
            for path in args.wavs:
                await send_wav(path)
            closing = True
            await ws.send_json({"type": "session.close"})
            await asyncio.wait_for(task, 5)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    audio = np.concatenate(received) if received else np.zeros(0, "<i2")
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(args.save, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(audio.astype("<i2").tobytes())
    print(f"Output audio ({len(audio) / 16000:.1f} s, silence included): {args.save}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"files": args.wavs, "frame_ms": args.frame_ms,
                                          "turns": metrics, "client_audio": client_audio,
                                          "errors": errors},
                                         ensure_ascii=False, indent=2), encoding="utf-8")
    if errors or (want_metrics and len(metrics) != len(args.wavs)):
        raise SystemExit("Live smoke test failed: inspect warnings, timeouts and turn count")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (aiohttp.ClientError, OSError, RuntimeError, ValueError, wave.Error) as exc:
        raise SystemExit(f"Live client: {exc}") from exc
