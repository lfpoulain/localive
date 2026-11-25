"""Real Localive session: speak over the assistant and check what it decides.

Each scenario streams a question (80 ms frames, real time), then an interjection either while
the assistant is still thinking (right after the question) or once its voice has started.
Interjections use another Phonon voice than the assistant. Needs Localive running.

    .venv\\Scripts\\python.exe bench\\decision\\live_scenarios.py
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path

import aiohttp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ["RAYON_NUM_THREADS"] = "4"
os.chdir(ROOT)

from localive import hf  # noqa: E402
from localive.config import load_config  # noqa: E402

URL = "ws://127.0.0.1:8787/v1/live/sessions"
TOOLS = [
    {"type": "function", "name": "weather__read", "description": "Météo d'une ville et d'un jour.",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                    "day": {"type": "string"}}, "required": ["city"]}},
    {"type": "function", "name": "lights__set", "description": "Allumer ou éteindre une lumière.",
     "parameters": {"type": "object", "properties": {"room": {"type": "string"},
                    "on": {"type": "boolean"}}, "required": ["room", "on"]}},
]  # fmt: skip
SCENARIOS = [
    # (name, question, interjection, when, expected decision, audio cut expected)
    ("acquiescement", "Raconte-moi l'histoire de la tour Eiffel en quelques phrases.", "Mm-hm.", "speaking", "IGNORE", False),
    ("autre personne", "Raconte-moi l'histoire de la tour Eiffel en quelques phrases.",
     "Julie, tu peux sortir le plat du four ?", "speaking", "IGNORE", False),
    ("stop", "Raconte-moi l'histoire de la tour Eiffel en quelques phrases.", "Stop, ça suffit.", "speaking", "STOP", True),
    ("nouvelle demande", "Raconte-moi l'histoire de la tour Eiffel en quelques phrases.",
     "Allume la lumière de l'entrée.", "speaking", "NOW", False),
    ("correction (réflexion)", "Quel temps fera-t-il demain à Paris ?", "Non, à Lyon.",
     "thinking", "ADD", False),
    ("ajout (parole)", "Quel temps fera-t-il demain à Paris ?", "Et après-demain aussi.",
     "speaking", "ADD", False),
    ("Bordeaux 7 jours puis Nantes",
     "Quelle est la météo à Bordeaux pour les sept prochains jours, jour par jour ?",
     "Et Nantes ?", "speaking", "AFTER", False),
    ("liste coupée", "Tu peux me faire une liste de cinquante mots ?",
     "Coupe la liste après trois mots.", "speaking", "FIX", False),
    ("phrase longue par-dessus", "Tu peux me faire une liste de cinquante mots ?",
     "Arrête la liste et dis-moi plutôt quelle heure il est, s'il te plaît.", "speaking", "NEW",
     False),
    ("continuité porte Maillot", "Quel temps fait-il à la porte Maillot ?", "Et Bordeaux ?",
     "speaking", "ADD", False),
]  # fmt: skip
HESITATION = "Mets un minuteur de dix minutes pour les pâtes."  # cut after "de"


def synth(tts, text: str, voice: str) -> np.ndarray:
    chunks, rate = [], 24000
    with tts.stream(text, voice=voice) as stream:
        rate = stream.sample_rate
        for chunk in stream:
            chunks.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
    audio = np.concatenate(chunks)
    audio = np.interp(np.arange(0, len(audio), rate / 16000), np.arange(len(audio)), audio)
    return audio.astype(np.float32)


# Loudspeaker -> microphone loop (--echo GAIN): what the assistant says comes back into the
# microphone ECHO_DELAY_MS later through a small room, like a real speaker next to the mic.
ECHO = {"gain": 0.0, "brain": "qwen"}
ECHO_DELAY_MS = 120
_ROOM = np.exp(-np.arange(2400) / 500.0) * np.random.default_rng(3).normal(0, 1, 2400)
_ROOM[0] = 4.0
_ROOM /= np.abs(_ROOM).sum() / 2


class Loop:
    """Per session: the assistant audio received so far, and how much mic audio was sent."""

    def __init__(self) -> None:
        self.played: list[np.ndarray] = []
        self.sent = 0
        self.tail = np.zeros(len(_ROOM) - 1, dtype=np.float32)

    def echo(self, count: int) -> np.ndarray:
        played = np.concatenate(self.played) if self.played else np.zeros(0, np.float32)
        start = self.sent - ECHO_DELAY_MS * 16
        chunk = np.zeros(count, np.float32)
        if start + count > 0 and start < len(played):
            lo = max(0, start)
            piece = played[lo : start + count]
            chunk[lo - start : lo - start + len(piece)] = piece
        wet = np.convolve(np.concatenate([self.tail, chunk]), _ROOM, mode="valid")
        self.tail = np.concatenate([self.tail, chunk])[-(len(_ROOM) - 1) :]
        self.sent += count
        return (ECHO["gain"] * wet).astype(np.float32)


LOOP = Loop()


async def stream(ws, audio: np.ndarray) -> None:
    begin = time.monotonic()
    for n, i in enumerate(range(0, len(audio), 1280)):
        frame = audio[i : i + 1280]
        if ECHO["gain"]:
            frame = frame + LOOP.echo(len(frame))
        pcm = (np.clip(frame, -1, 1) * 32767).astype("<i2").tobytes()
        await ws.send_json({"type": "session.input_audio.append",
                            "audio": base64.b64encode(pcm).decode()})  # fmt: skip
        await asyncio.sleep(max(0.0, begin + (n + 1) * 0.08 - time.monotonic()))


async def run(name, question, interjection, when, expected, expect_cut, clips) -> dict:
    global LOOP
    LOOP = Loop()
    events: list[tuple[float, dict]] = []
    frames: list[bool] = []  # one flag per 40 ms output frame: audible or not
    voice_started = asyncio.Event()
    heard = asyncio.Event()
    async with aiohttp.ClientSession() as http, http.ws_connect(URL) as ws:
        session = {"model": "gpt-live-1", "instructions": "Tu es Cornelius, l'assistant de la maison.",
                   "audio": {"output": {"voice": "marin"}},
                   "delegation": {"type": "responses", "responses": {"model": ECHO["brain"], "tools": TOOLS}},
                   "localive": {"metrics": True}}  # fmt: skip
        await ws.send_json({"type": "session.start", "session": session})
        t0 = time.monotonic()

        async def reader():
            async for message in ws:
                event = json.loads(message.data)
                kind = event["type"]
                if kind == "session.output_audio.delta":
                    pcm = np.frombuffer(base64.b64decode(event["delta"]), "<i2")
                    LOOP.played.append(pcm.astype(np.float32) / 32768)
                    frames.append(bool(np.abs(pcm).max() > 500))
                    if frames[-1]:
                        voice_started.set()
                    continue
                events.append((time.monotonic() - t0, event))
                if kind == "session.input_transcript.delta":
                    heard.set()
                elif kind == "response.event":
                    inner = event["event"]
                    item = inner.get("item") or {}
                    if (
                        inner["type"] == "response.output_item.done"
                        and item.get("type") == "function_call"
                    ):
                        await ws.send_json({"type": "response.item.create", "item": {
                            "type": "function_call_output", "call_id": item["call_id"],
                            "output": json.dumps({"status": "succeeded", "result": {"ok": True,
                            "météo": "9 °C, averses toute la journée, vent fort", "7 jours": "lundi 18 °C soleil, mardi 16 °C nuageux, mercredi 14 °C averses, jeudi 17 °C soleil, vendredi 19 °C doux, samedi 20 °C, dimanche orages"}})}})  # fmt: skip
                elif kind == "session.closed":
                    return

        task = asyncio.create_task(reader())
        silence = np.zeros(1280, dtype=np.float32)
        await stream(ws, np.concatenate([np.zeros(4000, np.float32), clips[question]]))
        if when == "speaking":
            while not voice_started.is_set():
                await stream(ws, silence)
            await stream(ws, np.zeros(16000, np.float32))  # let it talk for 1 s
        else:
            await stream(ws, np.zeros(3200, np.float32))  # 200 ms: the end of turn fires
        interject_at = time.monotonic() - t0
        await stream(ws, clips[interjection])
        interject_end = time.monotonic() - t0
        # Keep the microphone open (silence) until the assistant has completely finished.
        quiet_since = time.monotonic()
        while time.monotonic() - quiet_since < 2.5 and time.monotonic() - t0 < 60:
            await stream(ws, np.zeros(1280 * 5, np.float32))
            if voice_started.is_set() and frames and any(frames[-10:]):
                quiet_since = time.monotonic()
        await ws.send_json({"type": "session.close"})
        await asyncio.wait_for(task, 10)

    decisions = [e for _, e in events if e["type"] == "localive.decision"]
    if expected in {"AFTER", "FIX", "NOW"}:  # judged on the "how to continue" decision
        decisions = [d for d in decisions if d["where"] == "enchaînement"] or decisions
    interrupted = any(e["type"] == "voice.interrupted" for _, e in events)
    final = decisions[-1]["label"] if decisions else ("STOP" if interrupted else "(aucune)")
    spoken = [e for t, e in events if e["type"] == "session.output_transcript.delta"]
    said = " ".join(e["delta"].strip() for t, e in events
                    if e["type"] == "session.output_transcript.delta" and t > interject_at)  # fmt: skip
    # Longest silence in the voice, measured on the audio itself (40 ms frames).
    audible = [i for i, a in enumerate(frames) if a]
    gaps = []
    if audible:
        run = 0
        for flag in frames[audible[0] : audible[-1] + 1]:
            run = 0 if flag else run + 1
            gaps.append(run * 40)
    decided = [
        t * 1000 for t, e in events if e["type"] == "localive.decision" and e["where"] == "tour"
    ]
    timeline = [f"{e['start_ms']}-{e['end_ms']} ms « {e['delta'].strip()[:70]} »" for e in spoken]
    interjection_ms = round(interject_at * 1000)
    timeline.insert(
        0,
        f"interjection envoyée à ~{interjection_ms} ms, finie à ~{round(interject_end * 1000)} ms",
    )
    if decided:
        timeline.append(
            f"=> décision {decided[-1] - interject_end * 1000:+.0f} ms après la fin de ta phrase"
        )
    if decided:
        timeline.append(f"décision sur la phrase entière à ~{decided[-1]:.0f} ms")
    tools = [e["event"]["item"] for _, e in events if e["type"] == "response.event"
             and e["event"].get("item", {}).get("type") == "function_call"]  # fmt: skip
    turns = [e for _, e in events if e["type"] == "localive.turn"]
    return {"name": name, "expected": expected, "decision": final, "expect_cut": expect_cut,
            "max_gap_ms": max(gaps) if gaps else None, "timeline": timeline, "all_said": " | ".join(e["delta"].strip() for e in spoken),
            "decisions": [(d["where"], d["label"], d["ms"], d["heard"]) for d in decisions],
            "interrupted": interrupted, "said_after": said[:160],
            "tools": [(t["name"], t["arguments"]) for t in tools],
            "silence_ms": [t.get("silence_ms") for t in turns]}  # fmt: skip


def split_after_de(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cut a fluent sentence right after the word "de" (word timestamps from Parakeet)."""
    import moondream

    with moondream.photon("moondream/parakeet-ultra") as speech:
        result = speech.transcribe(audio=audio, sample_rate=16000, timestamps="word")
    words = [w for seg in result.get("segments", []) for w in seg.get("words", [])]
    word = next(w for w in words if w["word"].strip(" ,.").lower() == "de")
    cut = int(float(word["end"]) * 16000) + 480
    return audio[:cut], audio[cut:]


async def hesitation(clips, pause_s: float = 0.9) -> str:
    """ "Mets un minuteur de… [pause] …dix minutes pour les pâtes." must stay one request."""
    heard: list[str] = []
    async with aiohttp.ClientSession() as http, http.ws_connect(URL) as ws:
        await ws.send_json({"type": "session.start", "session": {
            "model": "gpt-live-1", "delegation": {"type": "responses", "responses": {
                "model": "qwen", "tools": TOOLS}}, "localive": {"metrics": True}}})  # fmt: skip

        async def reader():
            async for message in ws:
                event = json.loads(message.data)
                if event["type"] == "session.input_transcript.delta":
                    heard.append(event["delta"])
                elif event["type"] == "session.closed":
                    return

        task = asyncio.create_task(reader())
        first, second = split_after_de(clips[HESITATION])
        await stream(ws, np.concatenate([np.zeros(4000, np.float32), first]))
        await stream(ws, np.zeros(int(16000 * pause_s), np.float32))
        await stream(ws, second)
        await stream(ws, np.zeros(16000 * 4, np.float32))
        await ws.send_json({"type": "session.close"})
        await asyncio.wait_for(task, 10)
    verdict = "OK " if len(heard) == 1 else "!! "
    return f"{verdict}hésitation de {pause_s} s : {len(heard)} transcription(s) : {heard}"


async def main() -> None:
    import argparse

    import ptts

    parser = argparse.ArgumentParser()
    parser.add_argument("--echo", type=float, default=0.0, help="speaker->mic loop gain (0: off)")
    parser.add_argument("--only", default="", help="comma-separated scenario names")
    parser.add_argument("--brain", default="qwen", help="delegation.responses.model (qwen, gemma)")
    args = parser.parse_args()
    ECHO["gain"] = args.echo
    ECHO["brain"] = args.brain

    hf.configure(load_config())
    tts = ptts.TTS(
        "hf://gradium/phonon-40m-fr", device="cpu", quant="q8", voice="Arnaud", lang="fr"
    )
    texts = {s[1] for s in SCENARIOS} | {s[2] for s in SCENARIOS} | {HESITATION}
    clips = {t: synth(tts, t, "Arnaud") for t in texts}
    results = []
    wanted = [s for s in SCENARIOS if not args.only or s[0] in args.only.split(",")]
    for scenario in wanted:
        result = await run(*scenario, clips)
        results.append(result)
        good = (
            result["decision"] == result["expected"]
            and result["interrupted"] == result["expect_cut"]
        )
        print(
            f"{'OK ' if good else '!! '}{result['name']:26s} attendu {result['expected']:6s} "
            f"décidé {result['decision']:8s} coupé={str(result['interrupted']):5s} "
            f"(attendu {result['expect_cut']}) · plus long silence dans la voix "
            f"{result['max_gap_ms']} ms"
        )
        for where, label, ms, heard in result["decisions"]:
            print(f"      décision {where}: {label} en {ms} ms ← « {heard} »")
        if result["tools"]:
            print(f"      outils : {result['tools']}")
        for line in result["timeline"]:
            print(f"      {line}")
    if not args.only:
        for pause in (0.9, 2.0):
            print(await hesitation(clips, pause))
    (ROOT / ".runtime" / "bench").mkdir(parents=True, exist_ok=True)
    (ROOT / ".runtime" / "bench" / "live_scenarios.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), "utf-8"
    )


if __name__ == "__main__":
    asyncio.run(main())
