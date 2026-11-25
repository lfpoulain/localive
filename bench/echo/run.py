"""Echo bench: does the server-side canceller remove the assistant's own voice from the mic?

Simulation: the far-end (what Localive sends) is Phonon speech; the mic gets that speech back
after ``delay`` ms through a small room (reverberation), plus a little noise, and — in the
"double talk" case — the user speaking at the same time. We compare, with and without the
canceller (src/localive/echo.py):
  ERLE       echo energy removed (dB, higher is better; > 20 dB is very good)
  VAD        how much of the echo Silero still calls speech (false barge-ins)
  double talk  the user speaks over the assistant: is it still detected and transcribed?

    .venv\\Scripts\\python.exe bench\\echo\\run.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
os.environ["RAYON_NUM_THREADS"] = "4"
os.chdir(ROOT)

from localive import hf  # noqa: E402
from localive.config import load_config  # noqa: E402
from localive.echo import EchoCanceller  # noqa: E402
from localive.vad import CHUNK, SileroVad  # noqa: E402
from tests.test_vad import FIXTURES, load_16k  # noqa: E402

RATE = 16000
FRAME = 640  # Localive sends 40 ms output frames


def phonon(text: str) -> np.ndarray:
    import ptts

    tts = ptts.TTS("hf://gradium/phonon-40m-fr", device="cpu", quant="q8", voice="Ninon", lang="fr")
    chunks, rate = [], 24000
    with tts.stream(text) as stream:
        rate = stream.sample_rate
        for chunk in stream:
            chunks.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
    audio = np.concatenate(chunks)
    return np.interp(np.arange(0, len(audio), rate / RATE), np.arange(len(audio)), audio).astype(
        np.float32
    )


def room(signal: np.ndarray, delay_ms: int, gain: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    length = int(0.25 * RATE)
    impulse = rng.normal(0, 1, length) * np.exp(-np.arange(length) / (0.05 * RATE))
    impulse[0] = 4.0
    impulse /= np.abs(impulse).sum() / 2
    wet = np.convolve(signal, impulse)[: len(signal)]
    delayed = np.concatenate([np.zeros(delay_ms * RATE // 1000, np.float32), wet])[: len(signal)]
    return (gain * delayed / max(1e-6, np.abs(delayed).max())).astype(np.float32)


def speech_ratio(audio: np.ndarray) -> float:
    vad = SileroVad()
    flags = [vad(audio[i : i + CHUNK]) >= 0.5 for i in range(0, len(audio) - CHUNK, CHUNK)]
    return sum(flags) / max(1, len(flags))


def run(far: np.ndarray, mic: np.ndarray, enabled: bool) -> np.ndarray:
    """Feed like Localive: far-end in 40 ms frames as sent, mic in 80 ms satellite frames."""
    config = load_config()
    config.echo.enabled = enabled
    canceller = EchoCanceller(config)
    out = []
    for i in range(0, len(mic), 1280):
        for j in range(i, min(i + 1280, len(far)), FRAME):
            canceller.far_end(far[j : j + FRAME])
        out.append(canceller.near_end(mic[i : i + 1280])[0])
    return np.concatenate(out)[: len(mic)]


def db(x: float) -> float:
    return 10 * np.log10(max(x, 1e-12))


def main() -> None:
    hf.configure(load_config())
    far = phonon(
        "Demain à Paris, il fera neuf degrés avec des averses toute la journée et un vent fort. "
        "Pense à prendre un parapluie si tu sors, et une veste chaude pour le soir."
    )
    far = np.concatenate([far, far])  # ~16 s: lets the adaptive filter converge, then measure
    user = load_16k(FIXTURES / "wake_clock.wav")
    rng = np.random.default_rng(0)
    import moondream

    speech = moondream.photon("moondream/parakeet-ultra")
    speech.__enter__()
    hear = lambda a: speech.transcribe(audio=a, sample_rate=RATE)["text"]  # noqa: E731
    print(f"utilisateur seul : « {hear(user)} »")
    print(
        f"{'cas':28s} {'ERLE':>8s} {'VAD écho sans':>13s} {'avec':>5s}  double parole (avec annulation)"
    )
    for delay in (40, 120, 250):
        for loud in (0.3, 0.7):
            echo = room(far, delay, loud, seed=delay)
            noise = rng.normal(0, 0.003, len(far)).astype(np.float32)
            mic = echo + noise
            raw, clean = run(far, mic, False), run(far, mic, True)
            half = len(far) // 2  # second half: converged
            erle = db(np.mean(raw[half:] ** 2)) - db(np.mean(clean[half:] ** 2))
            # Double talk: the user speaks during the second half.
            talk = np.zeros_like(far)
            start = half + RATE
            talk[start : start + len(user)] = user[: len(far) - start] * 0.6
            dt_raw, dt = mic + talk, run(far, mic + talk, True)
            window = slice(start - RATE // 4, start + len(user) + RATE // 4)
            print(f"délai {delay:3d} ms, écho {'fort' if loud > 0.5 else 'moyen':5s} "
                  f"{erle:6.1f} dB {speech_ratio(raw[half:]):12.0%} {speech_ratio(clean[half:]):5.0%}  "
                  f"sans : « {hear(dt_raw[window])} » / avec : « {hear(dt[window])} »")  # fmt: skip


if __name__ == "__main__":
    main()
