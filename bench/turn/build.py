"""Build the end-of-turn corpus in bench/turn/audio (16 kHz mono WAV) + cases.json.

complete    whole requests: Windows Hortense recordings + Phonon voices (4 speakers)
cut         the same audio cut right after a word that cannot end a French sentence
            (de, à, le, pour, que, est…), found with Parakeet word timestamps
hesitation  Phonon reading requests that trail off ("euh…", "attends…")

    .venv\\Scripts\\python.exe bench\\turn\\build.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
os.environ["RAYON_NUM_THREADS"] = "4"

from localive import hf  # noqa: E402
from localive.config import load_config  # noqa: E402
from tests.test_vad import load_16k  # noqa: E402

os.chdir(ROOT)
hf.configure(load_config())  # token + models/hf cache, like Localive

OUT = HERE / "audio"
REQUESTS = [
    "Quelle heure est-il ?",
    "Mets un minuteur de dix minutes pour les pâtes.",
    "Allume la lumière du salon.",
    "Quel temps fera-t-il demain à Lyon ?",
    "Rappelle-moi d'appeler le dentiste demain matin.",
    "Combien font trente-sept fois douze ?",
    "Est-ce que tu peux baisser le chauffage de la chambre ?",
    "Raconte-moi une blague courte.",
    "Ajoute du lait et des œufs à la liste de courses.",
    "Quand passe le prochain bus pour la gare ?",
    "Ferme les volets du salon et de la cuisine.",
    "Qui a gagné le match de rugby hier soir ?",
    "Traduis bonne soirée en espagnol.",
    "Lance la playlist du matin dans la cuisine.",
    "Je voudrais savoir si la pharmacie est ouverte dimanche.",
    "Tu peux me dire combien de temps il faut pour aller à Bordeaux en voiture ?",
]
HESITATIONS = [
    "Je voudrais savoir, euh…",
    "Est-ce que tu pourrais, attends…",
    "Mets-moi un minuteur de, euh…",
    "Allume la lumière de la, hmm…",
    "Dis-moi quel temps il fera à…",
    "Rappelle-moi de… euh…",
    "Combien font, attends, trente-sept fois…",
    "Ajoute à la liste de courses du, du…",
]
OPEN_WORDS = {
    "de", "d", "du", "des", "à", "a", "au", "aux", "le", "la", "les", "l", "un", "une", "pour",
    "avec", "que", "qu", "et", "ou", "mais", "dans", "sur", "en", "mon", "ma", "mes", "ton", "ta",
    "tes", "son", "sa", "ses", "est", "sont", "quel", "quelle", "si", "me", "te", "se", "je",
    "tu", "il", "elle", "on", "nous", "vous", "par", "ce", "cette", "ces", "très", "plus",
}  # fmt: skip
VOICES = ["Ninon", "Lucien", "Hélène", "Arnaud"]


def save(name: str, audio: np.ndarray) -> str:
    path = OUT / f"{name}.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    return path.name


def phonon_audio(tts, text: str, voice: str) -> np.ndarray:
    chunks, rate = [], 24000
    with tts.stream(text, voice=voice) as stream:
        rate = stream.sample_rate
        for chunk in stream:
            chunks.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
    audio = np.concatenate(chunks)
    positions = np.arange(0, len(audio), rate / 16000)
    audio = np.interp(positions, np.arange(len(audio)), audio).astype(np.float32)
    keep = np.nonzero(np.abs(audio) > 0.01)[0]  # trim TTS silence: the VAD decides that
    return audio[max(0, keep[0] - 800) : keep[-1] + 800] if len(keep) else audio


def main() -> None:
    import moondream
    import ptts

    OUT.mkdir(parents=True, exist_ok=True)
    random.seed(7)
    cases = []
    sources: list[tuple[str, str, np.ndarray]] = []  # (id, text, audio)
    hortense = json.loads((HERE / "asr" / "cases.json").read_text(encoding="utf-8"))["cases"]
    for case in hortense:
        path = HERE / "asr" / case["file"]
        sources.append((case.get("id", path.stem), case["text"], load_16k(path)))
    tts = ptts.TTS("hf://gradium/phonon-40m-fr", device="cpu", quant="q8", voice="Ninon", lang="fr")
    for i, text in enumerate(REQUESTS):
        voice = VOICES[i % len(VOICES)]
        sources.append((f"phonon_{voice.lower()}_{i:02d}", text, phonon_audio(tts, text, voice)))
    for i, text in enumerate(HESITATIONS):
        voice = VOICES[(i + 1) % len(VOICES)]
        name = save(f"hesit_{voice.lower()}_{i:02d}", phonon_audio(tts, text, voice))
        cases.append({"file": name, "text": text, "complete": False, "kind": "hesitation"})

    with moondream.photon("moondream/parakeet-ultra") as speech:
        for ident, text, audio in sources:
            cases.append(
                {"file": save(ident, audio), "text": text, "complete": True, "kind": "complete"}
            )
            result = speech.transcribe(audio=audio, sample_rate=16000, timestamps="word")
            words = [w for seg in result.get("segments", []) for w in seg.get("words", [])]
            spots = [
                (k, w) for k, w in enumerate(words[:-1])
                if 1 <= k and w.get("word", "").strip(" ,.?!'’").lower().split("'")[-1] in OPEN_WORDS
            ]  # fmt: skip
            if not spots:
                continue
            k, word = random.choice(spots)
            end = int(float(word["end"]) * 16000) + 480  # +30 ms: do not clip the word
            said = " ".join(w["word"].strip() for w in words[: k + 1])
            cases.append({"file": save(ident + "_cut", audio[:end]), "text": said + " …",
                          "complete": False, "kind": "cut"})  # fmt: skip
    (HERE / "cases.json").write_text(
        json.dumps({"cases": cases}, ensure_ascii=False, indent=1), "utf-8"
    )
    kinds = {k: sum(c["kind"] == k for c in cases) for k in ("complete", "cut", "hesitation")}
    print(len(cases), "cas", kinds)


if __name__ == "__main__":
    main()
