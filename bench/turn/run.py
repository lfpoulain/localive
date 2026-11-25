"""End-of-turn bench: Smart Turn v3.2 vs the silence-only rule (Silero) on bench/turn.

Each clip gets 200 ms of trailing silence: the moment the VAD first notices a pause.
  complete clip   -> the turn should end now   (answer ~400 ms earlier than today)
  cut/hesitation  -> the turn should NOT end   (today: cut off after 600 ms of silence)

    .venv\\Scripts\\python.exe bench\\turn\\run.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from localive.turn import SmartTurn  # noqa: E402
from tests.test_vad import load_16k  # noqa: E402

CONDITIONS = ("clean", "noise")


def with_noise(audio: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, len(audio)).astype(np.float32)
    speech = np.mean(audio[np.abs(audio) > 0.02] ** 2) if np.any(np.abs(audio) > 0.02) else 1e-4
    return np.clip(audio + noise * np.sqrt(speech / 10**1.5), -1, 1).astype(np.float32)  # 15 dB


def main() -> None:
    cases = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))["cases"]
    audios = {c["file"]: load_16k(HERE / "audio" / c["file"]) for c in cases}
    tail = np.zeros(3200, dtype=np.float32)  # 200 ms
    report = {}
    for model in ("smart-turn-v3.2-cpu.onnx", "smart-turn-v3.2-gpu.onnx"):
        detector = SmartTurn(ROOT / "models" / "turn" / model)
        detector(np.zeros(16000, np.float32))
        for condition in CONDITIONS:
            rows, times = [], []
            for i, case in enumerate(cases):
                audio = np.concatenate([audios[case["file"]], tail])
                if condition == "noise":
                    audio = with_noise(audio, i)
                probability, ms = detector(audio)
                times.append(ms)
                rows.append({**case, "p": round(probability, 3)})
            report[f"{model}/{condition}"] = rows
            print(
                f"\n{model} · {condition} · médiane {statistics.median(times):.1f} ms par décision (CPU)"
            )
            for threshold in (0.3, 0.5, 0.7):
                done = [r for r in rows if r["complete"]]
                open_ = [r for r in rows if not r["complete"]]
                fast = sum(r["p"] >= threshold for r in done)
                held = sum(r["p"] < threshold for r in open_)
                by = {k: f"{sum(r['p'] < threshold for r in open_ if r['kind'] == k)}/"
                      f"{sum(r['kind'] == k for r in open_)}" for k in ("cut", "hesitation")}  # fmt: skip
                print(f"  seuil {threshold}: fin détectée {fast}/{len(done)} phrases complètes · "
                      f"attend {held}/{len(open_)} phrases inachevées ({by['cut']} coupées, "
                      f"{by['hesitation']} hésitations)")  # fmt: skip
    print(
        "\nRègle actuelle (silence seul) : fin détectée 39/39 · attend 0/36 inachevées "
        "(toute pause de 600 ms termine la phrase)"
    )
    out = ROOT / ".runtime" / "bench"
    out.mkdir(parents=True, exist_ok=True)
    (out / "turn.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf-8")
    # Hybrid: Smart Turn + "the transcript ends on a word that cannot end a sentence".
    # Caveat: the cut clips were cut right after such a word, so this rule is favoured.
    import os

    os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))
    import moondream

    from localive.turn import ends_open

    with moondream.photon("moondream/parakeet-ultra") as speech:
        for model in ("smart-turn-v3.2-cpu.onnx", "smart-turn-v3.2-gpu.onnx"):
            rows = report[f"{model}/clean"]
            for r in rows:
                if "heard" not in r:
                    audio = np.concatenate([audios[r["file"]], tail])
                    r["heard"] = speech.transcribe(audio=audio, sample_rate=16000)["text"]
                r["hybrid_complete"] = r["p"] >= 0.5 and not ends_open(r["heard"])
            done = [r for r in rows if r["complete"]]
            open_ = [r for r in rows if not r["complete"]]
            fast = sum(r["hybrid_complete"] for r in done)
            held = {k: f"{sum(not r['hybrid_complete'] for r in open_ if r['kind'] == k)}/"
                    f"{sum(r['kind'] == k for r in open_)}" for k in ("cut", "hesitation")}  # fmt: skip
            print(f"Hybride {model} + transcription : fin détectée {fast}/{len(done)} · attend "
                  f"{held['cut']} coupées, {held['hesitation']} hésitations")  # fmt: skip
            for r in rows:
                if r["kind"] == "hesitation":
                    print(f"    hésitation p={r['p']:.2f} entendu « {r['heard']} »")
                if r["complete"] and not r["hybrid_complete"]:
                    print(f"    complète retenue à tort p={r['p']:.2f} « {r['heard']} »")
    (out / "turn.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf-8")
    wrong = [r for r in report["smart-turn-v3.2-cpu.onnx/clean"]
             if (r["p"] >= 0.5) != r["complete"]]  # fmt: skip
    print("\nErreurs (cpu, propre, seuil 0.5) :")
    for r in wrong:
        print(f"  {'complète ' if r['complete'] else 'inachevée'} p={r['p']:.2f}  {r['text']}")


if __name__ == "__main__":
    main()
