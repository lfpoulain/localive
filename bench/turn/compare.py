"""Compare local Smart Turn models/threads without loading ASR or downloading anything.

    .venv\\Scripts\\python.exe bench\\turn\\compare.py --threads 1 2 4

Uses the 75 existing synthetic clips, with clean audio and simulated 15 dB noise.
This is a regression/calibration set, not an independent real-microphone evaluation.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from bench.turn.run import with_noise  # noqa: E402
from localive.turn import SmartTurn  # noqa: E402
from tests.test_vad import load_16k  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--output", default=".runtime/bench/smart-turn-local.json")
    args = parser.parse_args()
    if min(args.threads) < 1 or args.repeats < 1:
        parser.error("threads and repeats must be positive")
    cases = json.loads((ROOT / "bench/turn/cases.json").read_text("utf-8"))["cases"]
    audios = [
        np.concatenate([
            load_16k(ROOT / "bench/turn/audio" / case["file"]),
            np.zeros(3200, np.float32),  # the first 200 ms pause
        ])
        for case in cases
    ]
    report = {"platform": platform.platform(), "processor": platform.processor(), "models": {}}
    for model in ("smart-turn-v3.2-gpu.onnx", "smart-turn-v3.2-cpu.onnx"):
        path = ROOT / "models/turn" / model
        if not path.is_file():
            print(f"Skipped missing model: {path}", flush=True)
            continue
        record = {"threads": {}, "conditions": {}}
        for threads in args.threads:
            detector = SmartTurn(path, threads=threads)
            detector(audios[0])
            times = [detector(audios[i % len(audios)])[1] for i in range(args.repeats)]
            timing = {"median_ms": statistics.median(times), "p90_ms": float(np.percentile(times, 90))}
            record["threads"][str(threads)] = timing
            print(f"{model} · {threads} threads · {timing}", flush=True)
        # Score at the first requested thread count to avoid multiplying the same
        # quality evaluation; thread count changes scheduling, not model weights.
        detector = SmartTurn(path, threads=args.threads[0])
        detector(audios[0])
        for condition in ("clean", "noise"):
            rows = []
            for index, (case, audio) in enumerate(zip(cases, audios, strict=True)):
                if condition == "noise":
                    audio = with_noise(audio, index)
                probability, milliseconds = detector(audio)
                rows.append({**case, "probability": probability, "ms": milliseconds,
                             "correct": (probability >= 0.5) == case["complete"]})
            done = [row for row in rows if row["complete"]]
            unfinished = [row for row in rows if not row["complete"]]
            summary = {
                "correct": sum(row["correct"] for row in rows), "total": len(rows),
                "complete_detected": sum(row["correct"] for row in done),
                "complete_total": len(done),
                "incomplete_held": sum(row["correct"] for row in unfinished),
                "incomplete_total": len(unfinished),
                "median_ms": statistics.median(row["ms"] for row in rows),
            }
            record["conditions"][condition] = {**summary, "rows": rows}
            print(f"{model} · {condition} · {summary}", flush=True)
        report["models"][model] = record
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(output)


if __name__ == "__main__":
    main()
