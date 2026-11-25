"""Re-score .runtime/bench/decision.json: 4 labels, and the binary duplex decision."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
data = json.loads((ROOT / ".runtime" / "bench" / "decision.json").read_text(encoding="utf-8"))


def accepted(row):
    return row["y"] if isinstance(row["y"], list) else [row["y"]]


print(f"{'moteur':16s} {'4 classes':>9s} {'continuer/couper':>16s} {'coupe à tort':>12s} "
      f"{'rate':>5s} {'ADD juste':>9s} {'médiane':>8s}")  # fmt: skip
for name, engine in data["engines"].items():
    rows = engine["rows"]
    binary = wrong_cut = missed = add_ok = 0
    for r in rows:
        want = accepted(r)
        cut = r["pred"] != "IGNORE"
        if (cut and any(a != "IGNORE" for a in want)) or (not cut and "IGNORE" in want):
            binary += 1
        if want == ["IGNORE"] and cut:
            wrong_cut += 1
        if "IGNORE" not in want and not cut:
            missed += 1
        if ("ADD" in want) == (r["pred"] == "ADD"):
            add_ok += 1
    n = len(rows)
    print(f"{name:16s} {engine['correct']:>4d}/{n:<4d} {binary:>11d}/{n:<4d} {wrong_cut:>12d} "
          f"{missed:>5d} {add_ok:>5d}/{n:<3d} {engine['latency_median_ms']:>5d} ms")  # fmt: skip
