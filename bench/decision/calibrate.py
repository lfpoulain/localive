"""Calibrate the "ask before acting" threshold on bench/decision/cases.json.

Uses Localive's own Arbiter (same prompt, same filters) against a running Qwen (Localive's
llama-server on :8081 by default). For each threshold: how many situations would become a
yes/no question, and how many of those were errors (good questions) or right (needless).

    .venv\\Scripts\\python.exe bench\\decision\\calibrate.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import aiohttp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from localive.config import Config  # noqa: E402
from localive.decision import Arbiter, Situation  # noqa: E402
from localive.live import is_echo, is_stop  # noqa: E402


def accepted(case):
    return case["y"] if isinstance(case["y"], list) else [case["y"]]


async def main() -> None:
    cases = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))["cases"]
    config = Config()
    config.decision.timeout_ms = 5000
    rows = []
    async with aiohttp.ClientSession() as http:
        arbiter = Arbiter(http, config)
        for case in cases:
            if case["p"] == "s" and is_echo(case["h"], case.get("s", "")):
                label, confidence = "IGNORE", 1.0
            elif is_stop(case["h"]):
                label, confidence = "STOP", 1.0
            else:
                situation = Situation(request=case["r"], heard=case["h"], speaking=case["p"] == "s",
                                      said=case.get("s", ""), tool=case.get("tool", ""))  # fmt: skip
                decision = await arbiter.decide(situation)
                label, confidence = decision.label, decision.confidence
            rows.append({**case, "pred": label, "conf": confidence, "ok": label in accepted(case)})
    ok = sum(r["ok"] for r in rows)
    print(f"Qwen3-4B (Arbiter Localive) : {ok}/{len(rows)} justes")
    asks = [r for r in rows if r["pred"] == "ASK"]
    by_gold = {}
    for r in asks:
        by_gold[accepted(r)[0]] = by_gold.get(accepted(r)[0], 0) + 1
    print(f"Questions (ASK) : {len(asks)} — par vraie réponse : {by_gold}")
    for r in asks:
        print(f"  ASK pour {str(r['y']):18s} {r['h']}")
    print(f"{'seuil':>6s} {'questions':>9s} {'dont erreurs':>12s} {'dont inutiles':>13s} "
          f"{'erreurs restantes':>17s}")  # fmt: skip
    for threshold in (0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95):
        asked = [r for r in rows if r["pred"] in {"ADD", "NEW"} and r["conf"] < threshold]
        caught = sum(not r["ok"] for r in asked)
        left = sum(not r["ok"] for r in rows if r not in asked)
        print(
            f"{threshold:>6.2f} {len(asked):>9d} {caught:>12d} {len(asked) - caught:>13d} {left:>17d}"
        )
    print("\nErreurs restantes et confiance :")
    for r in rows:
        if not r["ok"]:
            print(f"  {str(r['y']):18s} -> {r['pred']:6s} p={r['conf']:.2f}  {r['h']}")
    (ROOT / ".runtime" / "bench").mkdir(parents=True, exist_ok=True)
    (ROOT / ".runtime" / "bench" / "calibration.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), "utf-8"
    )


if __name__ == "__main__":
    asyncio.run(main())
