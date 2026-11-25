"""Check SystemOne tool selection on the Cornelius catalogue.

    python bench/prompts/route_tools.py --port 8085
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import tempfile
import time
from pathlib import Path

import aiohttp

from localive import search
from localive.config import AltBrainConfig, Config, RoutingConfig
from localive.engines import Brain
from localive.install import LLAMA_TAG
from localive.live import chat_tools
from localive.routing import ToolRouter

CLOCK, WEATHER, TRANSIT = "clock__read", "weather__read", "transit__read"
WEB, CARD, MEMORY = "web_search", "ui__show_luna_card", "memory__recall"
# Required capabilities; a returned superset still lets the conversational model choose.
CASES = [
    ("Quelle heure est-il ?", "", {CLOCK}),
    ("On est quel jour de la semaine ?", "", {CLOCK}),
    ("Est-ce que je prends un parapluie à Lyon demain ?", "", {WEATHER}),
    ("Et dimanche ?", "Quel temps fait-il à Lyon ? weather__read", {WEATHER}),
    ("Il passe quand, mon bus ?", "", {TRANSIT}),
    ("Et le suivant ?", "Quand arrive le prochain bus ? transit__read", {TRANSIT}),
    ("Tu te souviens de mon plat préféré ?", "", {MEMORY}),
    ("Qui a gagné le dernier Grand Prix de Formule 1 ?", "", {WEB, CARD}),
    ("Cherche sur le web le prix actuel du cuivre.", "", {WEB, CARD}),
    ("Quelle heure est-il et va-t-il pleuvoir demain ?", "", {CLOCK, WEATHER}),
    ("Affiche une carte qui compare deux et trois.", "", {CARD}),
    ("Raconte-moi une histoire de dragon.", "", set()),
    ("Merci, c'est tout.", "", set()),
    ("Combien font sept fois six ?", "", set()),
]

# Separate phrasings specified before seeing their decisions or timings.
HOLDOUT = [
    ("On est le combien, déjà ?", "", {CLOCK}),
    ("Je sors les plantes sur le balcon cette nuit ou il risque de geler ?", "", {WEATHER}),
    ("Et dans l’autre sens, vers la gare ?", "Quand passe le prochain bus vers le stade ? transit__read", {TRANSIT}),
    ("Et à Bordeaux à la place ?", "Est-ce qu’il pleuvra demain à Tours ? weather__read", {WEATHER}),
    ("C’est quoi déjà le prénom de la voisine dont je t’ai parlé ?", "", {MEMORY}),
    ("Vérifie qui a remporté le dernier Vendée Globe.", "", {WEB, CARD}),
    ("Mets à l’écran un tableau comparant le violon et l’alto, trois lignes.", "", {CARD}),
    ("Donne-moi la date du jour et les deux prochains bus de mon arrêt.", "", {CLOCK, TRANSIT}),
    ("Invente une devinette pour un enfant sur un escargot.", "", set()),
    ("Ça commence à quelle heure ?", "", None),  # requires clarification
]

# Frozen before comparing d3-lite, d3-nano, d1-2B and d1-Omni-2B on these requests.
EXTRA = [
    ("J’ai pas ma montre sur moi, il est quelle heure là ?", "", {CLOCK}),
    ("Tu crois que je dois gratter le pare-brise demain matin ?", "", {WEATHER}),
    ("Dans combien de minutes passe le prochain à mon arrêt ?", "On parlait du bus. transit__read", {TRANSIT}),
    ("Retrouve le nom du resto italien que je t’avais recommandé.", "", {MEMORY}),
    ("Cherche sur Internet qui vient de gagner le marathon de New York.", "", {WEB, CARD}),
    ("Fais-moi un tableau à l’écran avec deux colonnes, thé et café.", "", {CARD}),
    ("Tu peux me donner l’heure et me dire quand passe mon prochain bus ?", "", {CLOCK, TRANSIT}),
    ("Raconte l’histoire d’un bus qui se prend pour une horloge.", "", set()),
    ("Explique-moi en une phrase pourquoi la pluie tombe.", "", set()),
    ("Et pour celui de demain, c’est à quelle heure ?", "", None),
]


def size(schemas: list) -> int:
    return len(json.dumps(schemas, ensure_ascii=False).encode("utf-8")) if schemas else 0


class ObservedRouter(ToolRouter):
    """Record failed RPCs, including timeouts that production handles by keeping all schemas."""

    def __init__(self, http, config):
        super().__init__(http, config)
        self.failures: list[str] = []

    async def wait_for_health(self):
        async with self.http.get(self.url.removesuffix("/v1/systemone") + "/health") as reply:
            reply.raise_for_status()

    async def _ask(self, state, questions):
        try:
            answers = await super()._ask(state, questions)
            for key, question in questions.items():
                answer = answers[key]
                if question["type"] == "choice":
                    if answer["choice"] not in question["criteria"] or not 0 <= answer["confidence"] <= 1:
                        raise ValueError("invalid choice")
                elif not 0 <= answer["noul"] <= 1:
                    raise ValueError("invalid noul")
            return answers
        except (Exception, asyncio.CancelledError) as exc:
            self.failures.append(type(exc).__name__)
            raise


async def measure(router, tools, cases, *, wait_for_idle=False):
    rows = []
    for request, previous, required in cases:
        router.failures = []
        start = time.perf_counter()
        chosen = await router.select(tools, request, previous)
        elapsed = (time.perf_counter() - start) * 1000
        names = {tool["function"]["name"] for tool in chosen}
        if required is None:
            ok = not chosen or chosen == tools
        else:
            ok = required <= names if required else not names
        rows.append({"request": request, "required": sorted(required) if required is not None else None,
                     "selected": sorted(names), "covered": ok, "exact": names == (required or set()),
                     "full": chosen == tools, "schema_bytes": size(chosen), "ms": round(elapsed, 2),
                     "rpc_errors": router.failures})
        print(f"{'ok' if ok else 'MISS':4} {elapsed:5.0f} ms  {request}  {sorted(names)}", flush=True)
        if wait_for_idle and router.failures:
            # A synchronous server can finish a decision after Localive has timed out.
            # Wait outside the timed turn before admitting the next independent example.
            await router.wait_for_health()
    return rows


def summarize(rows, tools):
    times = sorted(row["ms"] for row in rows)
    exact = sum(row["exact"] for row in rows)
    covered = sum(row["covered"] for row in rows)
    omitted = sum(bool(row["required"]) and not row["covered"] for row in rows)
    full = sum(row["full"] for row in rows)
    failures = sum(bool(row["rpc_errors"]) for row in rows)
    payload = sum(row["schema_bytes"] for row in rows)
    saved = 1 - payload / (len(rows) * size(tools))
    p95 = times[math.ceil(len(rows) * 0.95) - 1]
    return (f"covered {covered}/{len(rows)}, exact {exact}, required tool omitted {omitted}, "
            f"full catalogue {full}, schema bytes saved {saved:.1%}, "
            f"median {statistics.median(times):.0f} ms, p95 {p95:.0f} ms, RPC failures {failures}")


def cpu_engine(model, folder, port):
    binary = Path(__file__).resolve().parents[2] / ".runtime" / "llama" / f"{LLAMA_TAG}-cpu"
    spec = AltBrainConfig(model=str(model.resolve()), llama_dir=str(binary), port=port,
                          context=4096, gpu_layers=0,
                          extra_args="-b 2048 -ub 2048 -fa auto -ctk f16 -ctv f16 "
                                     "--threads 6 --threads-batch 6")
    return Brain(Config(root=Path(folder)), spec, "tool-router")


def write_report(output, data):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


async def bench(port: int, timeout_ms: int = 0, *, model: Path | None = None,
                output: Path | None = None, existing_router: str | None = None,
                limit: int | None = None, wait_for_idle: bool = False) -> None:
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    session = json.loads(Path(__file__).with_name("cornelius_session.json").read_text("utf-8"))
    tools = chat_tools(session["delegation"]["responses"]["tools"]) + [search.TOOL]
    with tempfile.TemporaryDirectory(prefix="localive-tool-routing-") as folder:
        engine = cpu_engine(model, folder, port) if model else None
        async with aiohttp.ClientSession() as http:
            try:
                if engine:
                    await engine.start(http, reuse=False)
                router = ObservedRouter(http, RoutingConfig(port=port, timeout_ms=timeout_ms))
                await router.wait_for_health()
                # Warm the backend on an unrelated request and verify its native choice contract.
                await router._ask({"current_request": "Bonjour"}, {"check": {"type": "choice",
                                  "instructions": "Classify the message.", "criteria": {
                                      "hello": "A greeting", "other": "Another request"}}})
                router = ObservedRouter(http, RoutingConfig(port=port, timeout_ms=timeout_ms))
                groups = {}
                remaining = limit
                for name, cases in (("development", CASES), ("holdout", HOLDOUT), ("new", EXTRA)):
                    if remaining == 0:
                        break
                    cases = cases[:remaining]
                    groups[name] = await measure(router, tools, cases, wait_for_idle=wait_for_idle)
                    print(name, summarize(groups[name], tools), flush=True)
                    if remaining is not None:
                        remaining -= len(cases)
                controls = groups.get("holdout", []) + groups.get("new", [])
                if controls:
                    print("controls", summarize(controls, tools), flush=True)
                if output:
                    data = {"model": model.name if model else existing_router, "timeout_ms": timeout_ms,
                            "wait_for_idle_after_error": wait_for_idle,
                            "full_schema_bytes": size(tools), "groups": groups}
                    await asyncio.to_thread(write_report, output, data)
            finally:
                if engine:
                    engine.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8085)
    parser.add_argument("--timeout-ms", type=int, default=0, help="0 uses Localive's adaptive deadline")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model", type=Path, help="Start a private six-thread CPU server with this GGUF")
    source.add_argument("--existing-router", metavar="NAME", help="Name the running local server in the report")
    parser.add_argument("--output", type=Path, help="Write per-request results as JSON")
    parser.add_argument("--limit", type=int, help="Run only the first N cases, in corpus order")
    parser.add_argument("--wait-for-idle", action="store_true",
                        help="After an RPC error, wait for /health outside timing on a synchronous server")
    args = parser.parse_args()
    asyncio.run(bench(args.port, args.timeout_ms, model=args.model, output=args.output,
                      existing_router=args.existing_router, limit=args.limit, wait_for_idle=args.wait_for_idle))
