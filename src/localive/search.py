"""Web search for the hosted ``web_search`` tool, through a SearXNG instance."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import aiohttp

from .config import SearchConfig

TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Rechercher sur le web une information publique ou récente : actualité, résultats "
            "sportifs, horaires d'un lieu, prix, événement, personne, définition à vérifier."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Requête courte, en mots-clés."}
            },
            "required": ["query"],
        },
    },
}


class SearchError(RuntimeError):
    pass


@dataclass
class Result:
    title: str
    url: str
    content: str


def _clean(text: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


async def searxng(http: aiohttp.ClientSession, config: SearchConfig, query: str) -> list[Result]:
    url = config.url.rstrip("/") + "/search"
    params = {"q": query, "format": "json", "language": config.language, "safesearch": "0"}
    try:
        async with http.get(
            url, params=params, timeout=aiohttp.ClientTimeout(total=config.timeout_s)
        ) as response:
            if response.status != 200:
                raise SearchError(f"SearXNG HTTP {response.status} (format json activé ?)")
            data = await response.json(content_type=None)
    except (aiohttp.ClientError, TimeoutError) as exc:
        raise SearchError(f"SearXNG injoignable à {url} : {exc}") from exc
    results: list[Result] = []
    seen: set[str] = set()
    for item in data.get("results") or []:
        link = str(item.get("url") or "")
        if not link.startswith("http") or link in seen:
            continue
        seen.add(link)
        results.append(
            Result(_clean(item.get("title"), 140), link, _clean(item.get("content"), 400))
        )
        if len(results) >= config.max_results:
            break
    # Direct answers and infoboxes (e.g. Wikipedia summaries) are often the best evidence.
    for box in (data.get("infoboxes") or [])[:1]:
        content = _clean(box.get("content"), 600)
        if content:
            results.insert(
                0, Result(_clean(box.get("infobox"), 140), str(box.get("id") or ""), content)
            )
    for answer in (data.get("answers") or [])[:1]:
        text = answer.get("answer") if isinstance(answer, dict) else answer
        if text:
            results.insert(0, Result("Réponse directe", "", _clean(text, 400)))
    return results
