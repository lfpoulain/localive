"""Local SystemOne decisions for tool selection and optional spoken hangup."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from dataclasses import dataclass
from time import monotonic
from typing import Any

import aiohttp

from .config import RoutingConfig
from .text import normalize

log = logging.getLogger(__name__)
Tool = dict[str, Any]
# The demo's three functions use 777 bytes; the five-function Cornelius fixture uses 3,197.
MIN_ROUTED_TOOLS = 6
MIN_ROUTED_SCHEMA_BYTES = 3 * 1024
_THANKS = re.compile(r"(?:(ok|okay|oui|d accord) )?(?:super |parfait )?merci(?: beaucoup| bien)?")


def thanks_kind(text: str) -> str | None:
    """Only whole acknowledgements qualify; an action or farewell in the same turn still routes."""
    phrase = " ".join(normalize(text))
    if phrase in {"ok", "okay"}:
        return "bare_assent"
    match = _THANKS.fullmatch(phrase)
    if match:
        return "assent" if match[1] else "thanks"
    return None


_POSSIBLE_GOODBYE = {
    "type": "noul",
    "instructions": "Could this be a farewell or request to end this conversation?",
}
_GOODBYE_SCREEN_THRESHOLD = 0.20
_PENDING_OFFER = {
    "type": "noul",
    "instructions": "Is the assistant explicitly offering to do something or asking the user "
                    "to approve an outstanding action? Completed actions and generic offers "
                    "to help are not pending.",
}
_GOODBYE_QUESTION = {
    "type": "choice",
    "criteria": {
        "hang_up": "The user is signing off or asking to end their call now, explicitly or "
                   "with a closing courtesy. Example French sign-offs: « c’est bon merci », "
                   "« merci c’est bon », « merci, on se reparle demain ». No task remains.",
        "pending": "The speaker still wants an answer or action, including anything to do "
                   "before hanging up, saying goodbye to another person, stopping an activity, "
                   "or keeping the line open.",
        "other": "Everything else, including « merci » or « c'est bon » alone, uncertainty, "
                 "quoted or third-person goodbyes.",
    },
    "instructions": "Classify the intent of the speaker's entire utterance. Hang up only for "
                    "an unambiguous, immediate farewell to this assistant.",
}

_KNOWN_GOODBYE = {
    "type": "choice",
    "criteria": {
        "hang_up": "Finish talking to this assistant now: stop here, end this session, "
                   "disconnect or say goodbye. French: on peut s'arreter la, arreter la session.",
        "other": "Keep talking: a negation, quotation, unfinished thought, a remaining "
                 "request or stopping another activity.",
    },
    "instructions": "Is the speaker directly ending the current voice conversation now?",
}

_AFTER_THANKS = {
    "type": "choice",
    "criteria": {
        "hang_up": "Courtesy exchange is finished. No specific action or question remains. "
                   "The final speaker means that is all, goodbye.",
        "pending": "The assistant proposed a specific action or asked for details. "
                   "The user may be confirming it, or the context is unclear.",
    },
    "instructions": "After thanks, is this a completed exchange or is there still a specific "
                    "action to confirm? A generic offer of further help is a completed exchange.",
}


def should_route_tools(tools: list[Tool] | None) -> bool:
    """Use a decision round for large catalogues of normalized Qwen tool definitions."""
    if not tools or len(tools) < 2:
        return False
    if len(tools) >= MIN_ROUTED_TOOLS:
        return True
    schemas = json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
    return len(schemas.encode("utf-8", errors="backslashreplace")) >= MIN_ROUTED_SCHEMA_BYTES


def _probability(value: object) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("invalid SystemOne probability")
    return float(value)


def _description(tool: Tool) -> str:
    function = tool["function"]
    description = " ".join(function.get("description", "").split(maxsplit=5)[:5])
    return (function["name"] + " " + description[:120]).rstrip()


_LABEL_WORDS = re.compile(r"\w+")
_LABEL_STOP_WORDS = frozenset(
    "une le la les de l des du avec au ou un pour d sur dans ce qui et".split()
)


def _module_description(module: str, family: list[Tool]) -> str:
    actions = []
    for tool in family:
        function = tool["function"]
        action = function["name"].partition("__")[2]
        words = _LABEL_WORDS.findall(function.get("description", "")[:240])[:8]
        hint = " ".join([w for w in words if w.lower() not in _LABEL_STOP_WORDS][:3])[:120]
        actions.append(f"{action} {hint}".strip())
    return (module + " " + ", ".join(actions)).rstrip()


@dataclass
class RouteDecision:
    tools: list[Tool]
    confirm_goodbye: bool = False


class ToolRouter:
    def __init__(self, http: aiohttp.ClientSession, config: RoutingConfig) -> None:
        self.http = http
        self.config = config
        self.url = f"http://127.0.0.1:{config.port}/v1/systemone"
        self._auto_timeout_ms: float | None = None

    @property
    def timeout_ms(self) -> float:
        return self.config.timeout_ms or self._auto_timeout_ms or 8000

    async def _post(self, state: dict, questions: dict) -> dict:
        async with self.http.post(self.url, json={"state": state, "questions": questions}) as r:
            r.raise_for_status()
            return (await r.json())["answers"]

    async def verify(self) -> None:
        """Exercise the native choice endpoint before exposing a newly started router."""
        choices = {"hello": "A greeting", "other": "Another request"}
        answers = await self._post({"current_request": "Bonjour"}, {"check": {
            "type": "choice", "criteria": choices, "instructions": "Classify the message."}})
        answer = answers["check"]
        if answer["choice"] not in choices:
            raise ValueError("Réponse de vérification SystemOne invalide")
        _probability(answer["confidence"])

    async def should_hang_up(self, text: str, assistant_name: str, *,
                             known_phrase: bool = False, screen: bool = False,
                             after_thanks: tuple[str, str] | None = None) -> bool:
        """A disconnect needs a confident decision on the complete user utterance."""
        if not text.strip() or len(text) > 2000:
            return False
        state = {"assistant_name": assistant_name[:80], "user_utterance": text}
        if after_thanks:
            state = {"previous_user": after_thanks[0], "assistant": after_thanks[1],
                     "current_user": text}
        budget = min(self.timeout_ms, 800)
        started = monotonic()
        try:
            async with asyncio.timeout(budget / 1000):
                # A known farewell or a short timeout gets the whole budget for confirmation.
                if screen and not known_phrase and not after_thanks and budget == 800:
                    try:
                        probe = await self._post({"request": text}, {"intent": _POSSIBLE_GOODBYE})
                        possible = _probability(probe["intent"]["noul"])
                    except (aiohttp.ClientResponseError, KeyError, TypeError, ValueError) as exc:
                        # A custom SystemOne runtime may only implement choice questions.
                        log.debug("Présélection d3 indisponible, vérification complète (%r).", exc)
                    else:
                        if possible < _GOODBYE_SCREEN_THRESHOLD:
                            return False
                question = _KNOWN_GOODBYE if known_phrase else _GOODBYE_QUESTION
                if after_thanks:
                    question = _AFTER_THANKS
                answers = await self._post(state, {"goodbye": question})
            answer = answers["goodbye"]
            probability = _probability(answer["probabilities"]["hang_up"])
            log.debug("Raccrochage d3 (%d ms) %s, p(fin)=%.2f",
                      (monotonic() - started) * 1000, answer["choice"], probability)
            # An exact whole-utterance farewell corroborates the model's own positive choice.
            return answer["choice"] == "hang_up" and probability >= (0.70 if known_phrase else 0.80)
        except TimeoutError:
            log.info("Raccrochage d3 hors délai (%.0f ms), conversation ouverte.", budget)
        except (aiohttp.ClientError, OSError, KeyError, TypeError, ValueError) as exc:
            log.warning("Raccrochage d3 indisponible, conversation ouverte (%r).", exc)
        return False

    async def _ask(self, state: dict, questions: dict) -> dict:
        started = monotonic()
        answers = await self._post(state, questions)
        # Leave room for compound requests and for a longer transcript on the next turn.
        # Reduce the budget gradually after slow decisions, growing it immediately on a CPU.
        elapsed_ms = (monotonic() - started) * 1000
        self._auto_timeout_ms = min(8000, max(400, 4 * elapsed_ms,
                                              0.8 * (self._auto_timeout_ms or 0)))
        return answers

    async def select(self, tools: list[Tool], request: str, previous: str = "") -> list[Tool]:
        return (await self.decide(tools, request, previous)).tools

    async def decide(
        self, tools: list[Tool], request: str, previous: str = "", *,
        include_goodbye: bool = False, confirmation_offer: str | None = None,
    ) -> RouteDecision:
        """Route a request, or check the previous assistant reply before routing an assent.

        An uncertain request retains its tools. An uncertain assent grants none.
        Both checks share the turn's timeout and only see text and capability descriptions.
        """
        fallback = tools if confirmation_offer is None else []
        if len(tools) < 2 or not request.strip() or len(request) > 2000:
            return RouteDecision(fallback, confirm_goodbye=include_goodbye)
        subject = request if confirmation_offer is None else confirmation_offer[-600:]
        state = {"current_request": subject}
        if previous:
            state["previous_exchange"] = previous[-450:]
        # On large installations choose between modules rather than hundreds of functions.
        groups: dict[str, list[Tool]] = {}
        for tool in tools:
            name = tool["function"]["name"]
            key = name.partition("__")[0] if len(tools) > 16 else name
            groups.setdefault(key, []).append(tool)
        families = list(groups.values())
        if len(families) > 24:
            return RouteDecision(fallback, confirm_goodbye=include_goodbye)
        compact = include_goodbye and len(tools) > 16 and confirmation_offer is None
        options = {}
        for i, (module, family) in enumerate(groups.items()):
            options[f"t{i}"] = (_module_description(module, family) if compact else
                                " / ".join(_description(t) for t in family))
        criteria = {
            "direct": "Conversation, explanation or arithmetic; no tool.",
            "several": "Several independent capabilities.",
            "unsure": "Unclear capability.",
            **options,
        }
        if include_goodbye:
            criteria["hang_up"] = ("The user explicitly ends this call with the assistant now, "
                                   "including a closing courtesy. No task remains.")
        question = {"type": "choice", "criteria": criteria, "instructions":
                    "Select the capability for the current request. Use history for follow-ups. "
                    "Prefer a specific tool to web search."}
        if compact:
            criteria["hang_up"] = "Ends this call now. No remaining request."
            question["instructions"] = ("Route this request. Use history for follow-ups. "
                                        "Prefer a specific tool to web search.")
        if confirmation_offer is not None:
            question["criteria"] = {
                "direct": "Thanks, no pending action, or action already completed.", **options,
            }
            question["instructions"] = ("Choose the capability needed for the action PROPOSED by "
                                        "the assistant. Choose direct if the reply reports a "
                                        "completed action or has no specific action awaiting "
                                        "approval.")
        budget = self.timeout_ms
        try:
            async with asyncio.timeout(budget / 1000):
                if confirmation_offer is not None:
                    pending = await self._post({"assistant_reply": subject},
                                               {"pending": _PENDING_OFFER})
                    if _probability(pending["pending"]["noul"]) < 0.80:
                        return RouteDecision([])
                answer = (await self._ask(state, {"route": question}))["route"]
            choice = answer["choice"]
            confidence = _probability(answer["confidence"])
            confirm = include_goodbye and (
                choice == "hang_up" or _probability(answer["probabilities"]["hang_up"]) >= 0.05
            )
            if choice == "direct" and confidence >= 0.60:
                return RouteDecision([], confirm)
            if choice not in options or confidence < 0.55:
                return RouteDecision(fallback, confirm)
            selected = families[int(choice[1:])]
        except TimeoutError:
            log.info("Tri des outils interrompu après %.0f ms, catalogue complet.", budget)
            if not self.config.timeout_ms:
                self._auto_timeout_ms = min(8000, self.timeout_ms * 2)
            return RouteDecision(fallback)
        except (aiohttp.ClientError, OSError, KeyError, TypeError,
                ValueError, AttributeError) as exc:
            log.warning("Requête SystemOne en échec, catalogue complet (%r).", exc)
            return RouteDecision(fallback)
        names = {tool["function"]["name"] for tool in selected}
        if "web_search" in names:
            names.add("ui__show_luna_card")  # Cornelius presents the sourced result on a card
        return RouteDecision([t for t in tools if t["function"]["name"] in names], confirm)
