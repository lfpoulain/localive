"""What to do with speech heard while the assistant thinks or talks ("fake duplex").

IGNORE  not for the assistant (mm-hm, TV, another person, its own voice): keep going
ADD     completes or corrects the current request: restart with it
STOP    be quiet / give up
NEW     a new, unrelated request: drop the current answer, handle the new one

The decision is constrained by a grammar. See bench/decision for labelled situations.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from dataclasses import dataclass, field

import aiohttp

from .config import Config

log = logging.getLogger(__name__)

LABELS = ("IGNORE", "ADD", "STOP", "NEW")
GRAMMAR = 'root ::= "IGNORE" | "ADD" | "STOP" | "NEW"'
SYSTEM = """Tu décides de ce que fait l'assistant vocal {assistant_name} lorsqu'une personne \
parle pendant qu'il réfléchit ou répond. Lis la phrase entière dans son contexte.

IGNORE : il continue. La personne parle à quelqu'un d'autre (« Julie, viens voir », \
« Maman, ferme la porte »), c'est du bruit, de la télé, de l'écho ou une réaction d'écoute \
(« mm-hm », « oui oui », « d'accord », « je vois », rires). Une confirmation que l'étape \
précédente a réussi laisse aussi le guidage se poursuivre.
ADD : elle demande de modifier ou de compléter ce que l'assistant doit encore faire. Pour une \
météo, « Non, à Marseille » ou « Demain matin » sont ADD. Pour un récit, « Résume plutôt la \
fin » est ADD. Mentionner un résultat déjà obtenu ne demande pas de compléter la réponse.
NEW : elle adresse une autre demande à l'assistant, même si la phrase commence par clore la \
réponse en cours. Pendant un récit, « Laisse, dis-moi l'heure » est NEW. « Annule le minuteur » \
est NEW : le minuteur est une action distincte.
STOP : elle veut seulement que l'assistant cesse de répondre ou abandonne sa demande en cours. \
Un abandon (« oublie », « ne fais plus rien »), le fait d'avoir obtenu assez d'information ou \
une question de clôture (« On peut s'arrêter ici ? ») peuvent exprimer cette intention. Dire \
qu'on a déjà obtenu les éléments qu'on cherchait clôt la réponse même sans ordre explicite.

Exemples de lecture en contexte :
- Demande : donne beaucoup d'idées de cadeaux. L'assistant énumère des idées. \
« C'est bon » signifie que les idées déjà données suffisent : STOP.
- Demande : guide-moi pour monter l'étagère. L'assistant dit « Vérifie l'alignement avant de \
passer à la suite ». « C'est bon » confirme l'alignement : IGNORE.
- L'assistant continue son explication. « Ah, très bien, je comprends » : IGNORE.
- L'assistant lit les résultats du bilan. « Je crois que j'ai les chiffres que je cherchais. » \
La personne a reçu ce qu'elle voulait et n'attend plus rien : STOP.

Réponds uniquement par IGNORE, ADD, STOP ou NEW."""


@dataclass
class Situation:
    request: str  # what the assistant is working on
    heard: str  # what was just said
    speaking: bool = False  # True: talking; False: still thinking
    said: str = ""  # what the assistant has actually said so far (played audio)
    tool: str = ""  # tool being executed, if any
    history: list[tuple[str, str]] = field(default_factory=list)  # last (user, assistant) turns
    ignored: list[str] = field(default_factory=list)  # recent speech judged not for it

    def describe(self) -> str:
        lines = []
        if self.history:
            lines.append("Échanges précédents :")
            for user, assistant in self.history[-2:]:
                lines.append(
                    f"- utilisateur : « {user[:200]} » / assistant : « {assistant[:200]} »"
                )
        if self.ignored:
            lines.append("Paroles entendues juste avant et jugées non adressées à l'assistant : "
                         + " ; ".join(f"« {t[:120]} »" for t in self.ignored[-2:]))  # fmt: skip
        phase = (
            "l'assistant est en train de parler"
            if self.speaking
            else "l'assistant réfléchit, il n'a encore rien dit"
        )
        lines += [f"État : {phase}.", f"Demande en cours : « {self.request} »"]
        if self.said:
            lines.append(f"Ce que l'assistant a déjà dit : « {self.said[-400:]} »")
        if self.tool:
            lines.append(f"Outil en cours d'exécution : {self.tool}")
        lines.append(f"Ce qui vient d'être dit : « {self.heard} »")
        return "\n".join(lines)


def _reason(exc: BaseException, config) -> str:
    """A timeout has an empty message: say what it means (the GPU was too busy)."""
    if isinstance(exc, TimeoutError):
        return f"over {config.decision.timeout_ms} ms, GPU busy"
    return str(exc) or type(exc).__name__


@dataclass
class Decision:
    label: str | None  # None: model unavailable or too slow (fall back to rules)
    confidence: float = 1.0  # probability of ``label`` among the four labels
    probabilities: dict[str, float] = field(default_factory=dict)
    ms: float = 0.0
    rule: str = ""  # set when a deterministic rule decided instead of the model

    @property
    def unsure(self) -> bool:
        """Low-confidence choice between "not for me" and an action: ask before acting."""
        return self.label in {"ADD", "NEW"} and self.confidence < CONFIRM_BELOW


# Below this confidence an ADD/NEW becomes a yes/no question. Measured on bench/decision
# (calibrate.py): Qwen is rarely unsure, even when wrong, so this is a cheap safety net (0-1
# question on 89 situations), not a fix for its errors.
CONFIRM_BELOW = 0.75

QUESTION_SYSTEM = """Tu es un assistant vocal. Tu as entendu une phrase sans être sûr qu'elle \
t'était adressée. Pose UNE question très courte (moins de 12 mots), en français, pour vérifier \
si l'utilisateur veut que tu le fasses. Exemples : « Tu veux que j'allume la lumière de \
l'entrée ? », « Tu me demandes la météo de Bordeaux ? ». Réponds uniquement par la question."""


class Arbiter:
    def __init__(self, http: aiohttp.ClientSession | None, config: Config) -> None:
        self.http = http
        self.config = config

    async def _post(self, body: dict, timeout_ms: float) -> dict:
        llm = self.config.llm
        headers = {"Authorization": f"Bearer {llm.api_key}"} if llm.api_key else {}
        decision = self.config.decision
        own = self.config.local_arbiter
        url = f"http://127.0.0.1:{decision.port}/v1" if own else llm.url
        async with asyncio.timeout(timeout_ms / 1000):
            async with self.http.post(
                url.rstrip("/") + "/chat/completions", json=body, headers=headers
            ) as response:
                response.raise_for_status()
                data = await response.json()
                if not isinstance(data, dict):
                    raise ValueError("Decision response must be an object")
                return data

    async def decide(self, situation: Situation) -> Decision:
        body = {
            "model": self.config.llm.model,
            "messages": [
                {"role": "system", "content": SYSTEM.format(
                    assistant_name=self.config.decision.assistant_name,
                )},  # fmt: skip
                {"role": "user", "content": situation.describe()},
            ],
            "temperature": 0,
            "max_tokens": 4,
            "grammar": GRAMMAR,
            "cache_prompt": True,
            "logprobs": True,
            "top_logprobs": 8,
        }
        started = time.perf_counter()
        try:
            data = await self._post(body, self.config.decision.timeout_ms)
            label = _content(data)
        except (
            aiohttp.ClientError, TimeoutError, KeyError, IndexError, TypeError, ValueError
        ) as exc:
            reason = _reason(exc, self.config)
            log.warning("Decision unavailable (%s): falling back to rules", reason)
            return Decision(None, ms=(time.perf_counter() - started) * 1000)
        ms = (time.perf_counter() - started) * 1000
        if label not in LABELS:
            return Decision(None, ms=ms)
        # Renormalise the first-token distribution over the four labels. Logprobs are
        # optional on compatible servers; missing/malformed metadata must not lose a
        # perfectly valid decision or crash the microphone processing task.
        # Labels start with distinct letters (I, A, S, N): map every candidate first token, even
        # a partial one ("IGN", "AD"), to the label it begins.
        scores: dict[str, float] = {}
        choice = data["choices"][0]
        metadata = choice.get("logprobs")
        content = metadata.get("content") if isinstance(metadata, dict) else None
        first = content[0] if isinstance(content, list) and content else None
        top = first.get("top_logprobs") if isinstance(first, dict) else None
        top = top if isinstance(top, list) else []
        for candidate in top:
            if not isinstance(candidate, dict) or not isinstance(candidate.get("token"), str):
                continue
            logprob = candidate.get("logprob")
            if not isinstance(logprob, (int, float)) or not math.isfinite(logprob) or logprob > 0:
                continue
            token = candidate["token"].strip()
            owner = next((lab for lab in LABELS if token and lab.startswith(token)), None)
            if owner:
                scores[owner] = scores.get(owner, 0.0) + math.exp(logprob)
        total = sum(scores.values())
        probabilities = {k: v / total for k, v in scores.items()} if total else {label: 1.0}
        return Decision(label, probabilities.get(label, 0.0), probabilities, ms)

    async def plan(self, situation: Situation, remaining: str, generating: bool):
        return await plan(self, situation, remaining, generating)

    async def acknowledge(self, heard: str) -> str:
        return await acknowledge(self, heard)

    async def question(self, heard: str) -> str:
        """A short yes/no question checking that ``heard`` was a request for the assistant."""
        body = {
            "model": self.config.llm.model,
            "messages": [
                {"role": "system", "content": QUESTION_SYSTEM},
                {"role": "user", "content": f"Phrase entendue : « {heard} »"},
            ],
            "temperature": 0.2,
            "max_tokens": 32,
            "cache_prompt": True,
        }
        try:
            data = await self._post(body, 2000)
            text = _content(data).strip('«»" ')
        except (aiohttp.ClientError, TimeoutError, KeyError, IndexError, TypeError, ValueError):
            text = ""
        return text if 3 < len(text) < 120 and text.endswith("?") else "Tu me parles ?"


PLANS = ("AFTER", "FIX", "NOW")
NOW_SYSTEM = """L'assistant vocal est en train de parler ; l'utilisateur l'interrompt. Est-ce \
qu'il demande une action à faire immédiatement dans la maison (allumer, éteindre, ouvrir, \
fermer, monter, baisser, montrer la caméra), signale un problème ou une urgence (quelqu'un à la \
porte, un animal échappé, quelque chose qui brûle), ou demande explicitement d'abandonner ce que \
dit l'assistant (« laisse tomber », « oublie ») ? Réponds uniquement OUI ou NON."""
STILL_SYSTEM = """L'assistant vocal est en train de répondre ; il lui reste encore des choses à \
dire. L'utilisateur vient d'ajouter une phrase. Question : ce qu'il reste à dire est-il toujours \
juste et utile pour l'utilisateur, une fois sa phrase prise en compte ?
- OUI si la phrase ajoute une autre demande à traiter ensuite (« et à Rennes ? », « et le prix ? \
», « après tu me racontes une blague ») : la fin de la réponse en cours reste valable.
- NON si la phrase corrige, restreint ou modifie la demande en cours (autre lieu à la place, \
seulement une partie, autre période, autre quantité, autre forme, autre personnage) : la suite \
prévue ne correspond plus.
Réponds uniquement OUI ou NON."""
YES_NO = 'root ::= "OUI" | "NON"'


async def plan(
    arbiter: Arbiter, situation: Situation, remaining: str, generating: bool
) -> tuple[str, float]:
    """While talking: finish then answer (AFTER), correct now (FIX) or switch now (NOW).

    Two yes/no questions (asked in parallel) generalise better on a 4B model than one
    three-way choice: measured on bench/decision/plan.py, including sentences never used to
    write these prompts.
    """
    user = situation.describe()
    if remaining:
        user += f"\nCe qu'il lui reste à dire : « {remaining[:600]} »"
    if generating:
        user += "\n(La réponse en cours n'est pas finie : il est encore en train de la rédiger.)"
    started = time.perf_counter()
    cue = linking_cue(situation.heard)
    still_task = None
    try:
        # Keep the plan within the same latency budget as classification, rather than
        # adding up to two seconds of silence after the first decision.
        async with asyncio.timeout(arbiter.config.decision.timeout_ms / 1000):
            # The cue already overrides STILL's answer below. Do not evaluate a prompt
            # whose result would be discarded: it competes with the useful prompt for GPU
            # time and evicts the cached classification prompt on the dedicated server.
            if cue is None:
                still_task = asyncio.create_task(
                    _plain(arbiter, STILL_SYSTEM, user, max_tokens=2, grammar=YES_NO)
                )
            urgent = await _plain(arbiter, NOW_SYSTEM, user, max_tokens=2, grammar=YES_NO)
            if urgent == "OUI":
                return "NOW", (time.perf_counter() - started) * 1000
            if urgent != "NON":
                raise ValueError("Invalid urgency decision")
            if cue is not None:
                return cue, (time.perf_counter() - started) * 1000
            still = await still_task
            if still not in {"OUI", "NON"}:
                raise ValueError("Invalid continuation decision")
    except (aiohttp.ClientError, TimeoutError, KeyError, IndexError, TypeError, ValueError) as exc:
        log.warning("Plan unavailable (%s)", _reason(exc, arbiter.config))
        return "FIX", (time.perf_counter() - started) * 1000
    finally:
        # Cancellation must propagate to both HTTP requests; otherwise an obsolete
        # plan keeps occupying an inference slot after the user has interrupted again.
        if still_task is not None:
            if not still_task.done():
                still_task.cancel()
            await asyncio.gather(still_task, return_exceptions=True)
    how = "AFTER" if still == "OUI" else "FIX"
    return how, (time.perf_counter() - started) * 1000


_FIX_WORDS = re.compile(
    r"\b(plut[oô]t|non|en fait|finalement|seulement|uniquement|juste|à la place|au lieu|coupe|"
    r"arrête(?:-toi)? (?:la|à|après)|limite|pas (?:ça|"
    r"celle|celui)|plus (?:court|courte|long|longue|simple|lentement|vite)|moins)\b",
    re.IGNORECASE,
)
_AFTER_START = re.compile(r"^\s*(?:et\b|aussi\b|ensuite\b|après\b|puis\b|au fait\b)", re.IGNORECASE)


def linking_cue(heard: str) -> str | None:
    """French linking words are reliable where the 4B model is not: « et à Rennes ? » adds a
    request (AFTER), « plutôt vendredi », « seulement… », « non, … » correct it (FIX)."""
    if _FIX_WORDS.search(heard):
        return "FIX"
    if _AFTER_START.match(heard):
        return "AFTER"
    return None


ACK_SYSTEM = """Tu es un assistant vocal en train de parler. L'utilisateur vient de te demander \
autre chose ; tu vas d'abord finir ce que tu disais, puis lui répondre. Écris UNE phrase très \
courte (moins de 10 mots) pour le lui dire, en nommant le sujet. Exemples : « Je te donne \
Nantes juste après. », « Pour demain, je te dis ça dans un instant. » Réponds uniquement par \
la phrase."""


async def _plain(arbiter: Arbiter, system: str, user: str, **extra) -> str:
    body = {
        "model": arbiter.config.llm.model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
        "cache_prompt": True,
        **extra,
    }
    data = await arbiter._post(body, 2000)
    return _content(data)


def _content(data: dict) -> str:
    """Validate the small common subset returned by chat-completions servers."""
    text = data["choices"][0]["message"]["content"]
    if not isinstance(text, str):
        raise ValueError("Decision content must be text")
    return text.strip()


async def acknowledge(arbiter: Arbiter, heard: str) -> str:
    """ "Je te donne Nantes juste après." """
    try:
        text = await _plain(arbiter, ACK_SYSTEM, f"Nouvelle demande : « {heard} »", max_tokens=24,
                            temperature=0.2)  # fmt: skip
    except (aiohttp.ClientError, TimeoutError, KeyError, IndexError, TypeError, ValueError):
        text = ""
    text = text.strip('«»" ')
    return text if 3 < len(text) < 90 else "Je te réponds pour ça juste après."


async def warm_up_arbiter(arbiter: Arbiter) -> None:
    """First calls on a fresh llama-server are slow (GPU kernels, empty prompt cache)."""
    situation = Situation(request="Raconte une histoire.", heard="Mm-hm.", speaking=True,
                          said="Il était une fois un dragon.")  # fmt: skip
    timeout, arbiter.config.decision.timeout_ms = arbiter.config.decision.timeout_ms, 10_000
    try:
        await plan(arbiter, situation, "Un jour, un chevalier arriva.", False)
        await arbiter.decide(situation)  # leave the decision prompt in a slot's cache
    except Exception as exc:  # noqa: BLE001 - never block startup
        log.warning("Arbiter warm-up failed: %s", exc)
    finally:
        arbiter.config.decision.timeout_ms = timeout
