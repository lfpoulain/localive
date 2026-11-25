"""Text helpers: streaming sentence chunking and French speech preparation."""

from __future__ import annotations

import re
import unicodedata

from num2words import num2words


def normalize(text: str) -> list[str]:
    folded = "".join(
        c for c in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(c)
    )
    return re.sub(r"[^\w]+", " ", folded).split()

# ---------------------------------------------------------------- chunking

_BOUNDARY = re.compile(r"(?<=[.!?…;:])[\"»”)\]]*\s+|\n+")
_ABBREVIATIONS = ("M.", "Mme.", "Dr.", "St.", "etc.", "p.ex.", "env.", "av.", "apr.")


class SentenceChunker:
    """Accumulate streamed LLM text and release speakable chunks early.

    The first chunk is released as soon as a clause is complete, so the voice can
    start while the model is still writing; later chunks are whole sentences.
    """

    def __init__(self, first_min: int = 12, min_chars: int = 40, max_chars: int = 240) -> None:
        self.buffer = ""
        self.first = True
        self.first_min = first_min
        self.min_chars = min_chars
        self.max_chars = max_chars

    def push(self, delta: str) -> list[str]:
        self.buffer += delta
        out: list[str] = []
        while True:
            piece = self._take()
            if piece is None:
                return out
            out.append(piece)

    def _take(self) -> str | None:
        minimum = self.first_min if self.first else self.min_chars
        for match in _BOUNDARY.finditer(self.buffer):
            end = match.end()
            candidate = self.buffer[:end].strip()
            if len(candidate) < minimum or candidate.endswith(_ABBREVIATIONS):
                continue
            self.buffer = self.buffer[end:]
            self.first = False
            return candidate
        if len(self.buffer) > self.max_chars:  # no punctuation: cut at the last comma/space
            cut = max(
                self.buffer.rfind(", ", 0, self.max_chars),
                self.buffer.rfind(" ", 0, self.max_chars),
            )
            if cut > 0:
                candidate, self.buffer = self.buffer[: cut + 1].strip(), self.buffer[cut + 1 :]
                self.first = False
                return candidate
        return None

    def flush(self) -> str:
        rest, self.buffer = self.buffer.strip(), ""
        return rest


# ---------------------------------------------------------- speech cleanup

_EMOJI = re.compile(
    "[\U0001f000-\U0001faff\U00002600-\U000027bf\U0001f900-\U0001f9ff\u200d\ufe0f]+"
)


def clean_for_speech(text: str) -> str:
    """Remove what a voice cannot say: markdown, emojis, bullets, links."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"(\*\*|__|\*|_)(?=\S)(.+?)(?<=\S)\1", r"\2", text)
    text = re.sub(r"^\s*(?:#+|[-*•]|\d+[.)])\s+", "", text, flags=re.M)
    text = _EMOJI.sub("", text)
    text = text.replace("*", "").replace("#", "")
    return re.sub(r"\s+", " ", text).strip()


def _feminine(number: int) -> str:
    words = num2words(number, lang="fr")
    return words[:-2] + "une" if words.endswith("un") else words


def _clock(match: re.Match) -> str:
    hour, minute = int(match.group("h")), int(match.group("m") or 0)
    result = f"{_feminine(hour)} heure{'s' if hour > 1 else ''}"
    if minute:
        result += f" {_feminine(minute)}"
    return result


_CLOCK = re.compile(r"\b(?P<h>[01]?\d|2[0-3])\s*(?:h|:)\s*(?P<m>[0-5]\d)?\b(?!\s*(?:min|%))")
_PROTECTED = re.compile(r"https?://\S+|www\.\S+|[\w.+-]+@[\w-]+\.[\w.]+")
_DECIMAL = re.compile(r"(?<![\w.,])(-?\d+)[,.](\d+)(?![\w.,]*\d)")
_NUMBER = re.compile(r"(?<![\w])(-?)(\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)(\s*%)?(?![\w])")
_SYMBOLS = {
    "°C": " degrés",
    "°": " degrés",
    "€": " euros",
    "$": " dollars",
    "km/h": " kilomètres heure",
    " × ": " fois ",
    " x ": " fois ",
    " * ": " fois ",
    " ÷ ": " divisé par ",
    " / ": " divisé par ",
    " + ": " plus ",
    " = ": " égale ",
    " − ": " moins ",
    "&": " et ",
}


def speakable_french(text: str) -> str:
    """Spell out times, numbers and symbols so the French TTS reads them correctly."""
    text = clean_for_speech(text)
    text = re.sub(r"\s*[«“”»\"]\s*", " ", text)
    text = re.sub(r"\s+([.,…])", r"\1", text)
    parts, last = [], 0
    for match in _PROTECTED.finditer(text):
        parts.append(_prose(text[last : match.start()]))
        parts.append(match.group())
        last = match.end()
    parts.append(_prose(text[last:]))
    return re.sub(r"\s{2,}", " ", "".join(parts)).strip()


_ORDINAL = re.compile(r"\b(\d+)(er|re|ère|e|ème)\b")


def _ordinal(match: re.Match) -> str:
    number, suffix = int(match.group(1)), match.group(2)
    if number == 1:
        return "première" if suffix in {"re", "ère"} else "premier"
    return num2words(number, lang="fr", to="ordinal")


def _prose(span: str) -> str:
    span = _ORDINAL.sub(_ordinal, span)
    span = _CLOCK.sub(_clock, span)
    for symbol, words in _SYMBOLS.items():
        span = span.replace(symbol, words)
    span = re.sub(r"(?<=\d) - (?=\d)", " moins ", span)
    span = _DECIMAL.sub(
        lambda m: (
            f"{num2words(int(m.group(1)), lang='fr')} virgule "
            + " ".join(
                [num2words(int(m.group(2)), lang="fr")]
                if not m.group(2).startswith("0")
                else [num2words(int(d), lang="fr") for d in m.group(2)]
            )
        ),
        span,
    )

    def number(m: re.Match) -> str:
        value = int(re.sub(r"\D", "", m.group(2)))
        words = num2words(value, lang="fr")
        return ("moins " if m.group(1) else "") + words + (" pour cent" if m.group(3) else "")

    return _NUMBER.sub(number, span)


def speakable(text: str, language: str) -> str:
    return speakable_french(text) if language.startswith("fr") else clean_for_speech(text)
