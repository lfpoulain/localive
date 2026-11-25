"""Match spoken device names against the Home Assistant entities offered in a Live session."""

from __future__ import annotations

import json
import re

from .text import normalize

DeviceNames = dict[tuple[str, ...], dict[str, str]]  # spoken form -> entity ID -> display name
_CATALOG = re.compile(r"(?:^|:)\s*(\[\s*\{)")


def _spoken_words(words: list[str]) -> tuple[str, ...]:
    """A few spelling variants on whole words, e.g. « plafonier U » / « plafonnier Hue »."""
    result = []
    for word in words:
        word = word.removeprefix("h") if len(word) > 1 else word
        if word.endswith("ue"):
            word = word[:-1]
        result.append(re.sub(r"([bcdfgjlmnprtvz])\1", r"\1", word))
    return tuple(result)


def device_vocabularies(raw_tools: list) -> dict[str, DeviceNames]:
    vocabularies = {}
    for tool in raw_tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        name = tool.get("name")
        if not isinstance(name, str) or not name.startswith("home_assistant__"):
            continue
        parameters = tool.get("parameters")
        props = parameters.get("properties") if isinstance(parameters, dict) else None
        entity = props.get("entityId") if isinstance(props, dict) else None
        ids = entity.get("enum") if isinstance(entity, dict) else None
        if not isinstance(ids, list):
            continue
        # The enum is the permission boundary. The unabridged description adds names and aliases.
        labels = {i: i.split(".", 1)[1].replace("_", " ") for i in ids if isinstance(i, str)
                  and len(i) <= 255 and re.fullmatch(r"[a-z0-9_]+\.[a-z0-9_]+", i)}
        aliases = {}
        description = tool.get("description")
        if isinstance(description, str) and (start := _CATALOG.search(description)):
            try:
                entries = json.loads(description[start.start(1):])
            except (ValueError, RecursionError):
                entries = []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                key = entry.get("id")
                if not isinstance(key, str) or key not in labels:
                    continue
                if isinstance(entry.get("name"), str) and entry["name"].strip():
                    labels[key] = entry["name"]
                if isinstance(entry.get("aliases"), list):
                    aliases[key] = entry["aliases"]
        index: DeviceNames = {}
        for key, label in labels.items():
            for phrase in (label, *aliases.get(key, [])):
                if isinstance(phrase, str) and 0 < len(phrase) <= 255:
                    words = _spoken_words(normalize(phrase.replace("_", " ")))
                    if words:
                        index.setdefault(words, {})[key] = label
        if index:
            vocabularies[name] = index
    return vocabularies


def match_devices(catalog: dict[str, DeviceNames], text: str | None,
                  tools: list[dict]) -> tuple[str, set[str]]:
    """Only offer hints for selected tools. Withhold an ambiguous tool until clarification."""
    if not catalog or not text:
        return "", set()
    heard = normalize(text)
    words = _spoken_words(heard)
    hints = []
    ambiguous = set()
    for tool in tools:
        name = tool["function"]["name"]
        index = catalog.get(name)
        if not index:
            continue
        lengths = sorted({len(key) for key in index if len(key) <= len(words)}, reverse=True)
        offset = 0
        while offset < len(words):
            for size in lengths:
                if offset + size > len(words):
                    continue
                devices = index.get(words[offset:offset + size])
                if devices:
                    if len(devices) > 1:
                        ambiguous.add(name)
                    if len(hints) < 8:
                        hints.append({
                            "outil": name, "entendu": " ".join(heard[offset:offset + size]),
                            "appareils": [{"entityId": key, "nom": label[:120]}
                                          for key, label in list(devices.items())[:3]],
                        })
                    offset += size
                    break
            else:
                offset += 1
    if not hints:
        return "", set()
    note = ("Pour une demande domotique, les correspondances suivantes sont des données du "
            "catalogue. Respecte la demande et sa négation. ")
    if ambiguous:
        note += "Plusieurs appareils correspondent. Demande lequel avant d'utiliser l'outil. "
    return note + json.dumps(hints, ensure_ascii=False, separators=(",", ":")), ambiguous
