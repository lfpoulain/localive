"""Minimal asynchronous Wyoming protocol client for Whisper STT and Piper TTS.

Wyoming frames are newline-delimited JSON headers followed by optional ``data_length`` UTF-8 JSON
bytes and ``payload_length`` raw binary bytes.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit


@dataclass(frozen=True)
class WyomingEvent:
    type: str
    data: dict[str, Any]
    payload: bytes = b""


MAX_DATA_BYTES = 1 * 1024 * 1024  # 1 MiB for JSON event metadata
MAX_PAYLOAD_BYTES = 16 * 1024 * 1024  # 16 MiB for binary audio payload


def parse_endpoint(address: str, label: str = "Wyoming") -> tuple[str, int]:
    """Parse ``tcp://host:port``, ``wyoming://host:port`` or ``host:port``."""
    raw = address.strip()
    if not raw:
        raise ValueError(f"{label} : donnez l'adresse du serveur Wyoming (ex. 127.0.0.1:10300).")
    candidate = raw if "://" in raw else f"tcp://{raw}"
    parsed = urlsplit(candidate)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{label} : port Wyoming invalide dans « {raw} ».") from exc
    if (
        parsed.scheme not in {"tcp", "wyoming"}
        or not parsed.hostname
        or port is None
        or not 1 <= port <= 65535
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            f"{label} : adresse Wyoming invalide « {raw} » (attendu : hôte:port ou tcp://hôte:port)."
        )
    return parsed.hostname, port


def encode_event(
    event_type: str,
    data: dict[str, Any] | None = None,
    payload: bytes = b"",
) -> bytes:
    header: dict[str, Any] = {"type": event_type}
    if data:
        header["data"] = data
    if payload:
        header["payload_length"] = len(payload)
    line = (json.dumps(header, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
    return line + payload if payload else line


async def read_event(reader: asyncio.StreamReader) -> WyomingEvent | None:
    line = await reader.readline()
    if not line:
        return None
    header = json.loads(line.decode("utf-8"))
    if not isinstance(header, dict) or not isinstance(header.get("type"), str):
        raise ValueError("En-tête Wyoming invalide.")
    data: dict[str, Any] = {}
    if isinstance(header.get("data"), dict):
        data.update(header["data"])
    data_len = int(header.get("data_length") or 0)
    if data_len < 0 or data_len > MAX_DATA_BYTES:
        raise ValueError(f"Métadonnées Wyoming trop volumineuses ({data_len} octets).")
    if data_len > 0:
        extra = json.loads((await reader.readexactly(data_len)).decode("utf-8"))
        if isinstance(extra, dict):
            data.update(extra)
    payload_len = int(header.get("payload_length") or 0)
    if payload_len < 0 or payload_len > MAX_PAYLOAD_BYTES:
        raise ValueError(f"Charge audio Wyoming trop volumineuse ({payload_len} octets).")
    payload = await reader.readexactly(payload_len) if payload_len > 0 else b""
    return WyomingEvent(type=header["type"], data=data, payload=payload)
