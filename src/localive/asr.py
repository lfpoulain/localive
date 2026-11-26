"""Speech recognition: Parakeet Ultra (Photon), OpenAI-compatible HTTP (Speaches / Faster-Whisper)
or Wyoming Whisper.
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import aiohttp
import numpy as np

from . import wyoming
from .config import Config

log = logging.getLogger(__name__)
UNLOAD_TIMEOUT_S = 40.0
WYOMING_CHUNK_BYTES = 8192

# Phrases speech models are known to invent on silence, breathing or background noise.
_HALLUCINATIONS = re.compile(
    r"(?:merci d['’]avoir regardé(?: (?:cette|la) vidéo)?|"
    r"abonnez[- ]vous(?: à (?:ma|notre|la) chaîne)?|"
    r"thanks for watching(?: this video)?|"
    r"sous[- ]titres?(?: réalisés?| faits?)? par (?:la communauté )?amara\.org)"
    r"[\s.!…]*",
    re.IGNORECASE,
)


@dataclass
class Transcript:
    text: str
    latency_s: float


def clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    # Match standalone boilerplate only. Searching for fragments silently erased
    # real requests such as "Active les sous-titres" or "Traduis abonnez-vous".
    return "" if _HALLUCINATIONS.fullmatch(text) or not re.search(r"\w", text) else text


def encode_pcm16(audio: np.ndarray) -> bytes:
    """Convert mono float32 audio in [-1, 1] to little-endian signed 16-bit PCM bytes."""
    clipped = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def encode_wav_pcm16(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Encode mono float32 audio into a standard 16-bit RIFF/WAVE container."""
    pcm = encode_pcm16(audio)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buf.getvalue()


def extract_confident_transcript(payload: dict[str, Any]) -> str:
    """Filter low-confidence or silence-hallucinated Whisper segments from ``verbose_json``."""
    segments = payload.get("segments")
    if not isinstance(segments, list) or not segments:
        text = payload.get("text")
        return text.strip() if isinstance(text, str) else ""
    kept: list[str] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        text = seg.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        raw_no_speech = seg.get("no_speech_prob")
        raw_logprob = seg.get("avg_logprob")
        raw_compression = seg.get("compression_ratio")
        no_speech = float(raw_no_speech) if isinstance(raw_no_speech, int | float) else 0.0
        avg_logprob = float(raw_logprob) if isinstance(raw_logprob, int | float) else 0.0
        compression = float(raw_compression) if isinstance(raw_compression, int | float) else 1.0
        if no_speech > 0.72:
            continue
        if no_speech > 0.55 and avg_logprob < -0.95:
            continue
        if avg_logprob < -1.25 or compression > 2.4:
            continue
        kept.append(text.strip())
    return " ".join(kept).strip()


def _openai_transcriptions_url(base_url: str) -> str:
    clean_url = base_url.strip().rstrip("/")
    if clean_url.endswith("/audio/transcriptions"):
        return clean_url
    if clean_url.endswith("/v1"):
        return f"{clean_url}/audio/transcriptions"
    return f"{clean_url}/v1/audio/transcriptions"


def _openai_models_url(base_url: str) -> str:
    clean_url = base_url.strip().rstrip("/")
    if clean_url.endswith("/audio/transcriptions"):
        return clean_url.removesuffix("/audio/transcriptions") + "/models"
    if clean_url.endswith("/v1"):
        return f"{clean_url}/models"
    return f"{clean_url}/v1/models"


PROBE_TIMEOUT_S = 3.0


class Transcriber:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr")
        self._speech = None
        self._http: aiohttp.ClientSession | None = None
        self._wyoming: tuple[str, int] | None = None
        self._ready = False
        self._closed = False
        self.probe_error: str | None = None
        asr = config.asr
        if asr.backend == "openai":
            model = asr.external_model or "whisper"
            self.label = f"{model} (OpenAI, {asr.external_url.rstrip('/')})"
        elif asr.backend == "wyoming":
            self.label = f"Wyoming Whisper ({asr.external_url.strip()})"
        else:
            self.label = f"{asr.model} (Photon, {asr.device})"

    def _load(self) -> None:
        import moondream

        started = time.perf_counter()
        device = {} if self.config.asr.device == "auto" else {"device": self.config.asr.device}
        self._speech = moondream.photon(self.config.asr.model, **device)
        self._speech.__enter__()
        self._decode(np.zeros(16000, dtype=np.float32))  # warm-up: first call compiles kernels
        log.info("Parakeet ready in %.1f s (%s)", time.perf_counter() - started, self.label)

    async def _probe_openai(self) -> None:
        asr = self.config.asr
        url = _openai_models_url(asr.external_url)
        headers = {"Authorization": f"Bearer {asr.api_key}"} if asr.api_key else {}
        timeout = aiohttp.ClientTimeout(total=PROBE_TIMEOUT_S)
        try:
            async with self._http.get(url, headers=headers, timeout=timeout) as resp:
                if resp.status >= 400:
                    self.probe_error = f"Serveur de transcription HTTP {resp.status} ({url})."
                    log.warning("%s", self.probe_error)
                else:
                    self.probe_error = None
        except Exception as exc:  # noqa: BLE001 - non-blocking startup probe
            self.probe_error = f"Serveur de transcription injoignable ({url}) : {exc}"
            log.warning("%s", self.probe_error)

    async def _probe_wyoming(self) -> None:
        host, port = self._wyoming
        try:
            async with asyncio.timeout(PROBE_TIMEOUT_S):
                _reader, writer = await asyncio.open_connection(host, port)
                writer.close()
                try:
                    await writer.wait_closed()
                except OSError:
                    pass
            self.probe_error = None
        except Exception as exc:  # noqa: BLE001 - non-blocking startup probe
            self.probe_error = f"Serveur Wyoming STT injoignable ({host}:{port}) : {exc}"
            log.warning("%s", self.probe_error)

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("Transcriber is closed")
        backend = self.config.asr.backend
        if backend == "openai":
            self._http = aiohttp.ClientSession()
            await self._probe_openai()
            self._ready = True
            log.info("STT externe prêt (%s)", self.label)
            return
        if backend == "wyoming":
            self._wyoming = wyoming.parse_endpoint(self.config.asr.external_url, "Transcription")
            await self._probe_wyoming()
            self._ready = True
            log.info("STT Wyoming prêt (%s)", self.label)
            return
        await asyncio.get_running_loop().run_in_executor(self._executor, self._load)
        self._ready = True

    def _decode(self, audio: np.ndarray) -> str:
        result = self._speech.transcribe(audio=audio.astype(np.float32), sample_rate=16000)
        return str(result.get("text") or "")

    def _run(self, audio: np.ndarray) -> Transcript:
        started = time.perf_counter()
        text = clean(self._decode(audio))
        return Transcript(text, time.perf_counter() - started)

    async def _transcribe_openai(self, audio: np.ndarray) -> Transcript:
        started = time.perf_counter()
        asr = self.config.asr
        wav_bytes = encode_wav_pcm16(audio, sample_rate=16000)
        form = aiohttp.FormData()
        form.add_field("file", wav_bytes, filename="speech.wav", content_type="audio/wav")
        if asr.external_model.strip():
            form.add_field("model", asr.external_model.strip())
        if asr.language.strip():
            form.add_field("language", asr.language.strip())
        if asr.prompt.strip():
            form.add_field("prompt", asr.prompt.strip())
        if asr.hotwords.strip():
            form.add_field("hotwords", asr.hotwords.strip())
        form.add_field("response_format", "verbose_json")
        form.add_field("temperature", "0")
        # Localive's Silero VAD already segmented the utterance; a second VAD drops short words.
        form.add_field("vad_filter", "false")
        headers = {"Authorization": f"Bearer {asr.api_key}"} if asr.api_key else {}
        url = _openai_transcriptions_url(asr.external_url)
        timeout = aiohttp.ClientTimeout(total=asr.timeout_s)
        owns_session = self._http is None
        session = self._http or aiohttp.ClientSession()
        try:
            async with session.post(url, data=form, headers=headers, timeout=timeout) as resp:
                if resp.status >= 400:
                    detail = (await resp.text())[:200]
                    raise RuntimeError(f"STT HTTP {resp.status} : {detail}")
                payload = await resp.json()
        finally:
            if owns_session:
                await session.close()
        text = clean(extract_confident_transcript(payload if isinstance(payload, dict) else {}))
        return Transcript(text, time.perf_counter() - started)

    async def _transcribe_wyoming(self, audio: np.ndarray) -> Transcript:
        started = time.perf_counter()
        asr = self.config.asr
        host, port = self._wyoming or wyoming.parse_endpoint(asr.external_url, "Transcription")
        pcm = encode_pcm16(audio)
        if not pcm:
            return Transcript("", time.perf_counter() - started)
        async with asyncio.timeout(asr.timeout_s):
            reader, writer = await asyncio.open_connection(host, port)
            try:
                lang = {"language": asr.language.strip()} if asr.language.strip() else {}
                writer.write(wyoming.encode_event("transcribe", lang))
                fmt = {"rate": 16000, "width": 2, "channels": 1}
                writer.write(wyoming.encode_event("audio-start", fmt))
                for offset in range(0, len(pcm), WYOMING_CHUNK_BYTES):
                    chunk = pcm[offset : offset + WYOMING_CHUNK_BYTES]
                    writer.write(wyoming.encode_event("audio-chunk", fmt, chunk))
                writer.write(wyoming.encode_event("audio-stop"))
                await writer.drain()
                while True:
                    event = await wyoming.read_event(reader)
                    if event is None:
                        raise RuntimeError("Connexion Wyoming STT fermée avant la transcription.")
                    if event.type == "transcript":
                        raw = event.data.get("text")
                        text = clean(raw if isinstance(raw, str) else "")
                        return Transcript(text, time.perf_counter() - started)
                    if event.type == "error":
                        detail = str(event.data.get("text") or "erreur inconnue")
                        raise RuntimeError(f"Wyoming STT : {detail}")
            finally:
                writer.close()
                try:
                    await writer.wait_closed()
                except OSError:
                    pass

    async def transcribe(self, audio: np.ndarray) -> Transcript:
        if self._closed:
            raise RuntimeError("Transcriber is closed")
        backend = self.config.asr.backend
        if backend == "openai":
            if not self._ready:
                raise RuntimeError("STT externe non initialisé")
            return await self._transcribe_openai(audio)
        if backend == "wyoming":
            if not self._ready:
                raise RuntimeError("STT Wyoming non initialisé")
            return await self._transcribe_wyoming(audio)
        if self._speech is None:
            raise RuntimeError("Parakeet is not loaded")
        return await asyncio.get_running_loop().run_in_executor(self._executor, self._run, audio)

    async def aclose(self) -> None:
        """Unload and wait for it: Photon's engine sends its last usage report while stopping,
        which fails ("cannot schedule new futures after shutdown") once Python is exiting."""
        if self._closed:
            return
        self._closed = True
        if self._http is not None:
            await self._http.close()
            self._http = None
        if self._speech is not None:
            unloading = asyncio.get_running_loop().run_in_executor(self._executor, self._unload)
            try:
                await asyncio.wait_for(unloading, UNLOAD_TIMEOUT_S)
            except TimeoutError:
                log.warning("Parakeet ne s'est pas arrêté en %.0f s", UNLOAD_TIMEOUT_S)
        self._executor.shutdown(wait=False)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Cancelling an asyncio waiter does not stop native inference already
        # running on this worker. Unload only once it (or startup) has finished.
        self._executor.submit(self._unload)
        self._executor.shutdown(wait=False)

    def _unload(self) -> None:
        if self._speech is not None:
            try:
                self._speech.__exit__(None, None, None)
            except Exception:  # noqa: BLE001 - shutting down anyway
                pass
            self._speech = None
