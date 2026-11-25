"""Streaming speech synthesis: Phonon 40m (ptts, CPU), OpenAI-compatible HTTP (Kokoro-FastAPI /
Speaches) or Wyoming Piper.

Yields float32 mono audio at ``TTS_RATE`` frame by frame; the Live session resamples it to the
16 kHz Cornelius expects. Generation stops promptly as soon as the reply is interrupted.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections.abc import AsyncIterator

import aiohttp
import numpy as np

from . import wyoming
from .audio import Resampler
from .config import Config

TTS_RATE = 24000
# A little look-ahead is enough to hide synthesis jitter. Without
# backpressure a slow client lets Phonon synthesize the entire reply after a barge-in.
BUFFER_CHUNKS = 8
CHUNK_SAMPLES = TTS_RATE * 80 // 1000  # at most 640 ms queued; native chunks grow to >1 s
log = logging.getLogger(__name__)


class TtsError(RuntimeError):
    pass


def _openai_speech_url(base_url: str) -> str:
    clean_url = base_url.strip().rstrip("/")
    if clean_url.endswith("/audio/speech"):
        return clean_url
    if clean_url.endswith("/v1"):
        return f"{clean_url}/audio/speech"
    return f"{clean_url}/v1/audio/speech"


def _openai_models_url(base_url: str) -> str:
    clean_url = base_url.strip().rstrip("/")
    if clean_url.endswith("/audio/speech"):
        return clean_url.removesuffix("/audio/speech") + "/models"
    if clean_url.endswith("/v1"):
        return f"{clean_url}/models"
    return f"{clean_url}/v1/models"


PROBE_TIMEOUT_S = 3.0


def _pcm16_to_float32(raw: bytes, channels: int = 1) -> np.ndarray:
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        usable = (len(samples) // channels) * channels
        samples = samples[:usable].reshape(-1, channels)[:, 0]
    return samples


class Synthesizer:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._tts = None
        self._http: aiohttp.ClientSession | None = None
        self._wyoming: tuple[str, int] | None = None
        self._ready = False
        self.probe_error: str | None = None
        self._lock = threading.Lock()
        voice = config.voice
        if voice.backend == "openai":
            model = voice.external_model or "tts"
            self.label = f"{model} (OpenAI, {voice.external_url.rstrip('/')})"
        elif voice.backend == "wyoming":
            self.label = f"Wyoming Piper ({voice.external_url.strip()})"
        else:
            self.label = f"Phonon {voice.model.removeprefix('hf://gradium/')} (CPU)"

    def _voice(self, gender: str) -> str:
        name = self.config.voice.male if gender == "male" else self.config.voice.female
        if name.endswith(".safetensors"):  # a cloned / designed voice file
            path = self.config.path(name)
            self._tts.add_voice(path.stem, str(path))
            return path.stem
        return name

    def _external_voice(self, gender: str) -> str:
        voice = self.config.voice
        chosen = voice.external_male if gender == "male" else voice.external_female
        return chosen.strip()

    def _load(self) -> None:
        # ptts sizes its Rayon pool to every logical core and ignores set_num_threads():
        # the environment variable must be set before the import.
        os.environ["RAYON_NUM_THREADS"] = str(max(1, self.config.voice.threads))
        import ptts

        from . import hf

        voice = self.config.voice
        model = (
            voice.model if voice.model.startswith("hf://") else str(self.config.path(voice.model))
        )
        started = time.perf_counter()
        try:
            self._tts = ptts.TTS(model, device="cpu", quant="q8", voice=voice.female,
                                 lang=voice.language)  # fmt: skip
        except OSError as exc:
            raise hf.explain(exc, voice.model) from exc
        for gender in ("female", "male"):  # resolve (and download) both voices up front
            for _ in self._tts.stream("Bonjour.", voice=self._voice(gender)):
                pass
        log.info("Phonon ready in %.1f s (%s / %s, %d threads)", time.perf_counter() - started,
                 voice.female, voice.male, voice.threads)  # fmt: skip

    async def _probe_openai(self) -> None:
        voice = self.config.voice
        url = _openai_models_url(voice.external_url)
        headers = {"Authorization": f"Bearer {voice.api_key}"} if voice.api_key else {}
        timeout = aiohttp.ClientTimeout(total=PROBE_TIMEOUT_S)
        try:
            async with self._http.get(url, headers=headers, timeout=timeout) as resp:
                if resp.status >= 400:
                    self.probe_error = f"Serveur vocal HTTP {resp.status} ({url})."
                    log.warning("%s", self.probe_error)
                else:
                    self.probe_error = None
        except Exception as exc:  # noqa: BLE001 - non-blocking startup probe
            self.probe_error = f"Serveur vocal injoignable ({url}) : {exc}"
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
            self.probe_error = f"Serveur Wyoming TTS injoignable ({host}:{port}) : {exc}"
            log.warning("%s", self.probe_error)

    async def start(self) -> None:
        backend = self.config.voice.backend
        if backend == "openai":
            self._http = aiohttp.ClientSession()
            await self._probe_openai()
            self._ready = True
            log.info("TTS externe prêt (%s)", self.label)
            return
        if backend == "wyoming":
            self._wyoming = wyoming.parse_endpoint(self.config.voice.external_url, "Voix")
            await self._probe_wyoming()
            self._ready = True
            log.info("TTS Wyoming prêt (%s)", self.label)
            return
        await asyncio.get_running_loop().run_in_executor(None, self._load)
        self._ready = True

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.close()
            self._http = None

    async def _stream_openai(self, text: str, gender: str) -> AsyncIterator[np.ndarray]:
        if not self._ready:
            raise TtsError("TTS externe non initialisé")
        voice = self.config.voice
        payload: dict[str, object] = {
            "model": voice.external_model or "kokoro",
            "voice": self._external_voice(gender) or "ff_siwis",
            "input": text,
            "response_format": "pcm",
            "stream": True,
        }
        if voice.lang_code.strip():
            payload["lang_code"] = voice.lang_code.strip()
        if voice.speed != 1.0:
            payload["speed"] = voice.speed
        headers = {"Authorization": f"Bearer {voice.api_key}"} if voice.api_key else {}
        url = _openai_speech_url(voice.external_url)
        timeout = aiohttp.ClientTimeout(total=voice.timeout_s)
        owns_session = self._http is None
        session = self._http or aiohttp.ClientSession()
        resampler = Resampler(voice.sample_rate, TTS_RATE)
        carry = b""
        try:
            async with session.post(url, json=payload, headers=headers, timeout=timeout) as resp:
                if resp.status >= 400:
                    detail = (await resp.text())[:200]
                    raise TtsError(f"TTS HTTP {resp.status} : {detail}")
                async for raw in resp.content.iter_chunked(CHUNK_SAMPLES * 2):
                    if not raw:
                        continue
                    data = carry + raw
                    usable = len(data) - (len(data) % 2)
                    carry = data[usable:]
                    if usable == 0:
                        continue
                    output = resampler(_pcm16_to_float32(data[:usable]))
                    for offset in range(0, len(output), CHUNK_SAMPLES):
                        yield output[offset : offset + CHUNK_SAMPLES]
        except TtsError:
            raise
        except Exception as exc:
            raise TtsError(f"TTS externe : {exc}") from exc
        finally:
            if owns_session:
                await session.close()

    async def _stream_wyoming(self, text: str, gender: str) -> AsyncIterator[np.ndarray]:
        if not self._ready:
            raise TtsError("TTS Wyoming non initialisé")
        voice = self.config.voice
        host, port = self._wyoming or wyoming.parse_endpoint(voice.external_url, "Voix")
        chosen = self._external_voice(gender)
        req_data: dict[str, object] = {"text": text}
        if chosen:
            req_data["voice"] = {"name": chosen}
        writer: asyncio.StreamWriter | None = None
        try:
            async with asyncio.timeout(voice.timeout_s):
                reader, writer = await asyncio.open_connection(host, port)
                writer.write(wyoming.encode_event("synthesize", req_data))
                await writer.drain()
                rate = 22050
                channels = 1
                resampler = Resampler(rate, TTS_RATE)
                carry = b""
                while True:
                    event = await wyoming.read_event(reader)
                    if event is None:
                        raise TtsError("Connexion Wyoming TTS fermée prématurément.")
                    if event.type == "audio-start":
                        if isinstance(event.data.get("rate"), int):
                            rate = event.data["rate"]
                        if isinstance(event.data.get("channels"), int):
                            channels = max(1, event.data["channels"])
                        resampler = Resampler(rate, TTS_RATE)
                        carry = b""
                    elif event.type == "audio-chunk" and event.payload:
                        frame_bytes = 2 * channels
                        data = carry + event.payload
                        usable = (len(data) // frame_bytes) * frame_bytes
                        carry = data[usable:]
                        if usable == 0:
                            continue
                        output = resampler(_pcm16_to_float32(data[:usable], channels=channels))
                        for offset in range(0, len(output), CHUNK_SAMPLES):
                            yield output[offset : offset + CHUNK_SAMPLES]
                    elif event.type == "audio-stop":
                        break
                    elif event.type == "error":
                        detail = str(event.data.get("text") or "erreur inconnue")
                        raise TtsError(f"Wyoming TTS : {detail}")
        except TtsError:
            raise
        except Exception as exc:
            raise TtsError(f"Wyoming TTS : {exc}") from exc
        finally:
            if writer is not None:
                writer.close()
                try:
                    await writer.wait_closed()
                except OSError:
                    pass

    async def stream(self, text: str, gender: str) -> AsyncIterator[np.ndarray]:
        backend = self.config.voice.backend
        if backend == "openai":
            async for chunk in self._stream_openai(text, gender):
                yield chunk
            return
        if backend == "wyoming":
            async for chunk in self._stream_wyoming(text, gender):
                yield chunk
            return
        if self._tts is None:
            raise TtsError("Phonon n'est pas chargé")
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[np.ndarray | BaseException | None] = asyncio.Queue(
            maxsize=BUFFER_CHUNKS
        )
        slots = threading.BoundedSemaphore(BUFFER_CHUNKS)
        stop = threading.Event()

        def offer(item: np.ndarray | BaseException | None) -> bool:
            # Do not block the event loop, and wake a backpressured producer promptly
            # when its consumer has been cancelled or disconnected.
            while not stop.is_set():
                if not slots.acquire(timeout=0.05):
                    continue
                if stop.is_set():
                    slots.release()
                    return False
                try:
                    loop.call_soon_threadsafe(queue.put_nowait, item)
                except RuntimeError:  # event loop closed during application shutdown
                    slots.release()
                    stop.set()
                    return False
                return True
            return False

        def produce() -> None:
            acquired = False
            error = None
            try:
                while not stop.is_set():
                    if self._lock.acquire(timeout=0.05):
                        acquired = True
                        break
                if stop.is_set():
                    return
                # Voice registration and native synthesis share the same lock. A
                # cancelled sentence waiting for another session never starts work.
                with self._tts.stream(text, voice=self._voice(gender)) as audio:
                    resampler = Resampler(audio.sample_rate, TTS_RATE)
                    for chunk in audio:
                        if stop.is_set():
                            break  # leaving the `with` stops generation
                        samples = np.asarray(chunk, dtype=np.float32).reshape(-1)
                        # Keep filter state across frames; per-frame interpolation
                        # changes duration and clicks when a model uses another rate.
                        output = resampler(samples)
                        for offset in range(0, len(output), CHUNK_SAMPLES):
                            if not offer(output[offset : offset + CHUNK_SAMPLES]):
                                return
            except BaseException as exc:  # noqa: BLE001 - re-raised in the event loop
                error = exc
            finally:
                if acquired:
                    self._lock.release()
                if error is not None:
                    offer(error)
                offer(None)

        threading.Thread(target=produce, name="phonon", daemon=True).start()
        try:
            while True:
                item = await queue.get()
                slots.release()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise TtsError(f"Phonon : {item}") from item
                yield item
        finally:
            stop.set()
