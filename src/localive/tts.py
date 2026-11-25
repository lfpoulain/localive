"""Streaming speech synthesis with Phonon 40m (Gradium) through ``ptts``: Rust, CPU only.

Yields float32 mono audio at ``TTS_RATE`` frame by frame; the Live session resamples it to the
16 kHz Cornelius expects. One synthesis at a time (sentences are queued anyway); generation
runs on a worker thread and stops as soon as the reply is interrupted.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections.abc import AsyncIterator

import numpy as np

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


class Synthesizer:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._tts = None
        self._lock = threading.Lock()
        self.label = f"Phonon {config.voice.model.removeprefix('hf://gradium/')} (CPU)"

    def _voice(self, gender: str) -> str:
        name = self.config.voice.male if gender == "male" else self.config.voice.female
        if name.endswith(".safetensors"):  # a cloned / designed voice file
            path = self.config.path(name)
            self._tts.add_voice(path.stem, str(path))
            return path.stem
        return name

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

    async def start(self) -> None:
        await asyncio.get_running_loop().run_in_executor(None, self._load)

    async def stream(self, text: str, gender: str) -> AsyncIterator[np.ndarray]:
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
