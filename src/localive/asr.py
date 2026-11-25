"""Speech recognition with Parakeet Ultra (moondream/parakeet-ultra) through Moondream Photon.

0.6B TDT model post-trained from NVIDIA parakeet-tdt-0.6b-v3: on the 43-phrase French bench
2.5 / 6.3 / 7.5 % WER (clean / noise / reverberant room), ~40 ms per utterance on an RTX 4090.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from .config import Config

log = logging.getLogger(__name__)
UNLOAD_TIMEOUT_S = 40.0

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


class Transcriber:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr")
        self._speech = None
        self._closed = False
        self.label = f"{config.asr.model} (Photon, {config.asr.device})"

    def _load(self) -> None:
        import moondream

        started = time.perf_counter()
        device = {} if self.config.asr.device == "auto" else {"device": self.config.asr.device}
        self._speech = moondream.photon(self.config.asr.model, **device)
        self._speech.__enter__()
        self._decode(np.zeros(16000, dtype=np.float32))  # warm-up: first call compiles kernels
        log.info("Parakeet ready in %.1f s (%s)", time.perf_counter() - started, self.label)

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("Parakeet is closed")
        await asyncio.get_running_loop().run_in_executor(self._executor, self._load)

    def _decode(self, audio: np.ndarray) -> str:
        result = self._speech.transcribe(audio=audio.astype(np.float32), sample_rate=16000)
        return str(result.get("text") or "")

    def _run(self, audio: np.ndarray) -> Transcript:
        started = time.perf_counter()
        text = clean(self._decode(audio))
        return Transcript(text, time.perf_counter() - started)

    async def transcribe(self, audio: np.ndarray) -> Transcript:
        if self._speech is None or self._closed:
            raise RuntimeError("Parakeet is not loaded")
        return await asyncio.get_running_loop().run_in_executor(self._executor, self._run, audio)

    async def aclose(self) -> None:
        """Unload and wait for it: Photon's engine sends its last usage report while stopping,
        which fails ("cannot schedule new futures after shutdown") once Python is exiting."""
        if self._closed:
            return
        self._closed = True
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
