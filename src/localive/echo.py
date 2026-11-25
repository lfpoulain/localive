"""Server-side acoustic echo cancellation with WebRTC AudioProcessing (AEC3).

Localive knows exactly what the speaker plays: it synthesizes it. Every output frame is fed
to the canceller as the far-end reference when it is *sent*; every microphone chunk goes
through it before Silero, Smart Turn and Parakeet. AEC3 estimates the delay between the two
by itself (network + device buffers + room), so any client works: Cornelius in a browser,
a satellite, a raw microphone. The same engine as Chrome and Meet, through the ``livekit``
Python package (native libwebrtc, CPU, ~0.1 ms per 10 ms frame).
"""

from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np

from .config import Config

log = logging.getLogger(__name__)

RATE = 16000
STEP = 160  # WebRTC processes exactly 10 ms frames


class EchoCanceller:
    """One per microphone/speaker pair. Its adaptive filter takes a few seconds of speech to
    learn the room: ``EchoPool`` hands the same, already trained canceller to the next session
    (Cornelius opens one per conversation, always on the same device)."""

    def __init__(self, config: Config) -> None:
        settings = config.echo
        # Config is editable at runtime. Keep the effective native APM settings so
        # a pooled instance is not silently reused after a filter/latency change.
        self.settings = replace(settings)
        self.enabled = settings.enabled
        self.in_use = False
        self._apm = None
        self._near = np.zeros(0, dtype=np.int16)
        self._far = np.zeros(0, dtype=np.int16)
        if not self.enabled:
            return
        from livekit import rtc

        self._rtc = rtc
        self._apm = rtc.AudioProcessingModule(
            echo_cancellation=True,
            noise_suppression=settings.noise_suppression,
            high_pass_filter=settings.high_pass_filter,
            auto_gain_control=settings.auto_gain_control,
        )
        self._apm.set_stream_delay_ms(settings.delay_hint_ms)

    def reset_buffers(self) -> None:
        self._near = np.zeros(0, dtype=np.int16)
        self._far = np.zeros(0, dtype=np.int16)

    def _frame(self, samples: np.ndarray):
        data = samples.astype("<i2").tobytes()
        return self._rtc.AudioFrame(bytearray(data), RATE, 1, STEP)

    def far_end(self, audio: np.ndarray) -> None:
        """What is being sent to the speaker (float32, 16 kHz)."""
        if self._apm is None:
            return
        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        self._far = np.concatenate([self._far, pcm])
        while len(self._far) >= STEP:
            chunk, self._far = self._far[:STEP], self._far[STEP:]
            self._apm.process_reverse_stream(self._frame(chunk))

    def near_end(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Microphone chunk (float32, 16 kHz) -> (without the assistant's voice, untouched).

        Both are aligned and made of whole 10 ms frames: at most 9.9 ms is held back.
        """
        if self._apm is None:
            return audio, audio
        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        self._near = np.concatenate([self._near, pcm])
        out, raw = [], []
        while len(self._near) >= STEP:
            chunk, self._near = self._near[:STEP], self._near[STEP:]
            raw.append(chunk)
            frame = self._frame(chunk)
            self._apm.process_stream(frame)
            out.append(np.frombuffer(bytes(frame.data), dtype=np.int16))
        if not out:
            empty = np.zeros(0, dtype=np.float32)
            return empty, empty
        clean = np.concatenate(out).astype(np.float32) / 32768.0
        return clean, np.concatenate(raw).astype(np.float32) / 32768.0


class EchoPool:
    """Reuse trained cancellers across sessions; a second concurrent session gets its own."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._free: list[EchoCanceller] = []

    def acquire(self) -> EchoCanceller:
        while self._free:
            canceller = self._free.pop()
            if canceller.settings == self.config.echo:
                canceller.in_use = True
                canceller.reset_buffers()
                return canceller
        canceller = EchoCanceller(self.config)
        canceller.in_use = True
        return canceller

    def release(self, canceller: EchoCanceller) -> None:
        if not canceller.in_use:
            return
        canceller.in_use = False
        if canceller.enabled:
            self._free.append(canceller)
