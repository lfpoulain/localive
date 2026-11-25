"""PCM helpers and a small streaming polyphase resampler."""

from __future__ import annotations

import base64
from math import gcd

import numpy as np


def pcm16_to_float(data: bytes) -> np.ndarray:
    usable = len(data) - len(data) % 2
    return np.frombuffer(data[:usable], dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm16(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def b64encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def b64decode(text: str) -> bytes:
    return base64.b64decode(text, validate=False)


class Resampler:
    """Rational-ratio resampler (zero-stuffing + windowed-sinc FIR + decimation).

    Keeps its filter history between calls, so a stream can be fed in chunks of
    any size without clicks at the boundaries.
    """

    def __init__(self, source_rate: int, target_rate: int, taps_per_phase: int = 24) -> None:
        divisor = gcd(source_rate, target_rate)
        self.up = target_rate // divisor
        self.down = source_rate // divisor
        self.identity = self.up == self.down == 1
        length = taps_per_phase * self.up
        cutoff = 0.5 / max(self.up, self.down) * 0.92  # of the upsampled rate
        n = np.arange(length) - (length - 1) / 2
        kernel = 2 * cutoff * np.sinc(2 * cutoff * n) * np.kaiser(length, 8.0)
        self.kernel = (kernel / kernel.sum() * self.up).astype(np.float32)
        self.history = np.zeros(length - 1, dtype=np.float32)
        self.phase = 0  # index, in the upsampled stream, of the next sample to keep

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        if self.identity or not len(samples):
            return samples.astype(np.float32)
        stuffed = np.zeros(len(samples) * self.up, dtype=np.float32)
        stuffed[:: self.up] = samples
        extended = np.concatenate([self.history, stuffed])
        filtered = np.convolve(extended, self.kernel, mode="valid")
        self.history = extended[len(extended) - len(self.history) :]
        picked = filtered[self.phase :: self.down]
        self.phase = (self.phase - len(filtered)) % self.down
        return picked.astype(np.float32)
