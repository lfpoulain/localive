"""Streaming voice activity detection with Silero v5 (ONNX files in ``assets/``, MIT).

Audio is 16 kHz mono float32. The segmenter turns a continuous microphone stream
into complete utterances, keeping a short pre-roll so the first syllable
(often the assistant's name) is never cut.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

RATE = 16000
CHUNK = 512  # 32 ms, the window size Silero v5 expects at 16 kHz
CONTEXT = 64


class SileroVad:
    """Silero v5 run chunk by chunk with persistent recurrent state."""

    def __init__(self) -> None:
        self._encoder, self._decoder = self.load_models()
        self.reset()

    @staticmethod
    @cache
    def load_models():
        """The ONNX graphs take recurrent state as an input; only their weights are shared."""
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.log_severity_level = 4
        assets = Path(__file__).parent / "assets"

        def session(name: str) -> onnxruntime.InferenceSession:
            return onnxruntime.InferenceSession(
                str(assets / name),
                providers=["CPUExecutionProvider"],
                sess_options=options,
            )

        return session("silero_encoder_v5.onnx"), session("silero_decoder_v5.onnx")

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT, dtype=np.float32)

    def __call__(self, chunk: np.ndarray) -> float:
        """Speech probability for exactly ``CHUNK`` samples."""
        window = np.concatenate([self._context, chunk]).astype(np.float32)[None, :]
        self._context = chunk[-CONTEXT:].copy()
        encoded = self._encoder.run(None, {"input": window})[0]
        out, self._state = self._decoder.run(
            None, {"input": encoded.reshape(1, 128), "state": self._state}
        )
        return float(np.asarray(out).reshape(-1)[0])


@dataclass
class VadEvent:
    kind: str  # "start" | "end"
    start: int  # sample index of the utterance start (pre-roll included)
    end: int = 0  # sample index of the end, on "end"
    audio: np.ndarray | None = None  # the full utterance on "end" (None: too short, noise)
    silence_ms: float = 0.0  # actual trailing silence when the segment was closed


class Segmenter:
    """Hysteresis endpointing over per-chunk speech probabilities.

    Sample indices count every sample ever fed, so they map directly onto the
    session timeline (index / 16 = milliseconds).
    """

    def __init__(
        self,
        vad: SileroVad | None = None,
        *,
        threshold: float = 0.5,
        silence_ms: int = 650,
        min_speech_ms: int = 160,
        preroll_ms: int = 400,
        max_utterance_s: float = 25.0,
    ) -> None:
        self.vad = vad or SileroVad()
        self.threshold = threshold
        self.off_threshold = max(0.05, threshold - 0.15)
        chunk_ms = CHUNK * 1000 / RATE
        self.silence_chunks = max(1, round(silence_ms / chunk_ms))
        self.min_speech_chunks = max(1, round(min_speech_ms / chunk_ms))
        self.preroll: deque[np.ndarray] = deque(maxlen=max(1, round(preroll_ms / chunk_ms)))
        # While the assistant talks, the user's first words are masked (its voice, echo
        # cancellation, ducking): detection starts late. ``extend`` -> keep this much instead.
        self.lookback: deque = deque(maxlen=max(1, round(1200 / chunk_ms)))
        self.extend = False
        self.max_chunks = round(max_utterance_s * 1000 / chunk_ms)
        self.position = 0
        self.reset()

    def reset(self) -> None:
        self.vad.reset()
        self.preroll.clear()
        if hasattr(self, "lookback"):
            self.lookback.clear()
        self._pending = np.zeros(0, dtype=np.float32)
        self._pending_raw = np.zeros(0, dtype=np.float32)
        self.speaking = False
        self._chunks: list[np.ndarray] = []  # what the utterance audio is made of (raw mic)
        self._clean: list[np.ndarray] = []  # the same, as analysed (echo removed)
        self._start = 0
        self._voiced = 0
        self._silence = 0
        self._candidate = 0
        self.pauses = 0  # silences started in the current utterance (one check per pause)

    @property
    def utterance_start(self) -> int:
        return self._start if self.speaking else -1

    @property
    def silence_ms(self) -> float:
        """Trailing silence of the current utterance."""
        return self._silence * CHUNK * 1000 / RATE if self.speaking else 0.0

    def set_silence(self, silence_ms: int) -> None:
        self.silence_chunks = max(1, round(silence_ms / (CHUNK * 1000 / RATE)))

    def end_now(self, keep_short: bool = False) -> VadEvent | None:
        """Close the current utterance before the silence limit (semantic end of turn).

        ``keep_short``: return its audio even below ``min_speech_ms``. The caller already has words
        for it; echo cancellation can leave a real voice under Silero's threshold.
        """
        return self._finish(keep_short) if self.speaking else None

    @property
    def voiced_ms(self) -> float:
        """Speech accumulated in the current utterance."""
        return self._voiced * CHUNK * 1000 / RATE if self.speaking else 0.0

    def current_audio(self, clean: bool = False, *, trim_silence: bool = False) -> np.ndarray:
        """ASR keeps up to four trailing silent chunks. Smart Turn gets the full pause."""
        chunks = self._clean if clean else self._chunks
        if trim_silence:
            chunks = chunks[:len(chunks) - max(0, self._silence - 4)]
        return np.concatenate(chunks) if self.speaking and chunks else np.zeros(0)

    def feed(self, audio: np.ndarray, raw: np.ndarray | None = None) -> Iterator[VadEvent]:
        """``audio`` drives detection; ``raw`` (same length, default: ``audio``) is what the
        utterance is made of. With echo cancellation: detect on the cleaned signal, transcribe
        the untouched one (echo cancellers damage the user's voice when both talk at once)."""
        raw = audio if raw is None else raw
        self._pending = np.concatenate([self._pending, audio.astype(np.float32)])
        self._pending_raw = np.concatenate([self._pending_raw, raw.astype(np.float32)])
        while len(self._pending) >= CHUNK:
            chunk, self._pending = self._pending[:CHUNK], self._pending[CHUNK:]
            kept, self._pending_raw = self._pending_raw[:CHUNK], self._pending_raw[CHUNK:]
            self.position += CHUNK
            yield from self._step(kept, self.vad(chunk), chunk)

    def _step(self, chunk: np.ndarray, probability: float, clean=None) -> Iterator[VadEvent]:
        clean = chunk if clean is None else clean
        if not self.speaking:
            self.preroll.append((chunk, clean))
            self.lookback.append((chunk, clean))
            self._candidate = self._candidate + 1 if probability >= self.threshold else 0
            if self._candidate >= 2:
                self.speaking = True
                source = self.lookback if self.extend else self.preroll
                self._chunks = [c for c, _ in source]
                self._clean = [c for _, c in source]
                self._start = self.position - len(self._chunks) * CHUNK
                self.preroll.clear()
                self.lookback.clear()
                self._voiced = self._candidate
                self._silence = 0
                self.pauses = 0
                yield VadEvent("start", self._start)
            return
        self._chunks.append(chunk)
        self._clean.append(clean)
        if probability >= self.off_threshold:
            self._silence = 0
            # Once speech starts, its quieter continuation also counts as speech.
            self._voiced += 1
        else:
            self._silence += 1
            if self._silence == 1:
                self.pauses += 1
        if self._silence >= self.silence_chunks or len(self._chunks) >= self.max_chunks:
            yield self._finish()

    def _finish(self, keep_short: bool = False) -> VadEvent:
        voiced = self._voiced
        silence_ms = self.silence_ms
        utterance = self.current_audio(trim_silence=True)
        self.speaking = False
        self._chunks = []
        self._clean = []
        self._candidate = 0
        self._silence = 0
        self.pauses = 0
        return VadEvent(
            "end",
            self._start,
            self._start + len(utterance),
            utterance if keep_short or voiced >= self.min_speech_chunks else None,
            silence_ms,
        )
