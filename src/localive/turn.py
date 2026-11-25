"""Semantic end-of-turn detection with Pipecat Smart Turn v3 (ONNX, BSD-2, 8–32 MB, CPU).

Silero only hears silence; Smart Turn listens to the whole utterance (intonation, rhythm,
unfinished words) and says whether the speaker has finished. Input: up to the last 8 s of
16 kHz audio, as Whisper log-mel features (80 x 800), computed here in numpy so that the
``transformers`` package is not needed.
"""

from __future__ import annotations

import time
from functools import cache
from pathlib import Path

import numpy as np

RATE = 16000
N_FFT = 400
HOP = 160
N_MELS = 80
SECONDS = 8


@cache
def _mel_filters() -> np.ndarray:
    """librosa.filters.mel(sr=16000, n_fft=400, n_mels=80, htk=False, norm="slaney")."""

    def hz_to_mel(f):
        f = np.asarray(f, dtype=np.float64)
        mels = f / (200.0 / 3)
        log_region = f >= 1000.0
        return np.where(
            log_region, 15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) / (np.log(6.4) / 27.0), mels
        )

    def mel_to_hz(m):
        m = np.asarray(m, dtype=np.float64)
        freqs = m * (200.0 / 3)
        log_region = m >= 15.0
        return np.where(log_region, 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0)), freqs)

    fft_freqs = np.linspace(0, RATE / 2, N_FFT // 2 + 1)
    mel_points = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(RATE / 2), N_MELS + 2))
    differences = np.diff(mel_points)
    ramps = mel_points[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / differences[:-1, None]
    upper = ramps[2:] / differences[1:, None]
    weights = np.maximum(0, np.minimum(lower, upper))
    weights *= (2.0 / (mel_points[2 : N_MELS + 2] - mel_points[:N_MELS]))[:, None]
    return weights.astype(np.float32)


def features(audio: np.ndarray) -> np.ndarray:
    """Whisper log-mel of the last 8 s (zero-padded at the start), shape (1, 80, 800)."""
    samples = SECONDS * RATE
    audio = np.asarray(audio, dtype=np.float32)[-samples:]
    audio = np.pad(audio, (samples - len(audio), 0))
    audio = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)  # do_normalize=True
    padded = np.pad(audio, (N_FFT // 2, N_FFT // 2), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP]
    window = np.hanning(N_FFT + 1)[:-1].astype(np.float32)  # periodic Hann
    power = np.abs(np.fft.rfft(frames * window, axis=-1)) ** 2
    mel = _mel_filters() @ power.T[:, :-1].astype(np.float32)  # drop the last frame (800)
    log = np.log10(np.maximum(mel, 1e-10))
    log = np.maximum(log, log.max() - 8.0)
    return ((log + 4.0) / 4.0)[None].astype(np.float32)


_OPEN_WORDS = {
    "de", "d", "du", "des", "à", "a", "au", "aux", "le", "la", "les", "l", "un", "une", "pour",
    "avec", "que", "qu", "et", "ou", "mais", "dans", "sur", "en", "mon", "ma", "mes", "ton", "ta",
    "tes", "son", "sa", "ses", "est", "sont", "quel", "quelle", "quels", "quelles", "si", "me",
    "te", "se", "je", "j", "tu", "il", "elle", "on", "nous", "vous", "par", "ce", "cette", "ces",
    "très", "plus", "euh", "heu", "hum", "hmm", "attends", "genre", "donc", "parce", "comme",
}  # fmt: skip


def ends_open(text: str) -> bool:
    """The transcript stops on a word that cannot end a French sentence ("…pour la", "euh")."""
    words = text.lower().replace("’", "'").replace("'", " ").split()
    last = words[-1].strip(' ,.;:!?…-"«»') if words else ""
    return bool(last) and (last in _OPEN_WORDS or text.rstrip().endswith(("…", ",", "...")))


class SmartTurn:
    def __init__(self, model: str | Path, threads: int = 2) -> None:
        # Zero means 'use all cores' to ONNX Runtime, starving ASR and synthesis.
        if threads < 1:
            raise ValueError("Smart Turn threads must be at least 1")
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = threads
        options.log_severity_level = 4
        self.session = onnxruntime.InferenceSession(
            str(model), sess_options=options, providers=["CPUExecutionProvider"]
        )

    def __call__(self, audio: np.ndarray) -> tuple[float, float]:
        """(probability that the turn is complete, milliseconds spent)."""
        started = time.perf_counter()
        output = self.session.run(None, {"input_features": features(audio)})[0]
        return float(np.asarray(output).reshape(-1)[0]), (time.perf_counter() - started) * 1000
