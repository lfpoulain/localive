import numpy as np

from localive.audio import Resampler


def tone(rate: int, seconds: float, freq: float) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def dominant(samples: np.ndarray, rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    return np.fft.rfftfreq(len(samples), 1 / rate)[np.argmax(spectrum)]


def test_24k_to_16k_keeps_pitch_and_level():
    out = Resampler(24000, 16000)(tone(24000, 1.0, 1000))
    assert abs(len(out) - 16000) <= 1
    assert abs(dominant(out[2000:], 16000) - 1000) < 5
    assert 0.45 < np.abs(out[2000:]).max() < 0.55


def test_chunked_equals_whole():
    signal = tone(24000, 0.5, 440)
    whole = Resampler(24000, 16000)(signal)
    stream = Resampler(24000, 16000)
    parts = np.concatenate([stream(signal[i : i + 777]) for i in range(0, len(signal), 777)])
    assert len(parts) == len(whole)
    assert np.allclose(parts, whole, atol=1e-5)


def test_silence_trimmer_shortens_edges_keeps_pauses():
    from localive.live import SilenceTrimmer

    rate = 16000
    voice = tone(rate, 0.5, 300)
    gap = lambda ms: np.zeros(rate * ms // 1000, dtype=np.float32)  # noqa: E731
    sentence = np.concatenate([gap(150), voice, gap(300), voice, gap(700)])
    trim = SilenceTrimmer()
    out = np.concatenate([trim(sentence[i : i + 4000]) for i in range(0, len(sentence), 4000)])
    out = np.concatenate([out, trim.finish()])
    expected = 0.120 + 0.5 + 0.300 + 0.5 + 0.120  # edges cut to 120 ms, comma pause kept
    assert abs(len(out) / rate - expected) < 0.02


def test_aliasing_is_attenuated():
    out = Resampler(24000, 16000)(tone(24000, 1.0, 10000))  # above the new Nyquist (8 kHz)
    assert np.abs(out[2000:]).max() < 0.02
