import numpy as np

from localive.config import Config
from localive.echo import EchoCanceller

RATE = 16000


def simulate(enabled: bool, seconds: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    """Far-end noise-like 'speech', heard back 80 ms later through a short room."""
    rng = np.random.default_rng(1)
    far = (rng.normal(0, 0.2, int(RATE * seconds)) * np.repeat(rng.random(int(seconds * 10)) > 0.3,
           RATE // 10)).astype(np.float32)  # fmt: skip
    impulse = np.exp(-np.arange(800) / 120.0) * rng.normal(0, 1, 800)
    impulse[0] = 3
    echo = np.convolve(far, impulse / np.abs(impulse).sum())[: len(far)]
    mic = np.concatenate([np.zeros(80 * RATE // 1000), echo])[: len(far)].astype(np.float32)
    config = Config()
    config.echo.enabled = enabled
    canceller = EchoCanceller(config)
    out = []
    for i in range(0, len(mic), 1280):
        for j in range(i, i + 1280, 640):
            canceller.far_end(far[j : j + 640])
        out.append(canceller.near_end(mic[i : i + 1280]))
    clean = np.concatenate([c for c, _ in out])
    raw = np.concatenate([r for _, r in out])
    assert len(clean) == len(raw)
    if enabled:
        assert np.allclose(
            raw, mic[: len(raw)], atol=2 / 32768
        )  # untouched copy, aligned (int16 rounding)
    return mic, clean


def test_pool_reuses_a_trained_canceller():
    from localive.echo import EchoPool

    config = Config()
    config.echo.enabled = True
    pool = EchoPool(config)
    first = pool.acquire()
    second = pool.acquire()  # a concurrent session gets its own
    assert first is not second
    pool.release(first)
    assert pool.acquire() is first  # the next session starts with the room already learned
    config.echo.enabled = False
    pool.release(first)
    assert pool.acquire().enabled is False  # settings changed: a fresh one


def test_disabled_is_transparent():
    mic, out = simulate(False, 1.0)
    assert np.array_equal(mic, out)


def test_pool_applies_changed_native_filter_settings():
    from localive.echo import EchoPool

    config = Config()
    config.echo.enabled = True
    pool = EchoPool(config)
    first = pool.acquire()
    pool.release(first)
    config.echo.noise_suppression = True
    second = pool.acquire()
    assert second is not first
    assert second.settings.noise_suppression


def test_releasing_twice_cannot_share_one_native_filter_between_sessions():
    from localive.echo import EchoPool

    config = Config()
    config.echo.enabled = True
    pool = EchoPool(config)
    first = pool.acquire()
    pool.release(first)
    pool.release(first)
    assert pool.acquire() is first
    assert pool.acquire() is not first


def test_removes_most_of_the_echo():
    mic, out = simulate(True)
    half = len(mic) // 2
    removed = 10 * np.log10(
        np.mean(mic[half:] ** 2) / max(1e-12, np.mean(out[half : len(mic)] ** 2))
    )
    assert removed > 15  # dB, after the adaptive filter has converged
    assert abs(len(out) - len(mic)) < 160  # never more than one 10 ms frame held back
