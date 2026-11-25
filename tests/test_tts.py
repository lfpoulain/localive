import asyncio
import threading
from contextlib import contextmanager

import numpy as np
import pytest

from localive.audio import Resampler
from localive.config import Config
from localive.tts import BUFFER_CHUNKS, CHUNK_SAMPLES, Synthesizer, TtsError


class FakeAudio:
    def __init__(self, chunks, rate=24000):
        self.chunks = chunks
        self.sample_rate = rate
        self.generated = 0

    def __iter__(self):
        for chunk in self.chunks:
            self.generated += 1
            yield chunk


class FakeTts:
    def __init__(self, audio):
        self.audio = audio
        self.opened = threading.Event()
        self.closed = threading.Event()

    @contextmanager
    def stream(self, text, voice):
        self.opened.set()
        try:
            yield self.audio
        finally:
            self.closed.set()


async def wait_for(predicate):
    async with asyncio.timeout(2):
        while not predicate():  # noqa: ASYNC110 - observe a native worker thread
            await asyncio.sleep(0.01)


async def test_slow_consumer_bounds_synthesis_and_cancellation_releases_engine():
    audio = FakeAudio([np.zeros(240, np.float32)] * 1000)
    native = FakeTts(audio)
    synth = Synthesizer(Config())
    synth._tts = native
    stream = synth.stream("Une très longue réponse.", "female")
    await anext(stream)
    await wait_for(lambda: audio.generated >= BUFFER_CHUNKS + 1)
    await asyncio.sleep(0.05)
    assert audio.generated <= BUFFER_CHUNKS + 2  # queued + consumed + one in flight
    await stream.aclose()
    await wait_for(native.closed.is_set)
    assert synth._lock.acquire(blocking=False)
    synth._lock.release()


async def test_cancelled_waiter_never_starts_native_synthesis():
    native = FakeTts(FakeAudio([np.zeros(240, np.float32)]))
    synth = Synthesizer(Config())
    synth._tts = native
    synth._lock.acquire()
    stream = synth.stream("Réponse abandonnée.", "female")
    pending = asyncio.create_task(anext(stream))
    await asyncio.sleep(0.02)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    synth._lock.release()
    await asyncio.sleep(0.1)
    assert not native.opened.is_set()


async def test_large_native_chunks_are_split_to_bound_buffered_audio_duration():
    audio = FakeAudio([np.zeros(24000, np.float32)] * 100)
    native = FakeTts(audio)
    synth = Synthesizer(Config())
    synth._tts = native
    stream = synth.stream("Une réponse lente à écouter.", "female")
    assert len(await anext(stream)) == CHUNK_SAMPLES
    await asyncio.sleep(0.05)
    assert audio.generated == 1  # the remaining second blocks before further synthesis
    await stream.aclose()
    await wait_for(native.closed.is_set)


async def test_resampling_preserves_phase_and_duration_across_native_frames():
    samples = np.sin(np.arange(3200, dtype=np.float32) * 0.3)
    chunks = [samples[i : i + 127] for i in range(0, len(samples), 127)]
    synth = Synthesizer(Config())
    synth._tts = FakeTts(FakeAudio(chunks, rate=16000))
    output = np.concatenate([chunk async for chunk in synth.stream("Bonjour.", "female")])
    expected = Resampler(16000, 24000)(samples)
    np.testing.assert_allclose(output, expected, atol=1e-6)


async def test_native_error_reaches_consumer_and_releases_engine():
    class BrokenAudio:
        sample_rate = 24000

        def __iter__(self):
            raise ValueError("failed synthesis")

    synth = Synthesizer(Config())
    synth._tts = FakeTts(BrokenAudio())
    with pytest.raises(TtsError, match="failed synthesis"):
        async for _ in synth.stream("Bonjour.", "female"):
            pass
    await wait_for(synth._tts.closed.is_set)
    assert synth._lock.acquire(blocking=False)
    synth._lock.release()
