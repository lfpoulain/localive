import asyncio
import threading
from contextlib import contextmanager

import numpy as np
import pytest
from aiohttp import web

from localive import wyoming
from localive.audio import Resampler
from localive.config import Config
from localive.tts import BUFFER_CHUNKS, CHUNK_SAMPLES, TTS_RATE, Synthesizer, TtsError


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


async def test_openai_tts_streams_pcm_with_odd_byte_boundaries():
    seen: dict[str, object] = {}
    pcm_samples = (np.sin(np.arange(4800, dtype=np.float32) * 0.2) * 16000).astype("<i2")
    raw_bytes = pcm_samples.tobytes()

    async def handle(request: web.Request) -> web.StreamResponse:
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = await request.json()
        resp = web.StreamResponse(status=200, headers={"Content-Type": "audio/pcm"})
        await resp.prepare(request)
        # Split on an odd byte boundary to verify carry handling.
        await resp.write(raw_bytes[:1001])
        await resp.write(raw_bytes[1001:])
        await resp.write_eof()
        return resp

    async def handle_models(_req: web.Request) -> web.Response:
        return web.json_response({"data": []})

    app = web.Application()
    app.router.add_get("/v1/models", handle_models)
    app.router.add_post("/v1/audio/speech", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    config = Config()
    config.voice.backend = "openai"
    config.voice.external_url = f"http://127.0.0.1:{port}"
    config.voice.external_model = "kokoro"
    config.voice.external_female = "ff_siwis"
    config.voice.lang_code = "f"
    config.voice.speed = 1.1
    config.voice.api_key = "secret-tts"
    synth = Synthesizer(config)
    await synth.start()
    try:
        chunks = [c async for c in synth.stream("Bonjour le monde.", "female")]
    finally:
        await synth.aclose()
        await runner.cleanup()

    assert seen["auth"] == "Bearer secret-tts"
    assert seen["body"] == {
        "model": "kokoro",
        "voice": "ff_siwis",
        "input": "Bonjour le monde.",
        "response_format": "pcm",
        "stream": True,
        "lang_code": "f",
        "speed": 1.1,
    }
    output = np.concatenate(chunks)
    np.testing.assert_allclose(output, pcm_samples.astype(np.float32) / 32768.0, atol=1e-5)


async def test_wyoming_tts_streams_and_resamples_piper_audio():
    seen: dict[str, object] = {}
    samples_22k = (np.sin(np.arange(2205, dtype=np.float32) * 0.25) * 16000).astype("<i2")
    raw = samples_22k.tobytes()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            event = await wyoming.read_event(reader)
            assert event is not None and event.type == "synthesize"
            seen["data"] = event.data
            fmt = {"rate": 22050, "width": 2, "channels": 1}
            writer.write(wyoming.encode_event("audio-start", fmt))
            writer.write(wyoming.encode_event("audio-chunk", fmt, raw[:513]))
            writer.write(wyoming.encode_event("audio-chunk", fmt, raw[513:]))
            writer.write(wyoming.encode_event("audio-stop"))
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    config = Config()
    config.voice.backend = "wyoming"
    config.voice.external_url = f"tcp://127.0.0.1:{port}"
    config.voice.external_female = "fr_FR-siwis-medium"
    synth = Synthesizer(config)
    await synth.start()
    try:
        chunks = [c async for c in synth.stream("Bonjour.", "female")]
    finally:
        await synth.aclose()
        server.close()
        await server.wait_closed()

    assert seen["data"] == {"text": "Bonjour.", "voice": {"name": "fr_FR-siwis-medium"}}
    output = np.concatenate(chunks)
    expected = Resampler(22050, TTS_RATE)(samples_22k.astype(np.float32) / 32768.0)
    np.testing.assert_allclose(output, expected, atol=1e-5)


async def test_wyoming_tts_uses_server_default_voice_when_unconfigured():
    seen: dict[str, object] = {}

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            event = await wyoming.read_event(reader)
            if event is not None:
                seen["data"] = event.data
                fmt = {"rate": 22050, "width": 2, "channels": 1}
                writer.write(wyoming.encode_event("audio-start", fmt))
                writer.write(wyoming.encode_event("audio-stop"))
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    config = Config()
    config.voice.backend = "wyoming"
    config.voice.external_url = f"127.0.0.1:{port}"
    assert config.voice.external_female == "" and config.voice.external_male == ""
    synth = Synthesizer(config)
    await synth.start()
    try:
        assert synth.probe_error is None
        _ = [c async for c in synth.stream("Salut.", "female")]
    finally:
        await synth.aclose()
        server.close()
        await server.wait_closed()

    assert seen["data"] == {"text": "Salut."}


async def test_external_tts_probe_reports_unreachable_server_without_blocking_startup():
    config = Config()
    config.voice.backend = "wyoming"
    config.voice.external_url = "127.0.0.1:1"
    synth = Synthesizer(config)
    await synth.start()
    try:
        assert synth.probe_error is not None
        assert "injoignable" in synth.probe_error
    finally:
        await synth.aclose()
