"""Run the public WAV client over a real WebSocket with simulated speech engines."""

import asyncio
import json
import wave

import numpy as np
import pytest
from aiohttp.test_utils import TestServer

from localive.config import Config
from localive.server import App
from scripts import live_client
from tests.test_live import Client
from tests.test_vad import FIXTURES


@pytest.mark.parametrize("cornelius", [False, True])
async def test_wav_round_trip(tmp_path, cornelius):
    engines = Client(script=["Quatre."], tts_seconds=0.12, heard="Combien font deux et deux ?")
    instance = App(engines.session.config)
    instance.services = engines.session.s
    instance.ready.set()
    app = instance.build()
    app.on_startup.clear()
    output = tmp_path / "audio" / "reply.wav"
    report = tmp_path / "metrics.json"
    async with TestServer(app) as server:
        path = "/v1/live/sessions" if cornelius else "/live"
        url = str(server.make_url(path).with_scheme("ws"))
        options = ["--profile", "cornelius", "--report", str(report)] if cornelius else []
        await asyncio.wait_for(live_client.main([
            str(FIXTURES / "wake_calculation.wav"), "--url", url, "--save", str(output), *options,
        ]), 15)
    offered = engines.chat.calls[0][1]
    if cornelius:
        assert "clock__read" in [tool["function"]["name"] for tool in offered]
        measurements = json.loads(report.read_text("utf-8"))
        assert len(measurements["turns"]) == 1
        assert measurements["client_audio"][0]["first_sound_ms"] > 0
    else:
        assert not offered
        assert "Tu es Cornelius" not in engines.chat.calls[0][0][0]["content"]
    with wave.open(str(output), "rb") as wav:
        assert (wav.getnchannels(), wav.getframerate(), wav.getsampwidth()) == (1, 16000, 2)
        assert np.frombuffer(wav.readframes(wav.getnframes()), "<i2").any()


async def test_server_start_error_reaches_the_client_immediately(tmp_path, monkeypatch):
    instance = App(Config())

    async def unavailable():
        return "Modèles en cours de chargement"

    monkeypatch.setattr(instance, "_wait_ready", unavailable)
    app = instance.build()
    app.on_startup.clear()
    async with TestServer(app) as server:
        url = str(server.make_url("/v1/live/sessions").with_scheme("ws"))
        with pytest.raises(RuntimeError, match="engine_not_ready: Modèles en cours"):
            await asyncio.wait_for(live_client.main([
                str(FIXTURES / "wake_calculation.wav"), "--url", url,
                "--save", str(tmp_path / "reply.wav"),
            ]), 2)
