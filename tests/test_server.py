import asyncio
import base64

import pytest
from aiohttp.test_utils import TestClient, TestServer

from localive.config import Config
from localive.server import (
    MAX_AUDIO_BYTES,
    App,
    is_loopback,
    send_live_event,
    validate_live_event,
)


@pytest.mark.parametrize("event", [
    [], None, {"type": 2}, {"type": "session.start", "session": []},
    {"type": "session.start", "session": {"audio": {"output": []}}},
    {"type": "session.update", "session": {"delegation": {"responses": 1}}},
    {"type": "session.update", "session": {"tools": "clock"}},
    {"type": "session.update", "session": {"localive": {"end_on_goodbye": "false"}}},
    {"type": "session.start", "session": {"localive": {"goodbye_mode": "auto"}}},
    {"type": "session.update", "session": {"localive": {"goodbye_mode": []}}},
    {"type": "response.item.create", "item": "bad"},
    {"type": "session.input_audio.append", "audio": "not base64!"},
    {"type": "session.input_audio.append", "audio": base64.b64encode(b"x").decode()},
])
def test_invalid_events_are_rejected(event):
    assert validate_live_event(event)


@pytest.mark.parametrize("rate", [8000, 24000, 48000])
def test_wrong_audio_rate_cannot_silently_change_pitch(rate):
    event = {"type": "session.start", "session": {
        "audio": {"format": {"type": "audio/pcm", "rate": rate}},
    }}
    assert "16000" in validate_live_event(event)


def test_audio_backlog_is_bounded_but_normal_satellite_frames_are_accepted():
    frame = {"type": "session.input_audio.append",
             "audio": base64.b64encode(bytes(2560)).decode()}
    assert validate_live_event(frame) is None
    frame["audio"] = base64.b64encode(bytes(MAX_AUDIO_BYTES + 2)).decode()
    assert "seconde" in validate_live_event(frame)


def test_realtime_client_gets_explicit_protocol_error():
    problem = validate_live_event({"type": "input_audio_buffer.append", "audio": ""})
    assert "pas le protocole Realtime" in problem


@pytest.mark.parametrize("path", ["/live", "/v1/live/sessions"])
async def test_bad_event_returns_error_and_connection_accepts_next_event(monkeypatch, path):
    class Session:
        closed = False

        def __init__(self, _services, send):
            self.send = send

        async def handle(self, event):
            await self.send({"type": "accepted", "original": event["type"]})

        async def warn(self, code, message):
            await self.send({"type": "voice.provider_warning", "error": {"code": code}})

        async def close(self, _reason):
            self.closed = True

    monkeypatch.setattr("localive.server.LiveSession", Session)
    instance = App(Config())
    instance.ready.set()
    app = instance.build()
    app.on_startup.clear()  # no model downloads/processes in protocol tests
    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect(path) as ws:
            await ws.send_str("{bad json")
            warning = await ws.receive_json()  # Cornelius ends the conversation on "error"
            assert warning == {"type": "voice.provider_warning", "error": {"code": "invalid_json"}}
            await ws.send_json({"type": "session.start", "event_id": "bad", "session": []})
            error = (await ws.receive_json())["error"]  # a broken session config is fatal
            assert error["code"] == "invalid_event" and error["client_event_id"] == "bad"
            await ws.send_json({"type": "response.item.create", "event_id": "x", "item": "bad"})
            warning = await ws.receive_json()
            assert warning["type"] == "voice.provider_warning"
            assert warning["client_event_id"] == "x"
            await ws.send_json({"type": "session.start", "session": {}})
            assert (await ws.receive_json())["type"] == "accepted"


async def test_health_identifies_live_wire_format(monkeypatch):
    from unittest.mock import AsyncMock

    instance = App(Config())
    instance.config.search.managed = False
    probe = AsyncMock(return_value=True)
    monkeypatch.setattr(instance, "boot", AsyncMock())
    monkeypatch.setattr(instance.searxng, "_wait_ready", probe)
    app = instance.build()
    async with TestClient(TestServer(app)) as client:
        health = await (await client.get("/health")).json()
        assert health["protocols"]["live"] == {
            "path": "/v1/live/sessions", "audio_rate": 16000,
            "transports": ["websocket", "webrtc"],
        }
        assert health["search"]["state"] == "idle"
        probe.assert_not_awaited()
        reply = await (await client.post("/api/search/retry")).json()
        assert reply["state"] == "starting"
        await instance.searxng.startup_task
        status = await (await client.get("/health")).json()
        assert status["search"]["state"] == "ready"


@pytest.mark.parametrize(("mode", "farewell"), [
    ("phrases", "Au revoir."), ("d3", "Je file, on se reparle demain."),
])
async def test_websocket_goodbye_closes_an_idle_socket_after_the_transcript(mode, farewell):
    import json
    from unittest.mock import AsyncMock

    import numpy as np

    from tests.test_live import Client
    from tests.test_vad import FIXTURES, load_16k

    peer = Client(heard=farewell)
    router = peer.session.s.router = AsyncMock()
    router.should_hang_up.return_value = True
    instance = App(peer.session.config)
    instance.services = peer.session.s
    instance.ready.set()
    app = instance.build()
    app.on_startup.clear()
    speech = np.concatenate([load_16k(FIXTURES / "wake_clock.wav"), np.zeros(16000)])
    pcm = (speech * 32767).astype("<i2").tobytes()
    async with TestClient(TestServer(app)) as client, client.ws_connect("/live") as ws:
        await ws.send_json({"type": "session.start", "session": {
            "localive": {"end_on_goodbye": True, "goodbye_mode": mode}}})
        for offset in range(0, len(pcm), 2560):
            audio = base64.b64encode(pcm[offset:offset + 2560]).decode()
            await ws.send_json({"type": "session.input_audio.append", "audio": audio})
        async with asyncio.timeout(5):
            events = [json.loads(message.data) async for message in ws]
        assert events[-1]["type"] == "session.closed" and events[-1]["reason"] == "user_goodbye"
        assert [e["delta"] for e in events if e["type"] == "session.input_transcript.delta"] == [
            farewell]
        assert ws.closed and not peer.chat.calls
        assert router.should_hang_up.await_count == int(mode == "d3")


@pytest.mark.parametrize("base", ["/live", "/v1/live/sessions"])
async def test_configured_token_for_live_session_control(base):
    instance = App(Config())
    instance.config.server.api_key = "session-test"
    app = instance.build()
    app.on_startup.clear()
    async with TestClient(TestServer(app)) as client:
        path = f"{base}/unused-session/hangup"
        assert (await client.post(path)).status == 401
        reply = await client.post(path, headers={"Authorization": "Bearer session-test"})
        assert reply.status == 200


@pytest.mark.parametrize("close_stalls", [False, True])
async def test_stalled_audio_socket_is_closed_without_waiting_for_drain(monkeypatch, close_stalls):
    class Socket:
        close_options = None

        async def send_json(self, _event):
            await asyncio.Event().wait()

        async def close(self, **options):
            self.close_options = options
            if close_stalls:
                await asyncio.Event().wait()

    class Transport:
        closed = False

        def close(self):
            self.closed = True

    monkeypatch.setattr("localive.server.SEND_TIMEOUT_S", 0.01)
    socket, transport = Socket(), Transport()
    with pytest.raises(ConnectionError):
        await asyncio.wait_for(send_live_event(socket, {"type": "audio"}, transport), 1)
    assert socket.close_options["drain"] is False
    assert transport.closed


async def test_foreign_web_page_cannot_reach_the_api_but_the_playground_can():
    instance = App(Config())
    app = instance.build()
    app.on_startup.clear()
    async with TestClient(TestServer(app)) as client:
        evil = await client.get("/api/settings", headers={"Origin": "https://evil.example"})
        assert evil.status == 403
        assert (await evil.json())["error"]["code"] == "forbidden_origin"
        host = f"http://{client.host}:{client.port}"
        assert (await client.get("/api/settings", headers={"Origin": host})).status == 200
        assert (await client.get("/api/settings")).status == 200  # native client, no Origin


def test_settings_changes_from_the_network_require_an_api_key():
    assert is_loopback("127.0.0.1") and is_loopback("::1") and is_loopback("localhost")
    assert not is_loopback("192.168.1.20") and not is_loopback("0.0.0.0")
    assert not is_loopback(None)
