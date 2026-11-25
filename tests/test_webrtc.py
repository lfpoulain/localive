"""OpenAI-compatible WebRTC flow, end to end, with aiortc playing the browser."""

import asyncio
import fractions
import json
from unittest.mock import AsyncMock

import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer

aiortc = pytest.importorskip("aiortc")
av = pytest.importorskip("av")

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription  # noqa: E402
from aiortc.mediastreams import MediaStreamError  # noqa: E402

from localive.config import Config  # noqa: E402
from localive.live import Services  # noqa: E402
from localive.routing import RouteDecision  # noqa: E402
from localive.server import App  # noqa: E402
from localive.vad import Segmenter  # noqa: E402
from tests.test_live import FakeChat, FakeTranscriber, FakeTts  # noqa: E402
from tests.test_vad import FIXTURES, load_16k  # noqa: E402

SESSION = {
    "model": "gpt-live-1",
    "instructions": "Tu es Cornelius.",
    "audio": {"output": {"voice": "cedar"}},
    "client": {"data_channel": {"allowed_client_events": [], "allowed_server_events": [
        {"type": "session.started"}, {"type": "session.closed"}, {"type": "error"}]}},
    "delegation": {"type": "responses", "responses": {"instructions": "", "tools": []}},
}


class Microphone(MediaStreamTrack):
    """The browser's microphone: a spoken question, then silence, in 20 ms frames."""

    kind = "audio"

    def __init__(self, samples: np.ndarray) -> None:
        super().__init__()
        self.samples = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
        self.pts = 0

    async def recv(self):
        await asyncio.sleep(0.02)
        chunk = self.samples[self.pts : self.pts + 320]
        if len(chunk) < 320:
            chunk = np.concatenate([chunk, np.zeros(320 - len(chunk), np.int16)])
        frame = av.AudioFrame.from_ndarray(chunk.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate, frame.pts = 16000, self.pts
        frame.time_base = fractions.Fraction(1, 16000)
        self.pts += 320
        return frame


def make_app(chat: FakeChat) -> App:
    config = Config()
    config.echo.enabled = False
    app = App(config)
    app.services = Services(
        config=config, http=None, transcriber=FakeTranscriber(), chat=chat, tts=FakeTts(0.6),
        make_segmenter=Segmenter,
    )  # fmt: skip
    app.ready.set()
    return app


@pytest.mark.parametrize(("path", "trickle", "goodbye_mode"), [
    ("/v1/live/sessions", False, None), ("/v1/live/sessions", True, None),
    ("/live", True, "phrases"), ("/live", True, "d3"),
])
async def test_webrtc_session_like_openai(path, trickle, goodbye_mode):
    """``trickle``: like a browser, the offer is sent before ICE gathering (no candidates);
    Localive learns the browser's address from its connectivity checks."""
    chat = FakeChat(["Il est quatorze heures."])
    instance = make_app(chat)
    instance.config.turn.end_on_goodbye = goodbye_mode is not None
    if goodbye_mode:
        instance.config.turn.goodbye_mode = goodbye_mode
    session_config = SESSION
    if goodbye_mode == "d3":
        catalog = [{"type": "function", "name": f"tool_{i}", "parameters": {"type": "object"}}
                   for i in range(6)]
        session_config = {**SESSION, "delegation": {"type": "responses",
                                                    "responses": {"tools": catalog}}}
        instance.services.router = AsyncMock()
        instance.services.router.decide.side_effect = [RouteDecision([]), RouteDecision([], True)]
        instance.services.router.should_hang_up.return_value = True
    web_app = instance.build()
    web_app.on_startup.clear()
    speech = load_16k(FIXTURES / "wake_clock.wav")
    silence = np.zeros(16000, np.float32)
    pc = RTCPeerConnection()
    mic = Microphone(np.concatenate([silence, speech, silence, silence]))
    pc.addTrack(mic)
    channel = pc.createDataChannel("oai-events")
    channel_events: list[dict] = []
    channel.on("message", lambda m: channel_events.append(json.loads(m)))
    heard: list[float] = []

    @pc.on("track")
    def on_track(track):
        async def listen():
            while True:
                try:
                    frame = await track.recv()
                except MediaStreamError:
                    return
                pcm = frame.to_ndarray().astype(np.float32) / 32768
                heard.append(float(np.abs(pcm).max()))

        asyncio.ensure_future(listen())

    await pc.setLocalDescription(await pc.createOffer())
    try:
        async with TestClient(TestServer(web_app)) as client:
            offer = pc.localDescription.sdp
            if trickle:
                offer = "".join(line for line in offer.splitlines(keepends=True)
                                if not line.startswith(("a=candidate", "a=end-of-candidates")))
                assert "a=candidate" not in offer
            created = await client.post(path, json={
                "session": session_config, "transport": {"type": "webrtc", "sdp": offer},
            })  # fmt: skip
            assert created.status == 200, await created.text()
            body = await created.json()
            assert body["transport"]["type"] == "webrtc" and "m=audio" in body["transport"]["sdp"]
            session_id = body["session"]["id"]
            await pc.setRemoteDescription(RTCSessionDescription(body["transport"]["sdp"], "answer"))

            events: list[dict] = []
            async with client.ws_connect(f"{path}/{session_id}/attach") as ws:
                first = await asyncio.wait_for(ws.receive_json(), 5)
                assert first["type"] == "session.started" and first["session"]["id"] == session_id

                async def collect():
                    async for message in ws:
                        events.append(json.loads(message.data))

                reader = asyncio.ensure_future(collect())
                for _ in range(400):  # the question is spoken, transcribed, answered aloud
                    await asyncio.sleep(0.05)
                    if any(e["type"] == "voice.turn_completed" for e in events):
                        break
                kinds = [e["type"] for e in events]
                assert "session.input_transcript.delta" in kinds
                assert kinds.index("session.input_transcript.delta") < kinds.index(
                    "session.delegation.created") < kinds.index("session.output_transcript.delta")
                assert "session.delegation.created" not in [e["type"] for e in channel_events]
                assert "session.output_transcript.delta" in kinds
                assert "voice.turn_completed" in kinds
                assert "session.output_audio.delta" not in kinds  # audio travels over RTP
                assert max(heard, default=0) > 0.05  # the assistant's voice reached the browser
                assert any(e["type"] == "session.started" for e in channel_events)
                if goodbye_mode:
                    farewell = ("Je file, on se reparle demain." if goodbye_mode == "d3"
                                else "OK, c’est bon, merci.")
                    instance.services.transcriber.text = farewell
                    mic.samples = np.concatenate([np.zeros(mic.pts + 3200, np.int16),
                                                  (speech * 32767).astype(np.int16)])
                else:
                    await ws.send_json({"type": "session.close"})
                await asyncio.wait_for(reader, 8)
                reasons = [e["reason"] for e in events if e["type"] == "session.closed"]
                assert reasons == ["user_goodbye" if goodbye_mode else "close_requested"]
                channel_reasons = [e["reason"] for e in channel_events
                                   if e["type"] == "session.closed"]
                assert channel_reasons == reasons
                if goodbye_mode == "d3":
                    assert instance.services.router.decide.await_count == 2
                    instance.services.router.should_hang_up.assert_awaited_once()
                assert ws.closed
            assert session_id not in instance.rtc  # cleaned up
    finally:
        await pc.close()


async def test_webrtc_rejects_bad_requests():
    instance = make_app(FakeChat([]))
    web_app = instance.build()
    web_app.on_startup.clear()
    async with TestClient(TestServer(web_app)) as client:
        bad = await client.post("/v1/live/sessions", json={"session": {}, "transport": {}})
        assert bad.status == 400
        no_audio = await client.post("/v1/live/sessions", json={
            "session": {}, "transport": {"type": "webrtc", "sdp": "v=0\r\n"}})
        assert no_audio.status == 400
        async with client.ws_connect("/v1/live/sessions/sess_unknown/attach") as ws:
            assert (await ws.receive_json())["error"]["code"] == "session_not_found"
