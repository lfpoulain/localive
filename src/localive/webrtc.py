"""OpenAI-compatible Live WebRTC transport: what a browser (Cornelius' mirror) uses.

The flow mirrors OpenAI's unified WebRTC interface:

1. ``POST /v1/live/sessions`` with ``{"session": {...}, "transport": {"type": "webrtc",
   "sdp": "<offer>"}}``. The session starts at once; the answer is
   ``{"session": {"id": ...}, "transport": {"type": "webrtc", "sdp": "<answer>"}}``.
2. The browser sends its microphone as Opus over RTP and plays the assistant's track.
3. Its ``oai-events`` data channel only receives ``session.started``, ``session.closed`` and
   ``error``.
4. The application server attaches a control WebSocket,
   ``/v1/live/sessions/{id}/attach``: every other event (transcripts, tool calls, warnings)
   goes there, and client events (tool outputs, instructions, close) come from there.

The browser's offer carries no ICE candidates (it is sent before gathering, as OpenAI
answers with its own). Its connectivity checks reach the host candidates of the answer,
which is enough on a local network. No STUN server is contacted.
"""

from __future__ import annotations

import asyncio
import contextlib
import fractions
import json
import logging
import time
from collections import deque
from typing import Any

import numpy as np

from .live import LiveSession, Services

log = logging.getLogger(__name__)

RATE = 16000
PTIME = 0.02  # 20 ms media frames, as browsers send
FRAME = int(RATE * PTIME)
MAX_BUFFER_S = 1.0  # output audio held for the track; Output paces ~120 ms ahead
ATTACH_TIMEOUT_S = 15.0  # the control WebSocket must come within this delay
DATA_CHANNEL_EVENTS = {"session.started", "session.closed", "error"}
MAX_PENDING_EVENTS = 500


class WebRtcError(ValueError):
    """The offer cannot be answered (bad SDP, no audio, missing dependency)."""


def _media():
    try:
        import av
        from aiortc import (
            MediaStreamTrack,
            RTCConfiguration,
            RTCPeerConnection,
            RTCSessionDescription,
        )
        from aiortc.mediastreams import MediaStreamError
    except ImportError as exc:  # pragma: no cover - aiortc is a dependency
        raise WebRtcError("WebRTC indisponible : installer aiortc") from exc
    return av, MediaStreamTrack, MediaStreamError, RTCConfiguration, RTCPeerConnection, \
        RTCSessionDescription


class OutputTrack:
    """The assistant's voice as an aiortc audio track: 20 ms frames of 16 kHz mono PCM, in
    real time, silence when nothing is queued. aiortc's Opus encoder resamples to 48 kHz."""

    def __new__(cls):
        av, base, media_error, *_ = _media()

        class _Track(base):
            kind = "audio"

            def __init__(self) -> None:
                super().__init__()
                self._buffer: deque[np.ndarray] = deque()
                self._buffered = 0
                self._start: float | None = None
                self._pts = 0

            # AudioSink (LiveSession.Output) -----------------------------------
            def push(self, frame: np.ndarray) -> None:
                pcm = (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16)
                self._buffer.append(pcm)
                self._buffered += len(pcm)
                while self._buffered > MAX_BUFFER_S * RATE and self._buffer:
                    self._buffered -= len(self._buffer.popleft())

            def clear(self) -> None:
                self._buffer.clear()
                self._buffered = 0

            # aiortc -----------------------------------------------------------
            def _take(self) -> np.ndarray:
                out = np.zeros(FRAME, dtype=np.int16)
                filled = 0
                while filled < FRAME and self._buffer:
                    head = self._buffer[0]
                    n = min(FRAME - filled, len(head))
                    out[filled : filled + n] = head[:n]
                    filled += n
                    if n == len(head):
                        self._buffer.popleft()
                    else:
                        self._buffer[0] = head[n:]
                self._buffered -= filled
                return out

            async def recv(self):
                if self.readyState != "live":
                    raise media_error
                now = time.monotonic()
                if self._start is None:
                    self._start = now
                else:
                    wait = self._start + self._pts / RATE - now
                    if wait > 0:
                        await asyncio.sleep(wait)
                    elif wait < -0.5:  # stalled loop: do not burst, restart the clock
                        self._start = now - self._pts / RATE
                frame = av.AudioFrame.from_ndarray(self._take().reshape(1, -1), format="s16",
                                                   layout="mono")  # fmt: skip
                frame.sample_rate = RATE
                frame.pts = self._pts
                frame.time_base = fractions.Fraction(1, RATE)
                self._pts += FRAME
                return frame

        return _Track()


class WebRtcSession:
    """One browser peer connection, its LiveSession and its control WebSocket."""

    def __init__(self, services: Services) -> None:
        self.services = services
        self.pc = None
        self.track = None
        self.session: LiveSession | None = None
        self.channel = None
        self.ws = None
        self.closed = False
        self._pending: deque[dict[str, Any]] = deque(maxlen=MAX_PENDING_EVENTS)
        self._send_lock = asyncio.Lock()
        self._tasks: list[asyncio.Task] = []
        self.on_closed = None  # callback(self) for the server's registry

    @property
    def id(self) -> str:
        return self.session.id if self.session else ""

    # ----------------------------------------------------------------- setup
    async def answer(self, session_config: dict[str, Any], offer_sdp: str) -> str:
        av, _, media_error, configuration, peer_connection, description = _media()
        if "m=audio" not in offer_sdp:
            raise WebRtcError("L'offre WebRTC doit contenir une piste audio.")
        self.pc = peer_connection(configuration(iceServers=[]))  # local network: no STUN
        self.track = OutputTrack()
        self.session = LiveSession(self.services, self._send, audio_sink=self.track)

        @self.pc.on("datachannel")
        def on_datachannel(channel) -> None:
            self.channel = channel
            if self.session is not None and self.session.started:
                self._channel_send({"type": "session.started", "session": self._descriptor()})

        @self.pc.on("track")
        def on_track(track) -> None:
            if track.kind == "audio":
                self._tasks.append(asyncio.create_task(self._microphone(track, av, media_error)))

        @self.pc.on("connectionstatechange")
        async def on_state() -> None:
            state = self.pc.connectionState if self.pc else "closed"
            log.info("WebRTC %s: %s", self.id, state)
            if state in ("failed", "closed"):
                await self.close("disconnected")

        try:
            await self.pc.setRemoteDescription(description(sdp=offer_sdp, type="offer"))
            audio = [t for t in self.pc.getTransceivers() if t.kind == "audio"]
            if not audio:
                raise WebRtcError("L'offre WebRTC doit contenir une piste audio.")
            audio[0].sender.replaceTrack(self.track)
            audio[0].direction = "sendrecv"
            await self.pc.setLocalDescription(await self.pc.createAnswer())
        except WebRtcError:
            await self._close_pc()
            raise
        except Exception as exc:  # noqa: BLE001 - aiortc raises plain errors on bad SDP
            await self._close_pc()
            raise WebRtcError(f"Offre WebRTC refusée : {exc}") from exc
        await self.session.start(session_config)
        self._tasks.append(asyncio.create_task(self._attach_deadline()))
        return self.pc.localDescription.sdp

    def _descriptor(self) -> dict[str, Any]:
        return {"id": self.id, "object": "live.session", "model": "localive"}

    # ----------------------------------------------------------------- media
    async def _microphone(self, track, av, media_error) -> None:
        """Browser Opus (48 kHz) -> 16 kHz mono float -> the session's microphone input."""
        resampler = av.AudioResampler(format="s16", layout="mono", rate=RATE)
        try:
            while not self.closed:
                frame = await track.recv()
                for out in resampler.resample(frame):
                    pcm = out.to_ndarray().reshape(-1).astype(np.float32) / 32768.0
                    if self.session is not None and not self.session.muted and len(pcm):
                        await self.session.on_audio(pcm)
        except media_error:
            pass
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a broken stream ends the session, not the server
            log.exception("WebRTC microphone failed")
            await self.close("media_error")

    # ---------------------------------------------------------------- events
    def _channel_send(self, event: dict[str, Any]) -> None:
        channel = self.channel
        if channel is not None and channel.readyState == "open":
            with contextlib.suppress(Exception):
                channel.send(json.dumps(event, ensure_ascii=False))

    async def _send(self, event: dict[str, Any]) -> None:
        """LiveSession's event sink: the control WebSocket, or a queue until it attaches."""
        if event.get("type") in DATA_CHANNEL_EVENTS:
            self._channel_send(event)
        ws = self.ws
        if ws is None:
            if event.get("type") == "session.started" or len(self._pending) < MAX_PENDING_EVENTS:
                self._pending.append(event)
        else:
            from .server import send_live_event

            try:
                await send_live_event(ws, event)
            except ConnectionError:
                asyncio.get_running_loop().call_soon(
                    lambda: asyncio.ensure_future(self.close("disconnected"))
                )
                raise
        if event["type"] == "session.closed" and not self.closed:
            await self.close(event["reason"])

    async def attach(self, ws) -> bool:
        """Bind the control WebSocket; replay what happened before (session.started first)."""
        if self.closed or self.ws is not None:
            return False
        from .server import send_live_event

        async with self._send_lock:
            # Events emitted during the replay are queued and replayed by this same loop;
            # the socket is published only once the queue is empty, so order is kept.
            while self._pending:
                await send_live_event(ws, self._pending.popleft())
            self.ws = ws
        return True

    async def _attach_deadline(self) -> None:
        await asyncio.sleep(ATTACH_TIMEOUT_S)
        if self.ws is None and not self.closed:
            log.warning("WebRTC %s: no control WebSocket after %.0f s", self.id,
                        ATTACH_TIMEOUT_S)  # fmt: skip
            await self.close("attach_timeout")

    # ------------------------------------------------------------- teardown
    async def _close_pc(self) -> None:
        pc, self.pc = self.pc, None
        if pc is not None:
            with contextlib.suppress(Exception):
                await pc.close()

    async def close(self, reason: str) -> None:
        if self.closed:
            return
        self.closed = True
        current = asyncio.current_task()
        tasks = [task for task in self._tasks if task is not current]
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self.session is not None:
            await self.session.close(reason)  # session.closed: data channel + WebSocket
        if self.track is not None:
            self.track.stop()
        await self._close_pc()
        if self.on_closed is not None:
            self.on_closed(self)
        ws, self.ws = self.ws, None
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.close()
