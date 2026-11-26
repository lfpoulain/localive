"""HTTP + WebSocket server: an OpenAI-Live-compatible endpoint backed by local models."""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import hmac
import ipaddress
import json
import logging
import os
import socket
import time
import webbrowser
from collections import deque
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp
from aiohttp import web

from . import __version__, hf, install, settings
from .asr import Transcriber
from .config import (
    BRAIN_SERVERS,
    DEFAULT_ROUTER,
    GOODBYE_MODES,
    VULKAN_RUNTIME,
    AltBrainConfig,
    Config,
)
from .decision import Arbiter, warm_up_arbiter
from .echo import EchoPool
from .engines import Brain
from .live import LiveSession, Services, warm_from_saved
from .llm import ChatClient
from .models import Downloads, repo_files
from .routing import ToolRouter
from .searxng import SearxngManager
from .tts import Synthesizer
from .turn import SmartTurn
from .vad import Segmenter, SileroVad
from .webrtc import WebRtcError, WebRtcSession

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
LIVE_PATH = "/v1/live/sessions"
LIVE_ALIAS = "/live"
MAX_AUDIO_BYTES = 16000 * 2  # one second; live clients normally send 20-80 ms
RESTART_EXIT_CODE = 75  # the launch scripts start Localive again after this exit code


class LogBuffer(logging.Handler):
    """The last log lines, for the admin page's "Journal"."""

    def __init__(self, size: int = 800) -> None:
        super().__init__(logging.INFO)
        self.lines: deque[tuple[int, str, str]] = deque(maxlen=size)
        self.count = 0
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                                            datefmt="%H:%M:%S"))  # fmt: skip

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.count += 1
            self.lines.append((self.count, record.levelname, self.format(record)))
        except Exception:  # noqa: BLE001 - logging must never raise
            self.handleError(record)

    def after(self, index: int) -> list[dict]:
        return [{"n": n, "level": level, "text": text} for n, level, text in self.lines
                if n > index]  # fmt: skip


LOG_BUFFER = LogBuffer()


def _http_origin(host: str, port: int) -> str:
    authority = f"[{host}]" if ":" in host else host or "127.0.0.1"
    return f"http://{authority}:{port}"


def lan_addresses(family: int = socket.AF_INET) -> list[str]:
    """Addresses reachable by Live clients using the server's listening socket family."""
    found: list[str] = []
    with contextlib.suppress(OSError):
        for info in socket.getaddrinfo(socket.gethostname(), None, family):
            address = info[4][0]
            ip = ipaddress.ip_address(address)
            if not ip.is_loopback and not ip.is_link_local and address not in found:
                found.append(address)
    if family == socket.AF_INET6:
        return found
    with contextlib.suppress(OSError):  # the address of the default route, when available
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect(("192.0.2.1", 9))  # TEST-NET: no packet is sent for UDP connect
            address = probe.getsockname()[0]
        finally:
            probe.close()
        if address not in found and not address.startswith("127."):
            found.insert(0, address)
    return sorted(found, key=_lan_rank)  # stable: same rank keeps the default route first


def _lan_rank(address: str) -> int:
    """Home networks first; VPN (Tailscale's 100.64/10) and virtual switches (WSL, Hyper-V and
    Docker use 172.16/12) after, since Cornelius usually sits on the home network."""
    ip = ipaddress.ip_address(address)
    if ip in ipaddress.ip_network("192.168.0.0/16") or ip in ipaddress.ip_network("10.0.0.0/8"):
        return 0
    if ip in ipaddress.ip_network("172.16.0.0/12"):
        return 1
    if ip in ipaddress.ip_network("100.64.0.0/10"):
        return 2
    return 3
SEND_TIMEOUT_S = 2.0


async def send_live_event(ws, event: dict, transport=None) -> None:
    """A stalled client must release the shared synthesis engine promptly."""
    try:
        async with asyncio.timeout(SEND_TIMEOUT_S):
            await ws.send_json(event)
    except (TimeoutError, OSError) as exc:
        # Do not close the LiveSession here: its send lock is held by the caller.
        # Closing the socket wakes the request handler, which owns session cleanup.
        with contextlib.suppress(TimeoutError, OSError, RuntimeError):
            async with asyncio.timeout(0.25):
                await ws.close(code=1013, message=b"audio client too slow", drain=False)
        if transport is not None:
            transport.close()
        raise ConnectionError("Live audio client disconnected or too slow") from exc


def validate_live_event(event: object) -> str | None:
    """Reject malformed payloads before they reach the real-time pipeline.

    Live is PCM16 mono at 16 kHz. Realtime's 24 kHz wire format cannot be silently
    accepted here: it would change both the pitch and the duration of the audio.
    """
    if not isinstance(event, dict) or not isinstance(event.get("type"), str):
        return "Un événement Live doit être un objet avec un champ type."
    kind = event["type"]
    if kind.startswith(("input_audio_buffer.", "conversation.item.")):
        return "Cet endpoint utilise Live (session.start), pas le protocole Realtime."
    if kind in {"session.start", "session.update"}:
        session = event.get("session", {})
        if not isinstance(session, dict):
            return "session doit être un objet."
        if session.get("type") in ("realtime", "transcription"):
            return "Cet endpoint utilise Live (session.start), pas le protocole Realtime."
        for key in ("audio", "delegation", "localive"):
            if session.get(key) is not None and not isinstance(session[key], dict):
                return f"session.{key} doit être un objet."
        local = session.get("localive") or {}
        if "end_on_goodbye" in local and not isinstance(local["end_on_goodbye"], bool):
            return "session.localive.end_on_goodbye doit être un booléen."
        if "goodbye_mode" in local and local["goodbye_mode"] not in GOODBYE_MODES:
            return "session.localive.goodbye_mode doit valoir phrases ou d3."
        audio = session.get("audio") or {}
        for key in ("input", "output", "format"):
            if audio.get(key) is not None and not isinstance(audio[key], dict):
                return f"session.audio.{key} doit être un objet."
        formats = [audio.get("format")]
        for direction in ("input", "output"):
            options = audio.get(direction) or {}
            formats.append(options.get("format"))
            voice = options.get("voice")
            if voice is not None and not isinstance(voice, str):
                return "La voix doit être une chaîne."
        for fmt in formats:
            if fmt is None:
                continue
            if not isinstance(fmt, dict) or fmt.get("type", "audio/pcm") != "audio/pcm":
                return "Localive Live accepte uniquement audio/pcm (PCM16 mono)."
            if fmt.get("rate", 16000) != 16000:
                return "Localive Live attend du PCM16 mono à 16000 Hz."
        delegation = (session.get("delegation") or {}).get("responses")
        if delegation is not None and not isinstance(delegation, dict):
            return "session.delegation.responses doit être un objet."
        for options in (session, delegation or {}):
            if options.get("tools") is not None and not isinstance(options["tools"], list):
                return "tools doit être une liste."
    elif kind == "response.item.create":
        if not isinstance(event.get("item"), dict):
            return "item doit être un objet."
    elif kind == "session.input_audio.append":
        audio = event.get("audio")
        if not isinstance(audio, str):
            return "audio doit contenir du PCM16 encodé en base64."
        if len(audio) > ((MAX_AUDIO_BYTES + 2) // 3) * 4:
            return "Envoyer l'audio par blocs d'une seconde maximum (20 à 80 ms conseillé)."
        try:
            data = base64.b64decode(audio, validate=True)
        except (ValueError, binascii.Error):
            return "audio n'est pas un encodage base64 valide."
        if len(data) > MAX_AUDIO_BYTES:
            return "Envoyer l'audio par blocs d'une seconde maximum (20 à 80 ms conseillé)."
        if len(data) % 2:
            return "Un échantillon PCM16 doit contenir exactement deux octets."
    return None


def problem_is_fatal(event: object) -> bool:
    """Cornelius closes the whole conversation on any ``error`` event. Only a broken session
    configuration or the wrong protocol deserves that; one malformed event is a warning."""
    if not isinstance(event, dict) or not isinstance(event.get("type"), str):
        return False
    kind = event["type"]
    return kind in {"session.start", "session.update"} or kind.startswith(
        ("input_audio_buffer.", "conversation.item.")
    )


def is_loopback(host: str | None) -> bool:
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def same_origin(request: web.Request) -> bool:
    """Browsers send Origin; native clients (Cornelius, satellites) usually do not."""
    origin = request.headers.get("Origin")
    if origin is None:
        return True
    parsed = urlsplit(origin)
    return bool(parsed.netloc) and parsed.netloc.lower() == request.host.lower()


def lmstudio_reasoning_options(catalogue: dict) -> dict[str, list[str]]:
    options = {}
    for model in catalogue.get("models", []):
        capabilities = model.get("capabilities") or {}
        levels = (capabilities.get("reasoning") or {}).get("allowed_options") or []
        names = [model.get("key"), *(i.get("id") for i in model.get("loaded_instances", []))]
        for name in filter(None, names):
            options[name] = levels
    return options


def _error(status: int, code: str, message: str) -> web.Response:
    return web.json_response(
        {"error": {"type": "invalid_request_error", "code": code, "message": message}},
        status=status,
    )


class App:
    def __init__(self, config: Config, *, open_browser: bool = False) -> None:
        self.config = config
        self.open_browser = open_browser
        self.brain = Brain(config)
        self.router_engine: Brain | None = None
        self._router_task: asyncio.Task | None = None
        self._router_download: asyncio.Task | None = None
        self._router_wanted = False
        self.alts: list[Brain] = []  # the dedicated arbiter, when enabled
        if config.local_arbiter:
            decision = config.decision
            arbiter = AltBrainConfig(
                name="arbiter", model=decision.model or config.brain.model,
                llama_dir=decision.llama_dir, port=decision.port,
                context=decision.context, extra_args="",
            )  # fmt: skip
            self.alts.append(Brain(config, arbiter, "arbiter"))
        self.services: Services | None = None
        self.ready = asyncio.Event()
        self.status = "Démarrage"
        self.failure: str | None = None
        self.failure_kind = ""  # "hf_token" / "hf_terms": the admin page guides the fix
        self.sessions: set[LiveSession] = set()
        self.rtc: dict[str, WebRtcSession] = {}  # WebRTC sessions, by session id
        self.searxng = SearxngManager(config)
        self.downloads = Downloads(config)
        self.started_at = time.time()
        self.restart_pending: set[str] = set()  # settings saved for the next start
        self.restart_requested = False
        self._negotiating_webrtc = 0
        self._warm_task: asyncio.Task | None = None
        self._http: aiohttp.ClientSession | None = None
        self._boot: asyncio.Task | None = None
        self._transcriber: Transcriber | None = None
        self._synthesizer: Synthesizer | None = None
        self.tts_label = ""

    # ------------------------------------------------------------ lifecycle
    async def on_startup(self, _app: web.Application) -> None:
        # llama-server drops idle keep-alive sockets; reusing one fails with
        # "Server disconnected". Fresh local connections cost nothing.
        self._http = aiohttp.ClientSession(connector=aiohttp.TCPConnector(force_close=True))
        self._boot = asyncio.create_task(self.boot())

    async def boot(self) -> None:
        config, http = self.config, self._http
        try:
            self.status = "Chargement des modèles"
            hf.configure(config)
            self._transcriber = Transcriber(config)
            self._synthesizer = Synthesizer(config)
            synthesizer = self._synthesizer
            await asyncio.gather(
                self.brain.start(http),
                *(alt.start(http) for alt in self.alts),
                self._transcriber.start(),
                synthesizer.start(),
                asyncio.to_thread(SileroVad.load_models),
            )
            self.tts_label = synthesizer.label
            vad = config.vad
            if not config.routing.enabled:
                log.info("Tri des outils désactivé dans la configuration, catalogue direct.")
            self.services = Services(
                config=config,
                http=http,
                transcriber=self._transcriber,
                chat=ChatClient(http, config.llm, lmstudio=config.brain.backend == "lmstudio"),
                tts=synthesizer,
                search_manager=self.searxng,
                request_router=self.request_router,
                turn=SmartTurn(config.path(config.turn.model), threads=config.turn.threads)
                if config.turn.enabled else None,
                arbiter=Arbiter(http, config) if config.decision.enabled else None,
                echo=EchoPool(config),
                make_segmenter=lambda: Segmenter(
                    SileroVad(),
                    threshold=vad.threshold,
                    silence_ms=vad.silence_ms,
                    min_speech_ms=vad.min_speech_ms,
                    preroll_ms=vad.preroll_ms,
                    max_utterance_s=vad.max_utterance_s,
                ),
            )
            if self.services.arbiter is not None:  # compile kernels, cache the prompts
                await warm_up_arbiter(self.services.arbiter)
            self.status = "Prêt"
            self.ready.set()
            if config.turn.uses_d3_goodbye:
                self.request_router()
            elif config.routing.enabled and not config.decision_model_installed:
                self._ensure_router_weights()
            if ((config.routing.enabled or config.turn.uses_d3_goodbye)
                    and not config.decision_model_installed and self._router_download is None):
                log.warning("Décideur absent (%s). En mode hors ligne ou avec un modèle "
                            "personnalisé, placez le fichier à cet emplacement.",
                            config.routing.model)
            # In the background: a conversation may start meanwhile (it then waits less).
            self._warm_task = asyncio.create_task(warm_from_saved(self.services))
            url = _http_origin(config.server.host, config.server.port)
            log.info("Localive prêt : %s/v1 (administration : %s)", url, url)
            self._open_admin()
        except hf.GatedModelError as exc:
            # A fresh install without Hugging Face access: no traceback, a way out.
            log.error("%s", exc)
            self.failure = str(exc)
            self.failure_kind = f"hf_{exc.kind}" if exc.kind in ("token", "terms") else ""
            self.status = "Échec du démarrage"
            self._open_admin()
        except Exception as exc:  # noqa: BLE001 - reported on /health and to clients
            log.exception("Startup failed")
            self.failure = str(exc)
            self.status = "Échec du démarrage"
            self._open_admin()

    def _router_exited(self) -> bool:
        process = self.router_engine.process if self.router_engine else None
        return process is not None and process.poll() is not None

    def request_router(self) -> None:
        services = self.services
        self._router_wanted = True
        if services.router_starting:
            return
        if services.router is not None:
            if not self._router_exited():
                return
            log.warning("Le processus du décideur s’est arrêté, redémarrage en arrière-plan.")
            services.router = None
            self.router_engine.stop()
        if time.monotonic() < services.router_retry_at:
            return
        if not self.config.decision_model_installed:
            self._ensure_router_weights()
            return
        services.router_starting = True
        self._router_task = asyncio.create_task(self._start_router())

    def _ensure_router_weights(self) -> None:
        if self._router_download and not self._router_download.done():
            return
        config = self.config
        offline = os.getenv("HF_HUB_OFFLINE", "").strip().upper() in {"1", "ON", "YES", "TRUE"}
        model = config.path(config.routing.model).resolve()
        if (config.huggingface.offline or offline
                or model != config.path(DEFAULT_ROUTER).resolve()):
            return
        self._router_download = asyncio.create_task(self._fetch_router_weights())

    async def _fetch_router_weights(self) -> None:
        log.info("Téléchargement de d3-mini en arrière-plan (5,16 Go). "
                 "Pendant le transfert, Qwen reçoit le catalogue complet.")
        try:
            await install.fetch_default_router(self.config.root)
        except (OSError, install.InstallError) as exc:
            self.services.router_retry_at = time.monotonic() + 30
            log.warning("Téléchargement de d3-mini interrompu (%s). Un prochain tour "
                        "pourra le reprendre après 30 s.", exc)
            return
        self.services.router_retry_at = 0
        log.info("Poids de d3-mini vérifiés.")
        if self._router_wanted:
            self.request_router()

    async def _start_router(self) -> None:
        """Load optional decision weights after the voice engines have their resources."""
        config = self.config
        layers = config.routing.gpu_layers
        runtime = config.path(config.routing.llama_dir or config.brain.llama_dir).resolve()
        if layers == -1:
            main_runtime = config.path(config.brain.llama_dir).resolve()
            layers = config.brain.gpu_layers if runtime == main_runtime else 99
        extra = "-b 2048 -ub 2048 -fa auto -ctk f16 -ctv f16"
        if (layers > 0 and runtime == config.path(VULKAN_RUNTIME).resolve()
                and not os.getenv("LLAMA_ARG_DEVICE")):
            # Without an explicit device, llama.cpp can silently accept -ngl on a CPU-only host.
            extra += " --device Vulkan0"
        spec = AltBrainConfig(name="tool-router", model=config.routing.model,
                              llama_dir=config.routing.llama_dir,
                              gpu_layers=layers,
                              port=config.routing.port, context=4096,
                              extra_args=extra)
        self.router_engine = Brain(config, spec, "tool-router")
        router = ToolRouter(self._http, config.routing)
        try:
            await self.router_engine.start(self._http, reuse=False)
            async with asyncio.timeout(config.brain.startup_timeout):
                await router.verify()
        except (aiohttp.ClientError, OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            self.router_engine.stop()
            self.services.router_retry_at = time.monotonic() + 30
            log.warning("Décideur indisponible, catalogue complet et conversation ouverte (%r). "
                        "Nouvel essai possible dans 30 s.", exc)
            return
        finally:
            self.services.router_starting = False
        self.services.router_retry_at = 0
        self.services.router = router
        log.info("Décideur prêt : %s (décision SystemOne vérifiée)",
                 Path(config.routing.model).name)

    async def on_cleanup(self, _app: web.Application) -> None:
        for link in list(self.rtc.values()):
            await link.close("server_shutdown")
        for session in list(self.sessions):
            await session.close("server_shutdown")
        if self._boot and not self._boot.done():
            self._boot.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._boot
        tasks = [task for task in (self.searxng.startup_task, self._warm_task,
                                   self._router_download, self._router_task)
                 if task and not task.done()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self._transcriber:
            await self._transcriber.aclose()
        if self._synthesizer:
            await self._synthesizer.aclose()
        self.brain.stop()
        if self.router_engine:
            self.router_engine.stop()
        for alt in self.alts:
            alt.stop()
        if self._http:
            await self._http.close()

    def _open_admin(self) -> None:
        """Once ready, or when startup failed: the admin page explains what to do."""
        if self.open_browser:
            self.open_browser = False
            server = self.config.server
            host = {"::": "::1", "0.0.0.0": "127.0.0.1"}.get(server.host, server.host)
            webbrowser.open(_http_origin(host, server.port))

    # ------------------------------------------------------------------ auth
    def authorized(self, request: web.Request) -> bool:
        expected = self.config.server.api_key
        if not expected:
            return True
        header = request.headers.get("Authorization", "")
        token = header[7:] if header.startswith("Bearer ") else request.query.get("api_key", "")
        return hmac.compare_digest(token.encode(), expected.encode())

    @web.middleware
    async def auth_middleware(self, request: web.Request, handler):
        protected = ("/v1/", "/api/", LIVE_ALIAS + "/")
        if request.path != LIVE_ALIAS and not request.path.startswith(protected):
            return await handler(request)
        if not same_origin(request):
            # Any web page the user visits could otherwise open a Live session or change the
            # settings of this local server. Cornelius connects from Node: no Origin header.
            return _error(403, "forbidden_origin", "Origine non autorisée.")
        if not self.authorized(request):
            return _error(401, "invalid_api_key", "Jeton Localive invalide.")
        if (request.path.startswith("/api/") and not self.config.server.api_key
                and not is_loopback(request.remote)):
            return _error(403, "api_key_required",
                          "Accéder à l'administration depuis le réseau demande server.api_key.")
        return await handler(request)

    # --------------------------------------------------------------- routes
    async def index(self, _request: web.Request) -> web.Response:
        """The admin page. Asset URLs carry a version: an update is never hidden by a cache."""
        stamp = max(int(p.stat().st_mtime) for p in STATIC.glob("admin.*"))
        page = (STATIC / "index.html").read_text(encoding="utf-8")
        return web.Response(text=page.replace("{{version}}", f"{__version__}-{stamp}"),
                            content_type="text/html", headers={"Cache-Control": "no-cache"})

    async def health(self, _request: web.Request) -> web.Response:
        services = self.services
        download_pct = None
        if services and services.router and not self._router_exited():
            router_state = "ready"
        elif self._router_download and not self._router_download.done():
            router_state = "downloading"
            model = self.config.path(DEFAULT_ROUTER)
            download_pct = 100 if model.is_file() else 0
            with contextlib.suppress(OSError):
                size = model.with_suffix(".gguf.part").stat().st_size
                download_pct = min(100, size * 100 // install.ROUTER_SIZE)
        elif self._router_download and not self.config.decision_model_installed:
            router_state = "download_failed"
        elif self._router_task is not None:
            router_state = "unavailable" if self._router_task.done() else "loading"
        elif self.config.decision_model_installed:
            router_state = "idle"
        else:
            router_state = "missing"
        tool_router_state = router_state if self.config.routing.enabled else "off"
        return web.json_response(
            {
                "ready": self.ready.is_set() and not self.restart_requested,
                "restarting": self.restart_requested,
                "status": self.status,
                "error": self.failure,
                "error_kind": self.failure_kind or None,
                "sessions": len(self.sessions),
                "asr": services.transcriber.label if services else None,
                "asr_error": services.transcriber.probe_error if services else None,
                "tts": self.tts_label or None,
                "tts_error": self._synthesizer.probe_error if self._synthesizer else None,
                "voices": {
                    "female": (
                        self.config.voice.female
                        if self.config.voice.backend == "phonon"
                        else (self.config.voice.external_female or "par défaut")
                    ),
                    "male": (
                        self.config.voice.male
                        if self.config.voice.backend == "phonon"
                        else (self.config.voice.external_male or "par défaut")
                    ),
                    "default": self.config.voice.default,
                },  # fmt: skip
                "brain": self.config.llm.model,
                "decision_model_state": router_state,
                "decision_model_download_pct": download_pct,
                "tool_router": tool_router_state == "ready",
                "tool_router_state": tool_router_state,
                "tool_router_file": (Path(self.config.routing.model).name
                                     if tool_router_state == "ready" else None),
                "brains": list(self.config.brains()),
                "brain_file": (Path(self.config.brain.model).name
                               if self.config.brain.backend == "localive"
                               else f"{self.config.llm.upstream_model or '?'} "
                                    f"({self.config.brain.backend})"),
                "search": self.searxng.status(),
                "protocols": {"live": {"path": LIVE_PATH, "audio_rate": 16000,
                                       "transports": ["websocket", "webrtc"]}},
            }
        )

    def _track_restart(self, keys: list[str], result: dict) -> None:
        """A key set back to the value running now no longer waits for a restart."""
        self.restart_pending.difference_update(keys)
        self.restart_pending.update(result["restart_required"])
        # Switching providers can implicitly remove a key saved for the next start.
        if self.config.file_values.get("llm", {}).get("api_key", "") == self.config.llm.api_key:
            self.restart_pending.discard("llm.api_key")

    async def get_settings(self, _request: web.Request) -> web.Response:
        return web.json_response({**settings.schema(self.config, self.restart_pending),
                                  "restart_pending": sorted(self.restart_pending)})

    async def reset_settings(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
            keys = body.get("keys") if isinstance(body, dict) else None
            if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
                raise ValueError("Liste de réglages attendue.")
            result = settings.reset(self.config, keys)
        except (ValueError, json.JSONDecodeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except OSError as exc:
            return web.json_response({"error": f"Enregistrement impossible : {exc}"}, status=500)
        self._track_restart(keys, result)
        log.info("Réglages remis par défaut : %s", ", ".join(keys))
        return web.json_response({**result, **settings.schema(self.config, self.restart_pending),
                                  "restart_pending": sorted(self.restart_pending)})

    def connection(self) -> dict:
        """Reachable base URLs for Live clients."""
        server = self.config.server
        everywhere = server.host in ("0.0.0.0", "::", "")
        if everywhere:
            ipv6 = server.host == "::"
            hosts = lan_addresses(socket.AF_INET6 if ipv6 else socket.AF_INET)
            hosts.append("::1" if ipv6 else "127.0.0.1")
        else:
            hosts = [server.host]
        urls = [_http_origin(host, server.port) + "/v1" for host in dict.fromkeys(hosts)]
        return {"urls": urls, "lan": not is_loopback(server.host),
                "api_key_required": bool(server.api_key)}

    async def api_status(self, _request: web.Request) -> web.Response:
        sessions = []
        for session in list(self.sessions):
            sessions.append({
                "id": session.id, "voice": session.voice, "brain": session.brain,
                "transport": "webrtc" if session.id in self.rtc else "websocket",
                "since": round(session.created), "state": session.state,
            })  # fmt: skip
        health = await self.health(_request)
        return web.json_response({
            **json.loads(health.body), "version": __version__,
            "uptime_s": round(time.time() - self.started_at), "sessions": sessions,
            "config_file": str(self.config.file) if self.config.file else None,
            "restart_pending": sorted(self.restart_pending),
            "connection": await asyncio.to_thread(self.connection),
        })  # fmt: skip

    async def logs(self, request: web.Request) -> web.Response:
        try:
            after = int(request.query.get("after", "0"))
        except ValueError:
            after = 0
        return web.json_response({"lines": LOG_BUFFER.after(after)})

    async def check_hf(self, request: web.Request) -> web.Response:
        """Check a Hugging Face token against the voice model before saving it."""
        try:
            body = await request.json()
            token = body.get("token") if isinstance(body, dict) else None
            if not isinstance(token, str) or not token.strip():
                raise ValueError("Jeton attendu.")
        except (ValueError, json.JSONDecodeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        state, message = await hf.check_access(self._http, token, self.config.voice.model)
        return web.json_response({"state": state, "message": message,
                                  "page": "https://huggingface.co/" +
                                  self.config.voice.model.removeprefix("hf://").split("@")[0]})

    async def list_models(self, _request: web.Request) -> web.Response:
        return web.json_response({"models": self.downloads.listing()})

    async def download_model(self, request: web.Request) -> web.Response:
        """Download a reasoning model of the catalogue (page Cerveau)."""
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("Objet JSON attendu.")
            self.downloads.start(str(body.get("id") or ""), repo=str(body.get("repo") or ""),
                                 file=str(body.get("file") or ""))  # fmt: skip
        except (ValueError, json.JSONDecodeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except RuntimeError as exc:
            return web.json_response({"error": str(exc)}, status=409)
        return web.json_response({"models": self.downloads.listing()})

    async def model_files(self, request: web.Request) -> web.Response:
        """GGUF files of a Hugging Face repository, to add a model outside the catalogue."""
        try:
            body = await request.json()
            repo = str(body.get("repo") or "") if isinstance(body, dict) else ""
            files = await asyncio.to_thread(repo_files, repo)
        except (ValueError, json.JSONDecodeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        return web.json_response({"repo": repo.strip(), "files": files})

    async def external_models(self, request: web.Request) -> web.Response:
        """Models offered by an Ollama / LM Studio server (page Cerveau, before saving)."""
        names = {"ollama": "Ollama", "lmstudio": "LM Studio", "openai": "Le serveur"}
        try:
            body = await request.json()
            backend = str(body.get("backend") or "")
            url = settings.external_model_url(backend, str(body.get("url") or ""))
        except (ValueError, AttributeError, json.JSONDecodeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        brain = self.config.brain
        configured_url = (brain.external_url or BRAIN_SERVERS.get(brain.backend, "")).rstrip("/")
        key = self.config.llm.api_key if (backend, url) == (brain.backend, configured_url) else ""
        # A new address and key may already be saved for the next restart: the settings page
        # should be able to list its models before making it the live provider.
        saved = self.config.file_values.get("brain", {})
        saved_backend = saved.get("backend", "localive")
        saved_url = saved.get("external_url") or BRAIN_SERVERS.get(saved_backend, "")
        if (backend, url) == (saved_backend, saved_url.rstrip("/")):
            key = self.config.file_values.get("llm", {}).get("api_key", "")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        reasoning = {}
        try:
            async with aiohttp.ClientSession() as http, http.get(
                url + "/models", headers=headers, allow_redirects=False,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as response:
                if response.status != 200:
                    return web.json_response({"url": url, "models": [], "error":
                        f"{names[backend]} a répondu HTTP {response.status} sur {url}."})
                data = await response.json(content_type=None)
                if backend == "lmstudio" and url.endswith("/v1"):
                    # Older LM Studio servers expose /v1/models only.
                    with contextlib.suppress(aiohttp.ClientError, TimeoutError, ValueError,
                                             AttributeError, TypeError):
                        native_url = url.removesuffix("/v1") + "/api/v1/models"
                        async with http.get(native_url, headers=headers, allow_redirects=False,
                                            timeout=aiohttp.ClientTimeout(total=3)) as native:
                            if native.status == 200:
                                catalogue = await native.json()
                                reasoning = lmstudio_reasoning_options(catalogue)
            models = sorted({str(m["id"]) for m in data.get("data", []) if m.get("id")})
        except (aiohttp.ClientError, TimeoutError, ValueError, AttributeError, KeyError,
                TypeError) as exc:  # fmt: skip
            if isinstance(exc, TimeoutError):
                error = f"{names[backend]} n'a pas répondu en 20 s sur {url} : réessayez."
            else:
                hint = "" if backend == "openai" else " : lancez-le et activez son serveur"
                error = f"{names[backend]} ne répond pas sur {url}{hint} ({exc})."
            return web.json_response({"url": url, "models": [], "error": error})
        return web.json_response({"url": url, "models": models, "reasoning": reasoning})

    async def retry_search(self, _request: web.Request) -> web.Response:
        self.searxng.start(self._http, retry=True)
        return web.json_response(self.searxng.status())

    async def restart(self, _request: web.Request) -> web.Response:
        """Stop gracefully; the launch scripts (or systemd) start Localive again."""
        if self.restart_requested:
            return web.json_response({"restarting": True})
        if self.sessions or self._negotiating_webrtc:
            return web.json_response(
                {"error": "Une conversation est en cours : réessayez une fois terminée."},
                status=409)  # fmt: skip
        self.restart_requested = True
        log.info("Redémarrage demandé depuis la page d'administration")

        def stop() -> None:
            raise web.GracefulExit()

        asyncio.get_running_loop().call_later(0.3, stop)
        return web.json_response({"restarting": True})

    async def put_settings(self, request: web.Request) -> web.Response:
        try:
            changes = await request.json()
            if not isinstance(changes, dict):
                raise ValueError("Objet JSON attendu.")
            result = settings.apply(self.config, changes)
        except (ValueError, json.JSONDecodeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except OSError as exc:
            return web.json_response({"error": f"Enregistrement impossible : {exc}"}, status=500)
        self._track_restart(list(changes), result)
        log.info("Réglages : appliqués %s, au redémarrage %s", result["applied"],
                 result["restart_required"])  # fmt: skip
        return web.json_response({**result, **settings.schema(self.config, self.restart_pending),
                                  "restart_pending": sorted(self.restart_pending)})

    async def models(self, _request: web.Request) -> web.Response:
        now = int(time.time())
        data = [
            {"id": m, "object": "model", "created": now, "owned_by": "localive"}
            for m in ("gpt-live-1", *self.config.brains())  # Live model + delegation brains
        ]
        return web.json_response({"object": "list", "data": data})

    async def model(self, request: web.Request) -> web.Response:
        return web.json_response(
            {"id": request.match_info["model"], "object": "model", "owned_by": "localive"}
        )

    async def hangup(self, request: web.Request) -> web.Response:
        target = request.match_info["session_id"]
        if target in self.rtc:
            await self.rtc[target].close("hangup")
        for session in list(self.sessions):
            if session.id == target:
                await session.close("hangup")
        return web.json_response({})

    async def _wait_ready(self) -> str | None:
        """Cornelius waits ~5 s for session.started: fail fast and clearly."""
        if not self.ready.is_set() and not self.restart_requested:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self.ready.wait(), timeout=3)
        if self.restart_requested:
            return "Localive redémarre."
        if self.ready.is_set():
            return None
        return self.failure or "Localive charge encore ses modèles."

    async def _pump(self, ws: web.WebSocketResponse, session: LiveSession,
                    audio: bool = True) -> None:  # fmt: skip
        """Read client events until the socket or the session closes."""
        async for message in ws:
            if session.closed or message.type != aiohttp.WSMsgType.TEXT:
                continue
            try:
                event = json.loads(message.data)
            except json.JSONDecodeError:
                await session.warn("invalid_json", "Événement Live ignoré : JSON invalide.")
                continue
            problem = validate_live_event(event)
            if (problem is None and not audio and isinstance(event, dict)
                    and event.get("type") == "session.input_audio.append"):
                problem = "En WebRTC, le micro passe par la piste audio, pas par la WebSocket."
            if problem:
                client_id = event.get("event_id") if isinstance(event, dict) else None
                if problem_is_fatal(event):
                    error = {"type": "invalid_request_error", "code": "invalid_event",
                             "message": problem}
                    if isinstance(client_id, str):
                        error["client_event_id"] = client_id
                    await session.send({"type": "error", "error": error})
                else:
                    warning = {"type": "voice.provider_warning",
                               "error": {"code": "invalid_event", "message": problem}}
                    if isinstance(client_id, str):
                        warning["client_event_id"] = client_id
                    await session.send(warning)
                continue
            await session.handle(event)

    async def live(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=20, max_msg_size=8 * 1024 * 1024)
        await ws.prepare(request)
        if (message := await self._wait_ready()) is not None:
            await ws.send_json(
                {"type": "error", "error": {"code": "engine_not_ready", "message": message}}
            )
            await ws.close()
            return ws

        async def send(event: dict) -> None:
            await send_live_event(ws, event, request.transport)
            if event["type"] == "session.closed":
                await ws.close()

        session = LiveSession(self.services, send)
        self.sessions.add(session)
        try:
            await self._pump(ws, session)
        finally:
            await session.close("disconnected")
            self.sessions.discard(session)
            await ws.close()
        return ws

    async def create_webrtc(self, request: web.Request) -> web.Response:
        """OpenAI ``client.live.create({session, transport: {type: "webrtc", sdp}})``."""

        def refuse(status: int, code: str, message: str) -> web.Response:
            error = {"type": "invalid_request_error", "code": code, "message": message}
            return web.json_response({"error": error}, status=status)

        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            return refuse(400, "invalid_json", "Corps JSON attendu.")
        transport = body.get("transport") if isinstance(body, dict) else None
        session_config = body.get("session", {}) if isinstance(body, dict) else None
        if (not isinstance(transport, dict) or transport.get("type") != "webrtc"
                or not isinstance(transport.get("sdp"), str)):
            return refuse(400, "invalid_transport",
                          'transport doit être {"type": "webrtc", "sdp": "<offre>"}.')
        problem = validate_live_event({"type": "session.start", "session": session_config})
        if problem:
            return refuse(400, "invalid_session", problem)
        if (message := await self._wait_ready()) is not None:
            return refuse(503, "engine_not_ready", message)
        link = WebRtcSession(self.services)
        self._negotiating_webrtc += 1
        try:
            answer = await link.answer(session_config, transport["sdp"])
        except WebRtcError as exc:
            return refuse(400, "invalid_offer", str(exc))
        finally:
            self._negotiating_webrtc -= 1
        self.rtc[link.id] = link
        self.sessions.add(link.session)

        def forget(closed: WebRtcSession) -> None:
            self.rtc.pop(closed.id, None)
            self.sessions.discard(closed.session)

        link.on_closed = forget
        log.info("WebRTC session %s created", link.id)
        return web.json_response({
            "session": {"id": link.id, "object": "live.session", "model": "localive"},
            "transport": {"type": "webrtc", "sdp": answer},
        })  # fmt: skip

    async def attach(self, request: web.Request) -> web.WebSocketResponse:
        """Control WebSocket of a WebRTC session (events only; audio is on the media track)."""
        ws = web.WebSocketResponse(heartbeat=20, max_msg_size=8 * 1024 * 1024)
        await ws.prepare(request)
        link = self.rtc.get(request.match_info["session_id"])
        if link is None or not await link.attach(ws):
            await ws.send_json({"type": "error", "error": {
                "type": "invalid_request_error", "code": "session_not_found",
                "message": "Session WebRTC inconnue, fermée ou déjà attachée."}})
            await ws.close()
            return ws
        try:
            await self._pump(ws, link.session, audio=False)
        finally:
            await link.close("disconnected")
        return ws

    def build(self) -> web.Application:
        app = web.Application(middlewares=[self.auth_middleware])
        app.on_startup.append(self.on_startup)
        app.on_cleanup.append(self.on_cleanup)
        app.router.add_get("/", self.index)
        app.router.add_get("/health", self.health)
        app.router.add_get("/api/settings", self.get_settings)
        app.router.add_put("/api/settings", self.put_settings)
        app.router.add_post("/api/settings/reset", self.reset_settings)
        app.router.add_get("/api/status", self.api_status)
        app.router.add_get("/api/logs", self.logs)
        app.router.add_post("/api/search/retry", self.retry_search)
        app.router.add_post("/api/hf/check", self.check_hf)
        app.router.add_post("/api/restart", self.restart)
        app.router.add_get("/api/models", self.list_models)
        app.router.add_post("/api/brain/models", self.external_models)
        app.router.add_post("/api/models/download", self.download_model)
        app.router.add_post("/api/models/files", self.model_files)
        app.router.add_get("/v1/models", self.models)
        app.router.add_get("/v1/models/{model}", self.model)
        for path in (LIVE_PATH, LIVE_ALIAS):
            app.router.add_get(path, self.live)
            app.router.add_post(path, self.create_webrtc)
            app.router.add_get(path + "/{session_id}/attach", self.attach)
            app.router.add_post(path + "/{session_id}/hangup", self.hangup)
        app.router.add_static("/static", STATIC, append_version=True)
        return app


def run(config: Config, *, open_browser: bool = False) -> bool:
    """Serve until stopped. True when the admin page asked for a restart."""
    if not config.server.api_key and not is_loopback(config.server.host):
        log.warning(
            "Localive écoute sur %s sans server.api_key : toute machine du réseau peut ouvrir "
            "une session vocale. Les réglages restent modifiables depuis ce PC uniquement.",
            config.server.host,
        )
    app = App(config, open_browser=open_browser)
    web.run_app(
        app.build(),
        host=config.server.host,
        port=config.server.port,
        print=lambda *_: None,
        access_log=None,
    )
    return app.restart_requested
