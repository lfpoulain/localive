"""SearXNG for the Live API's hosted ``web_search`` tool, run by Localive in Docker.

When a session first declares web_search, or the administrator requests it:

1. find Docker; on Windows, start Docker Desktop if it is installed but stopped;
2. write ``.runtime/searxng/settings.yml`` (JSON output on, no rate limiter, a secret key
   generated once and kept);
3. create the ``localive-searxng`` container (bound to 127.0.0.1 only, restarted with
   Docker), or start it if it exists;
4. wait until ``/search?format=json`` answers.

With ``search.managed = false`` Localive only checks the configured URL: any SearXNG works.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import shutil
import sys
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp

from .config import Config
from .search import SearchError

log = logging.getLogger(__name__)
OWNER_LABEL = "io.localive.searxng.settings"

DOCKER_DESKTOP = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Docker" / \
    "Docker" / "Docker Desktop.exe"
SETTINGS = """\
# Written by Localive. Edit freely: kept across restarts (the secret key is preserved).
use_default_settings: true
general:
  instance_name: localive
search:
  safe_search: 0
  default_lang: {language}
  formats: [html, json]
server:
  secret_key: "{secret}"
  limiter: false
  image_proxy: false
"""

STATES = {
    "off": "Recherche web désactivée.",
    "idle": "En attente d’une session qui utilise la recherche web.",
    "starting": "Démarrage de SearXNG…",
    "ready": "SearXNG répond.",
    "docker_missing": "Docker n'est pas installé : la recherche web est indisponible.",
    "docker_stopped": "Docker ne répond pas : démarrez Docker Desktop.",
    "unreachable": "SearXNG ne répond pas à l'adresse configurée.",
    "error": "SearXNG n'a pas pu démarrer.",
}


class SearxngManager:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._state = "idle"
        self.detail = ""
        self.startup_task: asyncio.Task[None] | None = None

    @property
    def state(self) -> str:
        return "off" if self.config.search.engine == "off" else self._state

    def status(self) -> dict:
        search = self.config.search
        state = self.state
        detail = self.detail if state != "off" else ""
        return {"state": state, "message": detail or STATES.get(state, ""),
                "url": search.url, "managed": search.managed}  # fmt: skip

    def _set(self, state: str, detail: str = "") -> None:
        self._state, self.detail = state, detail
        level = logging.INFO if state in ("ready", "starting", "off") else logging.WARNING
        log.log(level, "SearXNG: %s", detail or STATES.get(state, state))

    # -------------------------------------------------------------- startup
    def start(self, http: aiohttp.ClientSession, *, retry: bool = False) -> None:
        """Start one shared background check for all sessions that declare web search."""
        if self.config.search.engine == "off" or (self.state == "ready" and not retry):
            return
        if self.startup_task is None or self.startup_task.done():
            self._state, self.detail = "starting", ""
            self.startup_task = asyncio.create_task(self.ensure(http))

    async def wait_ready(self, http: aiohttp.ClientSession) -> None:
        """A cancelled voice turn must leave startup running for other clients."""
        if self.config.search.engine == "off":
            raise SearchError(STATES["off"])
        self.start(http)
        if self.startup_task is not None:
            await asyncio.shield(self.startup_task)
        if self.state != "ready":
            raise SearchError(self.detail or STATES[self.state])

    async def ensure(self, http: aiohttp.ClientSession) -> None:
        search = self.config.search
        if search.engine == "off":
            self._state, self.detail = "idle", ""
            return
        self._set("starting")
        try:
            if search.managed and self._local_url():
                if not await self._run_container():
                    return
            if await self._wait_ready(http, 90 if search.managed else 5):
                self._set("ready")
            elif self.state != "error":
                self._set("unreachable", f"SearXNG ne répond pas sur {search.url}.")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - web search is optional, never fatal
            self._set("error", f"SearXNG n'a pas pu démarrer : {exc}")

    def _local_url(self) -> bool:
        host = urlsplit(self.config.search.url).hostname or ""
        if host in ("127.0.0.1", "localhost", "::1"):
            return True
        self._set("starting", "Adresse SearXNG distante : Localive ne gère pas ce conteneur.")
        return False

    @staticmethod
    def _docker() -> str | None:
        found = shutil.which("docker")
        if found:
            return found
        bundled = DOCKER_DESKTOP.parent / "resources" / "bin" / "docker.exe"
        return str(bundled) if sys.platform == "win32" and bundled.is_file() else None

    async def _docker_cmd(self, docker: str, *args: str, limit_s: float = 30) -> tuple[int, str]:
        process = await asyncio.create_subprocess_exec(
            docker, *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL,
        )  # fmt: skip
        try:
            async with asyncio.timeout(limit_s):
                out, _ = await process.communicate()
        except TimeoutError:
            process.kill()
            await process.wait()
            return 124, "délai dépassé"
        return process.returncode or 0, out.decode("utf-8", "replace").strip()

    async def _docker_ready(self, docker: str) -> bool:
        code, _ = await self._docker_cmd(docker, "info", "--format", "{{.ServerVersion}}")
        if code == 0:
            return True
        if not (self.config.search.start_docker and sys.platform == "win32"
                and DOCKER_DESKTOP.is_file()):  # fmt: skip
            return False
        self._set("starting", "Démarrage de Docker Desktop…")
        await asyncio.create_subprocess_exec(str(DOCKER_DESKTOP), stdin=asyncio.subprocess.DEVNULL,
                                             stdout=asyncio.subprocess.DEVNULL,
                                             stderr=asyncio.subprocess.DEVNULL)  # fmt: skip
        for _ in range(60):  # Docker Desktop takes 20-60 s to start its engine
            await asyncio.sleep(3)
            code, _ = await self._docker_cmd(docker, "info", "--format", "{{.ServerVersion}}")
            if code == 0:
                return True
        return False

    def write_settings(self) -> Path:
        """settings.yml with JSON output; the secret key survives rewrites."""
        folder = self.config.path(".runtime/searxng")
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "settings.yml"
        if path.is_file():
            text = path.read_text(encoding="utf-8-sig")
            if "json" in text and re.search(r"secret_key:", text):
                return folder  # the user's (or our previous) file: keep it as it is
            found = re.search(r'secret_key:\s*"?([^"\s]+)"?', text)
            secret = found.group(1) if found else secrets.token_hex(32)
        else:
            secret = secrets.token_hex(32)
        path.write_text(SETTINGS.format(language=self.config.search.language, secret=secret),
                        encoding="utf-8")  # fmt: skip
        return folder

    async def _run_container(self) -> bool:
        search = self.config.search
        docker = self._docker()
        if docker is None:
            self._set("docker_missing")
            return False
        if not await self._docker_ready(docker):
            self._set("docker_stopped")
            return False
        folder = self.config.path(".runtime/searxng").resolve()
        url = urlsplit(search.url)
        host = "::1" if url.hostname == "::1" else "127.0.0.1"
        port = str(url.port or (443 if url.scheme == "https" else 80))
        code, out = await self._docker_cmd(
            docker, "inspect", "--type", "container", "--format", "{{json .}}", search.container)
        if code == 0:
            info = json.loads(out)
            conflict = self._container_conflict(info, folder, host, port)
            if conflict:
                self._set("error", f"Conteneur « {search.container} » : {conflict}. "
                          "Choisissez un autre nom de conteneur et un port disponible dans "
                          "Recherche web, puis redémarrez Localive.")
                return False
            if info["State"]["Running"]:
                return True
            self.write_settings()
            self._set("starting", "Redémarrage du conteneur SearXNG…")
            code, out = await self._docker_cmd(docker, "start", search.container)
        elif "no such object" in out.lower() or "no such container" in out.lower():
            self.write_settings()
            self._set("starting", "Installation de SearXNG (premier téléchargement)…")
            bind = f"[{host}]" if host == "::1" else host
            code, out = await self._docker_cmd(
                docker, "run", "-d", "--name", search.container, "--restart", "unless-stopped",
                "--label", f"{OWNER_LABEL}={folder}", "-p", f"{bind}:{port}:8080",
                "-v", f"{folder}:/etc/searxng", search.image, limit_s=900,
            )  # fmt: skip
        else:
            self._set("error", f"Docker ne peut pas inspecter le conteneur : {out[-300:]}")
            return False
        if code != 0:
            self._set("error", f"Docker a refusé de lancer SearXNG : {out[-300:]}")
            return False
        return True

    def _container_conflict(self, info: dict, folder: Path, host: str, port: str) -> str:
        configured = info["Config"]
        owner = (configured.get("Labels") or {}).get(OWNER_LABEL)
        if owner is not None and Path(owner) != folder:
            return "ce nom appartient à une autre installation"
        # Older Localive containers have no label; their bind mount identifies the checkout.
        for mount in info.get("Mounts", []):
            if mount.get("Type") != "bind" or mount.get("Destination") != "/etc/searxng":
                continue
            source = mount.get("Source", "")
            if sys.platform == "win32":
                # Docker Desktop can return the Linux VM's path for a Windows bind mount.
                source = re.sub(r"^/(?:run/desktop/mnt/host|host_mnt)/([a-zA-Z])/",
                                r"\1:/", source)
            if Path(source) == folder:
                break
        else:
            return "son dossier de configuration appartient à une autre installation"
        if configured.get("Image") != self.config.search.image:
            return (f"image installée {configured.get('Image')}, "
                    f"image demandée {self.config.search.image}")
        ports = info["HostConfig"].get("PortBindings") or {}
        if ports.get("8080/tcp") != [{"HostIp": host, "HostPort": port}]:
            return f"la publication du port diffère de l'adresse demandée ({host}, port {port})"
        return ""

    async def _wait_ready(self, http: aiohttp.ClientSession, seconds: float) -> bool:
        url = self.config.search.url.rstrip("/") + "/search"
        params = {"q": "localive", "format": "json"}
        end = asyncio.get_running_loop().time() + seconds
        while True:
            try:
                async with http.get(url, params=params,
                                    timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        try:
                            data = await response.json(content_type=None)
                        except (ValueError, UnicodeError):
                            data = None
                        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
                            self._set("error", "L'adresse répond sans résultats JSON SearXNG. "
                                      "Vérifiez search.url et le format json de settings.yml.")
                            return False
                        return True
                    if response.status == 403:  # JSON output disabled in settings.yml
                        self._set("error", "SearXNG refuse le format JSON (settings.yml).")
                        return False
            except (aiohttp.ClientError, TimeoutError, OSError):
                pass
            if asyncio.get_running_loop().time() > end:
                return False
            await asyncio.sleep(2)
