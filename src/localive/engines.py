"""Start, watch and stop llama.cpp for the conversational model and the dedicated arbiter."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import subprocess
import sys
import time

import aiohttp

from . import children
from .catalog import QWEN_4B_MTP
from .config import Config

log = logging.getLogger(__name__)


class EngineError(RuntimeError):
    pass


def fast_attention(extra: list[str], mode: str = "on") -> list[str]:
    """Flash attention and an 8-bit KV cache, unless the extra options choose otherwise.

    Generation reads the whole KV cache for every token, and Cornelius's prompt is ~8,600
    tokens long. Measured on a Radeon 890M (Vulkan, Qwen3-4B, 8,192 tokens of context):
    13.0 tokens/s by default ("auto" leaves flash attention off there), 20.0 with it,
    22.6 with the 8-bit cache too. Prompt reading is unchanged (~700 tokens/s).
    """
    defaults = []
    if not {"-fa", "--flash-attn"} & set(extra):
        defaults += ["-fa", mode]
    if not {"-ctk", "--cache-type-k", "-ctv", "--cache-type-v"} & set(extra):
        defaults += ["-ctk", "q8_0", "-ctv", "q8_0"]
    return defaults


class Brain:
    """One llama-server child process; its log goes to ``.runtime/logs/brain.log``."""

    def __init__(self, config: Config, spec=None, name: str = "qwen") -> None:
        self.config = config
        self.spec = spec or config.brain
        self.name = name
        self.main = spec is None
        decision = config.decision
        # Two slots: barge-in + the two parallel "how to continue" questions.
        self.serves_arbiter = name == "arbiter" or (
            self.main and decision.enabled and not decision.dedicated
        )
        self.timeout = self.config.brain.startup_timeout
        self.url = f"http://127.0.0.1:{self.spec.port}"
        self.process: subprocess.Popen | None = None
        self._log_file = None

    def command(self) -> list[str]:
        spec = self.spec
        llama_dir = spec.llama_dir or self.config.brain.llama_dir
        binary = self.config.path(llama_dir) / (
            "llama-server.exe" if sys.platform == "win32" else "llama-server"
        )
        model = self.config.path(spec.model)
        for required in (binary, model):
            if not required.is_file():
                raise EngineError(f"Missing file: {required} (run scripts/setup.ps1)")
        extra = str(getattr(spec, "extra_args", "") or "").split()
        mtp = self.main and model == self.config.path(QWEN_4B_MTP.path)
        profile_args = []
        if mtp and "--spec-type" not in extra:
            profile_args.extend(["--spec-type", "draft-mtp"])
            if "--spec-draft-n-max" not in extra:
                profile_args.extend(["--spec-draft-n-max", "3"])
        return [
            str(binary), "-m", str(model), "--host", "127.0.0.1", "--port", str(spec.port),
            "-c", str(spec.context), "-ngl", str(spec.gpu_layers), "--reasoning", "off",
            "--no-webui",
            *fast_attention(extra, "auto" if mtp else "on"),
            *profile_args,
            *extra,
            # Second slot for the barge-in arbiter, sharing one KV cache (no context split).
            *(["--parallel", "2", "--kv-unified"] if self.serves_arbiter
              else ["--parallel", "1"]),
        ]  # fmt: skip

    async def healthy(self, http: aiohttp.ClientSession) -> bool:
        try:
            async with http.get(self.url + "/health", timeout=aiohttp.ClientTimeout(total=2)) as r:
                return r.status == 200
        except (aiohttp.ClientError, TimeoutError, OSError):
            return False

    async def start(self, http: aiohttp.ClientSession, *, reuse: bool = True) -> None:
        if not getattr(self.spec, "managed", True):
            return
        if await self.healthy(http):
            if not reuse:
                raise EngineError(f"Port {self.spec.port} déjà occupé")
            log.info("%s: already running on %s, reusing it", self.name, self.url)
            return
        log.info("%s: starting %s", self.name, self.config.path(self.spec.model).name)
        self._spawn()
        deadline = time.monotonic() + self.timeout
        try:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise EngineError(
                        f"llama-server exited with code {self.process.returncode}; "
                        "see .runtime/logs/brain-*.log"
                    )
                if await self.healthy(http):
                    log.info("%s: ready on %s", self.name, self.url)
                    self._log_offload()
                    return
                await asyncio.sleep(0.5)
            raise EngineError(f"{self.name} was not ready after {self.timeout:.0f} s")
        except BaseException:
            self.stop()
            raise

    def _log_offload(self) -> None:
        """Report what this llama.cpp process actually loaded on the accelerator."""
        if not self._log_file:
            return
        try:
            with open(self._log_file.name, encoding="utf-8", errors="replace") as startup_log:
                startup = startup_log.read(256 * 1024)
        except OSError:
            return
        layers = re.search(r"offloaded (\d+/\d+) layers to GPU", startup)
        if layers:
            devices = re.findall(r"using device ([^\r\n]+?) - [\d.]+ MiB free", startup)
            log.info("%s · %s couches sur GPU (%s)", self.name, layers[1],
                     ", ".join(devices) or "llama.cpp")

    def _spawn(self) -> None:
        logs = self.config.path(".runtime/logs")
        logs.mkdir(parents=True, exist_ok=True)
        self._log_file = open(logs / f"brain-{self.name}.log", "wb")  # noqa: SIM115 - closed in stop()
        self.process = subprocess.Popen(
            self.command(),
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=str(self.config.root),
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            preexec_fn=children.preexec(),  # noqa: PLW1509 - only sets the parent-death signal
        )
        children.bind(self.process)

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.process.wait(timeout=5)
            if self.process.poll() is None:
                self.process.kill()
        self.process = None
        if self._log_file:
            self._log_file.close()
            self._log_file = None
