"""Settings: built-in defaults, optionally overridden by a ``localive.toml`` file.

The pipeline is fixed: Parakeet Ultra (ears) -> Qwen through llama-server (brain) -> Phonon
(voice). Every value here is a tuning of that pipeline.
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8787
    api_key: str = ""  # when set, clients must send "Authorization: Bearer <api_key>"


@dataclass
class VoiceConfig:
    language: str = "fr"
    default: str = "female"  # when the client does not ask for a voice: female | male
    female: str = "Ninon"  # gradium/phonon-voices catalog name or id, or a .safetensors file
    male: str = "Lucien"
    wait_notice: bool = False
    # OpenAI Live voice names (sent by Cornelius) spoken with the male voice; others: female.
    male_names: list[str] = field(
        default_factory=lambda: ["ash", "ballad", "cedar", "echo", "verse", "onyx", "fable"]
    )
    model: str = "hf://gradium/phonon-40m-fr"  # Phonon checkpoint (gated: accept its terms)
    threads: int = 4  # Phonon CPU threads; more is slower (measured on an i9-13900K)


@dataclass
class AsrConfig:
    model: str = "moondream/parakeet-ultra"
    device: str = "auto"  # auto | cuda | cpu
    device_vocabulary: bool = False


@dataclass
class VadConfig:
    threshold: float = 0.5
    silence_ms: int = 600
    min_speech_ms: int = 160
    preroll_ms: int = 300
    max_utterance_s: float = 25.0
    barge_in_ms: int = 350  # speech needed before the assistant may be interrupted
    duck_early_level: float = 0.4  # voice volume as soon as a voice is heard over it
    duck_level: float = 0.12  # volume once it is a person (not its own echo), while listening


@dataclass
class EchoConfig:
    """Server-side echo cancellation (WebRTC AEC3) using the voice Localive sends as reference.

    Off by default: on simulated echo it removes 21-33 dB and stops false voice detections,
    but in full sessions it made double-talk decisions slightly worse than lowering the voice
    plus the text echo filter (bench/echo, bench/decision/live_scenarios.py --echo). Try it
    in your room: Réglages -> Écho."""

    enabled: bool = False
    # False: the cleaned signal only drives detection (Silero, Smart Turn, barge-in) and
    # Parakeet hears the untouched mic. True: Parakeet hears the cleaned signal too (measured
    # worse: AEC3 damages the user's voice when both speak at once).
    clean_transcription: bool = False
    noise_suppression: bool = False  # may hurt transcription: measure before enabling
    high_pass_filter: bool = True
    auto_gain_control: bool = False
    delay_hint_ms: int = 100  # initial guess; AEC3 estimates the real delay by itself


GOODBYE_MODES = ("phrases", "d3")


@dataclass
class TurnConfig:
    """Semantic end of turn (Pipecat Smart Turn v3.2) on top of the silence detector."""

    enabled: bool = True
    end_on_goodbye: bool = False
    goodbye_mode: str = "phrases"
    wait_for_reply: bool = False  # ignore the microphone from the request to the reply's voice
    model: str = "models/turn/smart-turn-v3.2-gpu.onnx"  # fp32 file; runs on the CPU (~55 ms)
    threads: int = 2  # FP32: ~32 ms on Ryzen AI 9 HX 370, without quantization quality loss
    check_ms: int = 200  # first pause long enough to ask "has the user finished?"
    threshold: float = 0.5
    max_silence_ms: int = 3000  # unfinished ("mets un minuteur de…"): wait this long (Pipecat: 3s)
    over_silence_ms: int = 700  # speaking over the assistant: end of sentence after this pause

    @property
    def uses_d3_goodbye(self) -> bool:
        return self.end_on_goodbye and self.goodbye_mode == "d3"


DEFAULT_BRAIN = "models/Qwen3-4B-Instruct-2507/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
DEFAULT_ROUTER = "models/decision/vllm-d3-mini-Q8_0.gguf"
LLAMA_TAG = "b11960"
VULKAN_RUNTIME = f".runtime/llama/{LLAMA_TAG}-vulkan"


@dataclass
class RoutingConfig:
    enabled: bool = True
    model: str = DEFAULT_ROUTER
    llama_dir: str = VULKAN_RUNTIME  # an empty override inherits brain.llama_dir
    gpu_layers: int = 99  # offload the full router even when the main brain runs on CPU
    port: int = 8085
    timeout_ms: int = 0  # automatic deadline based on observed local router latency


@dataclass
class DecisionConfig:
    """Barge-in arbiter: Qwen classifies speech heard while it thinks or talks."""

    enabled: bool = True
    timeout_ms: int = 600  # past this, fall back to the simple rules
    # Own llama-server (small context): on the brain's server the arbiter
    # waits whenever a reply prompt is being read (0.3-5 s measured). Default Qwen: ~3 GB VRAM.
    dedicated: bool = True
    # The measured arbiter, whatever brain is chosen (a 9B brain must not load a 9B arbiter
    # too). Empty: the brain's model. Missing default file: falls back to the brain's model.
    model: str = DEFAULT_BRAIN
    llama_dir: str = ""  # empty: inherit brain.llama_dir; allows an independent GPU backend
    port: int = 8084
    context: int = 8192
    assistant_name: str = "Cornelius"  # "Julie, …" is for someone else, "Cornelius, …" is not


@dataclass
class LlmConfig:
    url: str = "http://127.0.0.1:8081/v1"  # any chat-completions server; managed Qwen by default
    model: str = "qwen"  # id served on /v1/models; Cornelius sends it as delegation.responses.model
    api_key: str = ""
    temperature: float = 0.2
    max_tokens: int = 400
    history_turns: int = 12
    tool_timeout_s: float = 60.0
    # Set by finalize() for an Ollama / LM Studio brain: the model id that server expects
    # (Cornelius still sees ``model``).
    upstream_model: str = ""
    # Set by finalize() for Ollama: ask the server not to think first. A thinking
    # model (Qwen3.5, bonsai...) otherwise spends max_tokens on its reasoning and says nothing.
    no_thinking: bool = False
    # Set by finalize() for Ollama: its own API, with this context (its OpenAI endpoint keeps
    # the server's default, 4096, and silently drops the start of Cornelius' instructions).
    ollama_context: int = 0


@dataclass
class SearchConfig:
    engine: str = "searxng"  # searxng | off ; used when the client declares web_search
    url: str = "http://127.0.0.1:8888"
    managed: bool = True  # Localive runs SearXNG in Docker (local URL only)
    image: str = "searxng/searxng:latest"
    container: str = "localive-searxng"
    start_docker: bool = True  # Windows: start Docker Desktop if installed but stopped
    language: str = "fr-FR"
    max_results: int = 5
    timeout_s: float = 6.0


@dataclass
class HuggingFaceConfig:
    # For gated downloads (Phonon 40m). Prefer the HF_TOKEN environment variable or a Docker
    # secret (HF_TOKEN_FILE); the saved `hf auth login` token is used when both are empty.
    token: str = ""
    cache: str = "models/hf"
    offline: bool = False  # never contact the Hub: everything must already be in the cache


# Where the brain runs: Localive's own llama.cpp, or a model server already on this machine
# (or the network). Default addresses of their OpenAI-compatible endpoints:
BRAIN_SERVERS = {
    "ollama": "http://127.0.0.1:11434/v1",
    "lmstudio": "http://127.0.0.1:1234/v1",
    "openai": "",  # any other chat-completions server: its address must be given
}


@dataclass
class BrainConfig:
    backend: str = "localive"  # localive | ollama | lmstudio | openai
    external_url: str = ""  # empty: the backend's usual local address
    external_model: str = ""  # model id on that server (its /v1/models list)
    ollama_context: int = 16384  # Ollama only: context asked for (0: the server's default)
    managed: bool = True  # start/stop llama-server automatically
    llama_dir: str = VULKAN_RUNTIME
    model: str = DEFAULT_BRAIN
    port: int = 8081
    context: int = 24576  # Cornelius sends long instructions and ~20 tool schemas
    gpu_layers: int = 99
    startup_timeout: float = 180.0
    extra_args: str = ""  # additional llama-server options for the main brain only


@dataclass
class AltBrainConfig:
    """An extra llama-server next to the brain: the dedicated arbiter, and the benches."""

    enabled: bool = False
    name: str = "qwen3.5"
    model: str = "models/Qwen3.5-4B/Qwen3.5-4B-Q4_K_M.gguf"
    llama_dir: str = ""  # empty: inherit brain.llama_dir; override for a different GPU backend
    port: int = 8082
    context: int = 24576
    gpu_layers: int = 99
    # Gemma uses sliding-window attention: give it "--swa-full" (without a full-size SWA cache
    # llama.cpp sometimes re-evaluates the prompt slowly: 4-6 s stalls measured). MoE models
    # that do not fit in VRAM: "--n-cpu-moe N" keeps the experts of N layers in RAM.
    extra_args: str = ""


@dataclass
class Config:
    root: Path = field(default_factory=Path.cwd)
    file: Path | None = None  # the TOML file settings are saved to
    file_values: dict[str, Any] = field(default_factory=dict)  # what that file sets
    server: ServerConfig = field(default_factory=ServerConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    echo: EchoConfig = field(default_factory=EchoConfig)
    turn: TurnConfig = field(default_factory=TurnConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)

    def path(self, value: str) -> Path:
        """Resolve a configured path relative to the project root."""
        candidate = Path(value).expanduser()
        return candidate if candidate.is_absolute() else self.root / candidate

    def brains(self) -> dict[str, str]:
        """Brain name -> chat-completions URL. One brain: its model is chosen in the admin."""
        return {self.llm.model: self.llm.url}

    def context_for(self, name: str) -> int:
        """Context window (tokens) of the server answering ``name``."""
        if self.brain.backend == "ollama" and self.brain.ollama_context:
            return self.brain.ollama_context
        return self.brain.context

    def gender(self, requested: str | None) -> str:
        """OpenAI voice name (or female/male) -> the local voice gender."""
        if requested in {"female", "male"}:
            return requested
        if not requested:
            return self.voice.default
        return "male" if requested.lower() in self.voice.male_names else "female"

    @property
    def decision_model_installed(self) -> bool:
        return self.path(self.routing.model).is_file()

    @property
    def routing_installed(self) -> bool:
        return self.routing.enabled and self.decision_model_installed

    @property
    def local_arbiter(self) -> bool:
        """Whether the decision model runs on its own llama-server started by Localive
        (also with an Ollama / LM Studio brain: they offer no grammar nor logprobs)."""
        return (self.decision.enabled and self.decision.dedicated
                and (self.brain.managed or self.brain.backend != "localive"))  # fmt: skip

    def brain_identity(self) -> str:
        """What determines the brain's prompt format (a saved prompt is only valid for it)."""
        if self.brain.backend == "localive":
            return self.brain.model
        return f"{self.brain.backend}:{self.llm.url}:{self.llm.upstream_model}"

    def finalize(self) -> Config:
        if not -1 <= self.routing.gpu_layers <= 999:
            raise ValueError("routing.gpu_layers doit être compris entre -1 et 999.")
        if not 1 <= self.routing.port <= 65535 or not 0 <= self.routing.timeout_ms <= 10000:
            raise ValueError("Le port et le délai du décideur d’outils doivent être valides.")
        if self.voice.default not in {"female", "male"}:
            raise ValueError("voice.default must be 'female' or 'male'")
        if self.asr.device not in {"auto", "cuda", "cpu"}:
            raise ValueError("asr.device must be auto, cuda or cpu")
        if self.turn.threads < 1:
            raise ValueError("turn.threads must be positive")
        if self.turn.goodbye_mode not in GOODBYE_MODES:
            raise ValueError("turn.goodbye_mode doit valoir phrases ou d3.")
        if self.search.engine not in {"searxng", "off"}:
            raise ValueError("search.engine must be 'searxng' or 'off'")
        if self.brain.backend not in {"localive", *BRAIN_SERVERS}:
            raise ValueError("brain.backend must be localive, ollama, lmstudio or openai")
        if self.brain.backend != "localive":
            self.brain.managed = False
            url = self.brain.external_url or BRAIN_SERVERS[self.brain.backend]
            self.llm.url = url.rstrip("/")
            if not self.llm.url:
                raise ValueError("Cerveau : donnez l'adresse du serveur externe (page Cerveau)")
            self.llm.upstream_model = self.brain.external_model
            self.llm.no_thinking = self.brain.backend == "ollama"
            if self.brain.backend == "ollama":
                self.llm.ollama_context = self.brain.ollama_context
            if not self.brain.external_model:
                log.warning("Cerveau externe (%s) sans modèle choisi : le serveur utilisera son "
                            "modèle par défaut (page Cerveau).", self.brain.backend)  # fmt: skip
        if self.brain.managed and not self.path(self.brain.model).is_file():
            if self.brain.model != DEFAULT_BRAIN and self.path(DEFAULT_BRAIN).is_file():
                # A model chosen on another machine, or deleted: start with the installed one,
                # so the admin page is reachable to pick another.
                log.warning("Modèle de raisonnement introuvable (%s) : Localive utilise %s. "
                            "Choisissez-en un autre dans l'administration (page Cerveau).",
                            self.brain.model, DEFAULT_BRAIN)  # fmt: skip
                self.brain.model = DEFAULT_BRAIN
            else:
                raise FileNotFoundError(
                    f"Modèle de raisonnement introuvable : {self.path(self.brain.model)} "
                    "(lancez scripts/setup.ps1)")
        if (self.local_arbiter
                and self.decision.model and not self.path(self.decision.model).is_file()):
            if self.decision.model != DEFAULT_BRAIN:  # chosen explicitly: say it is missing
                raise FileNotFoundError(
                    f"Arbiter model not found: {self.path(self.decision.model)}")
            log.info("Décideur : %s absent, il prend le modèle du cerveau", DEFAULT_BRAIN)
            self.decision.model = ""
        return self


INTERNAL = {"root", "file", "file_values"}


def merge(target: Any, values: dict[str, Any], prefix: str = "") -> None:
    """Validate ``values`` (a TOML-like dict) and set them on the dataclass ``target``."""
    known = {f.name: f for f in fields(target)}
    for key, value in values.items():
        if key not in known or key in INTERNAL:
            raise ValueError(f"Unknown setting: {prefix}{key}")
        current = getattr(target, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"[{prefix}{key}] must be a table")
            merge(current, value, f"{prefix}{key}.")
        else:
            expected = type(current)
            if expected is float and isinstance(value, int) and not isinstance(value, bool):
                value = float(value)
            if not isinstance(value, expected) or (expected is int and isinstance(value, bool)):
                raise ValueError(f"{prefix}{key} must be of type {expected.__name__}")
            setattr(target, key, value)


# Ignore retired model sections in configurations saved by older versions.
LEGACY_SECTIONS = ("alt_brain", "alt_brain_2", "expert")


def load_config(path: str | Path | None = None) -> Config:
    """Load ``path``, or ``./localive.toml`` when it exists, over the defaults."""
    file = Path(path) if path else Path.cwd() / "localive.toml"
    config = Config(root=file.parent.resolve() if path else Path.cwd(), file=file)
    if file.is_file():
        # utf-8-sig: Notepad and Windows PowerShell 5 save UTF-8 with a BOM, which TOML rejects.
        config.file_values = tomllib.loads(file.read_text(encoding="utf-8-sig"))
        for legacy in LEGACY_SECTIONS:  # removed settings: ignored, dropped at the next save
            if config.file_values.pop(legacy, None) is not None:
                log.info("[%s] ignoré : réglage retiré de Localive", legacy)
        merge(config, config.file_values)
        log.info("Configuration: %s", file)
    elif path:
        raise FileNotFoundError(f"Configuration file not found: {file}")
    return config
