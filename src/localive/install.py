"""Installer shared by scripts/setup.ps1 (Windows) and scripts/install-linux.sh (Linux).

    python -m localive.install [--device auto|cpu|cuda] [--llama auto|vulkan|cuda|cpu]
                               [--no-tool-router] [--skip-voice]
                               [--models qwen3.5-4b,gemma-e4b|all]

Every download is checked against a published size and SHA-256 and resumes after an
interruption; what is already verified is kept, so running it again is safe.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import http.client
import json
import logging
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from . import children
from .catalog import BY_ID, CATALOG, DEFAULT_ID
from .config import DEFAULT_ROUTER, LLAMA_TAG, Config, load_config
from .settings import apply, to_toml

_RELEASE = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/"
# (platform, backend) -> (asset, size, sha256), from the GitHub release of LLAMA_TAG.
LLAMA = {
    ("win32", "cpu"): ("llama-b11960-bin-win-cpu-x64.zip", 24407745,
                       "73f962f82850d244db47eaacb9d913637027b82399124324298576d58c5f4287"),
    ("win32", "vulkan"): ("llama-b11960-bin-win-vulkan-x64.zip", 37345851,
                          "bac6914b5e83523eb008c6789eaa12669f93fbc196c3b9e8237b6672323b33e9"),
    ("linux", "vulkan"): ("llama-b11960-bin-ubuntu-vulkan-x64.tar.gz", 35474467,
                          "164199e978304bb32a8a6c24b8efa015528ddf286821596eab676c90b375feac"),
    ("linux", "cuda"): ("llama-b11960-bin-ubuntu-cuda-12.8-x64.tar.gz", 182524002,
                        "64d57beb9abb637c177c6afb313760a0c0782224cac71e74869eed5b8f9a1ea4"),
    ("linux", "cpu"): ("llama-b11960-bin-ubuntu-x64.tar.gz", 22686210,
                       "31cc3a6bc638ac6c48dc4b1b74888f1eaadf3d14227c1e5f67daf9a49541d97a"),
}  # fmt: skip
_DEFAULT = BY_ID[DEFAULT_ID]
BRAIN = (_DEFAULT.repo, _DEFAULT.file, _DEFAULT.folder)
# Other reasoning models, offered in the admin's dropdown once downloaded (--models).
EXTRA_BRAINS = {m.id: (m.repo, m.file, m.folder) for m in CATALOG if m.id != DEFAULT_ID}
SMART_TURN = ("pipecat-ai/smart-turn-v3", "smart-turn-v3.2-gpu.onnx", "models/turn")
ROUTER_REVISION = "8b29fd3b87373ea85fa59cc2f26251c1591d771a"
ROUTER_REPO = "ggml-org/vllm-d3-mini-GGUF"
ROUTER_SIZE = 5_157_835_264
ROUTER_SHA256 = "a4639297e407dd58d28dd2d23d112f853d0a78d13cbe86cdb644d23c8a84e62f"
USER_AGENT = "localive-installer"


class InstallError(RuntimeError):
    pass


def say(text: str) -> None:
    print(text, flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verified(path: Path, size: int, digest: str) -> bool:
    return path.is_file() and path.stat().st_size == size and sha256(path) == digest


def download(url: str, path: Path, size: int, digest: str, progress=None) -> Path:
    """Download ``url`` to ``path``, resuming a ``.part`` file, then check size and SHA-256.

    ``progress(done, size)`` is called as bytes arrive (the admin page's progress bar)."""
    if verified(path, size, digest):
        say(f"  ok  {path.name}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    done = part.stat().st_size if part.is_file() else 0
    if done > size:
        part.unlink()
        done = 0
    say(f"  ->  {path.name} ({size / 1e6:,.0f} Mo)".replace(",", " "))
    if done < size:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                       "Range": f"bytes={done}-"})  # fmt: skip
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                mode = "ab" if response.status == 206 else "wb"
                if mode == "wb":
                    done = 0
                with part.open(mode) as out:
                    while block := response.read(1 << 20):
                        out.write(block)
                        done += len(block)
                        if progress:
                            progress(done, size)
        except (OSError, http.client.HTTPException) as exc:
            raise InstallError(f"Téléchargement interrompu pour {path.name} ({exc}). "
                               "Le prochain essai reprendra le transfert.") from exc
    received = part.stat().st_size
    if received < size:
        raise InstallError(f"Téléchargement incomplet de {path.name} ({received}/{size} octets). "
                           "Le prochain essai reprendra le transfert.")
    if received != size or sha256(part) != digest:
        part.unlink()
        raise InstallError(f"Fichier corrompu (SHA-256 invalide) : {path.name}")
    part.replace(path)
    return path


def hub_file(repo: str, filename: str, folder: Path, progress=None) -> Path:
    """A Hugging Face file, checked against the size and SHA-256 published by the Hub."""
    api = f"https://huggingface.co/api/models/{repo}/tree/main"
    request = urllib.request.Request(api, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            tree = json.load(response)
    except OSError as exc:
        raise InstallError(f"Hugging Face injoignable ({repo}) : {exc}") from exc
    entry = next((e for e in tree if e.get("path") == filename), None)
    lfs = (entry or {}).get("lfs") or {}
    if not lfs.get("oid"):
        raise InstallError(f"Métadonnées Hugging Face introuvables : {repo}/{filename}")
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    return download(url, folder / filename, int(lfs["size"]), lfs["oid"], progress)


def nvidia_gpu() -> bool:
    if not shutil.which("nvidia-smi"):
        return False
    result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                            capture_output=True, text=True, check=False)  # fmt: skip
    return result.returncode == 0 and bool(result.stdout.strip())


def _llama_server(folder: Path) -> Path | None:
    name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    return next((p for p in folder.rglob(name) if p.is_file()), None)


def install_llama(config: Config, backend: str, device: str = "cpu") -> str:
    """llama.cpp for this platform; returns the llama_dir to use (relative to the project)."""
    system = "win32" if sys.platform == "win32" else "linux"
    if backend == "auto":
        current = config.path(config.brain.llama_dir)
        binary = current / ("llama-server.exe" if system == "win32" else "llama-server")
        if config.file.is_file() and binary.is_file():
            say(f"  ok  llama.cpp ({current})")
            return config.brain.llama_dir
        backend = "cuda" if system == "linux" and device == "cuda" and not shutil.which(
            "vulkaninfo") else "vulkan"
        if config.file_values.get("brain", {}).get("llama_dir"):
            variant = current.name.removeprefix(f"{LLAMA_TAG}-")
            if (system, variant) in LLAMA and current == config.path(
                f".runtime/llama/{LLAMA_TAG}-{variant}"
            ):
                backend = variant
    key = (system, backend)
    if key not in LLAMA:
        raise InstallError(f"llama.cpp {backend} n'est pas proposé ici ({sys.platform}).")
    if key[0] == "linux" and platform.machine() not in ("x86_64", "AMD64"):
        raise InstallError("Linux x86-64 seulement pour l'instant.")
    asset, size, digest = LLAMA[key]
    target = config.path(f".runtime/llama/{LLAMA_TAG}-{backend}")
    if _llama_server(target):
        say(f"  ok  llama.cpp {LLAMA_TAG} ({backend})")
        return str(target.relative_to(config.root)).replace("\\", "/")
    archive = download(_RELEASE + asset, config.path(f".runtime/downloads/{asset}"), size, digest)
    staging = target.with_name(target.name + ".tmp")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    if asset.endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(staging)
    else:
        with tarfile.open(archive) as bundle:
            bundle.extractall(staging, filter="data")
    server = _llama_server(staging)
    if server is None:
        raise InstallError(f"llama-server absent de {asset}")
    shutil.rmtree(target, ignore_errors=True)
    shutil.move(str(server.parent), str(target))  # binaries and their shared libraries
    shutil.rmtree(staging, ignore_errors=True)
    if sys.platform != "win32":
        for binary in target.iterdir():
            if binary.is_file() and not binary.suffix:
                binary.chmod(0o755)
    return str(target.relative_to(config.root)).replace("\\", "/")


def download_default_router(path: Path) -> None:
    """The installer and the running server share one resumable, verified file."""
    from filelock import FileLock

    path.parent.mkdir(parents=True, exist_ok=True)
    filename = Path(DEFAULT_ROUTER).name
    url = f"https://huggingface.co/{ROUTER_REPO}/resolve/{ROUTER_REVISION}/{filename}"
    with FileLock(str(path) + ".lock"):
        download(url, path, ROUTER_SIZE, ROUTER_SHA256)


async def fetch_default_router(root: Path) -> None:
    """Stop even a stalled network read on server shutdown, retaining the .part for the next run."""
    process = subprocess.Popen(  # noqa: ASYNC220 - bind the child to the server before yielding
        [sys.executable, "-X", "utf8", "-m", "localive.install", "--router-weights-only"],
        cwd=root, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        encoding="utf-8", errors="replace", preexec_fn=children.preexec(),
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    children.bind(process)
    result = asyncio.create_task(asyncio.to_thread(process.communicate))
    try:
        output, _ = await asyncio.shield(result)
    except asyncio.CancelledError:
        process.kill()
        await result
        raise
    if process.returncode:
        details = output.strip().splitlines()
        raise InstallError(details[-1] if details else f"Transfert arrêté ({process.returncode}).")


def install_router(config: Config, installed_llama_dir: str) -> None:
    """Prepare the configured decision engine independently of the main brain."""
    runtime = config.path(config.routing.llama_dir or config.brain.llama_dir).resolve()
    if runtime != config.path(installed_llama_dir).resolve():
        for backend in ("vulkan", "cuda", "cpu"):
            if runtime == config.path(f".runtime/llama/{LLAMA_TAG}-{backend}").resolve():
                install_llama(config, backend)
                break
        else:
            binary = "llama-server.exe" if sys.platform == "win32" else "llama-server"
            if not (runtime / binary).is_file():
                say(f"  Moteur personnalisé du décideur à fournir : {runtime / binary}")

    model = config.path(config.routing.model)
    if model == config.path(DEFAULT_ROUTER):
        say("d3-mini Q8_0 (sélection des outils)")
        download_default_router(model)
    elif model == config.path("models/decision/d1-Omni-2B-Q8_0.gguf"):
        hub_file("LiquidAI/d1-Omni-2B-GGUF", model.name, model.parent)
    elif not model.is_file():
        say(f"  Modèle personnalisé du décideur à fournir : {model}")


def first_config(config: Config, device: str, llama_dir: str, *,
                 tool_router: bool | None = None) -> Config:
    """localive.toml for this machine, written once (later edited from the admin page)."""
    if config.file.is_file():
        return config
    values: dict = {"asr": {"model": "moondream/parakeet-ultra", "device": device}}
    if device == "cpu":
        try:
            from kestrel_kernels.ternary import ternary_gemm_ready

            if ternary_gemm_ready():
                values["asr"]["model"] = "moondream/parakeet-redux"
        except ImportError:
            pass
    if llama_dir != config.brain.llama_dir:
        values["brain"] = {"llama_dir": llama_dir}
    if llama_dir == f".runtime/llama/{LLAMA_TAG}-cpu":
        values["routing"] = {"llama_dir": llama_dir, "gpu_layers": 0}
    if tool_router is False:
        values.setdefault("routing", {})["enabled"] = False
    config.file.write_text(to_toml(values), encoding="utf-8")
    return load_config(config.file)


async def prepare_speech(config: Config, skip_voice: bool) -> None:
    """Parakeet (and Phonon) are fetched by loading them once."""
    from . import hf
    from .asr import Transcriber
    from .tts import Synthesizer

    hf.configure(config)
    asr = Transcriber(config)
    try:
        await asr.start()
        if not skip_voice:
            await Synthesizer(config).start()
    finally:
        asr.close()


def check_docker() -> str:
    docker = shutil.which("docker")
    if docker is None:
        return ("Docker absent : la recherche web (SearXNG) sera indisponible. Installez Docker "
                "(Docker Desktop sous Windows), Localive s'occupera du reste.")
    result = subprocess.run([docker, "info", "--format", "{{.ServerVersion}}"],
                            capture_output=True, text=True, check=False)  # fmt: skip
    if result.returncode != 0:
        return ("Docker est arrêté. Démarrez-le pour utiliser la recherche web. "
                "Sous Windows, Localive peut démarrer Docker Desktop à la demande.")
    return f"Docker {result.stdout.strip()} : SearXNG sera préparé à la demande d’un client."


def extra_brains(value: str) -> list[str]:
    """``--models`` value: names from EXTRA_BRAINS, comma-separated, or "all"."""
    names = [n.strip().lower() for n in value.split(",") if n.strip()]
    if names == ["all"]:
        return list(EXTRA_BRAINS)
    unknown = [n for n in names if n not in EXTRA_BRAINS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"modèle inconnu : {', '.join(unknown)} (choix : {', '.join(EXTRA_BRAINS)}, all)")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(prog="localive.install", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="processeur de la transcription")
    parser.add_argument("--llama", choices=["auto", "vulkan", "cuda", "cpu"], default="auto",
                        help="moteur du cerveau (Vulkan : toutes les cartes graphiques)")
    parser.add_argument("--skip-voice", action="store_true",
                        help="ne pas préparer Phonon (accès Hugging Face pas encore réglé)")
    parser.add_argument("--tool-router", action=argparse.BooleanOptionalAction, default=None,
                        help="préparer le décideur d’outils (d3-mini par défaut). "
                             "--no-tool-router omet ses téléchargements et le désactive "
                             "sur une installation neuve")
    parser.add_argument("--models", type=extra_brains, default=[],
                        help="autres modèles de raisonnement à télécharger : "
                             + ", ".join(EXTRA_BRAINS) + " (séparés par des virgules) ou all")
    parser.add_argument("--router-weights-only", action="store_true",
                        help="télécharger uniquement les poids officiels de d3-mini")
    args = parser.parse_args()
    if args.router_weights_only:
        try:
            download_default_router(Path.cwd() / DEFAULT_ROUTER)
        except (OSError, InstallError) as exc:
            parser.exit(1, f"{exc}\n")
        return
    logging.basicConfig(level=logging.INFO, format="  %(message)s")
    for noisy in ("httpx", "httpcore", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    config = load_config()
    device = args.device if args.device != "auto" else ("cuda" if nvidia_gpu() else "cpu")
    try:
        say(f"llama.cpp {LLAMA_TAG}")
        llama_dir = install_llama(config, args.llama, device)
        config = first_config(config, device, llama_dir, tool_router=args.tool_router)
        say(f"  Transcription : {config.asr.model} / {config.asr.device}")
        say(f"{_DEFAULT.label} (cerveau)")
        repo, name, folder = BRAIN
        hub_file(repo, name, config.path(folder))
        prepare_router = config.routing.enabled or config.turn.uses_d3_goodbye
        if args.tool_router is not None:
            prepare_router = args.tool_router
        if prepare_router:
            install_router(config, llama_dir)
            if args.tool_router is True:
                saved = load_config(config.file)
                if not saved.path(saved.routing.model).is_file():
                    raise InstallError(f"Modèle du décideur à fournir : {saved.routing.model}")
                # Persist an explicit opt-in only once the requested weights are installed.
                if not saved.routing.enabled:
                    apply(saved, {"routing.enabled": True})
                    say("  Tri des outils activé pour le prochain démarrage de Localive.")
        for key in args.models:
            repo, name, folder = EXTRA_BRAINS[key]
            say(f"{name.removesuffix('.gguf')} (modèle de raisonnement au choix)")
            hub_file(repo, name, config.path(folder))
        say("Smart Turn v3.2 (fin de tour)")
        repo, name, folder = SMART_TURN
        hub_file(repo, name, config.path(folder))
        say("Parakeet (oreilles)" + ("" if args.skip_voice else " et Phonon 40m-fr (voix)"))
        asyncio.run(prepare_speech(config, args.skip_voice))
    except InstallError as exc:
        say(f"\nErreur : {exc}")
        raise SystemExit(1) from exc
    except Exception as exc:  # noqa: BLE001 - the most common one deserves an explanation
        say(f"\nErreur : {exc}")
        say("Si Phonon refuse l'accès : acceptez ses conditions sur "
            "https://huggingface.co/gradium/phonon-40m-fr, puis connectez Hugging Face "
            "(hf auth login), ou relancez avec --skip-voice.")
        raise SystemExit(1) from exc
    say("Recherche web : " + check_docker())
    if args.skip_voice:
        say("\nVoix non préparée : connectez Hugging Face puis relancez l'installation.")


if __name__ == "__main__":
    main()
