"""Installer helpers: verified, resumable downloads and llama.cpp archives."""

import argparse
import asyncio
import hashlib
import io
import tarfile
import threading
import tomllib
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from localive import install
from localive.config import DEFAULT_ROUTER, VULKAN_RUNTIME, load_config


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_download_checks_size_and_hash_and_resumes(tmp_path):
    data = b"llama" * 1000
    source = tmp_path / "source.bin"
    source.write_bytes(data)
    target = tmp_path / "out" / "model.bin"
    install.download(source.as_uri(), target, len(data), digest(data))
    assert target.read_bytes() == data
    install.download("file:///nowhere", target, len(data), digest(data))  # already verified
    target.unlink()
    with pytest.raises(install.InstallError, match="SHA-256"):
        install.download(source.as_uri(), target, len(data), digest(b"other"))
    assert not target.exists() and not target.with_name("model.bin.part").exists()


def test_short_download_keeps_partial_file_for_next_range_request(tmp_path, monkeypatch):
    data = b"official weights"
    ranges = []

    def fetch(request, timeout):
        ranges.append(request.get_header("Range"))
        response = io.BytesIO(data[:5] if len(ranges) == 1 else data[5:])
        response.status = 206
        return response

    monkeypatch.setattr(install.urllib.request, "urlopen", fetch)
    target = tmp_path / "router.gguf"
    with pytest.raises(install.InstallError, match="incomplet"):
        install.download("https://example.test/model", target, len(data), digest(data))
    assert not target.exists() and target.with_suffix(".gguf.part").read_bytes() == data[:5]
    install.download("https://example.test/model", target, len(data), digest(data))
    assert ranges == ["bytes=0-", "bytes=5-"]
    assert target.read_bytes() == data and not target.with_suffix(".gguf.part").exists()


def test_router_worker_downloads_only_the_pinned_weights(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(install.sys, "argv", ["localive.install", "--router-weights-only"])
    fetch = Mock()
    monkeypatch.setattr(install, "download", fetch)
    monkeypatch.setattr(install, "load_config", lambda: pytest.fail("full installer started"))
    install.main()
    model = tmp_path / DEFAULT_ROUTER
    url = f"https://huggingface.co/{install.ROUTER_REPO}/resolve/{install.ROUTER_REVISION}"
    fetch.assert_called_once_with(f"{url}/{model.name}", model,
                                  install.ROUTER_SIZE, install.ROUTER_SHA256)
    assert not (tmp_path / "localive.toml").exists()


async def test_cancelling_router_transfer_stops_and_reaps_the_worker(tmp_path, monkeypatch):
    started = threading.Event()
    stopped = threading.Event()

    def communicate():
        started.set()
        assert stopped.wait(5)
        return "", None

    process = Mock()
    process.communicate.side_effect = communicate
    process.kill.side_effect = stopped.set
    spawn = Mock(return_value=process)
    monkeypatch.setattr(install.subprocess, "Popen", spawn)
    monkeypatch.setattr(install.children, "bind", Mock())
    task = asyncio.create_task(install.fetch_default_router(tmp_path))
    assert await asyncio.to_thread(started.wait, 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert spawn.call_args.args[0][-1] == "--router-weights-only"
    assert spawn.call_args.kwargs["cwd"] == tmp_path
    process.kill.assert_called_once_with()
    process.communicate.assert_called_once_with()


def test_llama_server_is_found_in_zip_and_nested_tar(tmp_path):
    zipped = tmp_path / "a.zip"
    with zipfile.ZipFile(zipped, "w") as bundle:
        bundle.writestr("llama-server.exe", "x")
        bundle.writestr("ggml.dll", "x")
    nested = tmp_path / "b.tar.gz"
    payload = tmp_path / "build" / "bin"
    payload.mkdir(parents=True)
    (payload / "llama-server").write_text("x")
    (payload / "libggml.so").write_text("x")
    with tarfile.open(nested, "w:gz") as bundle:
        bundle.add(tmp_path / "build", arcname="build")
    for archive in (zipped, nested):
        out = tmp_path / f"x-{archive.stem}"
        if archive.suffix == ".zip":
            zipfile.ZipFile(archive).extractall(out)
        else:
            tarfile.open(archive).extractall(out, filter="data")
        found = [p.name for p in out.rglob("*") if p.name.startswith("llama-server")]
        assert found


def test_every_llama_asset_is_pinned():
    for (system, backend), (asset, size, sha) in install.LLAMA.items():
        assert install.LLAMA_TAG in asset and size > 1_000_000 and len(sha) == 64
        assert system in ("win32", "linux") and backend in ("vulkan", "cuda", "cpu")


def test_windows_cpu_install_and_auto_rerun_reuse_the_pinned_runtime(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(install.sys, "platform", "win32")
    config = load_config()
    archive = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("llama-server.exe", "test")
        bundle.writestr("ggml-cpu.dll", "test")
    asset = "llama-b11960-bin-win-cpu-x64.zip"
    fetched = []

    def fetch(url, path, size, digest):
        assert url.endswith("/b11960/" + asset) and path.name == asset
        assert size == 24_407_745
        assert digest == "73f962f82850d244db47eaacb9d913637027b82399124324298576d58c5f4287"
        fetched.append(asset)
        return archive

    monkeypatch.setattr(install, "download", fetch)
    folder = install.install_llama(config, "cpu")
    assert folder == ".runtime/llama/b11960-cpu"
    assert (config.path(folder) / "ggml-cpu.dll").is_file()
    config.file.write_text(install.to_toml({"brain": {"llama_dir": folder}}), encoding="utf-8")
    assert install.install_llama(load_config(), "auto") == folder
    assert fetched == [asset]


def test_first_config_is_written_once(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_config()
    written = install.first_config(config, "cuda", ".runtime/llama/b11960-cuda")
    values = tomllib.loads((tmp_path / "localive.toml").read_text(encoding="utf-8"))
    assert values["asr"]["device"] == "cuda"
    assert values["brain"]["llama_dir"] == ".runtime/llama/b11960-cuda"
    assert written.brain.llama_dir == ".runtime/llama/b11960-cuda"
    (tmp_path / "localive.toml").write_text("[voice]\nfemale = \"Hélène\"\n", encoding="utf-8")
    kept = install.first_config(load_config(), "cpu", ".runtime/llama/b11960-vulkan")
    assert kept.voice.female == "Hélène"  # an existing file is never overwritten


@pytest.mark.parametrize("options,existing,hangup,expected,routing_enabled", [
    ([], None, False, True, True),
    (["--tool-router"], None, False, True, True),
    (["--no-tool-router"], None, False, False, False),
    ([], False, False, False, False),
    (["--tool-router"], False, False, True, True),
    (["--no-tool-router"], True, False, False, True),
    ([], False, True, True, False),
    (["--no-tool-router"], False, True, False, False),
])
def test_router_download_matches_the_runtime_checkpoint(
    tmp_path, monkeypatch, options, existing, hangup, expected, routing_enabled,
):
    monkeypatch.chdir(tmp_path)
    config = load_config()
    if existing is not None:
        values = {"routing": {"enabled": existing}, "voice": {"default": "male"}}
        if hangup:
            values["turn"] = {"end_on_goodbye": True, "goodbye_mode": "d3"}
        config.file.write_text(install.to_toml(values), encoding="utf-8")
    model = config.path(config.routing.model)
    model.parent.mkdir(parents=True)
    model.touch()  # an opt-out must work even with cached weights
    brain = config.path(config.brain.model)
    brain.parent.mkdir(parents=True, exist_ok=True)
    brain.touch()
    monkeypatch.setattr(install.sys, "argv", ["localive.install", "--device", "cuda", *options])
    monkeypatch.setattr(install, "install_llama", lambda *_args: VULKAN_RUNTIME)
    monkeypatch.setattr(install, "prepare_speech", AsyncMock())
    monkeypatch.setattr(install, "check_docker", lambda: "ok")
    monkeypatch.setattr(install, "hub_file", lambda *_args: None)
    requested = []
    monkeypatch.setattr(install, "download", lambda *args: requested.append(args))
    install.main()
    repo = "https://huggingface.co/ggml-org/vllm-d3-mini-GGUF"
    assert requested == ([
        (f"{repo}/resolve/{install.ROUTER_REVISION}/{model.name}",
         model, 5_157_835_264, "a4639297e407dd58d28dd2d23d112f853d0a78d13cbe86cdb644d23c8a84e62f"),
    ] if expected else [])
    saved = load_config()
    assert saved.routing_installed == routing_enabled
    if existing is not None:
        assert saved.voice.default == "male"


def test_cpu_only_install_keeps_router_on_cpu(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cpu = f".runtime/llama/{install.LLAMA_TAG}-cpu"
    config = install.first_config(load_config(), "cuda", cpu)
    assert config.routing.llama_dir == cpu and config.routing.gpu_layers == 0
    assert config.routing.model == DEFAULT_ROUTER


def test_existing_cpu_brain_gets_independent_default_vulkan_router(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cpu = f".runtime/llama/{install.LLAMA_TAG}-cpu"
    values = {"brain": {"llama_dir": cpu}}
    (tmp_path / "localive.toml").write_text(install.to_toml(values), encoding="utf-8")
    config = load_config()
    fetched = []
    monkeypatch.setattr(install, "install_llama", lambda _cfg, backend: fetched.append(backend))
    monkeypatch.setattr(install, "download", lambda *_args: None)
    install.install_router(config, cpu)
    assert fetched == ["vulkan"]
    assert load_config().brain.llama_dir == cpu


def test_reinstall_keeps_custom_router_weights_and_runtime(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_config()
    config.routing.model = "custom/model.gguf"
    config.routing.llama_dir = "custom/llama"
    model = config.path(config.routing.model)
    model.parent.mkdir(parents=True)
    model.write_bytes(b"custom")
    monkeypatch.setattr(install, "install_llama", lambda *_: pytest.fail("custom runtime replaced"))
    monkeypatch.setattr(install, "download", lambda *_: pytest.fail("custom weights replaced"))
    install.install_router(config, VULKAN_RUNTIME)
    assert model.read_bytes() == b"custom"


def test_powershell_scripts_keep_their_bom():
    """Windows PowerShell 5 reads a .ps1 without BOM as ANSI: every accent breaks."""
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    for script in scripts.glob("*.ps1"):
        assert script.read_bytes().startswith(b"\xef\xbb\xbf"), script.name


def test_extra_brains_are_chosen_by_name():
    assert install.extra_brains("gemma-e4b, Qwen3.5-4B") == ["gemma-e4b", "qwen3.5-4b"]
    assert install.extra_brains("all") == list(install.EXTRA_BRAINS)
    with pytest.raises(argparse.ArgumentTypeError):
        install.extra_brains("llama-70b")
    doc = Path("docs/installation.md").read_text(encoding="utf-8")
    assert all(f"`{name}`" in doc for name in install.EXTRA_BRAINS)
