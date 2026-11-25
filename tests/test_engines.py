import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from localive import settings
from localive.catalog import QWEN_4B_MTP
from localive.config import VULKAN_RUNTIME, AltBrainConfig, Config
from localive.engines import Brain, EngineError
from localive.routing import ToolRouter
from localive.server import App

ARBITER = "models/arbiter/arbiter.gguf"


@pytest.fixture
def config(tmp_path):
    config = Config(root=tmp_path)
    config.brain.llama_dir = "runtime/default backend"
    for directory in (config.brain.llama_dir, "runtime/other backend", VULKAN_RUNTIME):
        binary = tmp_path / directory / (
            "llama-server.exe" if sys.platform == "win32" else "llama-server"
        )
        binary.parent.mkdir(parents=True)
        binary.touch()
    for model in (config.brain.model, ARBITER):
        path = config.path(model)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return config


def test_extra_server_inherits_the_backend_unless_overridden(config):
    main_command = Brain(config).command()
    inherited = Path(main_command[0])
    spec = AltBrainConfig(model=ARBITER, port=8084, context=8192)
    assert Brain(config, spec, "arbiter").command()[0] == str(inherited)
    spec.llama_dir = "runtime/other backend"
    command = Brain(config, spec, "arbiter").command()
    assert Path(command[0]) == config.path(spec.llama_dir) / inherited.name
    assert Brain(config).command() == main_command


def test_missing_override_binary_does_not_silently_use_default_backend(config):
    spec = AltBrainConfig(model=ARBITER, llama_dir="runtime/missing backend")
    with pytest.raises(EngineError, match="missing backend"):
        Brain(config, spec, "arbiter").command()


@pytest.mark.parametrize("model_override, backend_override", [
    (False, False), (True, False), (False, True), (True, True),
])
def test_dedicated_arbiter_can_use_an_independent_model_and_backend(
    config, model_override, backend_override,
):
    config.decision.model = ARBITER if model_override else ""
    if backend_override:
        config.decision.llama_dir = "runtime/other backend"
    instance = App(config.finalize())
    assert len(instance.alts) == 1  # one brain, plus its dedicated arbiter
    arbiter = instance.alts[0]
    main_command, arbiter_command = instance.brain.command(), arbiter.command()
    expected_model = ARBITER if model_override else config.brain.model
    expected_directory = config.decision.llama_dir if backend_override else config.brain.llama_dir
    assert arbiter_command[arbiter_command.index("-m") + 1] == str(config.path(expected_model))
    assert Path(arbiter_command[0]).parent == config.path(expected_directory)
    assert main_command[main_command.index("-m") + 1] == str(config.path(config.brain.model))
    assert Path(main_command[0]).parent == config.path(config.brain.llama_dir)
    assert arbiter_command[arbiter_command.index("--parallel") + 1] == "2"
    assert "--kv-unified" in arbiter_command


def test_a_bigger_brain_keeps_the_measured_arbiter(config):
    """Choosing a 9B brain must not load a second 9B model as the arbiter."""
    default_arbiter = config.decision.model
    config.brain.model = "models/big/brain-9b.gguf"
    config.path(config.brain.model).parent.mkdir(parents=True)
    config.path(config.brain.model).touch()
    config.path(default_arbiter).parent.mkdir(parents=True, exist_ok=True)
    config.path(default_arbiter).touch()
    instance = App(config.finalize())
    command = instance.alts[0].command()
    assert command[command.index("-m") + 1] == str(config.path(default_arbiter))


def test_main_extra_args_do_not_reach_the_arbiter(config):
    config.brain.extra_args = "--threads 4 --flash-attn on"
    instance = App(config)
    command = instance.brain.command()
    assert command[command.index("--threads") + 1] == "4"
    assert command[command.index("--flash-attn") + 1] == "on"
    assert [a.name for a in instance.alts] == ["arbiter"]
    assert "--threads" not in instance.alts[0].command()
    assert instance.alts[0].spec.extra_args == ""


def test_flash_attention_and_8_bit_cache_by_default(config):
    command = Brain(config).command()
    assert command[command.index("-fa") + 1] == "on"
    assert command[command.index("-ctk") + 1] == command[command.index("-ctv") + 1] == "q8_0"
    config.brain.extra_args = "--flash-attn off -ctk f16 -ctv f16"
    command = Brain(config).command()
    assert "-fa" not in command and command.count("-ctk") == 1  # the user's choice wins


def test_admin_mtp_profile_only_applies_to_its_own_brain(config):
    original = config.brain.model
    model = config.path(QWEN_4B_MTP.path)
    model.parent.mkdir(parents=True)
    model.touch()
    choices, labels = settings.choices_for(settings.BY_KEY["brain.model"], config)
    assert QWEN_4B_MTP.path in choices and labels[QWEN_4B_MTP.path] == QWEN_4B_MTP.label
    assert "--spec-type" not in Brain(config).command()
    config.brain.model = QWEN_4B_MTP.path
    command = Brain(config).command()
    for flag, value in (("-fa", "auto"), ("--spec-type", "draft-mtp"),
                        ("--spec-draft-n-max", "3"), ("-ctk", "q8_0"), ("-ctv", "q8_0")):
        assert command[command.index(flag) + 1] == value
    assert "--spec-type" not in Brain(config, AltBrainConfig(model=QWEN_4B_MTP.path)).command()
    config.brain.extra_args = "--spec-type none -fa on"
    command = Brain(config).command()
    assert "draft-mtp" not in command and command.count("-fa") == 1
    config.brain.extra_args = ""
    config.brain.model = original
    assert "--spec-type" not in Brain(config).command()


@pytest.mark.parametrize("enabled,state", [(True, "missing"), (False, "off")])
async def test_router_health_explains_why_weights_will_not_load(config, enabled, state):
    config.routing.enabled = enabled
    health = json.loads((await App(config).health(None)).body)
    assert health["tool_router_state"] == state and not health["tool_router"]


async def test_optional_tool_router_publishes_only_after_its_own_engine_starts(config, monkeypatch):
    model = config.path(config.routing.model)
    model.parent.mkdir(parents=True, exist_ok=True)
    model.touch()
    app = App(config)
    app.services = SimpleNamespace(router=None, transcriber=SimpleNamespace(label="ASR"))
    health = json.loads((await app.health(None)).body)
    assert not health["tool_router"] and health["tool_router_file"] is None
    assert health["tool_router_state"] == "idle"
    start = AsyncMock()
    monkeypatch.setattr(Brain, "start", start)
    verify = AsyncMock()
    monkeypatch.setattr(ToolRouter, "verify", verify)
    await app._start_router()
    verify.assert_awaited_once()
    start.assert_awaited_once_with(app._http, reuse=False)
    assert app.router_engine.spec.model == config.routing.model
    assert isinstance(app.services.router, ToolRouter)
    assert app.services.router.url == f"http://127.0.0.1:{config.routing.port}/v1/systemone"
    health = json.loads((await app.health(None)).body)
    assert health["tool_router"] and health["tool_router_file"] == Path(config.routing.model).name
    assert health["tool_router_state"] == "ready"


async def test_busy_router_port_leaves_voice_ready_and_existing_service_untouched(
    config, monkeypatch, caplog,
):
    app = App(config)
    app.services = SimpleNamespace(router=None)
    app.ready.set()
    monkeypatch.setattr(Brain, "healthy", AsyncMock(return_value=True))
    monkeypatch.setattr(Brain, "_spawn", lambda self: pytest.fail("occupied port reused"))
    await app._start_router()
    assert app.services.router is None and app.ready.is_set()
    assert "catalogue complet" in caplog.text and str(config.routing.port) in caplog.text
    assert app.router_engine.process is None


@pytest.mark.parametrize("directory,layers,expected", [
    ("", -1, 0), ("runtime/default backend", -1, 0), ("runtime/other backend", -1, 99),
    ("", 5, 5), ("runtime/other backend", 0, 0),
])
async def test_router_gpu_layers_reach_llama_server(
    config, monkeypatch, directory, layers, expected,
):
    config.brain.gpu_layers = 0
    config.routing.llama_dir = directory
    config.routing.gpu_layers = layers
    model = config.path(config.routing.model)
    model.parent.mkdir(parents=True, exist_ok=True)
    model.touch()
    app = App(config)
    app.services = SimpleNamespace(router=None)
    monkeypatch.setattr(Brain, "start", AsyncMock())
    monkeypatch.setattr(ToolRouter, "verify", AsyncMock())
    await app._start_router()
    command = app.router_engine.command()
    assert command[command.index("-ngl") + 1] == str(expected)
    assert "--device" not in command


@pytest.mark.parametrize("layers,device,explicit_vulkan", [
    (99, None, True), (0, None, False), (99, "Vulkan1", False),
])
async def test_default_router_offloads_to_vulkan_independently_of_cpu_brain(
    config, monkeypatch, layers, device, explicit_vulkan,
):
    if device:
        monkeypatch.setenv("LLAMA_ARG_DEVICE", device)
    else:
        monkeypatch.delenv("LLAMA_ARG_DEVICE", raising=False)
    config.brain.gpu_layers = 0
    config.routing.gpu_layers = layers
    model = config.path(config.routing.model)
    model.parent.mkdir(parents=True)
    model.touch()
    app = App(config)
    app.services = SimpleNamespace(router=None)
    monkeypatch.setattr(Brain, "start", AsyncMock())
    monkeypatch.setattr(ToolRouter, "verify", AsyncMock())
    await app._start_router()
    command = app.router_engine.command()
    assert Path(command[0]).parent == config.path(VULKAN_RUNTIME)
    assert command[command.index("-ngl") + 1] == str(layers)
    assert ("--device" in command) == explicit_vulkan
    if explicit_vulkan:
        assert command[command.index("--device") + 1] == "Vulkan0"
