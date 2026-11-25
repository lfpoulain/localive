import pytest

from localive.config import Config, load_config


def test_defaults_without_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_config()
    assert config.server.port == 8787
    assert config.asr.model == "moondream/parakeet-ultra"
    assert config.voice.model == "hf://gradium/phonon-40m-fr"
    assert config.llm.model == "qwen"
    assert config.decision.model == config.brain.model  # the measured Qwen3-4B arbiter
    assert config.decision.llama_dir == config.brain.extra_args == ""
    assert config.routing.enabled and config.routing.gpu_layers == 99
    assert config.routing.model == "models/decision/vllm-d3-mini-Q8_0.gguf"
    assert config.routing.llama_dir == ".runtime/llama/b11960-vulkan"


def test_toml_overrides(tmp_path):
    file = tmp_path / "localive.toml"
    file.write_text(
        '[server]\napi_key = "secret"\n[vad]\nsilence_ms = 800\n[llm]\ntool_timeout_s = 30\n'
        '[brain]\nport = 9000\nextra_args = "--threads 4"\n[voice]\nfemale = "Tiphaine"\n'
        '[decision]\nmodel = "models/arbiter.gguf"\nllama_dir = "runtime/arbiter"\n'
        '[alt_brain]\nllama_dir = ".runtime/llama/cuda"\n[alt_brain_2]\nenabled = true\n'
        '[expert]\nenabled = true\nbackend = "localive"\nlocal_model = "missing.gguf"\n',
        encoding="utf-8",
    )
    config = load_config(file)
    assert config.server.api_key == "secret"
    assert config.vad.silence_ms == 800
    assert config.llm.tool_timeout_s == 30.0
    assert config.brain.port == 9000
    assert config.brain.extra_args == "--threads 4"
    assert config.decision.model == "models/arbiter.gguf"
    assert config.decision.llama_dir == "runtime/arbiter"
    assert config.voice.female == "Tiphaine"
    assert "alt_brain" not in config.file_values  # removed sections: ignored, then dropped
    assert "expert" not in config.file_values and not hasattr(config, "expert")
    assert config.root == tmp_path


@pytest.mark.parametrize(
    "content", ["[server]\nprot = 1\n", '[server]\nport = "80"\n', "server = 3\n", "[tts]\nx = 1\n"]
)
def test_invalid_settings_are_rejected(tmp_path, content):
    file = tmp_path / "localive.toml"
    file.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(file)


def test_missing_qwen_is_a_clear_error(tmp_path):
    file = tmp_path / "localive.toml"
    file.write_text('[brain]\nmodel = "models/absent.gguf"\n', encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="setup.ps1"):
        load_config(file).finalize()


@pytest.mark.parametrize(
    "managed, enabled, dedicated",
    [(True, True, True), (False, True, True), (True, False, True), (True, True, False)],
)
def test_missing_arbiter_override_is_rejected_only_when_used(tmp_path, managed, enabled, dedicated):
    config = Config(root=tmp_path)
    config.brain.model = "main.gguf"
    (tmp_path / config.brain.model).touch()
    config.brain.managed = managed
    config.decision.enabled = enabled
    config.decision.dedicated = dedicated
    config.decision.model = "absent-arbiter.gguf"
    if managed and enabled and dedicated:
        with pytest.raises(FileNotFoundError, match="Arbiter model not found.*absent-arbiter"):
            config.finalize()
    else:
        assert config.finalize() is config


@pytest.mark.parametrize(
    "section, key", [("decision", "model"), ("decision", "llama_dir"), ("brain", "extra_args")]
)
def test_runtime_model_overrides_must_be_strings(tmp_path, section, key):
    file = tmp_path / "localive.toml"
    file.write_text(f"[{section}]\n{key} = 123\n", encoding="utf-8")
    with pytest.raises(ValueError, match=rf"{section}\.{key} must be of type str"):
        load_config(file)


def test_missing_default_arbiter_falls_back_to_the_brain(tmp_path):
    config = Config(root=tmp_path)
    config.brain.model = "models/other/brain.gguf"
    (tmp_path / "models/other").mkdir(parents=True)
    (tmp_path / config.brain.model).touch()
    assert config.finalize().decision.model == ""  # inherit: Qwen3-4B is not installed


@pytest.mark.parametrize(
    "requested, expected",
    [("cedar", "male"), ("marin", "female"), ("male", "male"), (None, "female"), ("", "female")],
)
def test_openai_voice_names_map_to_a_gender(requested, expected):
    assert Config().gender(requested) == expected


def test_toml_saved_with_a_bom_by_notepad_loads(tmp_path):
    file = tmp_path / "localive.toml"
    file.write_bytes("﻿[server]\nport = 9001\n".encode())
    assert load_config(file).server.port == 9001


def test_missing_chosen_brain_falls_back_to_the_installed_one(tmp_path):
    from localive.config import DEFAULT_BRAIN

    file = tmp_path / "localive.toml"
    file.write_text('[brain]\nmodel = "models/Qwen3.5-4B/Qwen3.5-4B-Q4_K_M.gguf"\n',
                    encoding="utf-8")
    (tmp_path / DEFAULT_BRAIN).parent.mkdir(parents=True)
    (tmp_path / DEFAULT_BRAIN).touch()
    config = load_config(file).finalize()
    assert config.brain.model == DEFAULT_BRAIN  # starts: the admin page can pick another


def test_ollama_brain_keeps_the_local_arbiter(tmp_path):
    file = tmp_path / "localive.toml"
    file.write_text('[brain]\nbackend = "ollama"\nexternal_model = "qwen3:8b"\n', encoding="utf-8")
    config = load_config(file)
    (tmp_path / config.decision.model).parent.mkdir(parents=True)
    (tmp_path / config.decision.model).write_bytes(b"gguf")
    config.finalize()
    assert not config.brain.managed and config.llm.url == "http://127.0.0.1:11434/v1"
    assert config.llm.upstream_model == "qwen3:8b" and config.local_arbiter
    assert config.brain_identity() == "ollama:http://127.0.0.1:11434/v1:qwen3:8b"
    config.decision.enabled = False  # classic mode: no decision model at all
    assert not config.local_arbiter


def test_other_server_needs_an_address(tmp_path):
    file = tmp_path / "localive.toml"
    file.write_text('[brain]\nbackend = "openai"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="adresse"):
        load_config(file).finalize()


@pytest.mark.parametrize(("backend", "expected"), [
    ("localive", False), ("ollama", True), ("lmstudio", False), ("openai", False)])
def test_thinking_is_turned_off_for_ollama(tmp_path, backend, expected):
    config = Config(root=tmp_path)
    config.brain.backend = backend
    config.brain.external_url = "http://127.0.0.1:9/v1"
    config.brain.managed = False
    config.decision.enabled = False
    config.finalize()
    assert config.llm.no_thinking is expected
