import tomllib

import pytest

from localive import settings
from localive.config import DEFAULT_BRAIN, load_config


@pytest.fixture
def config(tmp_path):
    model = tmp_path / DEFAULT_BRAIN
    model.parent.mkdir(parents=True)
    model.touch()
    file = tmp_path / "localive.toml"
    file.write_text("[server]\nport = 9999\n", encoding="utf-8")
    return load_config(file)


def test_schema_lists_values_and_defaults(config):
    entries = {s["key"]: s for g in settings.schema(config)["groups"] for s in g["settings"]}
    assert entries["server.port"]["value"] == 9999 and entries["server.port"]["default"] == 8787
    assert entries["voice.female"]["restart"]
    assert entries["voice.default"]["choices"] == ["female", "male"]
    assert entries["llm.temperature"]["type"] == "float"


def test_live_reasoning_level_is_persisted_and_scoped_to_the_local_brain(config):
    key = "brain.reasoning_level"
    entry = next(s for g in settings.schema(config)["groups"] for s in g["settings"]
                 if s["key"] == key)
    assert entry["default"] == "off" and entry["only"] == "brain.backend=localive"
    assert entry["choice_labels"]["medium"] == "Moyen (96 jetons)"
    assert settings.apply(config, {key: "medium"}) == {"applied": [key], "restart_required": []}
    assert load_config(config.file).brain.reasoning_level == "medium"
    assert config.brain.reasoning_budget == 0  # bundled non-thinking model
    config.brain.model = "Qwen3.5-4B.gguf"
    assert config.brain.reasoning_budget == 96
    config.brain.backend = "ollama"
    assert config.brain.reasoning_budget == 0
    config.brain.backend = "localive"
    config.brain.managed = False
    assert config.brain.reasoning_budget == 0
    config.brain.reasoning_level = "unknown"
    with pytest.raises(ValueError, match=key):
        config.finalize()


def test_lmstudio_effort_applies_live_and_survives_restart(config):
    key = "llm.lmstudio_reasoning"
    assert settings.BY_KEY[key].only == "brain.backend=lmstudio"
    assert settings.apply(config, {key: "off"}) == {"applied": [key], "restart_required": []}
    assert config.llm.lmstudio_reasoning == load_config(config.file).llm.lmstudio_reasoning == "off"


def test_device_vocabulary_is_opt_in_and_persisted(config):
    assert config.asr.device_vocabulary is False
    assert settings.apply(config, {"asr.device_vocabulary": True}) == {
        "applied": ["asr.device_vocabulary"], "restart_required": []}
    assert config.asr.device_vocabulary and load_config(config.file).asr.device_vocabulary
    settings.reset(config, ["asr.device_vocabulary"])
    assert not config.asr.device_vocabulary and not load_config(config.file).asr.device_vocabulary


def test_live_values_apply_and_everything_is_saved(config):
    result = settings.apply(
        config, {"llm.temperature": 1, "vad.silence_ms": 800, "brain.context": 32768}
    )
    assert config.llm.temperature == 1.0 and config.vad.silence_ms == 800
    assert config.brain.context == 24576  # restart-only: unchanged in memory
    assert result == {
        "applied": ["llm.temperature", "vad.silence_ms"],
        "restart_required": ["brain.context"],
    }
    saved = tomllib.loads(config.file.read_text(encoding="utf-8"))
    assert saved["server"]["port"] == 9999  # untouched settings are kept
    assert saved["llm"]["temperature"] == 1.0
    assert saved["brain"]["context"] == 32768
    assert load_config(config.file).brain.context == 32768


def test_goodbye_modes_are_saved_for_new_sessions_and_validate_toml(config):
    assert config.turn.goodbye_mode == "phrases" and not config.turn.uses_d3_goodbye
    mode = settings.BY_KEY["turn.goodbye_mode"]
    assert mode.only == "turn.end_on_goodbye=true" and mode.choices == ("phrases", "d3")
    result = settings.apply(config, {"turn.end_on_goodbye": True, "turn.goodbye_mode": "d3"})
    assert not result["restart_required"] and load_config(config.file).turn.uses_d3_goodbye
    with pytest.raises(ValueError, match="invalide"):
        settings.apply(config, {"turn.goodbye_mode": "auto"})
    config.turn.goodbye_mode = "auto"
    with pytest.raises(ValueError, match="turn.goodbye_mode"):
        config.finalize()


@pytest.mark.parametrize(
    "changes",
    [{"llm.nope": 1}, {"voice.default": "robot"}, {"vad.silence_ms": 10}, {"llm.tools": "yes"}],
)
def test_invalid_changes_touch_nothing(config, changes):
    before = config.file.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        settings.apply(config, changes)
    assert config.file.read_text(encoding="utf-8") == before


def test_validates_the_configuration_that_will_actually_restart(config):
    before = config.file.read_bytes()
    with pytest.raises(ValueError, match="adresse"):
        settings.apply(config, {"brain.backend": "openai", "llm.temperature": 1.0})
    assert config.file.read_bytes() == before and config.llm.temperature == 0.2
    settings.apply(config, {"brain.external_url": "https://first.example/v1"})
    settings.apply(config, {"brain.backend": "openai", "llm.api_key": "test-only"})
    with pytest.raises(ValueError, match="adresse"):
        settings.reset(config, ["brain.external_url"])
    assert load_config(config.file).finalize().llm.api_key == "test-only"
    settings.apply(config, {"brain.external_url": "https://second.example/v1"})
    assert load_config(config.file).finalize().llm.api_key == ""


def test_routing_settings_survive_restart_and_an_invalid_file_is_not_saved(config):
    enabled = not config.routing.enabled
    result = settings.apply(config, {"routing.enabled": enabled, "routing.port": 18085,
                                     "routing.timeout_ms": 700})
    assert result == {"applied": ["routing.timeout_ms"],
                      "restart_required": ["routing.enabled", "routing.port"]}
    routing = load_config(config.file).finalize().routing
    assert routing.enabled == enabled and routing.port == 18085 and routing.timeout_ms == 700
    # An invalid key already present in the file must also be checked on the next save.
    config.file_values["routing"]["port"] = 0
    before = config.file.read_bytes()
    with pytest.raises(ValueError, match="port"):
        settings.apply(config, {"llm.temperature": 1})
    assert config.file.read_bytes() == before


@pytest.mark.parametrize("action", ["apply", "reset"])
def test_failed_save_keeps_disk_and_running_state(config, monkeypatch, action):
    settings.apply(config, {"llm.temperature": 0.75})
    before = config.file.read_bytes()

    def fail(_source, _destination):
        raise OSError("disk unavailable")

    monkeypatch.setattr(settings.os, "replace", fail)
    with pytest.raises(OSError, match="disk unavailable"):
        if action == "apply":
            settings.apply(config, {"llm.temperature": 1.0})
        else:
            settings.reset(config, ["llm.temperature"])
    assert config.file.read_bytes() == before and config.llm.temperature == 0.75
    assert config.file_values["llm"]["temperature"] == 0.75
    assert not list(config.file.parent.glob("*.tmp"))


def test_toml_strings_are_escaped(config):
    settings.apply(config, {"voice.female": 'voix "maison" \\ é.safetensors'})
    assert load_config(config.file).voice.female == 'voix "maison" \\ é.safetensors'


def test_reasoning_model_is_chosen_among_the_models_on_disk(config):
    for name in ("Qwen3-4B/a.gguf", "Gemma/b.gguf", "Qwen3-4B/mmproj-a.gguf", "hf/hub/c.gguf"):
        path = config.root / "models" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    entries = {s["key"]: s for g in settings.schema(config)["groups"] for s in g["settings"]}
    brain = entries["brain.model"]
    assert {"models/Qwen3-4B/a.gguf", "models/Gemma/b.gguf"} <= set(brain["choices"])
    assert not any("mmproj" in c or "/hf/" in c for c in brain["choices"])
    assert brain["choice_labels"]["models/Gemma/b.gguf"] == "b"
    assert entries["decision.model"]["choices"][0] == ""  # "the same as the brain"
    result = settings.apply(config, {"brain.model": "models/Gemma/b.gguf"})
    assert result["restart_required"] == ["brain.model"]
    with pytest.raises(ValueError):
        settings.apply(config, {"brain.model": "models/absent.gguf"})


def test_systemone_weights_are_not_offered_for_chat_or_interruptions(config):
    renamed = "models/decision/renamed.gguf"
    by_name = ["models/custom/D1-Omni-2B-F16.gguf", "models/decision/vllm-d3-mini-Q8_0.gguf",
               "models/decision/vllm-d3-nano-Q8_0.gguf", "models/custom/D3-lite-Q4_K_M.gguf"]
    ordinary = "models/decision/chat.gguf"
    config.routing.model = renamed
    for name in (renamed, *by_name, ordinary):
        path = config.path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    for key in ("brain.model", "decision.model"):
        choices, _labels = settings.choices_for(settings.BY_KEY[key], config)
        assert ordinary in choices and not set(choices).intersection([renamed, *by_name])
        with pytest.raises(ValueError, match="invalide"):
            settings.apply(config, {key: renamed})
    with pytest.raises(ValueError, match="tri des outils"):
        settings.apply(config, {"routing.model": ordinary, "brain.model": ordinary})


def test_arbiter_overrides_and_main_options_are_saved_for_restart(config):
    arbiter = config.root / "models" / "arbiter.gguf"
    arbiter.parent.mkdir(parents=True, exist_ok=True)
    arbiter.touch()
    changes = {
        "decision.model": "models/arbiter.gguf",
        "decision.llama_dir": "runtime/arbiter",
        "brain.extra_args": "--threads 4",
    }
    entries = {s["key"]: s for g in settings.schema(config)["groups"] for s in g["settings"]}
    assert all(entries[key]["restart"] for key in changes)
    result = settings.apply(config, changes)
    assert result == {"applied": [], "restart_required": sorted(changes)}
    assert config.decision.llama_dir == config.brain.extra_args == ""
    reloaded = load_config(config.file)
    assert reloaded.decision.model == changes["decision.model"]
    assert reloaded.decision.llama_dir == changes["decision.llama_dir"]
    assert reloaded.brain.extra_args == changes["brain.extra_args"]
    assert reloaded.brain.model == config.brain.model
    assert reloaded.brain.llama_dir == config.brain.llama_dir


def test_settings_reference_in_docs_is_up_to_date():
    from pathlib import Path

    doc = Path(__file__).resolve().parents[1] / "docs" / "reglages.md"
    assert doc.read_text(encoding="utf-8") == settings.reference_markdown(), (
        "docs/reglages.md est périmé : python -m localive.settings > docs/reglages.md")


def test_asr_api_key_is_cleared_when_provider_or_url_changes(config):
    settings.apply(
        config,
        {
            "asr.backend": "openai",
            "asr.external_url": "https://stt-one.example",
            "asr.api_key": "secret-stt",
        },
    )
    assert load_config(config.file).finalize().asr.api_key == "secret-stt"
    # Unrelated setting keeps the saved key.
    settings.apply(config, {"asr.language": "en"})
    assert load_config(config.file).finalize().asr.api_key == "secret-stt"
    # Changing the host without editing the key clears it.
    settings.apply(config, {"asr.external_url": "https://stt-two.example"})
    assert load_config(config.file).finalize().asr.api_key == ""
    # Re-set and switch backend: key is cleared too.
    settings.apply(config, {"asr.api_key": "secret-stt-2"})
    settings.apply(config, {"asr.backend": "photon"})
    assert load_config(config.file).finalize().asr.api_key == ""


def test_voice_api_key_is_cleared_when_provider_or_url_changes(config):
    settings.apply(
        config,
        {
            "voice.backend": "openai",
            "voice.external_url": "https://tts-one.example",
            "voice.api_key": "secret-tts",
        },
    )
    assert load_config(config.file).finalize().voice.api_key == "secret-tts"
    # Unrelated setting keeps the saved key.
    settings.apply(config, {"voice.speed": 1.1})
    assert load_config(config.file).finalize().voice.api_key == "secret-tts"
    # Changing the host without editing the key clears it.
    settings.apply(config, {"voice.external_url": "https://tts-two.example"})
    assert load_config(config.file).finalize().voice.api_key == ""
    # Re-set and switch backend: key is cleared too.
    settings.apply(config, {"voice.api_key": "secret-tts-2"})
    settings.apply(config, {"voice.backend": "phonon"})
    assert load_config(config.file).finalize().voice.api_key == ""
