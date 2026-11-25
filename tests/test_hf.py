import pytest

from localive import hf
from localive.config import Config, load_config


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for name in (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HF_TOKEN_FILE",
        "HF_HOME",
        "HF_HUB_OFFLINE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(hf.Path, "home", lambda: tmp_path / "home")


def test_token_priority(monkeypatch, tmp_path):
    config = Config(root=tmp_path)
    assert hf.token(config) == ""
    saved = tmp_path / "home" / ".cache" / "huggingface"
    saved.mkdir(parents=True)
    (saved / "token").write_text("hf_saved\n")
    assert hf.token(config) == "hf_saved"
    config.huggingface.token = "hf_config"
    assert hf.token(config) == "hf_config"
    secret = tmp_path / "secret"
    secret.write_text("hf_secret")
    monkeypatch.setenv("HF_TOKEN_FILE", str(secret))
    assert hf.token(config) == "hf_secret"
    monkeypatch.setenv("HF_TOKEN", "hf_env")
    assert hf.token(config) == "hf_env"


def test_configure_sets_cache_token_and_offline(monkeypatch, tmp_path):
    file = tmp_path / "localive.toml"
    file.write_text('[huggingface]\ntoken = "hf_x"\noffline = true\n', encoding="utf-8")
    config = load_config(file)
    assert hf.configure(config) == "hf_x"
    assert hf.os.environ["HF_HOME"] == str(tmp_path / "models" / "hf")
    assert hf.os.environ["HF_TOKEN"] == "hf_x" and hf.os.environ["HF_HUB_OFFLINE"] == "1"


def test_gated_error_message():
    error = hf.explain(OSError("403 Forbidden: GatedRepo restricted"), "hf://gradium/phonon-40m-fr")
    assert "https://huggingface.co/gradium/phonon-40m-fr" in str(error)
    assert "conditions" in str(error)


# The message huggingface_hub raised on a fresh machine (no token), 9 November 2026.
FRESH = ("cannot fetch 'config.json' from hf://gradium/phonon-40m-fr: Authentication required: "
         "Access to model gradium/phonon-40m-fr is restricted. You must have access to it and be "
         "authenticated to access it. Please log in. (error_code=GatedRepo)")


def test_fresh_machine_without_token_asks_for_a_token():
    error = hf.explain(OSError(FRESH), "hf://gradium/phonon-40m-fr")
    assert error.kind == "token" and error.repo == "gradium/phonon-40m-fr"
    assert "administration" in str(error)


def test_token_without_accepted_terms_asks_for_the_terms(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    error = hf.explain(OSError(FRESH), "hf://gradium/phonon-40m-fr")
    assert error.kind == "terms"
    plain = hf.explain(OSError("403 Forbidden: GatedRepo restricted"), "hf://gradium/phonon-40m-fr")
    assert plain.kind == "terms"
    other = hf.explain(OSError("No space left on device"), "hf://gradium/phonon-40m-fr")
    assert other.kind == "download"


class FakeHead:
    def __init__(self, status):
        self.status = status
        self.headers = None

    def head(self, url, headers=None, **_):
        self.headers = headers
        status = self.status

        class Reply:
            async def __aenter__(self):
                return type("R", (), {"status": status})()

            async def __aexit__(self, *exc):
                return False

        return Reply()


@pytest.mark.parametrize("status, state", [(302, "ok"), (200, "ok"), (401, "token"),
                                           (403, "terms"), (500, "unreachable")])
async def test_check_access_reads_the_gated_file(status, state):
    http = FakeHead(status)
    result, message = await hf.check_access(http, " hf_abc ", "hf://gradium/phonon-40m-fr")
    assert result == state and message
    assert http.headers == {"Authorization": "Bearer hf_abc"}
