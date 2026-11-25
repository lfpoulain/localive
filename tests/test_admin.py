"""The admin page's API: status, explained settings, secrets, reset, restart, log."""

import asyncio
import logging
from unittest.mock import patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from localive import settings
from localive.config import DEFAULT_BRAIN, load_config
from localive.server import LOG_BUFFER, App
from localive.webrtc import WebRtcError, WebRtcSession


@pytest.fixture
def config(tmp_path):
    model = tmp_path / DEFAULT_BRAIN
    model.parent.mkdir(parents=True)
    model.touch()
    file = tmp_path / "localive.toml"
    file.write_text('[server]\napi_key = ""\n[huggingface]\ntoken = "hf_secret"\n',
                    encoding="utf-8")
    return load_config(file)


@pytest.fixture
async def client(config):
    instance = App(config)
    app = instance.build()
    app.on_startup.clear()  # no models, no Docker
    async with TestClient(TestServer(app)) as http:
        http.instance = instance
        yield http


def test_every_setting_is_explained_and_secrets_are_never_returned(config):
    groups = settings.schema(config)["groups"]
    assert len(groups) >= 8 and all(g["intro"] and g["icon"] for g in groups)
    entries = [s for g in groups for s in g["settings"]]
    assert len(entries) >= 55
    essential = [s for s in entries if not s["advanced"]]
    assert all(s["help"] or s["choices"] or s["type"] == "bool" for s in essential)
    token = next(s for s in entries if s["key"] == "huggingface.token")
    assert token["secret"] and token["value"] == "" and token["is_set"] is True
    assert "hf_secret" not in str(groups)


def test_reset_removes_the_key_from_the_file(config):
    settings.apply(config, {"turn.check_ms": 240})
    assert "check_ms" in config.file.read_text(encoding="utf-8")
    result = settings.reset(config, ["turn.check_ms"])
    assert config.turn.check_ms == 200 and result["applied"] == ["turn.check_ms"]
    assert "check_ms" not in config.file.read_text(encoding="utf-8")
    assert "hf_secret" in config.file.read_text(encoding="utf-8")  # other values are kept


def test_one_brain_and_old_comparison_sections_still_load(tmp_path):
    from localive.config import Config

    assert not hasattr(Config(), "alt_brain")
    file = tmp_path / "localive.toml"
    file.write_text('[alt_brain]\nenabled = true\n[voice]\nmale = "Jules"\n', encoding="utf-8")
    config = load_config(file)
    assert config.voice.male == "Jules" and list(config.brains()) == ["qwen"]


async def test_restart_pending_follows_what_is_saved(client):
    saved = await (await client.put("/api/settings", json={"voice.threads": 6})).json()
    assert saved["restart_pending"] == ["voice.threads"]
    threads = next(s for g in saved["groups"] for s in g["settings"] if s["key"] == "voice.threads")
    assert threads["value"] == 6 and client.instance.config.voice.threads == 4
    back = await (await client.put("/api/settings", json={"voice.threads": 4})).json()
    assert back["restart_pending"] == []  # the running value again: nothing to restart
    await client.put("/api/settings", json={"voice.threads": 6})
    reset = await (await client.post("/api/settings/reset",
                                     json={"keys": ["voice.threads"]})).json()
    assert reset["restart_pending"] == []


async def test_status_tells_cornelius_where_to_connect(client):
    status = await (await client.get("/api/status")).json()
    assert status["connection"]["urls"] == ["http://127.0.0.1:8787/v1"]
    assert status["connection"]["lan"] is False and status["sessions"] == []
    client.instance.config.server.host = "0.0.0.0"
    status = await (await client.get("/api/status")).json()
    assert status["connection"]["lan"] is True
    assert status["connection"]["urls"][-1] == "http://127.0.0.1:8787/v1"


@pytest.mark.parametrize(("host", "authority", "lan"), [
    ("::1", "[::1]", False), ("fd00::10", "[fd00::10]", True),
    ("192.168.1.20", "192.168.1.20", True),
])
def test_explicit_listen_address_in_client_url(config, host, authority, lan):
    config.server.host = host
    connection = App(config).connection()
    assert connection["urls"] == [f"http://{authority}:8787/v1"]
    assert connection["lan"] is lan


async def test_log_is_incremental(client):
    LOG_BUFFER.lines.clear()
    logging.getLogger("localive.test").addHandler(LOG_BUFFER)
    logging.getLogger("localive.test").setLevel(logging.INFO)
    logging.getLogger("localive.test").info("première ligne")
    first = (await (await client.get("/api/logs")).json())["lines"]
    assert first and first[-1]["text"].endswith("première ligne")
    logging.getLogger("localive.test").warning("deuxième ligne")
    more = (await (await client.get(f"/api/logs?after={first[-1]['n']}")).json())["lines"]
    assert [line["level"] for line in more] == ["WARNING"]


async def test_remote_admin_reads_require_a_configured_token(client, monkeypatch):
    monkeypatch.setattr("localive.server.is_loopback", lambda _address: False)
    for endpoint in ("/api/logs", "/api/settings", "/api/status"):
        assert (await client.get(endpoint)).status == 403
    assert (await client.get("/health")).status == 200
    client.instance.config.server.api_key = "admin-test"
    assert (await client.get("/api/logs")).status == 401
    assert (await client.get("/api/logs", headers={"Authorization": "Bearer admin-test"})).status \
        == 200


async def test_restart_waits_for_conversations_to_end(client):
    class Busy:
        id = "sess_busy"

    client.instance.sessions.add(Busy())
    response = await client.post("/api/restart")
    assert response.status == 409
    assert not client.instance.restart_requested
    client.instance.sessions.clear()


async def test_restart_marks_the_old_server_until_the_launcher_replaces_it(client):
    instance = client.instance
    instance.ready.set()
    health = await (await client.get("/health")).json()
    assert health["ready"] and not health["restarting"]
    # Keep the test server alive while checking what the browser sees during shutdown.
    with patch.object(asyncio.get_running_loop(), "call_later") as schedule:
        await instance.restart(None)
        await instance.restart(None)
        schedule.assert_called_once()
    health = await (await client.get("/health")).json()
    assert health["restarting"] and not health["ready"]
    async with client.ws_connect("/live") as ws:
        event = await ws.receive_json()
        assert event["error"] == {"code": "engine_not_ready", "message": "Localive redémarre."}
    response = await client.post("/v1/live/sessions", json={
        "session": {}, "transport": {"type": "webrtc", "sdp": "offer"}})
    assert response.status == 503 and not instance.sessions


async def test_restart_waits_during_webrtc_negotiation(client, monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()

    async def negotiate(*_args):
        entered.set()
        await release.wait()
        raise WebRtcError("Offre invalide")

    monkeypatch.setattr(WebRtcSession, "answer", negotiate)
    client.instance.ready.set()
    offer = asyncio.ensure_future(client.post("/v1/live/sessions", json={
        "session": {}, "transport": {"type": "webrtc", "sdp": "offer"}}))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert (await client.post("/api/restart")).status == 409
    finally:
        release.set()
        response = await offer
    assert response.status == 400
    with patch.object(asyncio.get_running_loop(), "call_later"):
        assert (await client.instance.restart(None)).status == 200


async def test_admin_page_versions_its_assets(client):
    page = await (await client.get("/")).text()
    assert "/static/admin.css?v=" in page and "{{version}}" not in page


async def test_missing_hugging_face_access_is_guided_not_a_traceback(client, monkeypatch):
    from localive import hf, server

    async def no_access(self):
        raise hf.GatedModelError("La voix demande un jeton", "token", "gradium/phonon-40m-fr")

    opened = []
    monkeypatch.setattr(server.webbrowser, "open", opened.append)
    instance = client.instance
    instance.open_browser = True
    monkeypatch.setattr(server.Synthesizer, "start", no_access)
    monkeypatch.setattr(server.Transcriber, "start", lambda self: _done())
    monkeypatch.setattr(server.Transcriber, "close", lambda self: None)
    instance.brain.config.brain.managed = False
    instance.alts = []  # no llama-server in this test (the dedicated arbiter included)
    await instance.boot()
    status = await (await client.get("/api/status")).json()
    assert status["error_kind"] == "hf_token" and not status["ready"]
    assert opened == ["http://127.0.0.1:8787"]  # the admin page shows the way out


async def _done():
    return None


async def test_token_check_endpoint(client, monkeypatch):
    from localive import hf

    async def fake(http, token, repo):
        return ("ok", "Jeton valide") if token == "hf_good" else ("token", "Jeton refusé")

    monkeypatch.setattr(hf, "check_access", fake)
    good = await (await client.post("/api/hf/check", json={"token": "hf_good"})).json()
    bad = await (await client.post("/api/hf/check", json={"token": "hf_bad"})).json()
    assert good["state"] == "ok" and bad["state"] == "token"
    assert good["page"] == "https://huggingface.co/gradium/phonon-40m-fr"
    assert (await client.post("/api/hf/check", json={})).status == 400


def test_home_network_addresses_come_before_vpn_and_virtual_switches(monkeypatch):
    from localive import server

    monkeypatch.setattr(server.socket, "getaddrinfo", lambda *a, **k: [
        (0, 0, 0, "", ("100.114.39.116", 0)), (0, 0, 0, "", ("172.21.80.1", 0)),
        (0, 0, 0, "", ("10.0.0.10", 0)), (0, 0, 0, "", ("127.0.0.1", 0))])

    class NoRoute:
        def __init__(self, *a):
            raise OSError

    monkeypatch.setattr(server.socket, "socket", NoRoute)
    assert server.lan_addresses() == ["10.0.0.10", "172.21.80.1", "100.114.39.116"]


async def test_models_are_downloaded_from_the_admin_page(client, monkeypatch):
    import asyncio

    from localive import install

    def fake_hub_file(repo, filename, folder, progress=None):
        progress(5, 10)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / filename).write_bytes(b"gguf")
        progress(10, 10)
        return folder / filename

    monkeypatch.setattr(install, "hub_file", fake_hub_file)
    rows = (await (await client.get("/api/models")).json())["models"]
    gemma = next(m for m in rows if m["id"] == "gemma-e4b")
    assert not gemma["installed"] and gemma["state"] == "idle"
    response = await client.post("/api/models/download", json={"id": "gemma-e4b"})
    assert response.status == 200
    for _ in range(50):
        rows = (await (await client.get("/api/models")).json())["models"]
        gemma = next(m for m in rows if m["id"] == "gemma-e4b")
        if gemma["state"] == "done":
            break
        await asyncio.sleep(0.02)
    assert gemma["installed"] and gemma["done"] == gemma["total"] == 10
    choices = next(s for g in settings.schema(client.instance.config)["groups"]
                   for s in g["settings"] if s["key"] == "brain.model")["choices"]
    assert gemma["path"] in choices  # offered in the dropdown right away
    assert (await client.post("/api/models/download", json={"id": "nope"})).status == 400



async def test_external_brain_models_are_listed(client):
    from aiohttp import web

    async def models(_request):
        return web.json_response({"data": [{"id": "qwen3:8b"}, {"id": "gemma3:4b"}]})

    async def native(_request):
        return web.json_response({"models": [{
            "key": "qwen", "loaded_instances": [{"id": "qwen3:8b"}],
            "capabilities": {"reasoning": {"allowed_options": ["off", "on"]}},
        }]})

    upstream = web.Application()
    upstream.router.add_get("/v1/models", models)
    upstream.router.add_get("/api/v1/models", native)
    async with TestClient(TestServer(upstream)) as fake:
        url = str(fake.make_url("/v1"))
        result = await (await client.post("/api/brain/models",
                                          json={"backend": "lmstudio", "url": url})).json()
        assert result["models"] == ["gemma3:4b", "qwen3:8b"] and "error" not in result
        assert result["reasoning"]["qwen3:8b"] == ["off", "on"]
    down = await (await client.post("/api/brain/models", json={
        "backend": "ollama", "url": "http://127.0.0.1:9/v1"})).json()
    assert down["models"] == [] and "Ollama ne répond pas" in down["error"]
    assert (await client.post("/api/brain/models", json={"backend": "x"})).status == 400
    # Saving the choice: any model id is accepted, the backend must be a known one.
    settings.apply(client.instance.config, {"brain.backend": "ollama",
                                            "brain.external_model": "qwen3:8b"})
    with pytest.raises(ValueError):
        settings.apply(client.instance.config, {"brain.backend": "vllm"})


async def test_model_discovery_uses_only_the_configured_credential(client):
    from aiohttp import web

    config = client.instance.config
    received = []

    async def catalogue(request):
        received.append(request.headers.get("Authorization"))
        return web.json_response({"error": "denied"}, status=401)

    upstream = web.Application()
    upstream.router.add_get("/{path}/models", catalogue)
    async with TestClient(TestServer(upstream)) as fake:
        config.brain.backend = "openai"
        config.brain.external_url = str(fake.make_url("/configured"))
        config.llm.api_key = "test-only"
        for backend, path in (("openai", "/configured"), ("openai", "/preview"),
                              ("lmstudio", "/configured")):
            response = await client.post("/api/brain/models", json={
                "backend": backend, "url": str(fake.make_url(path)),
            })
            assert "401" in (await response.json())["error"]
        saved_url = str(fake.make_url("/preview"))
        saved = await client.put("/api/settings", json={"brain.backend": "openai",
                                 "brain.external_url": saved_url, "llm.api_key": "next-key"})
        assert saved.status == 200
        await client.post("/api/brain/models", json={"backend": "openai", "url": saved_url})
    assert received == ["Bearer test-only", None, None, "Bearer next-key"]


async def test_any_hugging_face_gguf_can_be_added(client, monkeypatch):
    import asyncio

    from localive import install, models

    def fake_hub_file(repo, filename, folder, progress=None):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / filename).write_bytes(b"gguf")
        progress(4, 4)
        return folder / filename

    monkeypatch.setattr(install, "hub_file", fake_hub_file)
    monkeypatch.setattr(models, "repo_files", lambda repo: [{"file": "X-Q4_K_M.gguf",
                                                              "size_gb": 2.0}])
    import localive.server as server_module
    monkeypatch.setattr(server_module, "repo_files", models.repo_files)
    files = await (await client.post("/api/models/files", json={"repo": "someone/X-GGUF"})).json()
    assert files["files"][0]["file"] == "X-Q4_K_M.gguf"
    response = await client.post("/api/models/download",
                                 json={"repo": "someone/X-GGUF", "file": "X-Q4_K_M.gguf"})
    assert response.status == 200
    for _ in range(50):
        rows = (await (await client.get("/api/models")).json())["models"]
        added = next(m for m in rows if m["id"] == "custom:someone/X-GGUF/X-Q4_K_M.gguf")
        if added["state"] == "done":
            break
        await asyncio.sleep(0.02)
    assert added["installed"] and added["path"] == "models/X/X-Q4_K_M.gguf"
    for repo, file in (("../etc", "a.gguf"), ("a/b", "../a.gguf"), ("a/b", "x.bin"),
                       ("a/b", "mmproj-F16.gguf"), ("a/b", "X-00001-of-00002.gguf")):  # fmt: skip
        bad = await client.post("/api/models/download", json={"repo": repo, "file": file})
        assert bad.status == 400


def test_catalogue_keeps_the_folders_already_installed():
    from localive.catalog import BY_ID, CATALOG, FAMILIES

    assert BY_ID["qwen3.5-4b"].path == "models/Qwen3.5-4B/Qwen3.5-4B-Q4_K_M.gguf"
    assert BY_ID["gemma-e4b"].folder == "models/gemma-4-E4B-it"
    assert {m.family for m in CATALOG} == set(FAMILIES)
    assert {"Qwen", "Gemma", "Mistral", "OpenAI", "Liquid (LFM)"} <= set(FAMILIES)


def test_engine_specific_settings_are_marked(config):
    entries = {s["key"]: s for g in settings.schema(config)["groups"] for s in g["settings"]}
    assert entries["brain.model"]["only"] == "brain.backend=localive"
    assert entries["brain.external_model"]["only"] == "brain.backend!=localive"
    assert entries["llm.temperature"]["only"] == ""  # shared by every engine
