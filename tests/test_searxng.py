"""Localive's SearXNG container management, with Docker and HTTP faked."""

import asyncio
import json

import pytest

from localive.config import Config
from localive.searxng import OWNER_LABEL, SearxngManager


class FakeDocker(SearxngManager):
    def __init__(self, config, *, docker=True, engine_up=True, container=None, ready=True):
        super().__init__(config)
        self.calls: list[tuple[str, ...]] = []
        self.has_docker, self.engine_up = docker, engine_up
        self.container = container  # None: absent, "true"/"false": running or not
        self.ready = ready
        self.inspect = {
            "Config": {"Image": config.search.image, "Labels": {}},
            "Mounts": [{"Type": "bind", "Source": str(config.root / ".runtime" / "searxng"),
                        "Destination": "/etc/searxng"}],
            "HostConfig": {"PortBindings": {"8080/tcp": [
                {"HostIp": "127.0.0.1", "HostPort": "8888"}]}},
            "State": {"Running": container == "true"},
        }

    def _docker(self):
        return "docker" if self.has_docker else None

    async def _docker_cmd(self, docker, *args, limit_s=30):
        self.calls.append(args)
        if args[0] == "info":
            return (0, "29.0") if self.engine_up else (1, "Cannot connect")
        if args[0] == "inspect":
            return (0, json.dumps(self.inspect)) if self.container else (1, "No such object")
        return 0, "ok"

    async def _wait_ready(self, http, seconds):
        return self.ready


@pytest.fixture
def config(tmp_path):
    cfg = Config()
    cfg.root = tmp_path
    cfg.search.start_docker = False
    return cfg


async def test_first_start_creates_a_local_only_container(config):
    manager = FakeDocker(config)
    await manager.ensure(http=None)
    assert manager.state == "ready"
    run = next(c for c in manager.calls if c[0] == "run")
    assert "127.0.0.1:8888:8080" in run and "unless-stopped" in run
    assert run[-1] == config.search.image
    assert f"{OWNER_LABEL}={config.path('.runtime/searxng').resolve()}" in run
    settings = (config.root / ".runtime/searxng/settings.yml").read_text(encoding="utf-8")
    assert "formats: [html, json]" in settings and "limiter: false" in settings


async def test_existing_container_is_started_not_recreated(config):
    manager = FakeDocker(config, container="false")
    await manager.ensure(http=None)
    assert ("start", config.search.container) in manager.calls
    assert not any(c[0] == "run" for c in manager.calls)


async def test_running_container_is_left_alone(config):
    manager = FakeDocker(config, container="true")
    await manager.ensure(http=None)
    assert [c[0] for c in manager.calls] == ["info", "inspect"]
    assert manager.state == "ready"


@pytest.mark.parametrize("difference", ["port", "image", "mount", "owner"])
async def test_conflicting_container_remains_untouched(config, difference):
    manager = FakeDocker(config, container="false")
    if difference == "port":
        config.search.url = "http://127.0.0.1:8891"
    elif difference == "image":
        config.search.image = "searxng/searxng:new"
    elif difference == "mount":
        manager.inspect["Mounts"][0]["Source"] = str(config.root / "another")
    else:
        manager.inspect["Config"]["Labels"][OWNER_LABEL] = str(config.root / "another")
    await manager.ensure(http=None)
    assert manager.state == "error" and "autre nom" in manager.detail
    assert [c[0] for c in manager.calls] == ["info", "inspect"]
    assert not config.path(".runtime/searxng/settings.yml").exists()


@pytest.mark.parametrize("status,body", [(403, None), (200, "a login page"), (200, {})])
async def test_json_configuration_error_is_preserved(config, status, body):
    class Reply:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def json(self, **kwargs):
            if isinstance(body, str):
                raise ValueError("not JSON")
            return body

    class Http:
        def get(self, *args, **kwargs):
            response = Reply()
            response.status = status
            return response

    config.search.managed = False
    manager = SearxngManager(config)
    await manager.ensure(Http())
    assert manager.state == "error" and "JSON" in manager.detail


@pytest.mark.parametrize("docker, engine_up, state", [
    (False, True, "docker_missing"), (True, False, "docker_stopped")])
async def test_without_docker_the_voice_still_works(config, docker, engine_up, state):
    manager = FakeDocker(config, docker=docker, engine_up=engine_up)
    await manager.ensure(http=None)  # never raises: web search is optional
    assert manager.state == state
    assert manager.status()["message"]


async def test_secret_key_survives_and_user_settings_are_kept(config):
    folder = config.root / ".runtime/searxng"
    folder.mkdir(parents=True)
    (folder / "settings.yml").write_text('server:\n  secret_key: "abc123"\n', encoding="utf-8")
    FakeDocker(config).write_settings()  # no JSON format: rewritten, secret kept
    text = (folder / "settings.yml").read_text(encoding="utf-8")
    assert 'secret_key: "abc123"' in text and "json" in text
    (folder / "settings.yml").write_text("search:\n  formats: [json]\nserver:\n  secret_key: x\n",
                                         encoding="utf-8")
    FakeDocker(config).write_settings()  # already valid: the user's file is untouched
    assert (folder / "settings.yml").read_text(encoding="utf-8").startswith("search:")


async def test_remote_or_unmanaged_searxng_is_only_probed(config):
    config.search.managed = False
    manager = FakeDocker(config)
    await manager.ensure(http=None)
    assert manager.calls == [] and manager.state == "ready"
    config.search.engine = "off"
    off = FakeDocker(config)
    await off.ensure(http=None)
    assert off.state == "off"


async def test_background_check_shared_across_sessions_and_manual_retry(config, monkeypatch):
    manager = FakeDocker(config)
    release = asyncio.Event()
    entered = asyncio.Event()
    probes = []

    async def probe(http, seconds):
        probes.append(seconds)
        entered.set()
        await release.wait()
        return len(probes) > 1

    config.search.managed = False
    monkeypatch.setattr(manager, "_wait_ready", probe)
    assert manager.status()["state"] == "idle" and manager.startup_task is None
    manager.start(None)
    manager.start(None, retry=True)
    waiter = asyncio.create_task(manager.wait_ready(None))
    await entered.wait()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert probes == [5] and not manager.startup_task.cancelled()
    release.set()
    await manager.startup_task
    assert manager.status()["state"] == "unreachable"
    manager.start(None, retry=True)
    await manager.wait_ready(None)
    manager.start(None)
    assert manager.status()["state"] == "ready" and probes == [5, 5]
    config.search.engine = "off"
    manager.start(None, retry=True)
    assert manager.status()["state"] == "off" and probes == [5, 5]
