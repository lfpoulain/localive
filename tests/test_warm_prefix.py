"""The last conversation's opening prompt is read into the brain's cache at startup."""

import json

from localive import live
from localive.config import load_config
from localive.live import Services, save_prefix, warm_from_saved


class FakeChat:
    def __init__(self):
        self.calls = []

    async def stream(self, messages, tools, result, *, max_tokens=None, **_):
        self.calls.append((messages, tools, max_tokens))
        yield ""


async def test_saved_prefix_warms_the_same_model_only(tmp_path):
    (tmp_path / "localive.toml").write_text("", encoding="utf-8")
    config = load_config(tmp_path / "localive.toml")
    chat = FakeChat()
    services = Services(config, None, None, chat, None, None)
    await warm_from_saved(services)  # nothing saved yet: nothing sent
    assert chat.calls == []
    path = config.path(live.PREFIX_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": config.brain_identity(), "messages": [], "tools": []}),
                    encoding="utf-8")
    await warm_from_saved(services)  # older prefixes can contain retired instructions and tools
    assert chat.calls == []
    messages = [{"role": "system", "content": "Tu es Cornelius."},
                {"role": "user", "content": "Bonjour"}]
    tools = [{"type": "function", "function": {"name": "clock__read", "parameters": {}}}]
    save_prefix(config, messages, tools)
    await warm_from_saved(services)
    assert chat.calls == [(messages, tools, 1)]
    config.brain.model = "models/other.gguf"  # another model reads its own prompt format
    await warm_from_saved(services)
    assert len(chat.calls) == 1
