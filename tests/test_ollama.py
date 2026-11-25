"""Ollama through its own API: the only way to choose its context (num_ctx)."""

import json

import pytest

from localive.config import Config, LlmConfig
from localive.llm import ChatClient, Completion, LlmError, ollama_messages


class Response:
    status = 200

    def __init__(self, lines):
        self.content = self._lines(lines)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass

    @staticmethod
    async def _lines(lines):
        for line in lines:
            yield (json.dumps(line) + "\n").encode()


class Http:
    def __init__(self, lines):
        self.lines = lines

    def post(self, url, json=None, **_):
        self.url, self.body = url, json
        return Response(self.lines)


def chunk(content="", calls=None, thinking="", done=False, reason="stop"):
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = calls
    if thinking:
        message["thinking"] = thinking
    event = {"message": message, "done": done}
    if done:
        event["done_reason"] = reason
    return event


def config(**kwargs):
    return LlmConfig(url="http://10.0.0.4:11434/v1", upstream_model="bonsai",
                     ollama_context=16384, no_thinking=True, **kwargs)  # fmt: skip


async def run(lines, tools=None, messages=()):
    http = Http(lines)
    result = Completion()
    client = ChatClient(http, config())
    text = "".join([t async for t in client.stream(list(messages), tools, result)])
    return http, text, result


async def test_text_is_streamed_with_the_requested_context():
    http, text, result = await run([chunk("Bonjour "), chunk("à toi."), chunk(done=True)])
    assert text == result.text == "Bonjour à toi."
    assert http.url == "http://10.0.0.4:11434/api/chat"
    assert http.body["options"]["num_ctx"] == 16384
    assert http.body["think"] is False and http.body["model"] == "bonsai"


async def test_tool_calls_come_back_as_json_strings():
    call = {"function": {"name": "lights__set", "arguments": {"room": "bureau", "on": False}}}
    _, text, result = await run([chunk(calls=[call]), chunk(done=True)], tools=[{}])
    assert text == ""
    assert result.tool_calls[0].name == "lights__set"
    assert json.loads(result.tool_calls[0].arguments) == {"room": "bureau", "on": False}


async def test_tool_call_from_a_truncated_generation_is_rejected():
    call = {"function": {"name": "lights__set", "arguments": {"on": False}}}
    with pytest.raises(LlmError, match="did not complete its tool calls"):
        await run([chunk(calls=[call]), chunk(done=True, reason="length")], tools=[{}])


async def test_truncated_text_is_marked_for_the_written_tool_call_fallback():
    _, text, completion = await run([chunk("clock__read()"), chunk(done=True, reason="length")])
    assert text == "clock__read()" and completion.truncated


@pytest.mark.parametrize("lines", [
    [chunk("Bonjour")],  # cut before "done"
    [{"error": "model not found"}],
])
async def test_failures_raise(lines):
    with pytest.raises(LlmError):
        await run(lines)


async def test_thinking_without_answer_is_reported(caplog):
    await run([chunk(thinking="The user asks"), chunk(done=True, reason="length")])
    assert "sans rien répondre" in caplog.text


def test_messages_are_converted_to_ollama_format():
    call = {"id": "call_1", "type": "function",
            "function": {"name": "lights__set", "arguments": '{"room":"bureau"}'}}  # fmt: skip
    converted = ollama_messages([
        {"role": "user", "content": "Éteins le bureau"},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "call_1", "content": '{"ok":true}'},
    ])
    assert converted[1]["tool_calls"] == [
        {"function": {"name": "lights__set", "arguments": {"room": "bureau"}}}]
    assert converted[2] == {"role": "tool", "content": '{"ok":true}', "tool_name": "lights__set"}


async def test_a_prompt_bigger_than_the_context_is_reported(caplog):
    long = [{"role": "system", "content": "consigne " * 30000}]
    await run([chunk("Oui."), chunk(done=True)], messages=long)
    assert "Contexte demandé à Ollama" in caplog.text


@pytest.mark.parametrize(("backend", "context"), [("ollama", 16384), ("lmstudio", 0)])
def test_only_ollama_uses_its_own_api(tmp_path, backend, context):
    c = Config(root=tmp_path)
    c.brain.backend, c.brain.external_url = backend, "http://10.0.0.4:9/v1"
    c.decision.enabled = False
    c.finalize()
    assert c.llm.ollama_context == context
    assert c.context_for(c.llm.model) == (16384 if backend == "ollama" else c.brain.context)
