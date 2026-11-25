import json

import pytest

from localive.config import LlmConfig
from localive.llm import ChatClient, Completion, LlmError


class Response:
    status = 200

    def __init__(self, events):
        self.events = events
        self.content = self.lines()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass

    async def lines(self):
        for event in self.events:
            payload = event if isinstance(event, str) else json.dumps(event)
            yield f"data: {payload}\n".encode()


class Http:
    def __init__(self, events):
        self.events = events

    def post(self, *_args, **_kwargs):
        return Response(self.events)


def delta(content=None, calls=None, finish=None):
    return {"choices": [{"delta": {"content": content, "tool_calls": calls},
                         "finish_reason": finish}]}


def call(arguments, name="lights__set", index=0):
    return {"index": index, "id": "call_1", "function": {"name": name, "arguments": arguments}}


async def collect(events):
    result = Completion()
    client = ChatClient(Http(events), LlmConfig())
    text = "".join([piece async for piece in client.stream([], None, result)])
    return text, result


async def test_stream_collects_text_and_fragmented_tool_arguments():
    text, result = await collect([
        delta("Je regarde."), delta(calls=[call('{"room":')]),
        delta(calls=[call('"salon","on":true}', name="")]),
        delta(finish="tool_calls"), {"choices": [], "usage": {}}, "[DONE]",
    ])
    assert text == result.text == "Je regarde."
    assert result.tool_calls[0].name == "lights__set"
    assert json.loads(result.tool_calls[0].arguments) == {"room": "salon", "on": True}


@pytest.mark.parametrize("events", [
    [delta("Bonjour")],
    [delta("Bonjour"), {"error": {"message": "out of memory"}}, "[DONE]"],
    ["{broken", "[DONE]"],
    [None, "[DONE]"],
    [{"choices": [None]}, "[DONE]"],
])
async def test_stream_errors_are_not_reported_as_completed_answers(events):
    with pytest.raises(LlmError):
        await collect(events)


@pytest.mark.parametrize("arguments,finish", [
    ('{"room":', "tool_calls"), ("[]", "tool_calls"),
    ('{"on":true}', "length"), ("{}", None),
])
async def test_incomplete_actions_are_never_exposed_for_execution(arguments, finish):
    with pytest.raises(LlmError):
        await collect([delta(calls=[call(arguments)]), delta(finish=finish), "[DONE]"])


async def test_finish_reason_without_done_sentinel_is_supported():
    text, _ = await collect([delta("Bonjour"), delta(finish="stop")])
    assert text == "Bonjour"


class Recording(Http):
    def post(self, *_args, **kwargs):
        self.body = kwargs["json"]
        return Response(self.events)


@pytest.mark.parametrize("level,expected", [
    ("auto", None), ("off", "none"), ("on", "medium"), ("low", "low"),
    ("medium", "medium"), ("high", "high"),
])
async def test_lmstudio_reasoning_is_sent_with_tools_only_to_lmstudio(level, expected):
    http = Recording([delta("91", finish="stop")])
    config = LlmConfig(lmstudio_reasoning=level)
    tools = [{"type": "function", "function": {"name": "clock", "parameters": {"type": "object"}}}]
    for lmstudio in (True, False):
        client = ChatClient(http, config, lmstudio=lmstudio)
        assert [t async for t in client.stream([], tools, Completion())] == ["91"]
        assert http.body.get("reasoning_effort") == (expected if lmstudio else None)
        assert http.body["tools"] == tools


async def test_lmstudio_recovers_only_a_turn_exhausted_in_private_reasoning():
    exhausted = [{"choices": [{"delta": {"reasoning_content": "calcul"},
                               "finish_reason": "length"}]}]
    attempts = []

    class Upstream:
        def post(self, url, *, json, **kwargs):
            attempts.append(json.get("reasoning_effort"))
            return Response(exhausted if len(attempts) == 1 else [delta("91", finish="stop")])

    result = Completion()
    client = ChatClient(Upstream(), LlmConfig(lmstudio_reasoning="low"), lmstudio=True)
    assert [t async for t in client.stream([], None, result)] == ["91"]
    assert result.text == "91" and not result.truncated
    assert attempts == ["low", "none"]
    attempts.clear()
    client.config.lmstudio_reasoning = "off"
    with pytest.raises(LlmError, match="limite"):
        [t async for t in client.stream([], None, Completion())]
    assert attempts == ["none"]
    attempts.clear()
    client.config.lmstudio_reasoning = "high"
    assert not [t async for t in client.stream([], None, Completion(), max_tokens=1)]
    assert attempts == ["high"]


@pytest.mark.parametrize("no_thinking", [False, True])
async def test_external_servers_are_asked_not_to_think(no_thinking):
    http = Recording([delta("Oui.", finish="stop"), "[DONE]"])
    client = ChatClient(http, LlmConfig(no_thinking=no_thinking))
    [piece async for piece in client.stream([], None, Completion())]
    assert ("reasoning_effort" in http.body) is no_thinking


@pytest.mark.parametrize("budget", [0, 32, 96, 256])
async def test_reasoning_budget_preserves_room_for_the_spoken_response(budget):
    http = Recording([{"choices": [{"delta": {"reasoning_content": "Calcul interne"}}]},
                      delta("221", finish="stop"), "[DONE]"])
    client = ChatClient(http, LlmConfig(max_tokens=64))
    result = Completion()
    spoken = [piece async for piece in client.stream([], None, result, reasoning_budget=budget)]
    assert "".join(spoken) == result.text == "221"
    assert http.body["max_tokens"] == 64 + budget
    assert http.body.get("reasoning_budget_tokens", 0) == budget
    assert http.body.get("chat_template_kwargs") == ({"enable_thinking": True} if budget else None)


async def test_a_reply_spent_thinking_is_reported(caplog):
    thinking = {"choices": [{"delta": {"content": "", "reasoning": "The user"},
                             "finish_reason": None}]}
    text, _ = await collect([thinking, delta(finish="length"), "[DONE]"])
    assert text == ""
    assert "sans rien répondre" in caplog.text


async def test_a_truncated_prompt_is_reported_once(caplog):
    usage = {"choices": [], "usage": {"prompt_tokens": 2050}}
    client = ChatClient(Http([delta("Oui.", finish="stop"), usage, "[DONE]"]),
                        LlmConfig(upstream_model="bonsai"))  # fmt: skip
    long = [{"role": "system", "content": "consigne " * 4000}]
    for _ in range(2):
        [piece async for piece in client.stream(long, None, Completion())]
        client.http = Http([delta("Oui.", finish="stop"), usage, "[DONE]"])
    assert caplog.text.count("OLLAMA_CONTEXT_LENGTH") == 1
    short = ChatClient(Http([delta("Oui.", finish="stop"), usage, "[DONE]"]), LlmConfig())
    [piece async for piece in short.stream([], None, Completion())]
    assert caplog.text.count("OLLAMA_CONTEXT_LENGTH") == 1
