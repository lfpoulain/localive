import asyncio
import base64
import json
import time

import numpy as np
import pytest

from localive.asr import Transcript
from localive.config import Config
from localive.decision import Decision
from localive.live import (
    LOCAL_INSTRUCTIONS,
    MAX_TOOL_ROUNDS,
    TOOL_OUTPUT_LIMIT,
    WEB_ON,
    LiveSession,
    Services,
    Turn,
    Utterance,
    add_note,
    chat_tools,
    is_echo,
    is_goodbye,
    is_stop,
)
from localive import llm as live_errors
from localive.llm import ToolCall
from localive.vad import Segmenter
from tests.test_vad import FIXTURES, load_16k

CLOCK_TOOL = {
    "type": "function",
    "name": "clock__read",
    "description": "Lire l'heure.",
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
}
WEATHER_TOOL = {
    "type": "function", "name": "weather__read", "description": "Lire les prévisions d'une ville.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]},
}
SESSION = {
    "model": "gpt-live-1",
    "instructions": "Tu es Cornelius.",
    "audio": {"output": {"voice": "cedar"}, "format": {"type": "audio/pcm", "rate": 16000}},
    "delegation": {
        "type": "responses",
        "responses": {
            "instructions": "Utilise les outils.",
            "tools": [{"type": "web_search"}, CLOCK_TOOL],
        },
    },
}


@pytest.mark.parametrize("mode", ["slow", "fast", "interrupt", "off"])
async def test_wait_notice_only_during_slow_user_reply(client, monkeypatch, caplog, mode):
    from localive import live

    monkeypatch.setattr(live, "WAIT_NOTICE_SECONDS", .08)
    c = client(script=["Voilà la réponse."], tts_seconds=.03)
    c.session.config.voice.wait_notice = mode != "off"
    c.chat.delay = 0 if mode == "fast" else .40
    await c.session.start({"localive": {"metrics": True}})
    caplog.set_level("INFO", logger="localive.live")
    now = time.monotonic()
    await c.session.on_user_text("Quel temps fera-t-il ?", timings={
        "speech_end": now - .3, "transcript": now,
    })
    if mode == "interrupt":
        await asyncio.sleep(.02)
        await c.session.interrupt()
    await c.wait_turn()
    spoken = [text for text, voice in c.tts.texts]
    assert caplog.text.count("interjection prête") == int(mode == "slow")
    if mode == "slow":
        assert len(spoken) == 2 and spoken[0] in live.WAIT_NOTICES
        assert spoken[1] == "Voilà la réponse."
        assert c.of("localive.turn")[-1]["first_audio_ms"] >= 350
        assert spoken[0] not in str(c.session.history[-1].messages)
    elif mode == "interrupt":
        assert not spoken
    else:
        assert spoken == ["Voilà la réponse."]


async def test_wait_for_reply_holds_the_microphone_until_the_reply_speaks(client, monkeypatch):
    from localive import live

    monkeypatch.setattr(live, "WAIT_NOTICE_SECONDS", .05)
    c = client(script=["Voilà la réponse."], tts_seconds=.03)
    c.session.config.voice.wait_notice = True
    c.chat.delay = .40
    await c.session.start({"localive": {"wait_for_reply": True}})
    loud = np.sin(np.arange(16000) / 3).astype(np.float32) * .5
    held_during_notice = False
    now = time.monotonic()
    await c.session.on_user_text("Quel temps fera-t-il ?", timings={
        "speech_end": now - .3, "transcript": now,
    })
    assert c.session._mic_held
    for _ in range(100):
        await asyncio.sleep(.01)
        if c.tts.texts and not c.session._reply_spoke:
            held_during_notice = c.session._mic_held
            await c.session.on_audio(loud)  # a voice in the room while it thinks
            assert not c.session.segmenter.speaking
        if c.session._reply_spoke:
            break
    assert held_during_notice
    assert not c.session._mic_held  # the reply started: it can be interrupted
    await c.wait_turn()
    speaking = [e for e in c.of("voice.state") if e["state"] == "speaking"]
    assert [e["interjection"] for e in speaking] == [True, False]
    assert len(c.chat.calls) == 1


async def test_wait_for_reply_reopens_the_microphone_after_a_failed_turn(client):
    c = client(script=["Bonjour."])
    await c.session.start({"localive": {"wait_for_reply": True}})

    async def broken(*args, **kwargs):
        raise live_errors.LlmError("panne")
        yield  # pragma: no cover

    c.chat.stream = broken
    now = time.monotonic()
    await c.session.on_user_text("Allume la lumière.", timings={"transcript": now})
    await c.wait_turn()
    assert not c.session._mic_held


async def test_wait_for_reply_is_off_by_default_and_never_holds_for_an_announcement(client):
    c = client(script=["Bonjour."])
    await c.session.start({})
    await c.session.on_user_text("Bonjour.", timings={"transcript": time.monotonic()})
    assert not c.session._mic_held
    await c.wait_turn()
    await c.session.handle({"type": "session.update", "session": {"localive": {
        "wait_for_reply": True}}})
    await c.session.announce("L'agent a terminé.")
    assert not c.session._mic_held


def test_wait_notice_rotation(client, monkeypatch):
    from localive import live

    # Force a repeat at the boundary before the session applies its guard.
    orders = iter((False, True))

    def alternate_order(items):
        if next(orders):
            items.reverse()

    monkeypatch.setattr(live.random, "shuffle", alternate_order)
    session = client().session
    draws = [[session._next_wait_notice() for _ in live.WAIT_NOTICES] for _ in range(2)]
    for cycle in draws:
        assert len(set(cycle)) == len(live.WAIT_NOTICES)
        assert set(cycle) == set(live.WAIT_NOTICES)
    assert draws[0][-1] != draws[1][0]


class FakeChat:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.delay = 0.0  # seconds before the first token ("thinking")
        self.models = []

    async def stream(self, messages, tools, result, max_tokens=None, model=None, url=None):
        self.models.append(model)
        if max_tokens == 1:  # prompt-cache warm-up
            return
        self.calls.append((list(messages), tools))
        step = self.script.pop(0) if self.script else "D'accord."
        if self.delay:
            await asyncio.sleep(self.delay)
        if isinstance(step, list):
            result.tool_calls = [ToolCall(c.id, c.name, c.arguments) for c in step]
            return
        for word in step.split(" "):
            await asyncio.sleep(0)
            result.text += word + " "
            yield word + " "


class FakeTts:
    def __init__(self, seconds=0.3):
        self.texts = []
        self.seconds = seconds

    async def stream(self, text, voice):
        self.texts.append((text, voice))
        for _ in range(3):
            await asyncio.sleep(0)
            yield np.full(int(24000 * self.seconds / 3), 0.2, dtype=np.float32)


class FakeTranscriber:
    label = "fake"

    def __init__(self, text="Cornelius, quelle heure est-il ?"):
        self.text = text

    async def transcribe(self, audio):
        return Transcript(self.text, 0.01)


class Client:
    """A fake Cornelius: records events and answers tool calls like transport.ts."""

    def __init__(self, script=(), tts_seconds=0.3, heard="Cornelius, quelle heure est-il ?",
                 arbiter=None, turn=None):  # fmt: skip
        config = Config()
        config.echo.enabled = False  # fake TTS output is not speech: AEC is tested in test_echo
        self.events: list[dict] = []
        self.chat = FakeChat(script)
        self.tts = FakeTts(tts_seconds)
        services = Services(
            config=config, http=None, transcriber=FakeTranscriber(heard), chat=self.chat,
            tts=self.tts, make_segmenter=Segmenter, arbiter=arbiter, turn=turn,
        )  # fmt: skip
        self.session = LiveSession(services, self._receive)
        self.tool_answers = {"clock__read": {"status": "succeeded", "result": {"time": "14:05"}}}

    async def _receive(self, event):
        self.events.append(event)
        if event["type"] == "response.event":
            inner = event["event"]
            if (
                inner["type"] == "response.output_item.done"
                and inner["item"]["type"] == "function_call"
            ):
                item = inner["item"]
                output = json.dumps(self.tool_answers.get(item["name"], {"accepted": False}))
                reply = {
                    "type": "function_call_output",
                    "call_id": item["call_id"],
                    "output": output,
                }
                asyncio.get_running_loop().call_soon(
                    asyncio.ensure_future,
                    self.session.handle({"type": "response.item.create", "item": reply}),
                )

    def of(self, kind):
        return [e for e in self.events if e["type"] == kind]

    def audio_seconds(self):
        total = 0
        for e in self.of("session.output_audio.delta"):
            pcm = np.frombuffer(base64.b64decode(e["delta"]), "<i2")
            total += int(np.count_nonzero(np.abs(pcm) > 1000))
        return total / 16000

    async def wait_turn(self, seconds=5.0, *, count=1):
        for _ in range(int(seconds / 0.02)):
            if len(self.of("voice.turn_completed")) >= count:
                return
            await asyncio.sleep(0.02)
        raise AssertionError("no voice.turn_completed")


@pytest.fixture
async def client():
    made = []

    def make(*args, **kwargs):
        c = Client(*args, **kwargs)
        made.append(c)
        return c

    yield make
    for c in made:
        await c.session.close("test_end")


async def test_hosted_search_waits_for_startup_within_tool_budget(client, monkeypatch):
    from unittest.mock import AsyncMock

    from localive.live import ResponseDelegation
    from localive.searxng import SearxngManager

    c = client()
    session = c.session
    session.s.http = object()
    session.config.search.managed = False
    manager = session.s.search_manager = SearxngManager(session.config)
    release = asyncio.Event()

    async def probe(http, seconds):
        return await release.wait()

    monkeypatch.setattr(manager, "_wait_ready", probe)
    query = AsyncMock(return_value=[])
    monkeypatch.setattr("localive.live.search.searxng", query)
    session.configure({"tools": [CLOCK_TOOL]})
    assert manager.startup_task is None
    session.configure({"tools": [{"type": "web_search"}]}, update=True)
    assert manager.status()["state"] == "starting"
    session.config.search.timeout_s = 0.02
    call = ToolCall("w", "web_search", '{"query": "test"}')
    delegation = ResponseDelegation(session)
    await delegation.begin_response(first=True)
    result = await session._web_search(delegation, call)
    assert "error" in json.loads(result["w"])
    query.assert_not_awaited()
    assert not manager.startup_task.done()
    session.config.search.timeout_s = 1
    pending = asyncio.create_task(session._web_search(delegation, call))
    await asyncio.sleep(0)
    assert not pending.done()
    release.set()
    result = await pending
    assert json.loads(result["w"])["query"] == "test"
    query.assert_awaited_once()


async def test_start_announces_session_and_streams_silence(client):
    c = client()
    await c.session.handle({"type": "session.start", "session": SESSION})
    started = c.of("session.started")[0]
    assert isinstance(started["session"], dict) and started["session"]["id"]
    assert c.session.voice == "male"  # "cedar" -> LFM3 male voice
    assert [t["function"]["name"] for t in c.session.tools] == ["clock__read"]
    await asyncio.sleep(0.3)
    deltas = c.of("session.output_audio.delta")
    assert 5 <= len(deltas) <= 15  # ~real time (40 ms frames + 120 ms lead)
    assert all(len(base64.b64decode(d["delta"])) == 1280 for d in deltas)
    assert c.audio_seconds() == 0


async def test_spoken_question_with_tool_round_trip(client):
    c = client([[ToolCall("x", "clock__read", "{}")], "Il est quatorze heures cinq."])
    await c.session.handle({"type": "session.start", "session": SESSION})
    speech = load_16k(FIXTURES / "wake_clock.wav")
    silence = np.zeros(16000, dtype=np.float32)
    pcm = (np.concatenate([silence, speech, silence]) * 32767).astype("<i2").tobytes()
    for i in range(0, len(pcm), 2560):  # 80 ms frames, like a satellite
        await c.session.handle(
            {
                "type": "session.input_audio.append",
                "audio": base64.b64encode(pcm[i : i + 2560]).decode(),
            }
        )
    await c.wait_turn()

    heard = c.of("session.input_transcript.delta")[0]
    assert heard["delta"] == "Cornelius, quelle heure est-il ?"
    assert 500 <= heard["start_ms"] < heard["end_ms"] <= 1000 + len(speech) / 16 + 100

    envelopes = [e["event"] for e in c.of("response.event")]
    assert [e["type"] for e in envelopes] == [
        "response.created", "response.output_item.done", "response.completed",
        "response.created", "response.completed"]
    rid = envelopes[0]["response"]["id"]
    [delegated] = c.of("session.delegation.created")
    assert delegated["delegation"] == {
        "id": delegated["delegation_id"], "type": "delegation", "target": "responses",
        "response_id": rid,
    }
    assert delegated["event_id"] and delegated["offset_ms"] >= 0
    first_response = c.of("response.event")[0]
    assert c.events.index(heard) < c.events.index(delegated) < c.events.index(first_response)
    assert all(e["delegation_id"] == delegated["delegation"]["id"] for e in c.of("response.event"))
    assert envelopes[3]["response"]["id"] == envelopes[4]["response"]["id"] != rid
    assert [e["sequence_number"] for e in envelopes] == [0, 1, 2, 0, 1]
    assert envelopes[1]["response_id"] == rid == envelopes[2]["response"]["id"]
    call_id = envelopes[1]["item"]["call_id"]

    messages, tools = c.chat.calls[1]
    assert messages[-1] == {"role": "tool", "tool_call_id": call_id, "content": json.dumps(
        c.tool_answers["clock__read"])}  # fmt: skip
    assert messages[-2]["tool_calls"][0]["id"] == call_id
    assert (
        "Tu es Cornelius." in messages[0]["content"]
        and LOCAL_INSTRUCTIONS in messages[0]["content"]
    )

    assert c.tts.texts == [("Il est quatorze heures cinq.", "male")]
    spoken = c.of("session.output_transcript.delta")[0]
    assert spoken["delta"].strip() == "Il est quatorze heures cinq."
    assert spoken["end_ms"] - spoken["start_ms"] == pytest.approx(300, abs=5)
    states = [e["state"] for e in c.of("voice.state")]
    assert states[:3] == ["listening", "thinking", "speaking"] and states[-1] == "listening"
    assert c.audio_seconds() == pytest.approx(0.3, abs=0.05)


async def test_barge_in_flushes_audio_without_arbiter(client):
    c = client(["Il était une fois un très long conte."], tts_seconds=4.0, heard="Attends, stop.")
    await c.session.handle({"type": "session.start", "session": SESSION})
    await c.session.on_user_text("Raconte une histoire")
    for _ in range(50):
        await asyncio.sleep(0.02)
        if c.session.output.busy:
            break
    speech = load_16k(FIXTURES / "other_person.wav")
    for i in range(0, len(speech), 1280):
        await c.session.on_audio(speech[i : i + 1280])
        await asyncio.sleep(0)
    for _ in range(50):
        await asyncio.sleep(0.02)
        if c.of("voice.interrupted"):
            break
    assert c.of("voice.interrupted")
    await asyncio.sleep(0.3)
    assert not c.session.output.busy  # "Attends, stop." silences without a new answer
    assert len(c.tts.texts) == 1


async def test_hard_stop_keeps_only_fully_played_sentences_in_next_prompt(client):
    first = "Voici la première phrase."
    partial = "La deuxième continue encore."
    queued = "La troisième restera dans la file."
    c = client([[ToolCall("x", "clock__read", "{}")], f"{first} {partial} {queued}",
                "On reprend."], tts_seconds=0.5)
    await c.session.start(SESSION)
    await c.session.on_user_text("Quelle heure est-il ?")
    for _ in range(400):
        # The transport runs up to 120 ms ahead of the listener. At 800 ms of output
        # the first 500 ms sentence is over and the second is still playing.
        if c.audio_seconds() >= 0.8:
            break
        await asyncio.sleep(0.01)
    assert 0.8 <= c.audio_seconds() < 1.0
    await c.session.on_user_text("Stop.")
    assert c.of("voice.interrupted")
    await c.session.on_user_text("Où en étais-tu ?")
    await asyncio.wait_for(c.session._reply, 2)
    messages = c.chat.calls[-1][0]
    assert {"role": "assistant", "content": first} in messages
    assert partial not in str(messages) and queued not in str(messages)
    assert any(m.get("tool_calls") for m in messages) and any(m["role"] == "tool" for m in messages)


async def test_busy_commentary_is_delivered_in_order_once_without_user_prompt(client):
    c = client(["Une réponse en cours.", "Le minuteur est fini.", "Le four est prêt.",
                "Bonjour."], tts_seconds=0.2)
    await c.session.start(SESSION)
    await c.session.on_user_text("Parle-moi.")
    await speaking(c)
    for content in ("signal-minuteur", "signal-four"):
        await c.session.handle({"type": "session.commentary.append", "content": content})
    await c.wait_turn(count=3)
    assert [text for text, _ in c.tts.texts] == [
        "Une réponse en cours.", "Le minuteur est fini.", "Le four est prêt."
    ]
    assert "signal-minuteur" in str(c.chat.calls[1][0])
    assert "signal-four" in str(c.chat.calls[2][0])
    await c.session.on_user_text("Salut.")
    await asyncio.wait_for(c.session._reply, 2)
    assert "signal-" not in str(c.chat.calls[3][0])


async def test_echo_of_own_voice_does_not_interrupt(client):
    c = client(
        ["Il était une fois un petit chat."],
        tts_seconds=3.0,
        heard="il était une fois un petit chat",
    )
    await c.session.handle({"type": "session.start", "session": SESSION})
    await c.session.on_user_text("Raconte une histoire")
    await asyncio.sleep(0.2)
    speech = load_16k(FIXTURES / "other_person.wav")
    for i in range(0, len(speech), 1280):
        await c.session.on_audio(speech[i : i + 1280])
    await asyncio.sleep(0.2)
    assert not c.of("voice.interrupted")


async def test_close_and_thinking_ack(client):
    c = client()
    await c.session.handle({"type": "session.start", "session": SESSION})
    await c.session.handle(
        {"type": "session.thinking.append", "event_id": "cam1", "content": "Une personne sourit."}
    )
    assert c.of("session.thinking.appended")[0]["client_event_id"] == "cam1"
    assert "Une personne sourit." in c.session.context_note()  # per reply: cache-friendly
    assert "Une personne sourit." not in c.session.system_prompt()
    await c.session.handle({"type": "session.close"})
    assert c.of("session.closed")
    assert c.session.closed


async def test_session_update_replaces_tools_and_instructions(client):
    c = client()
    await c.session.handle({"type": "session.start", "session": SESSION})
    responses = {
        "instructions": "Nouveau profil.",
        "tools": [CLOCK_TOOL, {**CLOCK_TOOL, "name": "memory__recall"}],
    }
    update = {"delegation": {"type": "responses", "responses": responses}}
    await c.session.handle({"type": "session.update", "session": update})
    assert [t["function"]["name"] for t in c.session.tools] == ["clock__read", "memory__recall"]
    assert "Nouveau profil." in c.session.system_prompt()
    assert c.session.voice == "male"


@pytest.mark.parametrize("other_tools", [[], [CLOCK_TOOL]])
async def test_web_search_runs_locally_with_sources(client, monkeypatch, other_tools):
    from localive import search

    queries = []

    async def fake_searxng(http, config, query):
        queries.append(query)
        return [
            search.Result("Coupe du monde 2027", "https://fr.wikipedia.org/wiki/x", "En Australie.")
        ]

    monkeypatch.setattr(search, "searxng", fake_searxng)
    c = client([[ToolCall("x", "web_search", '{"query": "coupe du monde rugby 2027"}')],
                "Elle aura lieu en Australie."])  # fmt: skip
    c.session.s.http = object()  # any client: the search itself is faked
    await c.session.start({"tools": [{"type": "web_search"}, *other_tools]})
    assert [t["function"]["name"] for t in c.session.tools] == [
        *(t["name"] for t in other_tools), "web_search"
    ]
    await c.session.on_user_text("Où a lieu la coupe du monde de rugby 2027 ?")
    await c.wait_turn()
    assert queries == ["coupe du monde rugby 2027"]
    kinds = [e["event"]["type"] for e in c.of("response.event")]
    assert kinds == [
        "response.created", "response.output_item.added", "response.web_search_call.in_progress",
        "response.web_search_call.searching", "response.web_search_call.completed",
        "response.output_item.done", "response.output_item.done",
        "response.completed", "response.created", "response.completed",
    ]
    added = c.of("response.event")[1]["event"]["item"]
    assert added["type"] == "web_search_call" and added["id"].startswith("ws_")
    citation = next(e["event"]["item"] for e in c.of("response.event")
                    if e["event"].get("item", {}).get("type") == "message")
    note = citation["content"][0]["annotations"][0]
    assert note == {"type": "url_citation", "url": "https://fr.wikipedia.org/wiki/x",
                    "title": "Coupe du monde 2027"}  # fmt: skip
    tool_message = c.chat.calls[1][0][-1]
    assert tool_message["role"] == "tool" and "En Australie." in tool_message["content"]
    assert [t for t, _ in c.tts.texts] == ["Je regarde.", "Elle aura lieu en Australie."]
    assert WEB_ON in c.chat.calls[0][0][0]["content"]


async def test_delegation_model_and_opt_in_metrics(client):
    c = client(["Bonjour !"])
    session = {**SESSION, "localive": {"metrics": True}}
    session["delegation"] = {**SESSION["delegation"], "responses": {
        **SESSION["delegation"]["responses"], "model": "gpt-6-luna"}}  # fmt: skip
    await c.session.handle({"type": "session.start", "session": session})
    assert c.session.brain == "qwen" and c.session.metrics
    await c.session.on_user_text("Salut")
    await c.wait_turn()
    for _ in range(20):
        if c.of("localive.turn"):
            break
        await asyncio.sleep(0.02)
    turn = c.of("localive.turn")[0]
    assert turn["brain"] == "qwen" and turn["first_audio_ms"] is not None


async def test_metrics_are_never_sent_unless_asked(client):
    c = client(["Bonjour !"])
    await c.session.handle({"type": "session.start", "session": SESSION})
    await c.session.on_user_text("Salut")
    await c.wait_turn()
    await asyncio.sleep(0.1)
    assert not c.of("localive.turn")


class FakeArbiter:
    def __init__(self, label, confidence=1.0, question="Tu veux que j'allume la lumière ?"):
        self.label = label
        self.confidence = confidence
        self.seen = []
        self.decided = asyncio.Event()
        self.asked = []
        self.plans = []
        self.how = None  # plan to return (None: FIX for ADD, NOW for NEW)
        self._question = question

    async def decide(self, situation):
        self.seen.append(situation)
        self.decided.set()
        return Decision(self.label, self.confidence, {self.label: self.confidence}, 5.0)

    async def question(self, heard):
        self.asked.append(heard)
        return self._question

    async def plan(self, situation, remaining, generating):
        self.plans.append((situation.heard, remaining, generating))
        return self.how or ("FIX" if self.label == "ADD" else "NOW"), 5.0

    async def acknowledge(self, heard):
        return "Je te donne Nantes juste après."


async def speaking(c, seconds=2.0):
    for _ in range(int(seconds / 0.02)):
        await asyncio.sleep(0.02)
        if c.session.output.busy:
            return
    raise AssertionError("the assistant never started talking")


async def test_arbiter_ignore_keeps_talking(client, caplog):
    arbiter = FakeArbiter("IGNORE")
    c = client(["Il était une fois un très long conte."], tts_seconds=2.0, arbiter=arbiter)
    await c.session.handle({"type": "session.start", "session": SESSION})
    start = time.monotonic()
    caplog.set_level("INFO", logger="localive.live")
    await c.session.on_user_text("Raconte une histoire",
                                 timings={"speech_end": start - 0.4, "transcript": start})
    await speaking(c)
    c.session._utterances.put_nowait(Utterance(
        0, 1600, np.zeros(1600, dtype=np.float32), time.monotonic(),
        Transcript("Mm-hm.", 0.01), 300, False, True,
    ))
    await asyncio.wait_for(arbiter.decided.wait(), 2)
    assert arbiter.seen[0].speaking and arbiter.seen[0].request == "Raconte une histoire"
    assert not c.of("voice.interrupted") and c.session.output.busy
    assert len(c.chat.calls) == 1
    assert c.session._turn["speech_end"] == start - 0.4
    assert "premier audio prêt" in caplog.text
    assert not c.of("localive.turn")  # wire metrics are opt-in


async def test_add_while_speaking_continues_without_cutting(client):
    """ "Il fait beau à la porte Maillot ?" … « et Bordeaux ? » … the voice never stops."""
    script = [
        [ToolCall("x", "weather__read", '{"city": "Porte Maillot"}')],
        "Hier à Porte Maillot, il faisait neuf degrés avec des averses. Prépare-toi pour une "
        "journée humide, avec un ciel gris toute la journée.",
        "Et à Bordeaux, il fait doux et ensoleillé.",
    ]
    c = client(script, tts_seconds=1.5, arbiter=FakeArbiter("ADD"))
    c.tool_answers["weather__read"] = {"status": "succeeded", "result": {"c": 9}}
    await c.session.start({"tools": [WEATHER_TOOL]})
    await c.session.on_user_text("Il fait beau à la porte Maillot ?")
    await speaking(c)
    await c.session.on_user_text("Et Bordeaux ?")
    await c.wait_turn(seconds=8)
    assert not c.of("voice.interrupted")  # continuity: nothing was cut
    said = [e["delta"].strip() for e in c.of("session.output_transcript.delta")]
    assert said[0].startswith("Hier à Porte Maillot")
    assert "Prépare-toi" not in " ".join(said)  # the next sentence was dropped, not half-said
    assert said[-1] == "Et à Bordeaux, il fait doux et ensoleillé."
    last_call = c.chat.calls[-1][0]
    note = last_call[-1]["content"]
    assert last_call[-1]["role"] == "user" and "Hier à Porte Maillot" in note
    assert "Et Bordeaux ?" in note
    # The interrupted turn keeps what was actually said, and its tool result.
    assert {"role": "assistant", "content": said[0]} in last_call
    assert any(m["role"] == "tool" and json.loads(m["content"]) == c.tool_answers["weather__read"]
               for m in last_call)


async def test_addition_on_a_long_answer_is_answered_after_it(client):
    """7-day forecast for Bordeaux, « et Nantes ? »: finish Bordeaux, then Nantes."""
    bordeaux = ("Lundi à Bordeaux, il fera beau. Mardi, des nuages arriveront. Mercredi, il "
                "pleuvra un peu. Jeudi, retour du soleil. Vendredi, il fera doux.")  # fmt: skip
    arbiter = FakeArbiter("ADD")
    arbiter.how = "AFTER"
    c = client([bordeaux, "Pour Nantes, la semaine sera pluvieuse."], tts_seconds=0.6,
               arbiter=arbiter)  # fmt: skip
    await c.session.handle({"type": "session.start", "session": SESSION})
    await c.session.on_user_text("La météo de Bordeaux sur sept jours ?")
    await speaking(c)
    for _ in range(100):
        if c.session._reply_written:
            break
        await asyncio.sleep(0)
    await c.session.on_user_text("Et Nantes ?")
    await c.wait_turn(seconds=15)
    said = [e["delta"].strip() for e in c.of("session.output_transcript.delta")]
    assert not c.of("voice.interrupted")
    assert said[0] == "Lundi à Bordeaux, il fera beau."
    assert said[1] == "Je te donne Nantes juste après."  # right after the current sentence
    assert " ".join(said[2:-1]) == ("Mardi, des nuages arriveront. Mercredi, il pleuvra un peu. "
                                    "Jeudi, retour du soleil. Vendredi, il fera doux.")  # fmt: skip
    assert said[-1] == "Pour Nantes, la semaine sera pluvieuse."
    remaining = arbiter.plans[0][1]
    assert "Mardi" in remaining
    last = c.chat.calls[-1][0]
    assert last[-1]["role"] == "user" and last[-1]["content"].startswith("Et Nantes ?\n\n[")
    assert "juste après" in last[-1]["content"]


async def test_new_request_while_speaking_waits_for_the_end_of_the_sentence(client):
    c = client(["Une longue réponse qui parle. Et une deuxième phrase.", "Il est midi."],
               tts_seconds=1.0, arbiter=FakeArbiter("NEW"))  # fmt: skip
    await c.session.handle({"type": "session.start", "session": SESSION})
    await c.session.on_user_text("Explique la relativité")
    await speaking(c)
    await c.session.on_user_text("Quelle heure est-il ?")
    await c.wait_turn(seconds=6)
    assert not c.of("voice.interrupted")
    said = [e["delta"].strip() for e in c.of("session.output_transcript.delta")]
    assert said == ["Une longue réponse qui parle.", "Il est midi."]


@pytest.mark.parametrize("heard, label", [
    ("C’est bon.", "STOP"),
    ("C’est bon.", "IGNORE"),
    ("Ok, ok, c'est bon, merci, stop.", "STOP"),
    ("Je pense avoir tout ce qu'il me faut, on en reste là ?", "STOP"),
])
async def test_closure_depends_on_the_arbiter_and_stays_silent(client, heard, label):
    arbiter = FakeArbiter(label)
    c = client(["Une longue liste de légumes."], tts_seconds=4.0, arbiter=arbiter)
    await c.session.handle({"type": "session.start", "session": SESSION})
    await c.session.on_user_text("Donne cinquante légumes")
    await speaking(c)
    await c.session.on_user_text(heard)
    assert [s.heard for s in arbiter.seen] == [heard]
    assert not arbiter.plans
    assert bool(c.of("voice.interrupted")) == (label == "STOP")
    assert c.session.output.busy == (label == "IGNORE")
    await asyncio.sleep(0.1)
    assert len(c.chat.calls) == len(c.tts.texts) == 1


async def test_unsure_asks_a_yes_no_question(client):
    arbiter = FakeArbiter("NEW", confidence=0.4)
    c = client(["Une longue réponse qui parle."], tts_seconds=1.0, arbiter=arbiter)
    await c.session.handle({"type": "session.start", "session": SESSION})
    await c.session.on_user_text("Raconte une histoire")
    await speaking(c)
    await c.session.on_user_text("Allume la lumière.")
    await c.wait_turn(seconds=5)
    assert arbiter.asked == ["Allume la lumière."]
    assert c.tts.texts[-1][0] == "Tu veux que j'allume la lumière ?"
    assert c.session.history[-1].messages == [
        {"role": "user", "content": "Allume la lumière."},
        {"role": "assistant", "content": "Tu veux que j'allume la lumière ?"},
    ]  # the "oui" that follows reaches Qwen with this context


async def test_add_while_thinking_merges_the_request(client):
    c = client(["Il fera beau à Lyon."], arbiter=FakeArbiter("ADD"))
    c.chat.delay = 0.3  # Qwen still thinking when the user adds something
    await c.session.handle({"type": "session.start", "session": SESSION})
    await c.session.on_user_text("Quel temps fera-t-il demain ?")
    await asyncio.sleep(0.05)
    await c.session.on_user_text("À Lyon.")
    await c.wait_turn(seconds=5)
    last = c.chat.calls[-1][0][-1]
    last = {**last, "content": last["content"].split("\n\n[Contexte")[0]}
    assert last == {
        "role": "user",
        "content": "Quel temps fera-t-il demain ? À Lyon.",
    }


class GatedArbiter(FakeArbiter):
    """Holds its decision until released, to observe the voice while it listens."""

    def __init__(self, label):
        super().__init__(label)
        self.gate = asyncio.Event()

    async def decide(self, situation):
        await self.gate.wait()
        return await super().decide(situation)


async def test_speaking_over_lowers_the_voice_in_two_steps(client):
    arbiter = GatedArbiter("IGNORE")
    c = client(["Il était une fois un très long conte."], tts_seconds=4.0, heard="Euh attends",
               arbiter=arbiter)  # fmt: skip
    vad = c.session.config.vad
    await c.session.handle({"type": "session.start", "session": SESSION})
    await c.session.on_user_text("Raconte une histoire")
    await speaking(c)
    speech = load_16k(FIXTURES / "other_person.wav")
    levels = []
    for i in range(0, 16000, 1280):
        await c.session.on_audio(speech[i : i + 1280])
        await asyncio.sleep(0)
        levels.append(c.session.output.target_gain)
    await asyncio.sleep(0.05)
    assert vad.duck_early_level in levels  # lowered as soon as a voice is heard
    assert c.session.output.target_gain == vad.duck_level  # a person: much quieter
    arbiter.gate.set()
    await asyncio.sleep(0.1)
    assert c.session.output.target_gain == 1.0  # judged "not for me": full voice again
    assert not c.of("voice.interrupted")


async def test_own_echo_does_not_keep_the_voice_low(client):
    c = client(["Il était une fois un petit chat."], tts_seconds=3.0,
               heard="il était une fois un petit chat", arbiter=FakeArbiter("IGNORE"))  # fmt: skip
    await c.session.handle({"type": "session.start", "session": SESSION})
    await c.session.on_user_text("Raconte une histoire")
    await speaking(c)
    await asyncio.sleep(0.2)
    speech = load_16k(FIXTURES / "other_person.wav")
    for i in range(0, 16000, 1280):
        await c.session.on_audio(speech[i : i + 1280])
        await asyncio.sleep(0)
    await asyncio.sleep(0.1)
    assert c.session.output.target_gain == 1.0


def test_output_ducking_ramps_fast_down_slow_up():
    from localive.live import Output

    out = Output(None)
    out.push(np.full(16000, 0.5, dtype=np.float32))
    out.duck(0.12)
    frames = [out._take(640)[0] for _ in range(4)]
    assert abs(frames[1][-1] / 0.5 - 0.12) < 0.01  # reached in two 40 ms frames
    out.duck(1.0)
    up = [out._take(640)[0][-1] / 0.5 for _ in range(4)]
    assert up[0] < 0.3 and up[-1] < 0.7  # comes back gently


class FakeTurn:
    def __init__(self, probability):
        self.probability = probability

    def __call__(self, audio):
        return self.probability, 1.0


async def test_smart_turn_ends_the_turn_early(client):
    c = client(["D'accord."], turn=FakeTurn(0.95))
    await c.session.handle({"type": "session.start", "session": SESSION})
    speech = load_16k(FIXTURES / "wake_clock.wav")
    for i in range(0, len(speech), 1280):
        await c.session.on_audio(speech[i : i + 1280])
    silence = np.zeros(1280, dtype=np.float32)
    for _ in range(6):  # 480 ms: below the 1.5 s limit, above the 200 ms check
        await c.session.on_audio(silence)
        await asyncio.sleep(0.01)
    for _ in range(50):
        await asyncio.sleep(0.02)
        if c.of("session.input_transcript.delta"):
            break
    assert c.of("session.input_transcript.delta")
    assert c.session.segmenter.speaking is False


async def test_smart_turn_waits_when_unfinished(client):
    c = client(["D'accord."], turn=FakeTurn(0.1))
    await c.session.handle({"type": "session.start", "session": SESSION})
    speech = load_16k(FIXTURES / "wake_clock.wav")
    for i in range(0, len(speech), 1280):
        await c.session.on_audio(speech[i : i + 1280])
    silence = np.zeros(1280, dtype=np.float32)
    for _ in range(12):  # the clip ends with ~0.7 s of silence: ~1.7 s, past the old limits
        await c.session.on_audio(silence)
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.1)
    assert not c.of("session.input_transcript.delta")  # still waiting for the end
    for _ in range(30):  # past 3 s: the silence limit closes it anyway
        await c.session.on_audio(silence)
    for _ in range(50):
        await asyncio.sleep(0.02)
        if c.of("session.input_transcript.delta"):
            break
    assert c.of("session.input_transcript.delta")
    # Ended by the silence limit after Smart Turn said "not finished": Qwen is told so.
    await c.wait_turn()
    assert any(m["role"] == "user" and "inachevée" in m["content"] for m in c.chat.calls[-1][0])


@pytest.mark.parametrize("resumed", [True, False])
async def test_unfinished_transcript_reused_only_for_identical_microphone_range(client, resumed):
    from unittest.mock import AsyncMock

    from localive.vad import CHUNK
    from tests.test_vad import ScriptedVad

    c = client(turn=FakeTurn(0.1))
    session = c.session
    session.config.turn.check_ms = 128
    session.config.turn.max_silence_ms = 320
    first_pause = [1] * 6 + [0] * 6
    later = [1] * 2 + [0] * 10 if resumed else [0] * 4
    session.s.make_segmenter = lambda: Segmenter(ScriptedVad(first_pause + later))
    asr = AsyncMock(side_effect=[Transcript("Un minuteur pour", 0.1),
                                 Transcript("Un minuteur pour dix minutes.", 0.1)])
    session.s.transcriber.transcribe = asr
    await session.start({})
    await session.on_audio(np.zeros(len(first_pause) * CHUNK, dtype=np.float32))
    await session._turn_task
    assert asr.await_count == 1
    # Deliver the remaining audio in one packet, including the final silence timeout.
    await session.on_audio(np.zeros(len(later) * CHUNK, dtype=np.float32))
    await c.wait_turn()
    assert asr.await_count == (2 if resumed else 1)
    expected = "Un minuteur pour dix minutes." if resumed else "Un minuteur pour"
    assert c.of("session.input_transcript.delta")[0]["delta"] == expected


@pytest.mark.parametrize("resumed", [True, False])
async def test_smart_turn_never_commits_a_stale_partial_transcript(client, resumed):
    from types import SimpleNamespace

    c = client(turn=FakeTurn(0.95))
    segmenter = SimpleNamespace(
        speaking=True, utterance_start=0, pauses=1, silence_ms=224, voiced_ms=500,
        current_audio=lambda **kw: np.ones(1600, dtype=np.float32),
    )
    c.session.segmenter = segmenter
    started, release = asyncio.Event(), asyncio.Event()

    async def delayed_transcribe(audio):
        started.set()
        await release.wait()
        return Transcript("Je voudrais une pizza.", 0.1)

    c.session.s.transcriber.transcribe = delayed_transcribe
    checking = asyncio.create_task(c.session._check_end_of_turn((0, 1)))
    await started.wait()
    # pauses only increases when a NEW silence begins, not when voice resumes.
    segmenter.silence_ms = 0 if resumed else 256
    segmenter.voiced_ms = 532
    release.set()
    await checking
    assert c.session._utterances.empty()
    assert segmenter.speaking


async def test_smart_turn_failure_restores_silence_fallback(client):
    from types import SimpleNamespace

    def broken(audio):
        raise RuntimeError("inference unavailable")

    c = client(turn=broken)
    limits = []
    c.session.segmenter = SimpleNamespace(
        speaking=True, utterance_start=0, voiced_ms=500,
        current_audio=lambda **kw: np.ones(1600, dtype=np.float32),
        set_silence=limits.append,
    )
    await c.session._check_end_of_turn((0, 1))
    assert limits == [c.session.config.vad.silence_ms]
    assert c.of("voice.provider_warning")[0]["error"]["code"] == "turn_detection_failed"


async def test_output_does_not_burst_old_audio_after_stall():
    from localive.live import FRAME, RATE, Output

    frames = []

    async def send(event):
        frames.append(event)

    output = Output(send)
    output.started = time.monotonic() - 30  # simulate a blocked socket/event loop
    output.push(np.full(RATE * 2, 0.2, dtype=np.float32), sentence=1)
    runner = asyncio.create_task(output.run())
    await asyncio.sleep(0)
    runner.cancel()
    await asyncio.gather(runner, return_exceptions=True)
    assert 1 <= len(frames) <= 5
    assert output.queued == RATE * 2 - len(frames) * FRAME
    assert output.sent >= 29 * RATE


async def test_speech_generation_stays_close_to_playback(client):
    from localive.live import MAX_QUEUED_AUDIO_S, RATE

    class FastTts:
        closed = False
        started = 0

        async def stream(self, text, voice):
            self.started += 1
            try:
                for _ in range(3):
                    yield np.full(2400, 0.2, dtype=np.float32)
            finally:
                self.closed = True

    c = client(["Une phrase suffisamment longue pour être prononcée. " * 65])
    tts = c.session.s.tts = FastTts()
    await c.session.start(SESSION)
    await c.session.on_user_text("Parle longtemps")
    await asyncio.sleep(0.1)
    assert c.session.output.queued <= (MAX_QUEUED_AUDIO_S + 0.3) * RATE
    assert tts.started < 10
    assert not c.session._reply.done()  # no KeyError from pruning its queued sentences
    await c.session.interrupt()
    await asyncio.sleep(0.05)
    assert tts.closed
    assert not c.session.output.busy


async def test_session_close_awaits_dynamic_tasks(client):
    c = client()
    finished = asyncio.Event()

    async def pending():
        try:
            await asyncio.Event().wait()
        finally:
            finished.set()

    task = asyncio.create_task(pending())
    c.session._transcripts.add(task)
    await asyncio.sleep(0)
    await c.session.close("test")
    assert task.done() and finished.is_set()


async def test_turn_metrics_include_actual_endpoint_wait(client):
    c = client()
    c.session.metrics = True
    now = time.monotonic()
    await c.session._report_turn({"speech_end": now - 0.45, "transcript": now,
                                  "first_audio": now + 0.2})
    metrics = c.of("localive.turn")[0]
    assert metrics["endpoint_ms"] == 450
    assert metrics["audio_ready_latency_ms"] == 650


@pytest.mark.parametrize("phrase,ends", [
    ("OK, c’est bon, merci.", True), ("Merci beaucoup, au revoir Iris !", True),
    ("Iris, on va s’arrêter là.", True), ("Tu peux raccrocher ?", True),
    ("Merci !", False), ("C’est bon.", False), ("Stop.", False),
    ("Ne raccroche pas.", False), ("Comment on dit au revoir en allemand ?", False),
    ("Au revoir, mais avant ça donne-moi l’heure.", False), ("Julie, au revoir.", False),
])
def test_only_a_complete_farewell_ends_the_call(phrase, ends):
    assert is_goodbye(phrase, "Iris") is ends


async def test_goodbye_mode_can_be_configured_per_session(client):
    c = client()
    c.session.config.turn.end_on_goodbye = True
    await c.session.start({"localive": {"end_on_goodbye": False}})
    await c.session.on_user_text("Au revoir.")
    await c.wait_turn()
    assert not c.session.closed
    await c.session.handle({"type": "session.update", "session": {
        "localive": {"end_on_goodbye": True}}})
    await c.session.on_user_text("Tu peux raccrocher.")
    await c.session.close("already_closed")
    assert [e["reason"] for e in c.of("session.closed")] == ["user_goodbye"]
    assert len(c.chat.calls) == 1


async def test_goodbye_in_the_speakers_own_echo_keeps_session_open(client):
    c = client(tts_seconds=0.05)
    await c.session.start({"localive": {"end_on_goodbye": True}})
    c.session._spoken_recent = "En français, on dit au revoir."
    c.session.output.last_audible = time.monotonic()
    for start, words in [(0, "Au revoir."), (16000, "Et en anglais ?")]:
        c.session._utterances.put_nowait(Utterance(
            start, start + 16000, np.zeros(16000, dtype=np.float32), time.monotonic(),
            Transcript(words, 0.01), 300, False, False,
        ))
    await c.wait_turn()
    assert not c.session.closed
    assert [e["delta"] for e in c.of("session.input_transcript.delta")] == ["Et en anglais ?"]


@pytest.mark.parametrize("interrupt", [True, False], ids=["interrupted", "timeout"])
async def test_late_client_result_is_kept_for_the_next_question(client, interrupt):
    c = client([[ToolCall("a", "preview", "{}"), ToolCall("b", "palette", "{}")],
                "D'accord."], tts_seconds=0.05)
    calls = {}
    sent = asyncio.Event()

    async def receive(event):
        c.events.append(event)
        item = event.get("event", {}).get("item", {})
        if item.get("type") == "function_call":
            calls[item["name"]] = item["call_id"]
            if len(calls) == 2:
                sent.set()

    async def deliver(call_id, output):
        await c.session.handle({"type": "response.item.create", "item": {
            "type": "function_call_output", "call_id": call_id, "output": output}})

    c.session._raw_send = receive
    c.session.config.llm.tool_timeout_s = 60 if interrupt else 0.1
    tools = [{**CLOCK_TOOL, "name": name} for name in ("preview", "palette")]
    await c.session.start({"tools": tools})
    await c.session.on_user_text("Une illustration en couleurs.")
    await asyncio.wait_for(sent.wait(), 2)
    await deliver(calls["palette"], {"color": "blue"})
    if interrupt:
        await c.session.interrupt()
    else:
        await c.wait_turn()
    replies = len(c.chat.calls)
    await deliver(calls["preview"], {"image": "ready"})
    await deliver(calls["preview"], "duplicate")
    await deliver("unknown", "unrelated")
    assert len(c.chat.calls) == replies
    await c.session.on_user_text("Est-elle prête ?")
    await asyncio.wait_for(c.session._reply, 2)
    outputs = {m["tool_call_id"]: json.loads(m["content"]) for m in c.chat.calls[-1][0]
               if m["role"] == "tool"}
    assert outputs == {calls["preview"]: {"image": "ready"}, calls["palette"]: {"color": "blue"}}


async def test_closure_still_classified_if_reply_ended_during_utterance(client):
    arbiter = FakeArbiter("STOP")
    c = client(["Une liste de légumes."], tts_seconds=0.6,
               heard="On a fait le tour, je crois.", arbiter=arbiter)  # fmt: skip
    c.session.config.vad.barge_in_ms = 100_000  # exercise end-of-turn, not the early check
    await c.session.handle({"type": "session.start", "session": SESSION})
    await c.session.on_user_text("Donne cinquante légumes")
    await speaking(c)
    speech = load_16k(FIXTURES / "wake_clock.wav")
    await c.session.on_audio(speech[:16000])
    await c.wait_turn()  # the assistant finishes while the microphone utterance is still open
    assert not arbiter.seen
    await c.session.on_audio(np.concatenate([speech[16000:], np.zeros(16000, dtype=np.float32)]))
    for _ in range(50):
        if arbiter.seen:
            break
        await asyncio.sleep(0.02)
    assert [s.heard for s in arbiter.seen] == ["On a fait le tour, je crois."]
    assert len(c.chat.calls) == len(c.tts.texts) == 1


@pytest.mark.parametrize("text", [
    "C'est bon pour la santé ?", "C'est bon ?", "Ok, c'est bon, et demain ?",
    "Stop le minuteur des pâtes", "Arrête la liste et dis-moi l'heure", "Ok, merci.",
])
def test_stop_does_not_swallow_questions_or_other_requests(text):
    assert not is_stop(text)


def test_ends_open():
    from localive.turn import ends_open

    assert ends_open("Mets un minuteur de") and ends_open("Je voudrais, euh…")
    assert not ends_open("Quelle heure est-il ?")


def test_strip_echo_prefix():
    from localive.live import strip_echo_prefix

    spoken = "Voici 50 mots : maison, lumière, soleil, vent, eau, arbre, table, chaise,"
    assert strip_echo_prefix("table, chaise. Coupe la liste après trois mots.", spoken) == (
        "Coupe la liste après trois mots."
    )
    assert strip_echo_prefix("Coupe la liste.", spoken) == "Coupe la liste."
    assert (
        strip_echo_prefix("sûr. Arrête la liste.", "Bien sûr. Voici 50 mots") == "Arrête la liste."
    )
    assert strip_echo_prefix("table chaise", spoken) == "chaise"  # never empty
    spoken += " et une table"
    assert strip_echo_prefix("chaise. Et Nantes ?", spoken) == "Et Nantes ?"
    assert strip_echo_prefix("Et une autre liste.", spoken) == "Et une autre liste."


@pytest.mark.parametrize("heard", ["C'est bon merci.", "C'est bon. Merci beaucoup.", "Non merci."])
def test_short_closure_is_not_echo_from_shared_politeness_words(heard):
    from localive.live import strip_echo_prefix

    spoken = "Merci beaucoup pour votre question. C'est un bon exemple. Voici la suite."
    assert strip_echo_prefix(heard, spoken) == heard
    assert not is_echo(heard, spoken)
    assert is_echo("Merci beaucoup", spoken)  # a verbatim short echo is still recognized


def test_echo_prefix_with_omitted_articles_preserves_following_request():
    from localive.live import strip_echo_prefix

    spoken = "Voici des objets : une table, une chaise, une lampe, une porte."
    heard = "Table, chaise, lampe. Ferme la porte."
    assert not is_echo(strip_echo_prefix(heard, spoken), spoken)


@pytest.mark.parametrize("heard", ["C'est bon merci.", "C'est bon. Merci beaucoup."])
async def test_short_closure_reaches_transcript_and_arbiter_after_audio(client, heard):
    arbiter = FakeArbiter("STOP")
    c = client(["Une longue liste de légumes."], tts_seconds=4.0, arbiter=arbiter)
    await c.session.handle({"type": "session.start", "session": SESSION})
    await c.session.on_user_text("Donne cinquante légumes")
    await speaking(c)
    c.session._spoken_recent = "Merci beaucoup pour votre question. C'est un bon exemple."
    c.session._over_utterance = 0
    c.session.output.last_audible = time.monotonic()
    c.session._utterances.put_nowait(Utterance(
        0, 16000, np.zeros(16000, dtype=np.float32), time.monotonic(),
        Transcript(heard, 0.01), 300, False, True,
    ))
    for _ in range(50):
        if c.of("voice.interrupted"):
            break
        await asyncio.sleep(0.01)
    assert [event["delta"] for event in c.of("session.input_transcript.delta")] == [heard]
    assert [situation.heard for situation in arbiter.seen] == [heard]
    assert c.of("voice.interrupted")
    assert len(c.chat.calls) == 1  # closure stays silent rather than starting another answer


async def test_any_delegation_model_is_answered_by_the_one_brain(client):
    c = client(["Bonjour !"])
    session = {**SESSION}
    session["delegation"] = {**SESSION["delegation"], "responses": {
        **SESSION["delegation"]["responses"], "model": "gpt-6-luna"}}
    await c.session.handle({"type": "session.start", "session": session})
    assert c.session.brain == "qwen"
    assert c.session.brain_url == c.session.config.llm.url


def test_tool_call_written_as_text_is_recovered():
    from localive.llm import text_tool_call

    names = {"clock__read", "lights__set"}
    call = text_tool_call('lights__set{on:true,room:<|"|>cuisine<|"|>}', names)
    assert call.name == "lights__set" and json.loads(call.arguments) == {
        "on": True,
        "room": "cuisine",
    }
    assert text_tool_call("clock__read()", names).arguments == "{}"
    assert text_tool_call("Il est midi.", names) is None
    assert text_tool_call("unknown()", names) is None


async def test_truncated_written_call_never_reaches_cornelius(client, monkeypatch):
    c = client()

    async def stream(messages, tools, completion, **kwargs):
        completion.text, completion.truncated = "clock__read()", True
        yield completion.text

    monkeypatch.setattr(c.chat, "stream", stream)
    await c.session.start(SESSION)
    await c.session.on_user_text("Quelle heure est-il ?")
    await c.wait_turn()
    assert [e["event"]["type"] for e in c.of("response.event")] == [
        "response.created", "response.failed"]
    assert c.of("voice.provider_warning")[-1]["error"]["code"] == "engine_unavailable"


def test_helpers():
    assert is_stop("Attends, stop.") and is_stop("Chut Cornelius !")
    assert not is_stop("Stop le minuteur des pâtes")
    assert chat_tools([{"type": "web_search"}, CLOCK_TOOL])[0]["function"]["name"] == "clock__read"
    spoken = "Il fait sept degrés à Lyon avec un ciel couvert."
    assert is_echo("il fait sept degrés à Lyon", spoken)
    assert not is_echo("Cornelius, et demain ?", spoken)
    story = "La tour Eiffel a été construite pour l'Exposition universelle de 1889 par la société"
    assert not is_echo("Allume la lumière de l'entrée.", story)  # shared words: la, de, l'


async def test_tool_loop_ends_with_a_spoken_answer_without_tools(client):
    loop = [[ToolCall("x", "clock__read", "{}")] for _ in range(MAX_TOOL_ROUNDS)]
    c = client([*loop, "Il est quatorze heures cinq."])
    await c.session.handle({"type": "session.start", "session": SESSION})
    await c.session.on_user_text("Quelle heure est-il ?")
    await c.wait_turn()
    assert len(c.chat.calls) == MAX_TOOL_ROUNDS + 1
    messages, tools = c.chat.calls[-1]
    assert tools is None  # the last round cannot call a tool again
    assert "N'appelle plus d'outil" in messages[-1]["content"]
    assert any("quatorze" in text for text, _ in c.tts.texts)


async def test_huge_tool_output_is_capped_in_the_conversation(client):
    c = client([[ToolCall("x", "clock__read", "{}")], "Voilà."])
    c.tool_answers["clock__read"] = {"result": "x" * (TOOL_OUTPUT_LIMIT * 3)}
    await c.session.handle({"type": "session.start", "session": SESSION})
    await c.session.on_user_text("Quelle heure est-il ?")
    await c.wait_turn()
    messages, _ = c.chat.calls[1]
    assert len(messages[-1]["content"]) < TOOL_OUTPUT_LIMIT + 100
    assert messages[-1]["content"].endswith("[résultat tronqué]")


async def test_old_turns_are_dropped_to_fit_the_context_window(client):
    c = client(["D'accord."])
    c.session.config.brain.context = 4096
    await c.session.handle({"type": "session.start", "session": SESSION})
    filler = "mot " * 2000  # ~2 700 estimated tokens per turn
    c.session.history = [Turn([{"role": "user", "content": f"{i} {filler}"}]) for i in range(5)]
    await c.session.on_user_text("Et maintenant ?")
    await c.wait_turn()
    messages, _ = c.chat.calls[0]
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"].startswith("Et maintenant ?")
    assert not any(m["content"].startswith("0 ") for m in messages[1:] if m["role"] == "user")


def test_reply_notes_never_add_a_second_system_message():
    """Qwen3.5 and Gemma templates raise on a system message that is not first (HTTP 500)."""
    history_user = {"role": "user", "content": "Raconte une histoire."}
    messages = [{"role": "system", "content": "Tu es Cornelius."}, history_user]
    add_note(messages, "Réponds en anglais.")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[1]["content"].startswith("Raconte une histoire.\n\n[")
    assert history_user["content"] == "Raconte une histoire."  # history is not modified
    after_tool = [{"role": "system", "content": "x"}, {"role": "tool", "tool_call_id": "a",
                                                      "content": "{}"}]
    add_note(after_tool, "N'appelle plus d'outil.")
    assert [m["role"] for m in after_tool] == ["system", "tool", "user"]


async def test_correction_while_speaking_sends_one_system_message_only(client):
    c = client(["Il était une fois un dragon. Il vivait dans une grotte. Il aimait les livres.",
                "Once upon a time, a dragon."], tts_seconds=1.0, arbiter=FakeArbiter("ADD"))
    await c.session.handle({"type": "session.start", "session": SESSION})
    await c.session.on_user_text("Raconte une histoire.")
    await speaking(c)
    await c.session.on_user_text("Tu peux le faire en anglais ?")
    await c.wait_turn(seconds=8)
    for messages, _ in c.chat.calls:
        assert [m["role"] for m in messages].count("system") == 1
        assert messages[0]["role"] == "system"
    assert "en anglais" in c.chat.calls[-1][0][-1]["content"]


class CountingTurn(FakeTurn):
    def __init__(self, probability):
        super().__init__(probability)
        self.calls = 0

    def __call__(self, audio):
        self.calls += 1
        return super().__call__(audio)


async def test_speech_over_the_assistant_ends_on_words_not_smart_turn(client):
    """Over its own voice Smart Turn says "not finished" (3 s wait on every correction)."""
    turn = CountingTurn(0.1)  # would keep the sentence open until the 3 s silence limit
    c = client(["Il était une fois un dragon qui vivait dans une grotte très profonde."],
               tts_seconds=8.0, arbiter=FakeArbiter("IGNORE"), turn=turn)
    await c.session.handle({"type": "session.start", "session": SESSION})
    await c.session.on_user_text("Raconte une histoire.")
    await speaking(c)
    speech = load_16k(FIXTURES / "wake_clock.wav")  # ends with ~0.7 s of silence
    for i in range(0, len(speech), 1280):
        await c.session.on_audio(speech[i : i + 1280])
    silence = np.zeros(1280, dtype=np.float32)
    for _ in range(4):  # ~1 s of silence in all: far below the 3 s limit
        await c.session.on_audio(silence)
        await asyncio.sleep(0.01)
    for _ in range(50):
        await asyncio.sleep(0.02)
        if c.of("session.input_transcript.delta"):
            break
    assert c.of("session.input_transcript.delta")
    assert turn.calls == 0  # the mixed voices never reached Smart Turn


def test_correction_note_asks_to_redo_the_whole_answer():
    from localive.live import FIX_NOTE

    assert "refais toute la réponse" in FIX_NOTE and "sans annoncer" in FIX_NOTE


def test_system_prompt_is_stable_and_adapted_for_one_local_model(monkeypatch):
    from localive.live import adapt_instructions

    live = "\n".join([
        "Tu es Cornelius.",
        "Pendant l’attente d’un résultat : remplis le silence.",
        "Interruption policy: Arrête de parler.",
        "Delegation policy:",
        "Backend tools: x.",
        "Delegate to the backend when: y.",
        "Do not delegate to the backend when: z.",
        "Ne parle jamais de tes outils.",
    ])
    adapted = adapt_instructions(live, "Tu aides Cornelius. Utilise clock__read pour l’heure.")
    assert "Tu es Cornelius." in adapted and "Ne parle jamais de tes outils." in adapted
    for gone in ("Pendant l’attente", "Interruption policy", "Delegation policy", "Backend",
                 "delegate", "Tu aides Cornelius"):
        assert gone not in adapted
    assert "Utilise clock__read" in adapted


async def test_public_session_keeps_custom_instructions_and_only_mentions_available_tools(client):
    c = client()
    instructions = "Tu es le guide du musée.\nBackend tools are described by the curator."
    c.session.configure({"instructions": instructions, "tools": []})
    prompt = c.session.system_prompt()
    assert instructions in prompt
    assert "memory__recall" not in prompt and "web_search" not in prompt
    c.session.configure({"tools": [{**CLOCK_TOOL, "name": "memory__recall"}]}, update=True)
    assert "memory__recall" in c.session.system_prompt()
    c.session.configure({"tools": []}, update=True)
    assert c.session.system_prompt() == prompt


async def test_clock_and_camera_never_change_the_cached_system_prompt(client):
    import datetime as real_dt

    c = client(["D'accord."])
    session = dict(SESSION, delegation={"type": "responses", "responses": {"tools": []}})
    await c.session.handle({"type": "session.start", "session": session})

    class Later(real_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return real_dt.datetime(2030, 1, 1, 23, 59).astimezone()

    import localive.live as live_module

    live_module.dt.datetime = Later
    try:
        await c.session.handle({"type": "session.thinking.append", "event_id": "c",
                                "content": "Caméra locale : une tasse rouge."})
        assert "23 h 59" not in c.session.system_prompt()
        assert "23 h 59" in c.session.context_note()
        assert "tasse rouge" in c.session.context_note()
    finally:
        live_module.dt.datetime = real_dt.datetime


async def test_external_brain_receives_its_own_model_id():
    from localive.config import LlmConfig
    from localive.llm import ChatClient

    seen = {}

    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        @property
        def content(self):
            async def lines():
                yield b"data: [DONE]\n"
            return lines()

    class Http:
        def post(self, url, json, **_):
            seen.update(json)
            return Response()

    client = ChatClient(Http(), LlmConfig(upstream_model="qwen3:8b"))
    from localive.llm import Completion
    async for _ in client.stream([{"role": "user", "content": "x"}], None, Completion(),
                                 model="qwen"):
        pass
    assert seen["model"] == "qwen3:8b"
