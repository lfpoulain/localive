"""Tool routing in real Live turns, with inference and Cornelius' tools simulated."""

import asyncio
import json
import os
import time
from unittest.mock import AsyncMock, Mock

import aiohttp
import pytest

from localive import install, live, server
from localive.asr import Transcript
from localive.config import DEFAULT_ROUTER, Config, RoutingConfig
from localive.live import chat_tools, warm_from_saved
from localive.llm import ToolCall
from localive.routing import ToolRouter, should_route_tools, thanks_kind
from localive.server import App
from tests.test_live import CLOCK_TOOL, WEATHER_TOOL, Client
from tests.test_warm_prefix import FakeChat as WarmChat

TRANSIT = {**CLOCK_TOOL, "name": "transit__read", "description": "Read the next bus."}
WEATHER_WITH_STATIONS = {
    **WEATHER_TOOL,
    "parameters": {
        **WEATHER_TOOL["parameters"],
        "properties": {
            **WEATHER_TOOL["parameters"]["properties"],
            "station": {"type": "string", "enum": [f"station_{i:03}" for i in range(256)]},
        },
    },
}


@pytest.fixture
async def client(tmp_path):
    c = Client(tts_seconds=0.05)
    c.session.config.root = tmp_path
    c.session.s.router = AsyncMock()
    try:
        yield c
    finally:
        await c.session.close("test_end")


def names(tools):
    return [tool["function"]["name"] for tool in tools]


class Response:
    def __init__(self, answers):
        self.answers = answers

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass

    def raise_for_status(self):
        pass

    async def json(self):
        return {"answers": self.answers}


class SystemOne:
    def __init__(self, *answers):
        self.answers = iter(answers)
        self.bodies = []

    def post(self, url, json):
        assert url == "http://127.0.0.1:9001/v1/systemone"
        self.bodies.append(json)
        return Response(next(self.answers))


def choice(name, confidence):
    return {"route": {"choice": name, "confidence": confidence}}


def test_catalogue_size_and_count_thresholds():
    tools = chat_tools([{**CLOCK_TOOL, "name": f"clock_{i}"} for i in range(6)])
    assert not should_route_tools([])
    assert not should_route_tools(chat_tools([WEATHER_WITH_STATIONS]))
    assert not should_route_tools(tools[:5])
    assert should_route_tools(tools)

    pair = tools[:2]
    function = pair[0]["function"]
    function["description"] = ""
    size = len(json.dumps(pair, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    accents, leftover = divmod(3071 - size, 2)
    function["description"] = "é" * accents + "a" * leftover
    assert not should_route_tools(pair)
    function["description"] += "a"
    assert should_route_tools(pair)


@pytest.mark.parametrize("answer", [{"choice": "outside", "confidence": 1.0},
                                     {"choice": "hello", "confidence": float("nan")}])
async def test_router_probe_requires_a_valid_systemone_choice(answer):
    http = SystemOne({"check": answer}, {"check": {"choice": "hello", "confidence": 0.9}})
    router = ToolRouter(http, RoutingConfig(port=9001))
    with pytest.raises(ValueError):
        await router.verify()
    await router.verify()
    assert router.timeout_ms == 8000  # the tiny probe does not set the first real turn's budget


@pytest.mark.parametrize(("answer", "expected"), [
    ({"choice": "hang_up", "probabilities": {"hang_up": 0.8}}, True),
    ({"choice": "hang_up", "probabilities": {"hang_up": 0.79}}, False),
    ({"choice": "pending", "probabilities": {"hang_up": 0.1}}, False),
    ({"choice": "other", "probabilities": {"hang_up": 0.99}}, False),
    ({"choice": "hang_up", "probabilities": {"hang_up": float("nan")}}, False),
    ({"choice": "hang_up", "probabilities": {"hang_up": True}}, False),
    ({"choice": "hang_up", "confidence": 0.99}, False),
])
async def test_d3_goodbye_requires_a_confident_immediate_farewell(answer, expected):
    http = SystemOne({"goodbye": answer})
    router = ToolRouter(http, RoutingConfig(port=9001))
    phrase = "Je file, on se reparle demain."
    assert await router.should_hang_up(phrase, "Iris") is expected
    assert http.bodies[0]["state"] == {"assistant_name": "Iris", "user_utterance": phrase}
    assert router.timeout_ms == 8000  # hangups do not calibrate the tool selection budget
    assert not await router.should_hang_up("Au revoir." + "x" * 2000, "Iris")
    assert len(http.bodies) == 1


async def test_a_known_farewell_still_needs_the_model_to_agree():
    answers = [("hang_up", 0.74), ("hang_up", 0.74), ("hang_up", 0.69), ("other", 0.74)]
    http = SystemOne(*({"goodbye": {"choice": c, "probabilities": {"hang_up": p}}}
                       for c, p in answers))
    router = ToolRouter(http, RoutingConfig(port=9001))
    assert not await router.should_hang_up("Je pars.", "Iris")
    for expected in (True, False, False):
        assert await router.should_hang_up("Au revoir.", "Iris", known_phrase=True) is expected


@pytest.mark.parametrize("count", [0, 1, 3])
async def test_small_catalogue_keeps_all_tools_after_one_short_d3_question(client, count):
    session = client.session
    http = SystemOne({"intent": {"noul": 0.199}})
    session.s.router = ToolRouter(http, RoutingConfig(port=9001))
    await session.start({"tools": [CLOCK_TOOL, WEATHER_TOOL, TRANSIT][:count],
                         "localive": {"end_on_goodbye": True, "goodbye_mode": "d3"}})
    await session.on_user_text("Quelle heure est-il ?")
    await client.wait_turn()
    assert [list(body["questions"]) for body in http.bodies] == [["intent"]]
    assert (client.chat.calls[0][1] or []) == session.tools
    assert not session.closed


@pytest.mark.parametrize(("score", "decision", "expected"), [
    (0.20, "hang_up", True), (0.99, "pending", False), (float("nan"), "pending", False),
])
async def test_short_question_can_only_rule_out_a_goodbye(score, decision, expected):
    http = SystemOne({"intent": {"noul": score}},
                     {"goodbye": {"choice": decision, "probabilities": {"hang_up": 0.95}}})
    router = ToolRouter(http, RoutingConfig(port=9001))
    assert await router.should_hang_up("On se reparle.", "Iris", screen=True) is expected
    assert len(http.bodies) == 2


@pytest.mark.parametrize(("known", "budget"), [(True, 800), (False, 400)])
async def test_confirmation_gets_priority_for_known_goodbyes_and_short_budgets(known, budget):
    http = SystemOne({"goodbye": {"choice": "hang_up", "probabilities": {"hang_up": 0.95}}})
    router = ToolRouter(http, RoutingConfig(port=9001, timeout_ms=budget))
    assert await router.should_hang_up("Au revoir.", "Iris", known_phrase=known, screen=True)
    assert [list(body["questions"]) for body in http.bodies] == [["goodbye"]]


@pytest.mark.skipif(not os.getenv("LOCALIVE_TEST_D3_PORT"), reason="requires local d3-mini")
async def test_native_d3_distinguishes_signoffs_from_pending_requests():
    cases = [
        ("OK, c’est bon, merci.", True),
        ("Je file, on se reparle demain.", True),
        ("Tu peux raccrocher maintenant.", True),
        ("Cornelius, on va s’arrêter là.", True),
        ("Bah on peut s'arrêter là.", True),
        ("On peut arrêter la session, c'est bon.", True),
        ("Tu t'es pas arrêté là.", False),
        ("Merci.", False),
        ("C’est bon.", False),
        ("Ne raccroche pas.", False),
        ("Comment on dit au revoir en japonais ?", False),
        ("Tu peux raccrocher après m’avoir donné la météo ?", False),
        ("Julie, à la prochaine !", False),
    ]
    config = RoutingConfig(port=int(os.environ["LOCALIVE_TEST_D3_PORT"]))
    async with aiohttp.ClientSession() as http:
        router = ToolRouter(http, config)
        await router.verify()
        for phrase, expected in cases:
            known = live.is_goodbye(phrase, "Cornelius")
            hang_up = await router.should_hang_up(phrase, "Cornelius", known_phrase=known)
            assert hang_up is expected, phrase


async def test_hangup_timeout_keeps_the_session_open_and_cancellation_propagates():
    router = ToolRouter(None, RoutingConfig(timeout_ms=15))

    async def stall(*_):
        await asyncio.Future()

    router._post = stall
    assert not await asyncio.wait_for(router.should_hang_up("À bientôt.", "Iris"), 1)
    router.config.timeout_ms = 800
    task = asyncio.create_task(router.should_hang_up("À bientôt.", "Iris"))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_goodbye_starts_shared_model_without_enabling_tool_selection(client, monkeypatch):
    s = client.session
    s.config.routing.enabled = False
    s.config.routing.model = "d3.gguf"
    s.config.path(s.config.routing.model).touch()
    app = App(s.config)
    app.services = s.s
    s.s.router = None
    s.s.request_router = app.request_router
    start = AsyncMock()
    monkeypatch.setattr("localive.server.Brain.start", start)
    monkeypatch.setattr(ToolRouter, "verify", AsyncMock())
    decide = AsyncMock(side_effect=[False, True])
    select = AsyncMock()
    monkeypatch.setattr(ToolRouter, "should_hang_up", decide)
    monkeypatch.setattr(ToolRouter, "select", select)
    await s.start({})
    assert app._router_task is None
    s.configure({"localive": {"end_on_goodbye": True, "goodbye_mode": "d3"}}, update=True)
    await app._router_task
    health = json.loads((await app.health(None)).body)
    assert health["decision_model_state"] == "ready" and health["tool_router_state"] == "off"
    s.configure({"tools": [CLOCK_TOOL, WEATHER_WITH_STATIONS]}, update=True)
    assert s.s.warm_tools(s.tools) == s.tools
    await s.on_user_text("Et la météo demain ?")
    await client.wait_turn()
    assert client.chat.calls[0][1] == s.tools
    await s.on_user_text("Je file, on se reparle demain.")
    assert [e["reason"] for e in client.of("session.closed")] == ["user_goodbye"]
    start.assert_awaited_once()
    select.assert_not_awaited()
    assert decide.await_count == 2


@pytest.mark.parametrize("interrupt", ["disable", "change_mode", "speech", "queued_speech"])
async def test_pending_hangup_cannot_close_after_new_speech_or_option_change(client, interrupt):
    s = client.session
    s.configure({"localive": {"end_on_goodbye": True, "goodbye_mode": "d3"}})
    s.segmenter = s.s.make_segmenter()
    started, release = asyncio.Event(), asyncio.Event()

    async def delayed(*_, **options):
        started.set()
        await release.wait()
        return True

    s.s.router.should_hang_up.side_effect = delayed
    decision = asyncio.create_task(s.on_user_text("On s'en tient là pour aujourd'hui."))
    await asyncio.wait_for(started.wait(), 1)
    if interrupt == "speech":
        s.segmenter.speaking = True
    elif interrupt == "queued_speech":
        # A second utterance ended while the ASR worker was awaiting this decision.
        s._utterances.put_nowait(Mock(spec=live.Utterance))
    else:
        options = ({"end_on_goodbye": False} if interrupt == "disable"
                   else {"goodbye_mode": "phrases"})
        s.configure({"localive": options}, update=True)
    release.set()
    await decision
    assert not s.closed


async def test_unavailable_d3_does_not_downgrade_to_phrase_hangup(client):
    s = client.session
    await s.start({"localive": {"end_on_goodbye": True, "goodbye_mode": "d3"}})
    router, s.s.router = s.s.router, None
    await s.on_user_text("Au revoir.")
    await client.wait_turn()
    s.s.router = router
    await s.on_user_text("Stop.")
    await s.on_user_text("Je vais te laisser…", unfinished=True)
    router.should_hang_up.assert_not_awaited()
    assert not s.closed


async def test_automatic_deadline_adapts_to_real_cpu_and_gpu_responses(monkeypatch):
    http = SystemOne(*(choice("direct", 0.99) for _ in range(3)))
    config = RoutingConfig(port=9001)
    router = ToolRouter(http, config)
    tools = chat_tools([CLOCK_TOOL, WEATHER_TOOL])
    clock = iter([0, 0.02, 1, 2.8, 3, 3.02])
    monkeypatch.setattr("localive.routing.monotonic", lambda: next(clock))
    assert router.timeout_ms == 8000
    assert await router.select(tools, "Merci") == []
    assert router.timeout_ms == 400
    assert await router.select(tools, "Merci") == []
    assert router.timeout_ms == pytest.approx(7200)
    config.timeout_ms = 150
    assert await router.select(tools, "Merci") == []
    assert router.timeout_ms == 150
    config.timeout_ms = 0
    assert router.timeout_ms == pytest.approx(5760)


@pytest.mark.parametrize(("decision", "indexes"), [
    (choice("direct", 0.98), []),
    (choice("t1", 0.98), [1]),
    (choice("unsure", 0.98), [0, 1]),
    (choice("direct", 0.3), [0, 1]),
    (choice("t0", 0.3), [0, 1]),
    (choice("t77", 0.98), [0, 1]),
    (choice("t0", float("nan")), [0, 1]),
    ({}, [0, 1]),
])
async def test_short_router_request_and_confident_or_ambiguous_choice(decision, indexes):
    http = SystemOne(decision)
    router = ToolRouter(http, RoutingConfig(port=9001))
    tools = chat_tools([CLOCK_TOOL, WEATHER_TOOL])
    selected = await router.select(tools, "Et à Lyon ?", "Météo à Paris weather__read")
    assert selected == [tools[i] for i in indexes]
    [body] = http.bodies
    assert body["state"] == {"current_request": "Et à Lyon ?",
                             "previous_exchange": "Météo à Paris weather__read"}
    question = body["questions"]["route"]
    assert question["type"] == "choice" and "weather__read" in question["criteria"]["t1"]
    assert "properties" not in json.dumps(body) and "parameters" not in json.dumps(body)


async def test_compound_request_immediately_keeps_the_full_catalogue():
    tools = chat_tools([CLOCK_TOOL, WEATHER_TOOL])
    http = SystemOne(choice("several", 0.99))
    router = ToolRouter(http, RoutingConfig(port=9001))
    assert await router.select(tools, "Donne-moi l'heure et la météo.") is tools
    assert len(http.bodies) == 1


@pytest.mark.parametrize(("choice_name", "p_end", "confirmed", "indexes"), [
    ("t0", 0.049, None, [0]),
    ("direct", 0.01, None, []),
    ("unsure", 0.02, None, [0, 1]),
    ("t1", 0.05, False, [1]),
    ("hang_up", 0.96, False, [0, 1]),
    ("hang_up", 0.70, True, []),
])
async def test_live_combines_routing_and_selective_goodbye(
        client, choice_name, p_end, confirmed, indexes):
    s = client.session
    answer = choice(choice_name, 0.96)
    answer["route"]["probabilities"] = {"hang_up": p_end}
    goodbye = {"goodbye": {"choice": "hang_up" if confirmed else "other",
                          "probabilities": {"hang_up": 0.95 if confirmed else 0.01}}}
    http = SystemOne(answer, goodbye)
    s.s.router = ToolRouter(http, RoutingConfig(port=9001))
    await s.start({"tools": [CLOCK_TOOL, WEATHER_WITH_STATIONS],
                   "localive": {"end_on_goodbye": True, "goodbye_mode": "d3"}})
    await s.on_user_text("Une demande.")
    await asyncio.wait_for(s._reply, 2)
    expected = [["route"]] if confirmed is None else [["route"], ["goodbye"]]
    assert [list(body["questions"]) for body in http.bodies] == expected
    assert "hang_up" in http.bodies[0]["questions"]["route"]["criteria"]
    assert s.closed is (confirmed is True)
    if confirmed:
        assert not client.chat.calls
        assert client.of("session.closed")[-1]["reason"] == "user_goodbye"
    else:
        assert (client.chat.calls[0][1] or []) == [s.tools[i] for i in indexes]


@pytest.mark.parametrize(("phrase", "unfinished"), [
    ("Bah on peut s'arrêter là.", False),
    ("On peut arrêter la session, c'est bon.", True),
])
async def test_explicit_signoff_only_asks_d3_once(client, phrase, unfinished):
    s = client.session
    http = SystemOne({"goodbye": {"choice": "hang_up", "probabilities": {"hang_up": .93}}})
    s.s.router = ToolRouter(http, RoutingConfig(port=9001))
    tools = [{**CLOCK_TOOL, "name": f"clock_{i}"} for i in range(18)]
    await s.start({"tools": tools, "localive": {"end_on_goodbye": True, "goodbye_mode": "d3"}})
    await s.on_user_text(phrase, unfinished=unfinished)
    assert [list(body["questions"]) for body in http.bodies] == [["goodbye"]]
    assert [e["reason"] for e in client.of("session.closed")] == ["user_goodbye"]
    assert not client.chat.calls


@pytest.mark.parametrize("phrase", [
    "On peut arrêter la session…", "Tu t'es pas arrêté là.", "On peut arrêter la musique.",
    "On peut arrêter la session après la météo ?", "Ne ferme pas la session.",
])
async def test_incomplete_or_unrelated_transcript_keeps_session_open(client, phrase):
    s = client.session
    s.configure({"localive": {"end_on_goodbye": True, "goodbye_mode": "d3"}})
    assert not await s._should_hang_up(phrase, unfinished=True)
    s.s.router.should_hang_up.assert_not_awaited()


@pytest.mark.parametrize("update", ["new_speech", "disable", "catalog"])
async def test_joint_decision_respects_changes_during_inference(client, update):
    s = client.session
    entered, release = asyncio.Event(), asyncio.Event()
    router = ToolRouter(None, RoutingConfig())
    s.s.router = router

    async def post(_state, questions):
        if "route" in questions:
            entered.set()
            await release.wait()
            return {"route": {"choice": "t1", "confidence": 0.90,
                              "probabilities": {"hang_up": 0.05}}}
        return {"goodbye": {"choice": "hang_up", "probabilities": {"hang_up": 0.95}}}

    router._post = AsyncMock(side_effect=post)
    await s.start({"tools": [CLOCK_TOOL, WEATHER_WITH_STATIONS],
                   "localive": {"end_on_goodbye": True, "goodbye_mode": "d3"}})
    await s.on_user_text("Ma demande.")
    await asyncio.wait_for(entered.wait(), 2)
    if update == "new_speech":
        s.segmenter.speaking = True
    else:
        s.configure({"localive": {"end_on_goodbye": False}}, update=True)
        if update == "catalog":
            s.configure({"tools": [CLOCK_TOOL]}, update=True)
    release.set()
    await asyncio.wait_for(s._reply, 2)
    assert not s.closed
    assert client.chat.calls[0][1] == ([s.tools[0]] if update == "catalog" else [s.tools[1]])


async def test_large_catalogue_with_too_many_families_skips_inference():
    tools = chat_tools([{**CLOCK_TOOL, "name": f"module{i}__read"} for i in range(25)])
    http = SystemOne()
    router = ToolRouter(http, RoutingConfig(port=9001))
    assert await router.select(tools, "Une demande composée") is tools
    assert not http.bodies


@pytest.mark.parametrize(("utterance", "kind"), [
    ("Merci !", "thanks"), ("OK, super, merci beaucoup.", "assent"),
    ("D’accord, merci.", "assent"), ("Merci, éteins la lampe.", None),
    ("C’est bon, merci.", None), ("Merci, on se reparle demain.", None),
    ("Okay.", "bare_assent"), ("OK, fais la recherche.", None),
])
def test_only_whole_thanks_get_the_courtesy_route(utterance, kind):
    assert thanks_kind(utterance) == kind


@pytest.mark.parametrize(("reply", "heard", "pending", "answer", "expected"), [
    ("Il est 17 heures.", "Merci.", None, None, []),
    ("Le réveil est réglé.", "OK super merci beaucoup.", .19, None, []),
    ("Le réveil est réglé.", "Okay.", .19, None, []),
    ("Voulez-vous que je consulte la météo ?", "OK.", .91, choice("t1", .91), [1]),
    ("Voulez-vous que je consulte la météo ?", "OK merci.", .91, choice("t1", .91), [1]),
    ("Tu confirmes ?", "Oui merci.", .94, choice("t1", .30), []),
    ("Tu confirmes ?", "OK merci.", .80, {}, []),
    ("Autre chose ?", "OK merci.", float("nan"), None, []),
])
async def test_thanks_route_uses_the_immediately_preceding_assistant_reply(
    client, reply, heard, pending, answer, expected,
):
    session = client.session
    answers = [] if pending is None else [{"pending": {"noul": pending}}]
    if answer is not None:
        answers.append(answer)
    http = SystemOne(*answers)
    session.s.router = ToolRouter(http, RoutingConfig(port=9001))
    await session.start({"tools": [CLOCK_TOOL, WEATHER_WITH_STATIONS],
                         "localive": {"end_on_goodbye": True, "goodbye_mode": "d3"}})
    session.history.append(live.Turn(messages=[
        {"role": "user", "content": "Et la météo ?"}, {"role": "assistant", "content": reply},
    ]))
    await session.on_user_text(heard)
    await client.wait_turn()
    assert not session.closed
    assert (client.chat.calls[-1][1] or []) == [session.tools[i] for i in expected]
    assert len(http.bodies) == len(answers)
    if http.bodies:
        assert http.bodies[0]["state"] == {"assistant_reply": reply}
    if answer is not None:
        assert http.bodies[1]["state"] == {
            "current_request": reply, "previous_exchange": "Et la météo ?",
        }


@pytest.mark.parametrize("catalog", [[], [CLOCK_TOOL, WEATHER_WITH_STATIONS]])
@pytest.mark.parametrize("closing", [True, False])
async def test_contextual_end_after_thanks(client, catalog, closing):
    s = client.session
    reply = "Avec plaisir !" if closing else "Veux-tu que je vérifie la météo ?"
    http = SystemOne({"goodbye": {"choice": "hang_up" if closing else "pending",
                                  "probabilities": {"hang_up": .97 if closing else .18}}},
                     choice("t1", .90))
    s.s.router = ToolRouter(http, RoutingConfig(port=9001))
    await s.start({"tools": catalog, "localive": {"end_on_goodbye": True, "goodbye_mode": "d3"}})
    s.history.append(live.Turn(messages=[{"role": "user", "content": "OK, merci."},
                                         {"role": "assistant", "content": reply}]))
    await s.on_user_text("C'est bon.")
    assert s.closed == closing
    assert http.bodies[0]["state"] == {"previous_user": "OK, merci.", "assistant": reply,
                                      "current_user": "C'est bon."}
    if closing:
        assert len(http.bodies) == 1
        assert not client.chat.calls
        assert client.of("session.closed")[-1]["reason"] == "user_goodbye"
    else:
        await client.wait_turn()
        assert client.chat.calls


async def test_contextual_signoff_in_echo_tail_reaches_d3(client):
    s = client.session
    closed = asyncio.Event()
    send = s.send

    async def observe(event):
        await send(event)
        if event["type"] == "session.closed":
            closed.set()

    s.send = observe
    s.s.router.should_hang_up.return_value = True
    await s.start({"localive": {"end_on_goodbye": True, "goodbye_mode": "d3"}})
    s.history.append(live.Turn(messages=[{"role": "user", "content": "OK, merci."},
                                         {"role": "assistant", "content": "Avec plaisir !"}]))
    s._over_utterance = 0
    s._spoken_recent = "Avec plaisir !"
    s.output.last_audible = time.monotonic()
    s._utterances.put_nowait(live.Utterance(0, 16000, [], time.monotonic(),
                                            Transcript("C'est bon.", .01), 350, False, False))
    await asyncio.wait_for(closed.wait(), 2)
    assert client.of("session.closed")[-1]["reason"] == "user_goodbye"
    assert not client.chat.calls


async def test_compact_description_and_web_companion_preserve_original_schemas():
    web = {**CLOCK_TOOL, "name": "web_search", "description": "one two three four five six seven"}
    card = {**CLOCK_TOOL, "name": "ui__show_luna_card"}
    tools = chat_tools([card, CLOCK_TOOL, web])
    http = SystemOne(choice("t2", 0.98))
    router = ToolRouter(http, RoutingConfig(port=9001))
    assert await router.select(tools, "Cherche le résultat du match.") == [tools[0], tools[2]]
    options = http.bodies[0]["questions"]["route"]["criteria"]
    assert options["t2"] == "web_search one two three four five"


async def test_large_catalogue_selects_a_module_and_single_tool_needs_no_decision():
    raw = [WEATHER_TOOL, CLOCK_TOOL, *[{**CLOCK_TOOL, "name": f"memory__read_{i}"}
                                       for i in range(15)]]
    tools = chat_tools(raw)
    http = SystemOne(choice("t2", 0.93))
    router = ToolRouter(http, RoutingConfig(port=9001))
    assert await router.select(tools, "Que sais-tu de moi ?") == tools[2:]
    only = tools[:1]
    assert await router.select(only, "Quel temps fait-il ?") == only
    assert len(http.bodies) == 1


async def test_joint_router_compacts_module_labels_and_preserves_client_tools():
    raw = [{**CLOCK_TOOL, "name": f"m{i}__read", "description": "Lire le prix des billets"}
           for i in range(16)]
    raw.extend([{**CLOCK_TOOL, "name": "m0__add", "description": "Ajouter un billet"},
                {**CLOCK_TOOL, "name": "inventory", "description": "Afficher le stock"}])
    tools = chat_tools(raw)
    answer = choice("t0", .95)
    answer["route"]["probabilities"] = {"hang_up": .01}
    http = SystemOne(answer)
    router = ToolRouter(http, RoutingConfig(port=9001))
    decision = await router.decide(tools, "Combien coûte le billet ?", include_goodbye=True)
    [body] = http.bodies
    labels = body["questions"]["route"]["criteria"]
    assert labels["t0"] == "m0 read Lire prix billets, add Ajouter billet"
    assert labels["t16"] == "inventory Afficher stock"
    assert decision.tools == [tools[0], tools[16]]
    assert not decision.confirm_goodbye


@pytest.mark.parametrize("cancel", [False, True])
async def test_router_deadline_falls_back_and_caller_cancellation_propagates(cancel):
    tools = chat_tools([CLOCK_TOOL, WEATHER_TOOL])
    router = ToolRouter(None, RoutingConfig(timeout_ms=5000 if cancel else 25))
    started, ended = asyncio.Event(), asyncio.Event()

    async def stalled(*_):
        started.set()
        try:
            await asyncio.Future()
        finally:
            ended.set()

    router._ask = stalled
    task = asyncio.create_task(router.select(tools, "Donne-moi l'heure."))
    await asyncio.wait_for(started.wait(), 2)
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        assert await asyncio.wait_for(task, 1) == tools
    assert ended.is_set()


async def test_main_model_receives_complete_selected_schemas_and_routes_the_followup(client):
    c = client
    await c.session.start({"tools": [CLOCK_TOOL, WEATHER_WITH_STATIONS, TRANSIT]})
    clock, weather, transit = c.session.tools
    c.session.s.router.select.side_effect = [[weather, transit], [weather], []]
    c.chat.script = [[ToolCall("id", "weather__read", '{"city":"Paris"}')],
                     "À Paris il pleut.", "À Lyon aussi.", "Bonjour."]
    c.tool_answers["weather__read"] = {"temperature": 12, "internal": "raw-tool-payload"}
    await c.session.on_user_text("La météo à Paris et le prochain bus ?")
    await c.wait_turn()
    await c.session.on_user_text("Et à Lyon ?")
    await c.wait_turn(count=2)
    await c.session.on_user_text("Bonjour.")
    await c.wait_turn(count=3)
    assert [tools for _, tools in c.chat.calls] == [
        [weather, transit], [weather, transit], [weather], None]
    assert weather["function"]["parameters"]["required"] == ["city"]
    assert clock not in c.chat.calls[0][1]
    queries = c.session.s.router.select.await_args_list
    assert len(queries) == 3
    _, followup, previous = queries[1].args
    assert followup == "Et à Lyon ?"
    assert "météo à Paris" in previous and "weather__read" in previous
    assert "raw-tool-payload" not in previous


async def test_unoffered_tools_are_not_executed_and_registered_tool_can_be_requested_again(client):
    c = client
    await c.session.start({"tools": [CLOCK_TOOL, WEATHER_WITH_STATIONS, TRANSIT]})
    clock, weather, _ = c.session.tools
    c.session.s.router.select.return_value = [weather]
    c.chat.script = [[ToolCall("a", "clock__read", "{}"), ToolCall("b", "unknown__action", "{}")],
                     [ToolCall("c", "clock__read", "{}")], "Il est midi."]
    await c.session.on_user_text("Quelle heure est-il ?")
    await c.wait_turn()
    assert [tools for _, tools in c.chat.calls] == [[weather], [clock, weather], [clock, weather]]
    rejected = [m for m in c.chat.calls[1][0] if m["role"] == "tool"]
    assert len(rejected) == 2 and all("error" in json.loads(m["content"]) for m in rejected)
    sent = [e["event"]["item"]["name"] for e in c.of("response.event")
            if e["event"]["type"] == "response.output_item.done"]
    assert sent == ["clock__read"]


@pytest.mark.parametrize("cancel", [False, True])
async def test_live_announces_and_keeps_thinking_while_the_router_is_busy(client, cancel):
    s = client.session
    entered, release = asyncio.Event(), asyncio.Event()
    renewed = asyncio.Event()

    async def receive(event):
        await client._receive(event)
        if event.get("state") == "thinking":
            if sum(e["state"] == "thinking" for e in client.of("voice.state")) == 2:
                renewed.set()

    s._raw_send = receive

    async def slow_router(*_):
        entered.set()
        await release.wait()
        return []

    s.s.router.select.side_effect = slow_router
    await s.start({"tools": [CLOCK_TOOL, WEATHER_WITH_STATIONS]})
    await s.on_user_text("Tu vas bien ?")
    await asyncio.wait_for(entered.wait(), 2)
    assert not client.chat.calls
    [delegation] = client.of("session.delegation.created")
    [created] = client.of("response.event")
    assert created["event"]["type"] == "response.created"
    assert created["event"]["response"]["id"] == delegation["delegation"]["response_id"]
    assert client.of("voice.state")[-1]["state"] == "thinking"
    await asyncio.wait_for(renewed.wait(), 3)
    assert len(client.of("session.delegation.created")) == 1
    if cancel:
        await s.interrupt()
    else:
        release.set()
    await client.wait_turn()
    created, terminal = [e["event"] for e in client.of("response.event")]
    assert terminal["type"] == ("response.incomplete" if cancel else "response.completed")
    assert terminal["response"]["id"] == created["response"]["id"]
    assert client.of("voice.state")[-1]["state"] == "listening"


async def test_catalogue_update_during_routing_replaces_the_selection(client):
    c = client
    await c.session.start({"tools": [CLOCK_TOOL, WEATHER_WITH_STATIONS]})
    entered, release = asyncio.Event(), asyncio.Event()

    async def select(tools, request, previous):
        entered.set()
        await release.wait()
        return [tools[1]]

    c.session.s.router.select.side_effect = select
    await c.session.on_user_text("Quel temps fait-il ?")
    await asyncio.wait_for(entered.wait(), 2)
    await c.session.handle({"type": "session.update", "session": {"tools": [TRANSIT]}})
    release.set()
    await c.wait_turn()
    assert names(c.chat.calls[0][1]) == ["transit__read"]


async def test_changed_schema_during_generation_invalidates_the_old_call(client):
    c = client
    await c.session.start({"tools": [CLOCK_TOOL, WEATHER_WITH_STATIONS]})
    c.session.s.router.select.return_value = [c.session.tools[1]]
    c.chat.script = [[ToolCall("old", "weather__read", '{"city":"Paris"}')], "D'accord."]
    entered, release = asyncio.Event(), asyncio.Event()
    stream = c.chat.stream

    async def gated(messages, tools, result, **kwargs):
        if kwargs.get("max_tokens") != 1 and not entered.is_set():
            entered.set()
            await release.wait()
        async for piece in stream(messages, tools, result, **kwargs):
            yield piece

    c.chat.stream = gated
    await c.session.on_user_text("La météo à Paris ?")
    await asyncio.wait_for(entered.wait(), 2)
    replacement = {**WEATHER_TOOL, "parameters": {
        "type": "object", "properties": {"postal_code": {"type": "string"}},
        "required": ["postal_code"]}}
    await c.session.handle({"type": "session.update", "session": {"tools": [replacement]}})
    release.set()
    await c.wait_turn()
    assert all(e["event"]["type"] != "response.output_item.done" for e in c.of("response.event"))
    messages, offered = c.chat.calls[1]
    assert offered[0]["function"]["parameters"]["required"] == ["postal_code"]
    assert "error" in json.loads(messages[-1]["content"])


@pytest.mark.parametrize(("weather", "routed"), [(WEATHER_TOOL, False),
                                                (WEATHER_WITH_STATIONS, True)],
                         ids=["direct", "routed"])
async def test_warmup_uses_the_same_policy_for_live_and_saved_catalogues(client, weather, routed):
    s = client.session
    s.configure({"tools": [CLOCK_TOOL, weather]})
    s.s.chat = WarmChat()
    await s._warm_up()
    messages = s.s.chat.calls[-1][0]
    expected = None if routed else s.tools
    assert json.loads(s.config.path(live.PREFIX_FILE).read_text("utf-8"))["tools"] == s.tools
    await warm_from_saved(s.s)
    s.s.router = None
    await warm_from_saved(s.s)  # same saved prefix, restarted without routing
    assert s.s.chat.calls == [(messages, expected, 1), (messages, expected, 1),
                              (messages, s.tools, 1)]


async def test_router_starts_once_on_first_catalogue_worth_routing(client, monkeypatch):
    session = client.session
    session.config.huggingface.offline = True
    app = App(session.config)
    app.services = session.s
    session.s.router = None
    session.s.request_router = app.request_router
    start = AsyncMock()
    monkeypatch.setattr("localive.server.Brain.start", start)
    monkeypatch.setattr(ToolRouter, "verify", AsyncMock())

    session.configure({"tools": []})
    session.configure({"tools": [WEATHER_WITH_STATIONS]}, update=True)
    session.configure({"tools": [CLOCK_TOOL, WEATHER_TOOL, TRANSIT]}, update=True)
    assert app._router_task is None
    assert session.s.warm_tools(session.tools) == session.tools

    # The model can be installed after Localive has started.
    session.configure({"tools": [CLOCK_TOOL, WEATHER_WITH_STATIONS]}, update=True)
    assert app._router_task is None
    assert session.s.warm_tools(session.tools) == session.tools
    model = session.config.path(session.config.routing.model)
    model.parent.mkdir(parents=True, exist_ok=True)
    model.touch()
    session.configure({"tools": [CLOCK_TOOL, WEATHER_WITH_STATIONS]}, update=True)
    pending = app._router_task
    assert pending is not None and session.s.router_starting
    assert session.s.warm_tools(session.tools) is None
    session.configure({"tools": [TRANSIT, WEATHER_WITH_STATIONS]}, update=True)
    assert app._router_task is pending
    await pending
    start.assert_awaited_once()
    assert session.s.router is not None and not session.s.router_starting


@pytest.mark.parametrize("routing,goodbye,offline,env,model,expected", [
    (True, False, False, "0", DEFAULT_ROUTER, "idle"),
    (False, True, False, "", DEFAULT_ROUTER, "ready"),
    (False, False, False, "", DEFAULT_ROUTER, "missing"),
    (True, False, True, "", DEFAULT_ROUTER, "missing"),
    (True, False, False, "yes", DEFAULT_ROUTER, "missing"),
    (True, False, False, "", "models/perso.gguf", "missing"),
])
async def test_boot_fetches_only_enabled_online_default_weights(
    tmp_path, monkeypatch, routing, goodbye, offline, env, model, expected,
):
    config = Config(root=tmp_path)
    config.routing.enabled, config.routing.model = routing, model
    config.huggingface.offline = offline
    config.turn.end_on_goodbye, config.turn.goodbye_mode = goodbye, "d3"
    config.turn.enabled = config.decision.enabled = False
    monkeypatch.setenv("HF_HUB_OFFLINE", env)
    monkeypatch.setattr(server.hf, "configure", Mock())
    monkeypatch.setattr(server.Transcriber, "start", AsyncMock())
    monkeypatch.setattr(server.Synthesizer, "start", AsyncMock())
    monkeypatch.setattr(server.SileroVad, "load_models", Mock())
    monkeypatch.setattr(server, "warm_from_saved", AsyncMock())
    monkeypatch.setattr(server.Brain, "start", AsyncMock())
    verify = AsyncMock()
    monkeypatch.setattr(ToolRouter, "verify", verify)

    async def fetch(root):
        path = root / DEFAULT_ROUTER
        path.parent.mkdir(parents=True)
        path.touch()

    worker = AsyncMock(side_effect=fetch)
    monkeypatch.setattr(install, "fetch_default_router", worker)
    app = App(config)
    try:
        await app.boot()
        assert app.ready.is_set()
        if expected != "missing":
            assert not config.decision_model_installed
            await app._router_download
            worker.assert_awaited_once_with(tmp_path)
        else:
            assert app._router_download is None
            worker.assert_not_awaited()
        if expected == "ready":
            await app._router_task
            verify.assert_awaited_once()
        else:
            assert app._router_task is None
        status = json.loads((await app.health(None)).body)
        assert status["decision_model_state"] == expected
        assert status["tool_router_state"] == (expected if routing else "off")
    finally:
        await app.on_cleanup(None)


async def test_open_live_session_routes_after_background_download(client, monkeypatch):
    session = client.session
    app = App(session.config)
    app.services = session.s
    session.s.router = None
    session.s.request_router = app.request_router
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(install, "ROUTER_SIZE", 10)
    monkeypatch.setattr(server.Brain, "start", AsyncMock())
    monkeypatch.setattr(ToolRouter, "verify", AsyncMock())
    monkeypatch.setattr(ToolRouter, "select", AsyncMock(return_value=chat_tools([CLOCK_TOOL])))
    entered, finish = asyncio.Event(), asyncio.Event()

    async def fetch(root):
        path = root / DEFAULT_ROUTER
        path.parent.mkdir(parents=True)
        part = path.with_suffix(".gguf.part")
        part.write_bytes(b"1234")
        entered.set()
        await finish.wait()
        part.replace(path)

    worker = AsyncMock(side_effect=fetch)
    monkeypatch.setattr(install, "fetch_default_router", worker)
    tools = [CLOCK_TOOL, *[{**TRANSIT, "name": f"transit__line_{i}"} for i in range(17)]]
    await session.start({"tools": tools, "localive": {"metrics": True}})
    await asyncio.wait_for(entered.wait(), 2)
    state = json.loads((await app.health(None)).body)
    assert state["decision_model_state"] == "downloading"
    assert state["decision_model_download_pct"] == 40
    await session.on_user_text("Quelle heure est-il ?")
    await client.wait_turn()
    assert len(client.chat.calls[0][1]) == 18
    finish.set()
    await app._router_download
    await app._router_task
    assert json.loads((await app.health(None)).body)["tool_router_state"] == "ready"
    await session.on_user_text("Et quelle heure est-il maintenant ?")
    await client.wait_turn(count=2)
    assert client.chat.calls[1][1] == chat_tools([CLOCK_TOOL])
    worker.assert_awaited_once()


async def test_failed_transfer_retries_on_demand_after_cooldown(client, monkeypatch):
    app = App(client.session.config)
    app.services = client.session.s
    app.services.router = None
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    worker = AsyncMock(side_effect=install.InstallError("hors connexion"))
    monkeypatch.setattr(install, "fetch_default_router", worker)
    app.request_router()
    await app._router_download
    assert json.loads((await app.health(None)).body)["tool_router_state"] == "download_failed"
    assert app.services.router_retry_at > 0
    app.request_router()
    worker.assert_awaited_once()
    app.services.router_retry_at = 0
    app.request_router()
    await app._router_download
    assert worker.await_count == 2


async def test_router_recovers_after_an_invalid_decision_service(client, monkeypatch):
    s = client.session
    s.config.routing.model = "router.gguf"
    s.config.path(s.config.routing.model).touch()
    app = App(s.config)
    app.services = s.s
    s.s.router = None
    s.s.request_router = app.request_router
    monkeypatch.setattr("localive.server.Brain.start", AsyncMock())
    verify = AsyncMock(side_effect=[ValueError("invalid choice endpoint"), None])
    monkeypatch.setattr(ToolRouter, "verify", verify)
    await s.start({"tools": [CLOCK_TOOL, WEATHER_WITH_STATIONS]})
    await app._router_task
    assert s.s.router is None and s.s.warm_tools(s.tools) == s.tools
    assert json.loads((await app.health(None)).body)["tool_router_state"] == "unavailable"
    await s.on_user_text("Bonjour.")
    await client.wait_turn()
    verify.assert_awaited_once()  # a voice turn during the cooldown uses the full catalogue
    s.s.router_retry_at = 0
    await s.on_user_text("Et demain ?")
    await client.wait_turn(count=2)
    await app._router_task
    assert verify.await_count == 2 and s.s.router is not None
    app.router_engine.process = Mock()
    app.router_engine.process.poll.return_value = 1
    assert json.loads((await app.health(None)).body)["tool_router"] is False
    s.configure({"tools": [CLOCK_TOOL, WEATHER_WITH_STATIONS]}, update=True)
    verify.side_effect = None
    await app._router_task
    assert verify.await_count == 3 and json.loads((await app.health(None)).body)["tool_router"]


async def test_only_available_tools_count_toward_routing(client):
    s = client.session
    s.s.router = None
    s.s.request_router = Mock()
    small = [{**CLOCK_TOOL, "name": f"clock_{i}"} for i in range(5)]
    advertised = {"tools": [*small, {"type": "web_search"}, {"type": "unsupported"}]}
    s.s.http = object()
    s.config.search.engine = "off"
    s.configure(advertised)
    assert len(s.tools) == 5
    s.s.request_router.assert_not_called()

    s.config.search.engine = "searxng"
    s.configure(advertised, update=True)
    assert len(s.tools) == 6 and names(s.tools)[-1] == "web_search"
    s.s.request_router.assert_called_once_with()


async def test_schema_update_switches_live_turns_with_resident_router(client, caplog):
    c = client
    caplog.set_level("INFO", logger="localive.live")
    compact = [CLOCK_TOOL, WEATHER_TOOL]
    expanded = [CLOCK_TOOL, WEATHER_WITH_STATIONS]
    await c.session.start({"tools": compact, "localive": {"metrics": True}})
    await c.session.on_user_text("Quelle heure est-il ?")
    await c.wait_turn()
    c.session.s.router.select.assert_not_awaited()

    await c.session.handle({"type": "session.update", "session": {"tools": expanded}})
    await c.session.handle({"type": "session.update", "session": {"instructions": "Sois bref."}})
    c.session.s.router.select.return_value = [c.session.tools[1]]
    await c.session.on_user_text("Quelle est la météo ?")
    await c.wait_turn(count=2)

    await c.session.handle({"type": "session.update", "session": {"tools": compact}})
    await c.session.on_user_text("Et à Paris ?")
    await c.wait_turn(count=3)
    c.session.s.router.select.assert_awaited_once()
    assert [tools for _, tools in c.chat.calls] == [
        chat_tools(compact), chat_tools([WEATHER_WITH_STATIONS]), chat_tools(compact)]
    first, second, last = c.of("localive.turn")
    assert [t["selected_tools"] for t in (first, second, last)] == [2, 1, 2]
    assert first["tool_routing_ms"] == last["tool_routing_ms"] == 0
    assert "catalogue direct · 2/2 outils" in caplog.text
    assert "routeur consulté · 1/2 outils" in caplog.text


@pytest.mark.parametrize("loading", [False, True], ids=["disabled", "loading"])
async def test_large_catalogue_can_answer_while_router_is_unavailable(client, loading):
    s = client.session
    s.config.routing.enabled = loading
    s.s.router = None
    s.s.router_starting = loading
    await s.start({"tools": [CLOCK_TOOL, WEATHER_WITH_STATIONS], "localive": {"metrics": True}})
    await s.on_user_text("Quel temps fait-il ?")
    await client.wait_turn()
    assert client.chat.calls[0][1] == s.tools
    [report] = client.of("localive.turn")
    assert report["selected_tools"] == report["available_tools"] == 2
    assert report["tool_routing_ms"] == 0


async def test_failed_router_restores_full_catalogue_for_both_warmup_paths(client, tmp_path,
                                                                         monkeypatch):
    s = client.session
    s.config.root = tmp_path
    s.config.routing.enabled = True
    s.config.routing.model = "decision.gguf"
    s.config.path(s.config.routing.model).touch()
    s.s.router = None
    s.s.router_starting = True
    s.s.chat = WarmChat()
    s.configure({"tools": [CLOCK_TOOL, WEATHER_WITH_STATIONS]})
    await s._warm_up()
    assert s.s.chat.calls[-1][1] is None  # give the router time to start
    app = App(s.config)
    app.services = s.s
    fail_start = AsyncMock(side_effect=OSError("unavailable"))
    monkeypatch.setattr("localive.server.Brain.start", fail_start)
    await app._start_router()
    assert s.config.routing_installed and not s.s.routed_prefix
    await warm_from_saved(s.s)
    await s._warm_up()
    assert [tools for _, tools, _ in s.s.chat.calls] == [None, s.tools, s.tools]
