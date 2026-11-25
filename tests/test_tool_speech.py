import asyncio
import json

import pytest

from localive import live
from localive.llm import ToolCall
from localive.tool_speech import TOOL_WAIT_NOTICE, model_tools, spoken_tool_result
from tests.test_live import CLOCK_TOOL, Client


def outcome(summary, **fields):
    return {"schemaVersion": 1, "status": "succeeded", "summary": summary, **fields}


@pytest.mark.parametrize(("payload", "expected"), [
    (outcome("**Quatre** vélos disponibles."), "Quatre vélos disponibles."),
    (outcome("Terminé.", result={"accepted": True}), "La demande a été transmise."),
    (outcome("Terminé.", result={"accepted": True, "confirmed": False}),
     "La demande a été transmise."),
    (outcome("Terminé.", result={"accepted": False, "confirmed": True}), ""),
    (outcome("Terminé.", result={"confirmed": "true"}), ""),
    (outcome("Terminé.", result=["running"]), ""),
    (outcome("Terminé.", status="pending_confirmation"), ""),
    (outcome("Terminé.", status="failed"), ""),
    (outcome("Terminé.", result={"stale": True}), ""),
    (outcome("Terminé.", error={"message": "Échec"}), ""),
    (outcome("x" * 301), ""),
    (outcome("👍"), ""),
    ({"summary": "Réussi"}, ""),
    ("invalid", ""),
])
def test_only_trust_successful_cornelius_results(payload, expected):
    assert spoken_tool_result(payload) == expected


@pytest.mark.parametrize(("tool", "direct", "valid", "multiple"), [
    ("velos__disponibles", True, True, False),
    ("imprimante__demarrer", True, True, False),
    ("atelier__outil", False, True, False),
    ("atelier__outil", True, False, False),
    ("atelier__outil", True, True, True),
])
async def test_tool_summary_in_live(tool, direct, valid, multiple):
    args = json.dumps({"speak_summary": direct})
    call = ToolCall("", tool, args)
    c = Client([[call, call] if multiple else [call], "Réponse de Qwen."], tts_seconds=.02)
    summary = "Le module a répondu."
    c.tool_answers[tool] = outcome(summary, result={"data": "x"*7000},
                                   status="succeeded" if valid else "stale")
    try:
        await c.session.start({"tools": [{**CLOCK_TOOL, "name": tool}],
                               "localive": {"metrics": True}})
        await c.session.on_user_text("Lance la demande.")
        await c.wait_turn()
        fast = direct and valid and not multiple
        assert len(c.chat.calls) == (1 if fast else 2)
        for event in c.of("response.event"):
            item = event["event"].get("item", {})
            if item.get("type") == "function_call":
                assert json.loads(item["arguments"]) == {}
        assert c.tts.texts[0][0] == (summary if fast else "Réponse de Qwen.")
        assert c.of("localive.turn")[-1]["direct_tool_reply"] is fast
        messages = c.session.history[-1].messages
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"].strip() == c.tts.texts[0][0]
        assert messages[-2]["role"] == "tool"
    finally:
        await c.session.close("test_end")


async def test_wait_notice_tracks_outstanding_client_call(monkeypatch):
    monkeypatch.setattr(live, "WAIT_NOTICE_SECONDS", .06)
    tool = "colis__suivre"
    c = Client([[ToolCall("", tool, '{"speak_summary":true}')]], tts_seconds=.02)
    c.session.config.voice.wait_notice = True
    called, heard = asyncio.Future(), asyncio.Event()

    async def receive(event):
        c.events.append(event)
        item = event.get("event", {}).get("item", {})
        if item.get("type") == "function_call":
            called.set_result(item["call_id"])
        if (event["type"] == "session.output_transcript.delta"
                and TOOL_WAIT_NOTICE in event["delta"]):
            heard.set()

    c.session._raw_send = receive
    try:
        await c.session.start({"tools": [{**CLOCK_TOOL, "name": tool}]})
        await c.session.on_user_text("Où est le colis ?")
        call_id = await asyncio.wait_for(called, 2)
        await asyncio.wait_for(heard.wait(), 2)
        await c.session.handle({"type": "response.item.create", "item": {
            "type": "function_call_output", "call_id": call_id,
            "output": outcome("Le colis est au dépôt.")}})
        await c.wait_turn()
        assert [t for t, _ in c.tts.texts] == [TOOL_WAIT_NOTICE, "Le colis est au dépôt."]
        assert TOOL_WAIT_NOTICE not in str(c.session.history[-1].messages)
    finally:
        await c.session.close("test_end")


def test_model_only_flag_preserves_original_and_business_schemas():
    tool = live.chat_tools([CLOCK_TOOL])[0]
    exposed = model_tools([tool])[0]
    assert "speak_summary" not in tool["function"]["parameters"]["properties"]
    assert exposed["function"]["parameters"]["properties"]["speak_summary"]["type"] == "boolean"
    for schema in ({"type": "object", "properties": {}},
                   {"type": "object", "additionalProperties": False,
                    "properties": {"speak_summary": {"type": "string"}}}):
        original = {"function": {"name": "custom", "parameters": schema}}
        assert model_tools([original])[0] is original
