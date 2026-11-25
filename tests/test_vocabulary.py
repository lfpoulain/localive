import asyncio
import json
import time

import numpy as np
import pytest

from localive.live import Utterance, chat_tools
from localive.llm import ToolCall
from localive.vocabulary import device_vocabularies, match_devices
from tests.test_live import CLOCK_TOOL, Client

HALL = {"id": "light.hall", "name": "Lustre du hall", "aliases": ["Plafonnier Hue"]}
OFFICE = {"id": "light.office", "name": "Spot du bureau", "aliases": ["Plafonnier Hue"]}


def control(entities, allowed):
    return {
        "type": "function", "name": "home_assistant__control",
        "description": "Consignes. " * 210 + "Appareils : " + json.dumps(entities),
        "parameters": {"type": "object", "properties": {
            "entityId": {"type": "string", "enum": allowed}}},
    }


def dispatched_tools(client):
    return [inner["item"]["name"] for envelope in client.of("response.event")
            if (inner := envelope["event"])["type"] == "response.output_item.done"
            and inner["item"]["type"] == "function_call"]


def test_vocabulary_respects_entity_enum_and_selected_tools():
    raw = control([HALL, OFFICE], [HALL["id"]])
    catalog = device_vocabularies([raw, {**raw, "name": "other__control"}])
    selected = chat_tools([raw])
    note, blocked = match_devices(catalog, "Éteins le plafonier U.", selected)
    assert HALL["id"] in note and OFFICE["id"] not in note and not blocked
    assert match_devices(catalog, "Les plafonniers utiles.", selected) == ("", set())
    assert match_devices(catalog, "Plafonnier U", chat_tools([CLOCK_TOOL])) == ("", set())
    assert match_devices(catalog, "Spot du bureau", selected) == ("", set())


def test_longest_name_resolves_a_shared_alias_and_handles_multiple_targets():
    hall = {**HALL, "name": "Plafonnier Hue du hall"}
    raw = control([hall, OFFICE], [HALL["id"], OFFICE["id"]])
    note, blocked = match_devices(device_vocabularies([raw]),
                                  "Allume le plafonnier U du hall et le spot du bureau.",
                                  chat_tools([raw]))
    assert not blocked and HALL["id"] in note and OFFICE["id"] in note


def test_entity_enum_supplies_spoken_names_if_no_catalog_is_available():
    raw = {**control([], ["light.entree_nord"]), "description": "Appareils : [{invalide"}
    note, blocked = match_devices(device_vocabularies([raw]), "Éteins l’entrée nord.",
                                  chat_tools([raw]))
    assert "light.entree_nord" in note and not blocked


@pytest.mark.parametrize("enabled", [False, True])
async def test_option_preserves_spoken_transcript_and_history(enabled, tmp_path):
    heard = "N’allume pas le plafonnier U."
    c = Client(heard=heard, tts_seconds=0.05)
    s = c.session
    s.config.root = tmp_path
    s.config.asr.device_vocabulary = enabled
    raw = control([HALL], [HALL["id"]])
    try:
        await s.start({"delegation": {"responses": {"tools": [raw]}}})
        s._utterances.put_nowait(Utterance(0, 16000, np.zeros(16000, dtype=np.float32),
                                           time.monotonic(), None, 300, False, False))
        await c.wait_turn()
        assert [event["delta"] for event in c.of("session.input_transcript.delta")] == [heard]
        prompt = c.chat.calls[0][0][-1]["content"]
        assert prompt.startswith(heard) and (HALL["id"] in prompt) == enabled
        assert s.history[-1].messages[0] == {"role": "user", "content": heard}
    finally:
        await s.close("test_end")


async def test_ambiguous_alias_blocks_call_until_catalog_update(tmp_path):
    call = ToolCall("x", "home_assistant__control", json.dumps({"entityId": HALL["id"]}))
    c = Client([[call], "Lequel ?", [call], "C'est fait."], tts_seconds=0.05)
    s = c.session
    s.config.root = tmp_path
    s.config.asr.device_vocabulary = True
    try:
        await s.start({"tools": [CLOCK_TOOL, control([HALL, OFFICE], [HALL["id"], OFFICE["id"]])]})
        await s.handle({"type": "session.update", "session": {"instructions": "Sois bref."}})
        await s.on_user_text("Allume le plafonnier U.")
        await c.wait_turn()
        messages, tools = c.chat.calls[0]
        assert HALL["id"] in messages[-1]["content"] and OFFICE["id"] in messages[-1]["content"]
        assert tools == chat_tools([CLOCK_TOOL])
        assert dispatched_tools(c) == []

        await s.handle({"type": "session.update", "session": {
            "tools": [control([HALL], [HALL["id"]])]}})
        await s.on_user_text("Allume le plafonnier U.")
        await c.wait_turn(count=2)
        assert dispatched_tools(c) == [call.name]
        assert OFFICE["id"] not in c.chat.calls[2][0][-1]["content"]
        await s.handle({"type": "session.update", "session": {"tools": []}})
        await s.on_user_text("Et le plafonnier U ?")
        await c.wait_turn(count=3)
        assert "Vocabulaire des appareils" not in c.chat.calls[-1][0][-1]["content"]
    finally:
        await s.close("test_end")


async def test_alias_changed_during_generation_is_checked_before_dispatch(tmp_path):
    ids = [HALL["id"], OFFICE["id"]]
    old = control([HALL, {**OFFICE, "aliases": []}], ids)
    new = control([HALL, OFFICE], ids)
    assert chat_tools([old]) == chat_tools([new])  # alias is beyond the 2,000-character cut
    call = ToolCall("x", old["name"], json.dumps({"entityId": HALL["id"]}))
    c = Client([[call], "Lequel ?"], tts_seconds=0.05)
    s = c.session
    s.config.root = tmp_path
    s.config.asr.device_vocabulary = True
    requested, resume = asyncio.Event(), asyncio.Event()
    stream = c.chat.stream

    async def delayed(*args, **kwargs):
        if kwargs.get("max_tokens") != 1 and not requested.is_set():
            requested.set()
            await resume.wait()
        async for chunk in stream(*args, **kwargs):
            yield chunk

    c.chat.stream = delayed
    try:
        await s.start({"tools": [old]})
        await s.on_user_text("Allume le plafonnier U.")
        await asyncio.wait_for(requested.wait(), 2)
        await s.handle({"type": "session.update", "session": {"tools": [new]}})
        resume.set()
        await c.wait_turn()
        assert dispatched_tools(c) == []
        assert c.chat.calls[1][1] is None
        assert OFFICE["id"] in c.chat.calls[1][0][1]["content"]
    finally:
        await s.close("test_end")
