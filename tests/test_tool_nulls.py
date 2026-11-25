"""Null-able tool arguments: the model may omit them, Cornelius still receives every key."""

import json

from localive.live import chat_tools, fill_nulls, tool_nulls

# Shape of Cornelius's alarms__set (OpenAI strict: every key required, null allowed).
ALARMS = {
    "type": "function",
    "name": "alarms__set",
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "required": ["kind", "time", "inMinutes", "durationSeconds", "label"],
        "properties": {
            "kind": {"type": "string", "enum": ["alarm", "timer"]},
            "time": {"type": ["string", "null"]},
            "inMinutes": {"type": ["integer", "null"]},
            "durationSeconds": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            "label": {"type": ["string", "null"]},
        },
    },
}


def test_null_able_keys_become_optional_for_the_model():
    (tool,) = chat_tools([ALARMS])
    assert tool["function"]["parameters"]["required"] == ["kind"]
    assert ALARMS["parameters"]["required"][1] == "time"  # the session's copy is untouched


def test_missing_null_able_keys_are_put_back_for_cornelius():
    nulls = tool_nulls([ALARMS])["alarms__set"]
    sent = json.loads(fill_nulls('{"kind": "timer", "durationSeconds": 900}', nulls))
    assert sent == {"kind": "timer", "durationSeconds": 900, "time": None, "inMinutes": None,
                    "label": None}  # fmt: skip
    complete = '{"kind": "alarm", "time": "07:00", "inMinutes": null, "durationSeconds": null, "label": null}'  # noqa: E501
    assert fill_nulls(complete, nulls) == complete
    assert fill_nulls("not json", nulls) == "not json"


def test_tools_without_null_able_keys_are_unchanged():
    plain = {"type": "function", "name": "lights", "parameters": {
        "type": "object", "required": ["room"], "properties": {"room": {"type": "string"}}}}
    assert chat_tools([plain])[0]["function"]["parameters"] is plain["parameters"]
    assert tool_nulls([plain]) == {}


def test_incomplete_client_schema_keeps_recognizable_nullable_keys():
    tool = {"type": "function", "name": "custom", "parameters": {
        "required": ["label", {}],
        "properties": {"label": {"anyOf": None, "oneOf": [{"type": "null"}]}},
    }}
    assert tool_nulls([tool]) == {"custom": ["label"]}


def test_the_time_in_a_tool_description_does_not_break_the_prompt_cache():
    from localive.live import stable_description, tools_tell_time

    text = ("Règle un réveil (« dans 20 minutes »). Nous sommes mardi 10 novembre 2026 à 17:03, "
            "heure du foyer. kind : alarm pour un réveil à une heure (07:00).")
    stable, found = stable_description(text)
    assert found and "17:03" not in stable and "07:00" in stable and stable.startswith("Règle")
    later = text.replace("17:03", "17:04")
    tool = {"type": "function", "name": "alarms__set", "description": text}
    assert chat_tools([tool]) == chat_tools([{**tool, "description": later}])
    assert tools_tell_time([tool]) and not tools_tell_time([{**tool, "description": "Rien."}])
