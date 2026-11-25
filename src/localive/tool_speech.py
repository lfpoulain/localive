"""Model-only choice to speak a Cornelius tool's summary as the final answer."""

import json

from .text import clean_for_speech

FINAL_SUMMARY = "speak_summary"
TOOL_WAIT_NOTICE = "J’attends le résultat."


def allows_summary_choice(tool: dict) -> bool:
    schema = tool.get("function", {}).get("parameters", {})
    props = schema.get("properties")
    # Only extend closed, plain object schemas. Open schemas may use arbitrary business keys.
    return (schema.get("type") == "object" and schema.get("additionalProperties") is False
            and isinstance(props, dict) and FINAL_SUMMARY not in props
            and not schema.keys() & {"allOf", "anyOf", "oneOf", "$ref", "patternProperties",
                                     "if", "not", "maxProperties", "propertyNames"})


def model_tools(tools: list[dict] | None) -> list[dict] | None:
    if not tools:
        return tools
    out = []
    for tool in tools:
        if allows_summary_choice(tool):
            function = tool["function"]
            schema = function["parameters"]
            props = {**schema["properties"], FINAL_SUMMARY: {
                "type": "boolean", "description": "Réponse vocale Localive"}}
            tool = {**tool, "function": {**function, "parameters": {**schema, "properties": props}}}
        out.append(tool)
    return out


def extract_summary_choice(arguments: str) -> tuple[str, bool]:
    values = json.loads(arguments or "{}")
    if not isinstance(values, dict) or FINAL_SUMMARY not in values:
        return arguments, False
    use_summary = values.pop(FINAL_SUMMARY) is True
    return json.dumps(values, ensure_ascii=False), use_summary


def spoken_tool_result(output: object) -> str:
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except (ValueError, RecursionError):
            return ""
    if (not isinstance(output, dict) or type(output.get("schemaVersion")) is not int
            or output["schemaVersion"] != 1 or output.get("status") != "succeeded"
            or output.get("error") is not None):
        return ""
    result = output.get("result")
    if result is not None and not isinstance(result, dict):
        return ""
    if isinstance(result, dict):
        if any(key in result and not isinstance(result[key], bool)
               for key in ("accepted", "confirmed")):
            return ""
        if result.get("stale") or result.get("error") or result.get("accepted") is False:
            return ""
        if result.get("accepted") is True and result.get("confirmed") is not True:
            return "La demande a été transmise."
        if "confirmed" in result and result["confirmed"] is not True:
            return ""
    summary = output.get("summary")
    if isinstance(summary, str) and 0 < len(summary) <= 300:
        cleaned = clean_for_speech(summary)
        if any(c.isalnum() for c in cleaned):
            return cleaned
    return ""
