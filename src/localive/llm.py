"""Streaming client for any OpenAI-compatible chat endpoint (llama.cpp, Ollama, LM Studio...)."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp

from .config import LlmConfig
from .tool_speech import model_tools

log = logging.getLogger(__name__)

# Diagnostic: while this file exists, every request to the brain is saved under
# .runtime/logs/prompts/ (conversations included: delete the folder afterwards).
DUMP_FLAG = Path(".runtime/dump-prompts")


def _dump(body: dict[str, Any]) -> None:
    if not DUMP_FLAG.exists():
        return
    folder = Path(".runtime/logs/prompts")
    try:
        folder.mkdir(parents=True, exist_ok=True)
        name = f"{time.strftime('%H%M%S')}-{time.perf_counter_ns() % 10**6:06d}.json"
        (folder / name).write_text(json.dumps(body, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass


class LlmError(RuntimeError):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str = ""


@dataclass
class Completion:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    truncated: bool = False
    exhausted_thinking: bool = False


class ChatClient:
    def __init__(self, http: aiohttp.ClientSession, config: LlmConfig, *,
                 lmstudio: bool = False) -> None:
        self.http = http
        self.config = config
        self.lmstudio = lmstudio
        self._warned_truncation = False

    def _check_truncation(self, body: dict[str, Any], read: Any) -> None:
        """Ollama keeps only the end of a prompt longer than its context (4096 by default),
        silently: Cornelius' instructions and tools are lost. Say so once."""
        if self._warned_truncation or not isinstance(read, int) or read <= 0:
            return
        sent = len(json.dumps([body["messages"], body.get("tools")], ensure_ascii=False)) // 4
        if sent > 3000 and read < sent * 0.6:
            self._warned_truncation = True
            log.warning("Le serveur %s n'a lu que %d jetons sur environ %d : son contexte est trop "
                        "petit et le début des consignes est perdu. Pour Ollama : variable "
                        "OLLAMA_CONTEXT_LENGTH=32768 sur sa machine, puis redémarrez-le.",
                        self.config.url, read, sent)  # fmt: skip

    async def _stream_ollama(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        result: Completion,
        max_tokens: int | None,
        model: str | None,
    ) -> AsyncIterator[str]:
        """Ollama's own /api/chat: the only way to choose its context (num_ctx) per request.
        Its OpenAI endpoint keeps the server's default (4096) and silently cuts the start."""
        body: dict[str, Any] = {
            "model": self.config.upstream_model or model or self.config.model,
            "messages": ollama_messages(messages),
            "stream": True,
            "think": not self.config.no_thinking,
            "options": {
                "num_ctx": self.config.ollama_context,
                "temperature": self.config.temperature,
                "num_predict": max_tokens or self.config.max_tokens,
            },
        }
        if tools:
            body["tools"] = tools
        headers = {"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {}
        _dump(body)
        sent = len(json.dumps([messages, tools], ensure_ascii=False)) // 4
        if sent > self.config.ollama_context * 0.9 and not self._warned_truncation:
            # Ollama would keep only the end: the instructions would be lost.
            self._warned_truncation = True
            log.warning("Prompt d'environ %d jetons pour un contexte Ollama de %d : augmentez "
                        "« Contexte demandé à Ollama » (page Cerveau).", sent,
                        self.config.ollama_context)  # fmt: skip
        base = self.config.url.rstrip("/").removesuffix("/v1")
        url = base + "/api/chat"
        calls: list[ToolCall] = []
        reasoned = 0
        done_reason = None
        try:
            async with self.http.post(
                url, json=body, headers=headers,
                timeout=aiohttp.ClientTimeout(total=600, sock_read=300),
            ) as response:  # fmt: skip
                if response.status != 200:
                    detail = (await response.text())[:300]
                    raise LlmError(f"LLM HTTP {response.status}: {detail}")
                async for raw in response.content:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise LlmError("Ollama returned invalid streaming JSON") from exc
                    if not isinstance(chunk, dict):
                        raise LlmError("Ollama returned an invalid streaming event")
                    if chunk.get("error"):
                        raise LlmError(f"Ollama: {str(chunk['error'])[:300]}")
                    message = chunk.get("message") or {}
                    reasoned += len(message.get("thinking") or "")
                    for item in message.get("tool_calls") or []:
                        function = item.get("function") or {}
                        arguments = function.get("arguments")
                        if not isinstance(arguments, str):
                            arguments = json.dumps(arguments or {}, ensure_ascii=False)
                        calls.append(ToolCall(id=item.get("id") or f"call_{len(calls)}",
                                              name=function.get("name") or "",
                                              arguments=arguments))  # fmt: skip
                    text = message.get("content")
                    if text:
                        result.text += text
                        yield text
                    if chunk.get("done"):
                        done_reason = chunk.get("done_reason") or "stop"
                        break
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise LlmError(f"LLM unreachable at {url}: {exc}") from exc
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise LlmError("Ollama returned malformed streaming content") from exc
        if done_reason is None:
            raise LlmError("LLM stream ended before completing the response")
        result.truncated = done_reason not in {"stop", "tool_calls"}
        if calls and result.truncated:
            raise LlmError("LLM did not complete its tool calls")
        if max_tokens != 1 and reasoned and not result.text.strip() and not calls:
            log.warning("Le modèle a réfléchi (%d caractères) sans rien répondre : augmentez "
                        "la longueur des réponses, ou n'activez pas la réflexion.",
                        reasoned)  # fmt: skip
        for call in calls:
            try:
                arguments = json.loads(call.arguments or "{}")
            except json.JSONDecodeError as exc:
                raise LlmError("LLM returned incomplete tool arguments") from exc
            if not call.name or not isinstance(arguments, dict):
                raise LlmError("LLM returned an invalid tool call")
        result.tool_calls = calls

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        result: Completion,
        *,
        max_tokens: int | None = None,
        reasoning_budget: int = 0,
        model: str | None = None,
        url: str | None = None,
    ) -> AsyncIterator[str]:
        """Yield text deltas; tool calls and the full text are collected into ``result``."""
        tools = model_tools(tools)
        if self.config.ollama_context:
            async for text in self._stream_ollama(messages, tools, result, max_tokens, model):
                yield text
            return
        body: dict[str, Any] = {
            # An Ollama / LM Studio brain needs its own model id; llama-server ignores it.
            "model": self.config.upstream_model or model or self.config.model,
            "messages": messages,
            "stream": True,
            "temperature": self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "cache_prompt": True,
        }
        if reasoning_budget:
            body["chat_template_kwargs"] = {"enable_thinking": True}
            body["reasoning_budget_tokens"] = reasoning_budget
            body["max_tokens"] += reasoning_budget
        if self.config.no_thinking:
            body["reasoning_effort"] = "none"  # Ollama: answer without thinking
        if self.lmstudio and self.config.lmstudio_reasoning != "auto":
            level = self.config.lmstudio_reasoning
            # LM Studio's chat endpoint accepts OpenAI effort names. On toggle-only
            # models, any positive effort enables thinking.
            body["reasoning_effort"] = {"off": "none", "on": "medium"}.get(level, level)
        if self.config.upstream_model:  # Ollama, LM Studio...: report the prompt size read
            body["stream_options"] = {"include_usage": True}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        endpoint = (url or self.config.url).rstrip("/") + "/chat/completions"
        async for text in self._chat_completion(body, endpoint, result):
            yield text
        if (self.lmstudio and result.exhausted_thinking and body["max_tokens"] > 1
                and body.get("reasoning_effort") != "none"):
            log.info("LM Studio a épuisé le budget en réflexion, nouvel essai sans réflexion.")
            async for text in self._chat_completion({**body, "reasoning_effort": "none"},
                                                    endpoint, result):
                yield text
        if self.lmstudio and result.exhausted_thinking and body["max_tokens"] > 1:
            raise LlmError("LM Studio a épuisé la limite de génération sans produire de réponse.")

    async def _chat_completion(self, body: dict, url: str,
                               result: Completion) -> AsyncIterator[str]:
        headers = {"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {}
        _dump(body)
        calls: dict[int, ToolCall] = {}
        reasoned = 0
        finished = False
        finish_reason = None
        try:
            async with self.http.post(
                url,
                json=body,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=120, sock_read=60),
            ) as response:
                if response.status != 200:
                    detail = (await response.text())[:300]
                    raise LlmError(f"LLM HTTP {response.status}: {detail}")
                async for raw in response.content:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        finished = True
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError as exc:
                        raise LlmError("LLM returned invalid streaming JSON") from exc
                    if not isinstance(chunk, dict):
                        raise LlmError("LLM returned an invalid streaming event")
                    if error := chunk.get("error"):
                        detail = error.get("message", "") if isinstance(error, dict) else error
                        raise LlmError(
                            f"LLM reported an error during generation: {str(detail)[:300]}")
                    if isinstance(chunk.get("usage"), dict):
                        self._check_truncation(body, chunk["usage"].get("prompt_tokens"))
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    finish_reason = choice.get("finish_reason") or finish_reason
                    delta = choice.get("delta") or {}
                    for item in delta.get("tool_calls") or []:
                        index = item.get("index", 0)
                        call = calls.setdefault(
                            index, ToolCall(id=item.get("id") or f"call_{index}", name="")
                        )
                        function = item.get("function") or {}
                        call.name += function.get("name") or ""
                        call.arguments += function.get("arguments") or ""
                        if item.get("id"):
                            call.id = item["id"]
                    reasoned += len(delta.get("reasoning") or delta.get("reasoning_content") or "")
                    text = delta.get("content")
                    if text:
                        result.text += text
                        yield text
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise LlmError(f"LLM unreachable at {url}: {exc}") from exc
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise LlmError("LLM returned malformed streaming content") from exc
        if not finished and finish_reason is None:
            raise LlmError("LLM stream ended before completing the response")
        result.truncated = finish_reason not in {None, "tool_calls", "stop"}
        thinking_only = bool(reasoned and not result.text.strip() and not calls)
        result.exhausted_thinking = thinking_only and finish_reason == "length"
        if body["max_tokens"] != 1 and thinking_only:
            log.warning("Le modèle a réfléchi (%d caractères) sans rien répondre : choisissez un "
                        "modèle sans réflexion, ou augmentez la longueur des réponses.", reasoned)
        if calls:
            # A truncated argument can still be valid JSON (or empty). Never execute
            # a partially generated action after a context/output limit was reached.
            if finish_reason not in {"tool_calls", "stop"}:
                raise LlmError("LLM did not complete its tool calls")
            for call in calls.values():
                try:
                    arguments = json.loads(call.arguments or "{}")
                except json.JSONDecodeError as exc:
                    raise LlmError("LLM returned incomplete tool arguments") from exc
                if not call.name or not isinstance(arguments, dict):
                    raise LlmError("LLM returned an invalid tool call")
        result.tool_calls = [c for _, c in sorted(calls.items()) if c.name]


def ollama_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI chat messages -> Ollama's own format (arguments as objects, tool names)."""
    names: dict[str, str] = {}
    converted = []
    for message in messages:
        message = dict(message)
        if message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                function = call.get("function") or {}
                names[call.get("id", "")] = function.get("name", "")
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                calls.append({"function": {"name": function.get("name", ""),
                                           "arguments": arguments}})  # fmt: skip
            message["tool_calls"] = calls
        if message.get("role") == "tool":
            name = names.get(message.pop("tool_call_id", ""), "")
            if name:
                message["tool_name"] = name
        converted.append(message)
    return converted


_TEXT_CALL = re.compile(r"^\s*`?([A-Za-z_][\w]*)\s*(\{.*\}|\(\s*\))\s*`?\s*$", re.S)


def text_tool_call(text: str, names: set[str]) -> ToolCall | None:
    """Some models (Gemma 4 with this llama.cpp) sometimes *write* the call instead of making
    it: ``clock__read()`` or ``lights__set{on:true,room:<|"|>cuisine<|"|>}``. Recover it."""
    match = _TEXT_CALL.match(text.strip())
    if not match or match.group(1) not in names:
        return None
    raw = match.group(2)
    arguments = "{}"
    if raw.startswith("{"):
        body = raw.replace('<|"|>', '"')
        body = re.sub(r"([{,]\s*)([A-Za-z_]\w*)\s*:", r'\1"\2":', body)
        try:
            arguments = json.dumps(json.loads(body), ensure_ascii=False)
        except json.JSONDecodeError:
            return None
    return ToolCall(id="call_text", name=match.group(1), arguments=arguments)
