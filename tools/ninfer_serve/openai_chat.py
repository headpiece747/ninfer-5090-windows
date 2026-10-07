"""OpenAI Chat Completions request encoding and stream classification."""

from __future__ import annotations

import json
from typing import Any, Sequence

from tools.ninfer_serve.client import ProtocolEvent, ProtocolRequest, ServeProtocolError
from tools.streaming_http.sse import SseEvent


def chat_request(
    model: str,
    messages: Sequence[dict[str, Any]],
    max_output_tokens: int,
    *,
    stream: bool = True,
    tools: Sequence[dict[str, Any]] | None = None,
    reasoning_effort: str | None = None,
) -> ProtocolRequest:
    if max_output_tokens <= 0:
        raise ValueError("Chat max_output_tokens must be positive")
    payload: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "max_completion_tokens": max_output_tokens,
        "temperature": 0,
        "stream": stream,
    }
    if reasoning_effort is None:
        payload["enable_thinking"] = False
    else:
        # The engine refuses a request that states both, so an effort value replaces the explicit
        # `enable_thinking: false` rather than sitting beside it. Thinking is then on, which is the
        # shape a client that sends `reasoning_effort` has.
        payload["reasoning_effort"] = reasoning_effort
    if stream:
        payload["stream_options"] = {"include_usage": True}
    if tools is not None:
        payload["tools"] = list(tools)
        payload["tool_choice"] = "auto"
    return ProtocolRequest(
        protocol="openai_chat",
        path="/v1/chat/completions",
        payload=payload,
        stream=stream,
    )


def openai_tools(anthropic_tools: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The OpenAI wire shape for tools the corpus stores in Anthropic's.

    `bench/fixtures/ttft/text/client_tools_32.json` is Anthropic-shaped (`input_schema`), and the
    Anthropic request builder takes it as it stands; the OpenAI wire wants `type: "function"` with the
    schema under `parameters`, and the engine refuses anything else with "tools entries must contain a
    string type" (measured 2026-10-07, one 400 per request). One place converts, so a case does not
    have to.
    """
    converted: list[dict[str, Any]] = []
    for tool in anthropic_tools:
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            raise ServeProtocolError("tool fixture entry has no name")
        function: dict[str, Any] = {"name": name}
        if isinstance(tool.get("description"), str):
            function["description"] = tool["description"]
        if isinstance(tool.get("input_schema"), dict):
            function["parameters"] = tool["input_schema"]
        converted.append({"type": "function", "function": function})
    return converted


class ChatStreamAdapter:
    def consume(self, event: SseEvent) -> list[ProtocolEvent]:
        if event.data == "[DONE]":
            return [ProtocolEvent("terminal", "done", event.received_ns)]
        try:
            payload = json.loads(event.data)
        except json.JSONDecodeError as error:
            raise ServeProtocolError("OpenAI Chat SSE data is not JSON") from error
        if not isinstance(payload, dict):
            raise ServeProtocolError("OpenAI Chat SSE payload is not an object")
        if isinstance(payload.get("error"), dict):
            error = payload["error"]
            return [
                ProtocolEvent(
                    "error",
                    "error",
                    event.received_ns,
                    payload=payload,
                    error_code=error.get("code") if isinstance(error.get("code"), str) else None,
                    error_message=(
                        error.get("message") if isinstance(error.get("message"), str) else None
                    ),
                )
            ]
        response_id = payload.get("id") if isinstance(payload.get("id"), str) else None
        choices = payload.get("choices")
        if not isinstance(choices, list):
            raise ServeProtocolError("OpenAI Chat SSE payload has no choices array")
        normalized: list[ProtocolEvent] = []
        for choice in choices:
            if not isinstance(choice, dict):
                raise ServeProtocolError("OpenAI Chat choice is not an object")
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                raise ServeProtocolError("OpenAI Chat choice has no delta object")
            if delta.get("role") == "assistant":
                normalized.append(
                    ProtocolEvent(
                        "accepted",
                        "assistant_role",
                        event.received_ns,
                        payload=payload,
                        response_id=response_id,
                    )
                )
            outputs: list[str] = []
            for key in ("content", "reasoning_content"):
                value = delta.get(key)
                if isinstance(value, str) and value:
                    outputs.append(value)
            tool_calls = delta.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                outputs.append(json.dumps(tool_calls, ensure_ascii=False, separators=(",", ":")))
            if outputs:
                normalized.append(
                    ProtocolEvent(
                        "model_output",
                        "delta",
                        event.received_ns,
                        payload=payload,
                        response_id=response_id,
                        output="".join(outputs),
                    )
                )
        if not normalized:
            normalized.append(
                ProtocolEvent(
                    "metadata",
                    "chunk",
                    event.received_ns,
                    payload=payload,
                    response_id=response_id,
                )
            )
        return normalized
