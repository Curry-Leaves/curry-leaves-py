"""The Anthropic provider — translates our neutral model to/from the Messages API.

Split into three pieces so the wire logic is testable without network:
  build_anthropic_request  (to_wire)   — neutral Context -> request JSON
  parse_anthropic_stream   (from_wire) — parsed SSE dicts -> StreamChunk/StreamDone
  AnthropicProvider.stream             — the thin HTTP + SSE glue
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterable, AsyncIterator, Optional

import httpx

from curry_leaves.core.events import Delta
from curry_leaves.core.messages import (
    AssistantMessage,
    Content,
    ImageBlock,
    StopReason,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    Usage,
    empty_assistant,
    empty_usage,
)
from curry_leaves.providers.base import (
    Context,
    Model,
    StreamChunk,
    StreamDone,
    StreamEvent,
    StreamOpts,
    apply_sampling,
)
from curry_leaves.providers.sse import iter_sse
from curry_leaves.util.retry import HttpError

_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"

_STOP_MAP: dict[str, StopReason] = {
    "tool_use": "tool_use",
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "pause_turn": "stop",
}


def _content_to_wire(block: Content) -> dict[str, Any]:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ThinkingBlock):
        w: dict[str, Any] = {"type": "thinking", "thinking": block.thinking}
        if block.signature:
            w["signature"] = block.signature
        return w
    if isinstance(block, ToolCallBlock):
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.arguments,
        }
    if isinstance(block, ImageBlock):
        if block.kind == "url":
            return {"type": "image", "source": {"type": "url", "url": block.source}}
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": block.media_type,
                "data": block.source,
            },
        }
    raise AssertionError(f"unreachable content block: {block!r}")


def build_anthropic_request(
    ctx: Context, model: Model, opts: StreamOpts
) -> dict[str, Any]:
    # Anthropic has no tool_result role: a tool result is a content block inside a USER
    # message, and consecutive tool results are grouped into a single user message.
    wire_messages: list[dict[str, Any]] = []
    pending_results: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal pending_results
        if pending_results:
            wire_messages.append({"role": "user", "content": pending_results})
            pending_results = []

    for msg in ctx.messages:
        if msg.role == "tool_result":
            pending_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": [_content_to_wire(b) for b in msg.content],
                    "is_error": msg.is_error,
                }
            )
            continue
        flush()
        wire_messages.append(
            {"role": msg.role, "content": [_content_to_wire(b) for b in msg.content]}
        )
    flush()

    body: dict[str, Any] = {
        "model": model.id,
        "max_tokens": opts.max_tokens if opts.max_tokens is not None else model.max_output_tokens,
        "messages": wire_messages,
        "stream": True,
    }

    cache = opts.cache if opts.cache is not None else True

    if ctx.system_prompt:
        if cache:
            blocks: list[dict[str, Any]] = [
                {"type": "text", "text": s} for s in ctx.system_prompt
            ]
            blocks[-1]["cache_control"] = {"type": "ephemeral"}
            body["system"] = blocks
        else:
            body["system"] = "\n\n".join(ctx.system_prompt)

    if ctx.tools:
        tools: list[dict[str, Any]] = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in ctx.tools
        ]
        if cache:
            tools[-1]["cache_control"] = {"type": "ephemeral"}
        body["tools"] = tools

    apply_sampling(body, opts)

    if opts.thinking_budget:
        body["thinking"] = {"type": "enabled", "budget_tokens": opts.thinking_budget}
        body["max_tokens"] = max(body["max_tokens"], opts.thinking_budget + 1024)

    return body


async def parse_anthropic_stream(
    events: AsyncIterable[Any],
) -> AsyncIterator[StreamEvent]:
    msg = empty_assistant()
    tool_json: dict[int, str] = {}

    def snapshot() -> AssistantMessage:
        return msg.model_copy(deep=True)

    async for evt in events:
        t = evt.get("type")

        if t == "message_start":
            m = evt.get("message") or {}
            msg.model = m.get("model")
            u = m.get("usage") or {}
            usage: Usage = empty_usage()
            usage.input = u.get("input_tokens") or 0
            usage.cache_write = u.get("cache_creation_input_tokens") or 0
            usage.cache_read = u.get("cache_read_input_tokens") or 0
            msg.usage = usage
        elif t == "content_block_start":
            idx = evt["index"]
            cb = evt.get("content_block") or {}
            while len(msg.content) <= idx:
                msg.content.append(TextBlock(text=""))
            if cb.get("type") == "text":
                msg.content[idx] = TextBlock(text=cb.get("text") or "")
            elif cb.get("type") == "thinking":
                msg.content[idx] = ThinkingBlock(
                    thinking=cb.get("thinking") or "", signature=None
                )
            elif cb.get("type") == "tool_use":
                msg.content[idx] = ToolCallBlock(
                    id=cb["id"], name=cb["name"], arguments={}
                )
                tool_json[idx] = ""
        elif t == "content_block_delta":
            idx = evt["index"]
            d = evt.get("delta") or {}
            if idx >= len(msg.content):
                continue
            block = msg.content[idx]
            dtype = d.get("type")
            if dtype == "text_delta" and isinstance(block, TextBlock):
                block.text += d["text"]
                yield StreamChunk(
                    delta=Delta(kind="text", block_index=idx, value=d["text"]),
                    partial=snapshot(),
                )
            elif dtype == "thinking_delta" and isinstance(block, ThinkingBlock):
                block.thinking += d["thinking"]
                yield StreamChunk(
                    delta=Delta(kind="thinking", block_index=idx, value=d["thinking"]),
                    partial=snapshot(),
                )
            elif dtype == "signature_delta" and isinstance(block, ThinkingBlock):
                block.signature = (block.signature or "") + d["signature"]
                yield StreamChunk(
                    delta=Delta(
                        kind="signature", block_index=idx, value=d["signature"]
                    ),
                    partial=snapshot(),
                )
            elif dtype == "input_json_delta":
                tool_json[idx] = tool_json.get(idx, "") + d["partial_json"]
                yield StreamChunk(
                    delta=Delta(
                        kind="tool_args", block_index=idx, value=d["partial_json"]
                    ),
                    partial=snapshot(),
                )
        elif t == "content_block_stop":
            idx = evt["index"]
            if idx in tool_json:
                raw = tool_json.pop(idx)
                block = msg.content[idx] if idx < len(msg.content) else None
                if isinstance(block, ToolCallBlock):
                    try:
                        block.arguments = json.loads(raw) if raw.strip() else {}
                    except (json.JSONDecodeError, ValueError):
                        block.arguments = {}
        elif t == "message_delta":
            stop = (evt.get("delta") or {}).get("stop_reason")
            if stop:
                msg.stop_reason = _STOP_MAP.get(stop, "stop")
            out = (evt.get("usage") or {}).get("output_tokens")
            if out is not None and msg.usage is not None:
                msg.usage.output = out
                # Cache reads/writes are prompt tokens too — leaving them out under-reports.
                msg.usage.total_tokens = (
                    msg.usage.input + msg.usage.cache_read + msg.usage.cache_write + out
                )
        elif t == "message_stop":
            break
        elif t == "error":
            msg.stop_reason = "error"
            msg.error_message = (evt.get("error") or {}).get("message") or "provider error"
            break

    yield StreamDone(message=snapshot())


class AnthropicProvider:
    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key

    async def stream(
        self, ctx: Context, model: Model, opts: Optional[StreamOpts] = None
    ) -> AsyncIterator[StreamEvent]:
        opts = opts if opts is not None else StreamOpts()
        key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")

        body = build_anthropic_request(ctx, model, opts)

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                _ANTHROPIC_URL,
                headers={
                    "x-api-key": key,
                    "anthropic-version": _ANTHROPIC_VERSION,
                    "content-type": "application/json",
                },
                json=body,
                timeout=None,
            ) as resp:
                if resp.status_code >= 400:
                    text = ""
                    try:
                        await resp.aread()
                        text = resp.text
                    except Exception:
                        text = ""
                    raise HttpError(
                        resp.status_code,
                        f"Anthropic {resp.status_code}: {text[:500]}",
                    )

                async for event in parse_anthropic_stream(iter_sse(resp)):
                    yield event
