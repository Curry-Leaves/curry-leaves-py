"""The OpenAI provider — translates our neutral model to/from the Chat Completions API.

Same three-piece split as Anthropic. Key differences from Anthropic:
  - system prompt is joined to ONE string; tool results use a dedicated `tool` role.
  - assistant content is a flat string + a `tool_calls` array; thinking blocks are
    DROPPED (OpenAI reasoning is output-only and can't be replayed).
  - tool-call arguments go on the wire as a JSON STRING (not an object).
  - streaming has no explicit block start/stop — everything interleaves in `delta`.
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterable, AsyncIterator

import httpx

from curry_leaves.core.events import Delta
from curry_leaves.core.messages import (
    AssistantMessage,
    Content,
    StopReason,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    empty_assistant,
    empty_usage,
    text_of,
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

OPENAI_BASE_URL = "https://api.openai.com/v1"

FINISH_MAP: dict[str, StopReason] = {
    "tool_calls": "tool_use",
    "stop": "stop",
    "length": "length",
    "content_filter": "stop",
    "function_call": "tool_use",
}


def _assistant_to_wire(content: list[Content]) -> dict[str, Any]:
    text = text_of(content)
    tool_calls = [
        {
            "id": b.id,
            "type": "function",
            "function": {"name": b.name, "arguments": json.dumps(b.arguments)},
        }
        for b in content
        if isinstance(b, ToolCallBlock)
    ]
    msg: dict[str, Any] = {"role": "assistant", "content": text or None}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def build_openai_request(ctx: Context, model: Model, opts: StreamOpts) -> dict[str, Any]:
    wire: list[dict[str, Any]] = []
    if ctx.system_prompt:
        wire.append({"role": "system", "content": "\n\n".join(ctx.system_prompt)})

    for msg in ctx.messages:
        if msg.role == "user":
            wire.append({"role": "user", "content": text_of(msg.content)})
        elif msg.role == "assistant":
            wire.append(_assistant_to_wire(msg.content))
        else:
            wire.append(
                {"role": "tool", "tool_call_id": msg.tool_call_id, "content": text_of(msg.content)}
            )

    # Reasoning-family OpenAI models (gpt-5, o-series) reject `max_tokens` and require
    # `max_completion_tokens`; older models and OpenAI-compatible gateways (Ollama, …)
    # still expect `max_tokens`, so pick the key by model id.
    mid = model.id.lower()
    reasoning_family = mid.startswith(("gpt-5", "o1", "o3", "o4"))
    limit_key = "max_completion_tokens" if reasoning_family else "max_tokens"

    body: dict[str, Any] = {
        "model": model.id,
        "messages": wire,
        "stream": True,
        "stream_options": {"include_usage": True},
        limit_key: opts.max_tokens if opts.max_tokens is not None else model.max_output_tokens,
    }

    if ctx.tools:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in ctx.tools
        ]

    apply_sampling(body, opts)
    if opts.reasoning_effort:
        body["reasoning_effort"] = opts.reasoning_effort
    if opts.response_format:
        body["response_format"] = opts.response_format

    return body


async def parse_openai_stream(chunks: AsyncIterable[Any]) -> AsyncIterator[StreamEvent]:
    msg = empty_assistant()
    text = TextBlock(text="")
    thinking = ThinkingBlock(thinking="", signature=None)
    have_text = False
    have_thinking = False
    tools: dict[int, ToolCallBlock] = {}
    tool_args_raw: dict[int, str] = {}

    def rebuild() -> None:
        ordered: list[Content] = []
        if have_thinking:
            ordered.append(thinking)
        if have_text:
            ordered.append(text)
        for i in sorted(tools.keys()):
            ordered.append(tools[i])
        msg.content = ordered

    def snapshot() -> AssistantMessage:
        return msg.model_copy(deep=True)

    async for chunk in chunks:
        if msg.model is None:
            msg.model = chunk.get("model")
        if chunk.get("usage"):
            u = empty_usage()
            usage = chunk["usage"]
            u.input = usage.get("prompt_tokens") or 0
            u.output = usage.get("completion_tokens") or 0
            u.total_tokens = usage.get("total_tokens") or 0
            msg.usage = u

        choices = chunk.get("choices") or []
        if len(choices) == 0:
            continue  # usage-only final chunk
        choice = choices[0]
        delta = choice.get("delta") or {}

        if delta.get("content"):
            have_text = True
            text.text += delta["content"]
            rebuild()
            yield StreamChunk(
                delta=Delta(kind="text", block_index=0, value=delta["content"]),
                partial=snapshot(),
            )

        # Reasoning/thinking field names vary by server: `reasoning_content` (DeepSeek, vLLM),
        # `reasoning_text` (some gateways), `reasoning` (Ollama's OpenAI endpoint, OpenRouter),
        # `thinking` (Ollama native-style). Capture whichever is present.
        reasoning = (
            delta.get("reasoning_content")
            or delta.get("reasoning_text")
            or delta.get("reasoning")
            or delta.get("thinking")
        )
        if reasoning:
            have_thinking = True
            thinking.thinking += reasoning
            rebuild()
            yield StreamChunk(
                delta=Delta(kind="thinking", block_index=0, value=reasoning),
                partial=snapshot(),
            )

        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index") or 0
            if idx not in tools:
                tools[idx] = ToolCallBlock(id=tc.get("id") or f"call_{idx}", name="", arguments={})
                tool_args_raw[idx] = ""
            block = tools[idx]
            if tc.get("id"):
                block.id = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                block.name = fn["name"]
            if fn.get("arguments"):
                tool_args_raw[idx] = tool_args_raw.get(idx, "") + fn["arguments"]
                rebuild()
                yield StreamChunk(
                    delta=Delta(kind="tool_args", block_index=idx, value=fn["arguments"]),
                    partial=snapshot(),
                )

        if choice.get("finish_reason"):
            msg.stop_reason = FINISH_MAP.get(choice["finish_reason"], "stop")

    for idx, raw in tool_args_raw.items():
        block = tools[idx]
        try:
            block.arguments = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            block.arguments = {}
    rebuild()
    if msg.stop_reason is None:
        msg.stop_reason = "stop"
    yield StreamDone(message=snapshot())


class OpenAIProviderOptions:
    """Plain options bag (mirrors the TS `OpenAIProviderOptions` interface)."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        require_key: bool | None = None,
    ) -> None:
        self.api_key = api_key
        # API base URL (no trailing /chat/completions), e.g. an OpenAI-compatible gateway.
        self.base_url = base_url
        # Whether a key is mandatory. False for keyless local servers (Ollama, LM Studio).
        self.require_key = require_key


class OpenAIProvider:
    """The OpenAI provider, and the base for any OpenAI-COMPATIBLE server (Ollama, LM Studio,
    Groq, Together, …) — they differ only by base URL, auth, and which opts they accept.
    """

    label = "OpenAI"

    def __init__(self, options: OpenAIProviderOptions | None = None) -> None:
        self.options = options if options is not None else OpenAIProviderOptions()

    def _resolve_key(self) -> str | None:
        return self.options.api_key or os.environ.get("OPENAI_API_KEY")

    def _resolve_base_url(self) -> str:
        base = self.options.base_url or os.environ.get("OPENAI_BASE_URL") or OPENAI_BASE_URL
        return base.rstrip("/")

    def _prepare_opts(self, opts: StreamOpts) -> StreamOpts:
        """Hook for compatible servers to drop opts they don't understand."""
        return opts

    async def stream(
        self, ctx: Context, model: Model, opts: StreamOpts | None = None
    ) -> AsyncIterator[StreamEvent]:
        if opts is None:
            opts = StreamOpts()
        key = self._resolve_key()
        require_key = self.options.require_key if self.options.require_key is not None else True
        if require_key and not key:
            raise RuntimeError(f"{self.label}: no API key set (OPENAI_API_KEY)")

        body = build_openai_request(ctx, model, self._prepare_opts(opts))
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self._resolve_base_url()}/chat/completions",
                headers=headers,
                json=body,
                timeout=None,
            ) as resp:
                if resp.status_code >= 400:
                    t = ""
                    try:
                        await resp.aread()
                        t = resp.text
                    except Exception:
                        t = ""
                    raise HttpError(
                        resp.status_code, f"{self.label} {resp.status_code}: {t[:500]}"
                    )

                async for event in parse_openai_stream(iter_sse(resp, done_sentinel="[DONE]")):
                    yield event


class OllamaProvider(OpenAIProvider):
    """Ollama — a local OpenAI-compatible server (default http://localhost:11434). Point it at
    any pulled model tag (e.g. "gemma4", "qwen3.6", "llama3.2"). No API key needed. Tool
    use requires a model with the `tools` capability. Reasoning knobs are dropped (Ollama's
    OpenAI endpoint doesn't accept `reasoning_effort`).
    """

    label = "Ollama"

    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        host = os.environ.get("OLLAMA_HOST")
        base = base_url or (f"{host.rstrip('/')}/v1" if host else "http://localhost:11434/v1")
        super().__init__(
            OpenAIProviderOptions(
                base_url=base, api_key=api_key or "ollama", require_key=False
            )
        )

    def _prepare_opts(self, opts: StreamOpts) -> StreamOpts:
        rest = opts.model_copy()
        rest.reasoning_effort = None
        rest.thinking_budget = None
        return rest
