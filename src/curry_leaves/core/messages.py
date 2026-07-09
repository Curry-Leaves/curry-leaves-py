"""The internal message model — the single most important type in the system.

Design rule: this model is PROVIDER-NEUTRAL. Nothing here knows about Anthropic
content blocks, OpenAI ``tool_calls``, or any wire format. Providers translate
to/from this model at their boundary and nowhere else.

The shape mirrors how modern tool-using LLMs actually converse::

    user        -> a turn from the human
    assistant   -> a turn from the model: text + thinking + tool_call blocks
    tool_result -> the harness's reply to a tool_call (paired by id)

These three message types, looping, ARE the conversation.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

# ── Content blocks — the pieces that make up a message's `content`. ──────────
# An assistant turn is not a single string; it's an ordered list of blocks. The
# model may think, then say something, then call two tools, all in one turn. We
# preserve that order because the provider needs it back verbatim on the next
# call (especially `thinking` blocks with reasoning signatures).


class TextBlock(BaseModel):
    """Visible prose from the model (or a user)."""

    type: Literal["text"] = "text"
    text: str


class ThinkingBlock(BaseModel):
    """The model's reasoning.

    ``signature`` is an opaque provider token that some APIs (e.g. Anthropic
    extended thinking) require echoed back unchanged on the next request — so
    we store it verbatim and never inspect it.
    """

    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str | None = None


class ToolCallBlock(BaseModel):
    """A request from the model to run a tool.

    ``id`` pairs this call with its eventual ToolResultMessage. ``arguments``
    is the parsed JSON object; we validate it against the tool's schema at
    execution time, not here, so a malformed call can still be represented and
    answered with an error result (keeping the pairing the API requires).
    """

    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict[str, object] = Field(default_factory=dict)


class ImageBlock(BaseModel):
    """An image in a message — multimodal input.

    ``source`` is either base64-encoded bytes (``kind="base64"``, with
    ``media_type``) or a publicly-fetchable URL.
    """

    type: Literal["image"] = "image"
    source: str  # base64 payload OR a URL
    media_type: str = "image/png"
    kind: Literal["base64", "url"] = "base64"


Content = Annotated[
    Union[TextBlock, ThinkingBlock, ToolCallBlock, ImageBlock],
    Field(discriminator="type"),
]

# ── Usage / cost — attached to assistant messages for accounting. ────────────


class Cost(BaseModel):
    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    total: float = 0.0


def empty_cost() -> Cost:
    return Cost()


def add_cost(a: Cost, b: Cost) -> Cost:
    return Cost(
        input=a.input + b.input,
        output=a.output + b.output,
        cache_read=a.cache_read + b.cache_read,
        cache_write=a.cache_write + b.cache_write,
        total=a.total + b.total,
    )


class Usage(BaseModel):
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    total_tokens: int = 0
    cost: Cost = Field(default_factory=empty_cost)


def empty_usage() -> Usage:
    return Usage()


def add_usage(a: Usage, b: Usage) -> Usage:
    return Usage(
        input=a.input + b.input,
        output=a.output + b.output,
        cache_read=a.cache_read + b.cache_read,
        cache_write=a.cache_write + b.cache_write,
        total_tokens=a.total_tokens + b.total_tokens,
        cost=add_cost(a.cost, b.cost),
    )


# ── Messages — the three roles. ──────────────────────────────────────────────

# Why the model stopped. Provider metadata; the loop uses it to decide whether to
# run tools (tool_use/stop) or abandon truncated calls (length).
StopReason = Literal["tool_use", "stop", "length", "error", "aborted"]


class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: list[Content] = Field(default_factory=list)
    # How this message entered the conversation. None for the normal prompt path;
    # "steering"/"follow_up" when injected via Runner.steer()/follow_up() so
    # consumers (UI, telemetry) can tell them apart from the event stream alone.
    origin: Literal["steering", "follow_up"] | None = None


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: list[Content] = Field(default_factory=list)
    stop_reason: StopReason | None = None
    usage: Usage | None = None
    model: str | None = None
    # Set when stop_reason is "error"/"aborted" so the UI/telemetry can surface it.
    error_message: str | None = None


class ToolResultMessage(BaseModel):
    role: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    tool_name: str
    content: list[Content] = Field(default_factory=list)
    is_error: bool = False


Message = Annotated[
    Union[UserMessage, AssistantMessage, ToolResultMessage],
    Field(discriminator="role"),
]


# ── Tiny ergonomic constructors — keep call sites readable. ──────────────────


def user_text(text: str, *, origin: Literal["steering", "follow_up"] | None = None) -> UserMessage:
    return UserMessage(content=[TextBlock(text=text)], origin=origin)


def user_image(
    source: str,
    *,
    kind: Literal["base64", "url"] = "base64",
    media_type: str = "image/png",
    text: str = "",
) -> UserMessage:
    blocks: list[Content] = []
    if text:
        blocks.append(TextBlock(text=text))
    blocks.append(ImageBlock(source=source, kind=kind, media_type=media_type))
    return UserMessage(content=blocks)


def assistant_text(text: str, **extra: object) -> AssistantMessage:
    base: dict[str, object] = {
        "content": [TextBlock(text=text)],
        "stop_reason": None,
        "usage": None,
        "model": None,
        "error_message": None,
    }
    base.update(extra)
    return AssistantMessage.model_validate(base)


def empty_assistant(**extra: object) -> AssistantMessage:
    base: dict[str, object] = {
        "content": [],
        "stop_reason": None,
        "usage": None,
        "model": None,
        "error_message": None,
    }
    base.update(extra)
    return AssistantMessage.model_validate(base)


def tool_result_text(
    tool_call_id: str,
    tool_name: str,
    text: str,
    is_error: bool = False,
) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        content=[TextBlock(text=text)],
        is_error=is_error,
    )


def text_of(content: list[Content]) -> str:
    """Concatenate the visible text of a message's content blocks."""
    return "".join(b.text for b in content if isinstance(b, TextBlock))
