"""The provider boundary — the only place that knows a wire format.

The loop knows ONLY this interface. Each provider translates our neutral Message
model to/from its API's JSON at its edge and nowhere else. That isolation is what
makes "multi-provider" a config choice instead of branching logic sprinkled through
the codebase.

## Who assembles the streaming message?

The PROVIDER does. Raw SSE is provider-specific, so each provider consumes its own
event shape and assembles our AssistantMessage, yielding:

    StreamChunk(delta, partial)   # repeatedly, as tokens arrive
    StreamDone(message)           # once, the finalized message

The loop just relays: StreamChunk -> MessageUpdate event, StreamDone -> return value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterable,
    Awaitable,
    Callable,
    Literal,
    Optional,
    Protocol,
    Union,
)

from pydantic import BaseModel

from curry_leaves.core.events import Delta
from curry_leaves.core.messages import AssistantMessage, Message

if TYPE_CHECKING:
    # Type-only: Context is a duck-typed bag tools read; providers ignore these
    # fields. Kept as string-annotated forward refs so this module doesn't force
    # blobs.py/host.py to exist yet (they're being ported in parallel).
    from curry_leaves.core.blobs import BlobStore
    from curry_leaves.core.host import Host


class Model(BaseModel):
    """A concrete model to call. `provider` selects which Provider handles it. Kept
    minimal; the catalog grows this with real context windows / pricing.
    """

    id: str
    provider: str
    max_output_tokens: int
    context_window: int
    supports_thinking: bool


def make_model(id: str, provider: str, **extra: Any) -> Model:
    defaults: dict[str, Any] = {
        "id": id,
        "provider": provider,
        "max_output_tokens": 4096,
        "context_window": 128_000,
        "supports_thinking": False,
    }
    defaults.update(extra)
    return Model.model_validate(defaults)


class ModelSettings(BaseModel):
    """Per-agent sampling/decoding settings. Only defined values are sent, so each
    provider falls back to its own defaults for anything unset.
    """

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None


class StreamOpts(BaseModel):
    """Loose per-call options bag passed to `provider.stream(ctx, model, opts)`."""

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    cache: Optional[bool] = None  # Anthropic prompt caching (default true)
    thinking_budget: Optional[int] = None  # Anthropic extended thinking
    reasoning_effort: Optional[str] = None  # OpenAI reasoning effort
    response_format: Optional[dict[str, Any]] = None  # OpenAI structured output


def settings_to_opts(s: ModelSettings) -> StreamOpts:
    o = StreamOpts()
    if s.temperature is not None:
        o.temperature = s.temperature
    if s.top_p is not None:
        o.top_p = s.top_p
    if s.max_tokens is not None:
        o.max_tokens = s.max_tokens
    return o


def apply_sampling(body: dict[str, Any], opts: StreamOpts) -> None:
    """Copy sampling params from opts into a request body (shared by providers)."""
    if opts.temperature is not None:
        body["temperature"] = opts.temperature
    if opts.top_p is not None:
        body["top_p"] = opts.top_p


class ToolSchema(BaseModel):
    """A tool as the model sees it on the wire: a name, a description, and a JSON
    Schema for its arguments. Produced from a Tool's pydantic args model.
    """

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class Context:
    """Everything needed to make one model call. The loop builds this from live state
    each turn (the "sync context" step). Most fields beyond system_prompt/messages/tools
    are duck-typed services tools read; providers ignore them.

    A dataclass (not a pydantic model): this is mutated in place each turn and holds
    live object references (blobs, host, spawn callable), not pure serializable data.
    """

    system_prompt: list[str]
    messages: list[Message]
    tools: list[ToolSchema]
    # service seams tools consult (providers ignore these)
    blobs: Optional["BlobStore"] = None
    resolve_local: Optional[Callable[[str], Optional[str]]] = None
    resolve_skill: Optional[Callable[[str], Optional[str]]] = None
    host: Optional["Host"] = None
    # Delegate to a subagent (agent-as-tool / task). None when depth-bounded.
    spawn: Optional[Callable[[Any, str], Awaitable[str]]] = None


# ── normalized streaming events (provider -> loop) ───────────────────────────


class StreamChunk(BaseModel):
    """One streaming step: what changed (`delta`) plus a snapshot of the message so far."""

    type: Literal["chunk"] = "chunk"
    delta: Delta
    partial: AssistantMessage


class StreamDone(BaseModel):
    """Terminal event: the finalized assistant message, with stop_reason + usage."""

    type: Literal["done"] = "done"
    message: AssistantMessage


StreamEvent = Union[StreamChunk, StreamDone]


class Provider(Protocol):
    """The one interface the loop depends on. Implement `stream`; the loop does the rest.

    (`build_request`/stream-parsers live as module functions in each provider so
    they're independently unit-testable without constructing the client.)
    """

    def stream(
        self, ctx: Context, model: Model, opts: Optional[StreamOpts] = None
    ) -> AsyncIterable[StreamEvent]: ...
