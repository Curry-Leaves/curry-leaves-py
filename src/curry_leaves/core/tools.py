"""The tool system — a registry of typed tools, and the executor the loop calls.

A tool is DATA + a callable. Its arguments are a pydantic model, which gives us the
JSON Schema the provider advertises to the model — for free. The loop never hardcodes
a tool; it just calls `execute_tools`, which dispatches by name and runs approved calls
CONCURRENTLY, streaming each result as it finishes.
"""

from __future__ import annotations

import asyncio
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Iterable,
    Literal,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import pydantic

from curry_leaves.core.events import AgentEvent, ev
from curry_leaves.core.messages import ToolCallBlock, ToolResultMessage, tool_result_text

if TYPE_CHECKING:
    from curry_leaves.core.blobs import BlobStore
    from curry_leaves.providers.base import Context, ToolSchema

# How dangerous a tool is — the input to the permission gate.
Risk = Literal["read", "write", "exec", "network"]


class AuthorizeResult(pydantic.BaseModel):
    ok: bool
    reason: str


@runtime_checkable
class Permission(Protocol):
    """The permission gate the executor consults before running a call — structural, so
    `core` doesn't depend on the PermissionEngine (which would be circular). The Runner
    injects one.
    """

    async def authorize(
        self, tool: str, risk: Risk, args: dict[str, object]
    ) -> AuthorizeResult: ...


class ToolResult(pydantic.BaseModel):
    """What a tool's `run` returns. `content` is the text shown back to the model."""

    content: str
    is_error: bool = False


# The args type each concrete tool's `run` takes — bound to pydantic.BaseModel so
# `schema`/`model_validate` stay meaningful. Mirrors the TS `Tool<A = any>` generic.
ArgsT = TypeVar("ArgsT", bound=pydantic.BaseModel)


@runtime_checkable
class Tool(Protocol[ArgsT]):
    """A tool: a name, a description, a pydantic args schema, and a `run`. `risk` is
    advisory; `timeout` (seconds) bounds a single call. `close` tears down any stateful
    resources.

    Structural (Protocol), matching the TS `interface Tool<A>` — any object with these
    attributes/methods satisfies it, no base class required. Generic over the args
    type so each concrete tool can narrow `run`'s parameter to its own pydantic model
    (e.g. `async def run(self, args: WriteArgs, ...)`) without violating Liskov —
    a non-generic Protocol would make every concrete `run` contravariantly
    incompatible with a `BaseModel`-typed base method.

    `close` is OPTIONAL (like the TS `close?()`) — not declared here, since a Protocol
    member is always required for structural matching and several concrete tools (e.g.
    `SearchTool`) don't define one. Callers must fetch it defensively, e.g.
    `getattr(tool, "close", None)` (see `Runner.close()`), mirroring `tool.close?.()`.
    """

    name: str
    description: str
    schema: type[ArgsT]
    risk: Risk | None
    timeout: float | None

    async def run(self, args: ArgsT, ctx: "Context", signal: asyncio.Event) -> ToolResult: ...


# Note: there is no `define_tool` helper in the Python port. TS's `defineTool` exists
# only to give an object literal a type annotation for terser call sites; Python has no
# equivalent need — a tool is just any object (dataclass, plain class, module-level
# instance) that structurally satisfies the `Tool` Protocol above.

# ── Universal large-result guard ─────────────────────────────────────────────
# No single tool result should dominate the context window. EVERY tool result passes
# through `cap_result_text`: anything over the budget is offloaded WHOLE to the blob
# store, and the model keeps a head+tail preview plus an `artifact://<id>` URL it can
# `read`.

MAX_RESULT_CHARS = 24_000
_MARKER_RESERVE = 300


def cap_result_text(
    text: str,
    tool_name: str,
    blobs: "BlobStore | None",
    limit: int = MAX_RESULT_CHARS,
) -> str:
    if len(text) <= limit:
        return text

    avail = max(200, limit - _MARKER_RESERVE)
    head_n = int(avail * 0.85)
    tail_n = avail - head_n
    head = text[:head_n]
    tail = text[len(text) - tail_n :]
    elided = len(text) - head_n - tail_n

    if blobs is None:
        return (
            f"{head}\n\n... [{tool_name} output truncated: {len(text)} chars, "
            f"{elided} elided — no artifact store to offload to] ...\n\n{tail}"
        )

    bid = blobs.put_text(text)
    return (
        f"{head}\n\n"
        f"... [{tool_name} output too large for context: {len(text)} chars; {elided} elided] ...\n"
        f"Full output saved to artifact://{bid} — `read` it (with offset/limit) to see the omitted middle.\n\n"
        f"{tail}"
    )


def _json_schema_of(schema: type[pydantic.BaseModel]) -> dict[str, object]:
    # pydantic's default model_json_schema() already treats fields with defaults as
    # optional in the schema — matching the "input" view the TS side had to opt into
    # via `io: "input"` (Zod 4 otherwise defaults to the stricter "output" view).
    return schema.model_json_schema()


class ToolRegistry:
    """Holds tools and produces their wire schemas. Tools are either ALWAYS-ON or
    DEFERRED: deferred tools aren't advertised upfront — the model finds them via
    `search_tools` and they're then activated. Keeps the advertised list small even
    with a big catalog.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool[Any]] = {}
        self._deferred: set[str] = set()

    def register(self, tool: Tool[Any], *, deferred: bool = False) -> None:
        self._tools[tool.name] = tool
        if deferred:
            self._deferred.add(tool.name)
        else:
            self._deferred.discard(tool.name)

    def get(self, name: str) -> Tool[Any] | None:
        return self._tools.get(name)

    def is_deferred(self, name: str) -> bool:
        return name in self._deferred

    def tools(self) -> list[Tool[Any]]:
        return list(self._tools.values())

    def schema(self, name: str) -> "ToolSchema | None":
        from curry_leaves.providers.base import ToolSchema

        t = self._tools.get(name)
        if t is None:
            return None
        return ToolSchema(
            name=t.name, description=t.description, input_schema=_json_schema_of(t.schema)
        )

    def advertised_schemas(self, active: Iterable[str] = ()) -> list["ToolSchema"]:
        """What to advertise: always-on tools + any activated deferred ones."""
        active_set = set(active)
        out: list["ToolSchema"] = []
        for name in self._tools.keys():
            if name not in self._deferred or name in active_set:
                s = self.schema(name)
                if s is not None:
                    out.append(s)
        return out

    def deferred_teasers(self, active: Iterable[str] = ()) -> list[tuple[str, str]]:
        """(name, first sentence of description) for deferred tools NOT yet activated —
        the prompt lists these so the model knows what `search_tools` can unlock. Without
        the listing, a prompt that references a deferred tool by name sends the model
        calling it directly; the provider's schema constraint then collapses the call
        onto the nearest advertised name (e.g. list_artifacts -> list_todos) and the
        model loops, baffled."""
        active_set = set(active)
        out: list[tuple[str, str]] = []
        for name in sorted(self._deferred):
            if name in active_set:
                continue
            desc = (self._tools[name].description or "").strip()
            first = desc.split(". ")[0].strip().rstrip(".")
            if len(first) > 120:
                first = first[:120].rsplit(" ", 1)[0] + "…"
            out.append((name, first))
        return out

    def search(self, query: str, limit: int = 5) -> list[Tool[Any]]:
        """Keyword-rank the DEFERRED tools for `search_tools`. Whole-WORD matching (set
        intersection), not substring; drops tokens < 3 chars; ties break by name.
        """
        import re

        terms = {t for t in re.split(r"\W+", query.lower()) if len(t) >= 3}
        scored: list[tuple[int, str, Tool[Any]]] = []
        for name in self._deferred:
            tool = self._tools[name]
            properties = _json_schema_of(tool.schema).get("properties")
            props = list(properties.keys()) if isinstance(properties, dict) else []
            hay = set(re.split(r"\W+", f"{name} {tool.description} {' '.join(props)}".lower()))
            score = sum(1 for t in terms if t in hay)
            if score:
                scored.append((score, name, tool))
        scored.sort(key=lambda s: (-s[0], s[1]))
        return [s[2] for s in scored[:limit]]


# ── the executor ─────────────────────────────────────────────────────────────


_T = TypeVar("_T")


async def _await_quietly(task: "asyncio.Task[_T]") -> None:
    """Await a task whose outcome (result or exception) we don't care about — used
    when reaping a cancelled/superseded task."""
    try:
        await task
    except BaseException:  # noqa: BLE001 - intentionally swallow everything here
        pass


def _swallow_late_result(task: "asyncio.Task[_T]") -> None:
    """Done-callback that retrieves (and discards) a task's exception, if any, so
    asyncio doesn't log "exception never retrieved" for an abandoned tool call."""
    if not task.cancelled():
        task.exception()


async def _run_one(
    registry: ToolRegistry,
    call: ToolCallBlock,
    ctx: "Context",
    signal: asyncio.Event,
    max_result_chars: int,
    tool_timeout: float | None,
) -> ToolResultMessage:
    blobs = getattr(ctx, "blobs", None)

    def finalize(text: str) -> str:
        return cap_result_text(text, call.name, blobs, max_result_chars)

    tool = registry.get(call.name)
    if tool is None:
        return tool_result_text(call.id, call.name, finalize(f"Unknown tool: '{call.name}'"), True)

    try:
        parsed = tool.schema.model_validate(call.arguments)
    except pydantic.ValidationError as e:
        msg = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '(root)'}: {err['msg']}" for err in e.errors()
        )
        return tool_result_text(call.id, call.name, finalize(f"Invalid arguments: {msg}"), True)

    deadline_s = tool.timeout if tool.timeout is not None else tool_timeout

    # A per-call cancellation signal, independent of the shared outer `signal` — mirrors
    # the TS AbortController created per call, chained to the outer AbortSignal so
    # aborting the whole run also cancels every in-flight call.
    own_cancel = asyncio.Event()

    async def watch_outer() -> None:
        await signal.wait()
        own_cancel.set()

    watcher = asyncio.ensure_future(watch_outer())

    try:
        run_task = asyncio.ensure_future(tool.run(parsed, ctx, own_cancel))
        try:
            if deadline_s is not None:
                try:
                    out = await asyncio.wait_for(asyncio.shield(run_task), timeout=deadline_s)
                except asyncio.TimeoutError:
                    own_cancel.set()
                    # The shielded task may still be running; if it later fails, retrieve the
                    # exception so asyncio doesn't log "Task exception was never retrieved".
                    run_task.add_done_callback(_swallow_late_result)
                    return tool_result_text(
                        call.id,
                        call.name,
                        finalize(f"Tool '{call.name}' timed out after {deadline_s}s and was cancelled."),
                        True,
                    )
            else:
                out = await run_task
        except Exception as e:  # noqa: BLE001 - mirrors TS catch(e) at the tool-run boundary
            return tool_result_text(call.id, call.name, finalize(f"{type(e).__name__}: {e}"), True)
        return tool_result_text(call.id, call.name, finalize(out.content), out.is_error)
    finally:
        watcher.cancel()
        await _await_quietly(watcher)


ToolExecutor = Callable[[list[ToolCallBlock], "Context", asyncio.Event], AsyncIterator[AgentEvent]]


def make_executor(
    registry: ToolRegistry,
    *,
    max_result_chars: int | None = None,
    tool_timeout: float | None = None,
    permission: Permission | None = None,
) -> ToolExecutor:
    """Build the `execute_tools` the loop calls. Emits ToolStart for every call, runs
    them CONCURRENTLY, and streams each ToolEnd as it finishes. When `signal` is set
    (the steering interrupt), still-running tools are reported as interrupted and
    cancelled.
    """
    resolved_max_result_chars = max_result_chars if max_result_chars is not None else MAX_RESULT_CHARS

    async def execute_tools(
        tool_calls: list[ToolCallBlock], ctx: "Context", signal: asyncio.Event
    ) -> AsyncIterator[AgentEvent]:
        for call in tool_calls:
            yield ev.tool_start(call.id, call.name, call.arguments)
        if len(tool_calls) == 0:
            return

        # Phase 1 — AUTHORIZE (serial): the permission gate. Prompts (if any) happen one
        # at a time, before anything runs. Denied calls get an error result (pairing
        # preserved) and never run.
        approved: list[ToolCallBlock] = []
        for call in tool_calls:
            if permission is not None:
                tool = registry.get(call.name)
                # A tool that doesn't declare its risk fails SAFE to "exec" (always prompts) —
                # defaulting to "read" would let an undeclared tool bypass the gate entirely.
                risk: Risk = tool.risk if tool is not None and tool.risk is not None else "exec"
                decision = await permission.authorize(call.name, risk, call.arguments)
                if not decision.ok:
                    yield ev.tool_end(
                        call.id,
                        call.name,
                        tool_result_text(call.id, call.name, f"Not run — {decision.reason}.", True),
                    )
                    continue
            approved.append(call)
        if len(approved) == 0:
            return

        # Phase 2 — RUN approved calls concurrently, streaming each result as it finishes.
        pending: dict[int, asyncio.Task[ToolResultMessage]] = {}
        task_to_index: dict[asyncio.Task[Any], int] = {}
        for i, call in enumerate(approved):
            task = asyncio.ensure_future(
                _run_one(registry, call, ctx, signal, resolved_max_result_chars, tool_timeout)
            )
            pending[i] = task
            task_to_index[task] = i

        interrupt_task: asyncio.Task[Any] = asyncio.ensure_future(signal.wait())

        try:
            while pending:
                waitable: list[asyncio.Task[Any]] = [*pending.values(), interrupt_task]
                done, _ = await asyncio.wait(waitable, return_when=asyncio.FIRST_COMPLETED)
                # Tool tasks that completed in the same wave as the interrupt still finished —
                # report their real results; only the still-pending ones are "interrupted".
                for task in done:
                    if task is interrupt_task:
                        continue
                    i = task_to_index[task]
                    del pending[i]
                    call = approved[i]
                    yield ev.tool_end(call.id, call.name, task.result())
                if interrupt_task in done:
                    # Mirror the TS: don't force-cancel the still-running tasks, just stop
                    # waiting on them here and swallow whatever they eventually resolve/raise
                    # — each `_run_one` call already has its own outer-signal watcher that
                    # cooperatively cancels the tool's own `run` via `own_cancel`. Attaching a
                    # no-op done-callback just prevents "exception never retrieved" warnings.
                    for p in pending.values():
                        p.add_done_callback(_swallow_late_result)
                    for i in pending.keys():
                        call = approved[i]
                        yield ev.tool_end(
                            call.id,
                            call.name,
                            tool_result_text(call.id, call.name, "Interrupted by user.", True),
                        )
                    return
        finally:
            if not interrupt_task.done():
                interrupt_task.cancel()
                await _await_quietly(interrupt_task)

    return execute_tools
