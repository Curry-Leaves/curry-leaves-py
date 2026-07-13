"""The agent loop — the engine.

The loop is an async generator that YIELDS events. It produces; the Runner and
frontends consume. The turn cycle:

    call provider -> stream an assistant message -> if it called tools, run them
    and feed results back -> repeat while there are tool calls.

The one decision that drives everything:

    runnable = stop_reason in ("tool_use","stop") AND tool_calls exist

Tools called -> loop again. None -> stop. That's the whole control flow.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Optional

from curry_leaves.core.events import AgentEvent, MessageEnd, ev
from curry_leaves.core.messages import (
    AssistantMessage,
    Message,
    ToolCallBlock,
    ToolResultMessage,
    empty_assistant,
    tool_result_text,
)
from curry_leaves.core.tools import ToolExecutor
from curry_leaves.providers.base import Context, Model, Provider, StreamOpts
from curry_leaves.util.retry import RetryPolicy


class Interrupt:
    """A resettable interrupt: `set()` aborts the current signal (cancelling running
    tools); `clear()` mints a fresh one for the next turn. Modeled this way because an
    asyncio.Event can't be un-set atomically with fresh waiters — the loop needs to fold
    steering in and keep going, so we swap in a brand new Event on clear().
    """

    def __init__(self) -> None:
        self._event = asyncio.Event()

    @property
    def signal(self) -> asyncio.Event:
        return self._event

    def is_set(self) -> bool:
        return self._event.is_set()

    def set(self) -> None:
        self._event.set()

    def clear(self) -> None:
        self._event = asyncio.Event()


@dataclass
class LoopConfig:
    provider: Provider
    model: Model
    execute_tools: ToolExecutor
    opts: StreamOpts
    max_turns: int
    interrupt: Interrupt
    # Live input, folded mid-run.
    drain_steering: Optional[Callable[[], list[Message]]] = None
    # Queued for after the run.
    drain_follow_ups: Optional[Callable[[], list[Message]]] = None
    # Called before EACH model turn to refresh ctx.tools / ctx.system_prompt from live state.
    sync_context: Optional[Callable[[Context], None]] = None
    # Checked before each turn; true ends the run now (lets a tool terminate mechanically).
    should_stop: Optional[Callable[[], bool]] = None
    # Retry policy for transient provider failures.
    retry: Optional[RetryPolicy] = None


# ── L1: stream one assistant message, relaying provider events as loop events. ──


async def stream_assistant(ctx: Context, cfg: LoopConfig) -> AsyncIterator[AgentEvent]:
    started = False
    last_partial: Optional[AssistantMessage] = None
    last_error = "provider stream ended without a completion"
    attempt = 0

    while True:
        try:
            async for sev in cfg.provider.stream(ctx, cfg.model, cfg.opts):
                if sev.type == "chunk":
                    last_partial = sev.partial
                    if not started:
                        yield ev.message_start(sev.partial)
                        started = True
                    yield ev.message_update(sev.partial, sev.delta)
                else:
                    message = sev.message
                    if not started:
                        yield ev.message_start(message)
                    yield ev.message_end(message)
                    return
            break  # stream ended without a "done" -> fall through to the fallback
        except Exception as e:
            # Retry only TRANSIENT failures, and only if nothing was streamed yet (retrying
            # mid-stream would duplicate output to the consumer).
            retry = cfg.retry
            name = type(e).__name__
            if not started and retry is not None and retry.is_transient(e) and attempt < retry.max_attempts:
                attempt += 1
                delay = retry.delay(attempt)
                yield ev.error(f"{name}: retrying in {delay:.1f}s ({attempt}/{retry.max_attempts})", False)
                await asyncio.sleep(delay)
                continue
            last_error = f"provider error: {name}: {e}"
            break

    # Stream ended without completion, or retries exhausted. Emit a terminal error message.
    fallback: AssistantMessage = (
        last_partial.model_copy(deep=True) if last_partial is not None else empty_assistant()
    )
    fallback.stop_reason = "error"
    fallback.error_message = last_error
    if not started:
        yield ev.message_start(fallback)
    yield ev.message_end(fallback)


# ── L2/L3: the turn cycle + the session loop. ────────────────────────────────


async def agent_loop(ctx: Context, cfg: LoopConfig) -> AsyncIterator[AgentEvent]:
    def steering() -> list[Message]:
        return cfg.drain_steering() if cfg.drain_steering is not None else []

    def follow_ups() -> list[Message]:
        return cfg.drain_follow_ups() if cfg.drain_follow_ups is not None else []

    pending: list[Message] = []
    turns = 0

    while True:
        # ── L3: session loop ──
        while True:  # inner: L2 turn cycle
            if turns >= cfg.max_turns:
                yield ev.error(f"max_turns ({cfg.max_turns}) exceeded", True)
                yield ev.agent_end()
                return

            if cfg.should_stop is not None and cfg.should_stop():
                yield ev.agent_end()
                return

            # fold any steering that arrived between turns, then inject pending msgs
            pending = pending + steering()
            for msg in pending:
                ctx.messages.append(msg)
                yield ev.message_start(msg)
                yield ev.message_end(msg)
            pending = []

            if turns > 0:
                yield ev.turn_start()  # first turn is implied by the user prompt
            turns += 1

            # refresh tools/system-prompt from live state (tools discovered via search_tools
            # last turn become available now)
            if cfg.sync_context is not None:
                cfg.sync_context(ctx)

            # stream the assistant message
            message: Optional[AssistantMessage] = None
            async for e in stream_assistant(ctx, cfg):
                yield e
                if isinstance(e, MessageEnd) and isinstance(e.message, AssistantMessage):
                    message = e.message
            if message is None:
                message = empty_assistant(stop_reason="error", error_message="no response from provider")
            ctx.messages.append(message)

            if message.stop_reason in ("error", "aborted"):
                results = _pair_aborted_calls(ctx, message)
                yield ev.error(message.error_message or message.stop_reason, True)
                yield ev.turn_end(message, results)
                yield ev.agent_end()
                return

            tool_calls = [b for b in message.content if isinstance(b, ToolCallBlock)]
            runnable = message.stop_reason in ("tool_use", "stop") and len(tool_calls) > 0

            if not runnable:
                # A non-runnable stop can still carry tool_use blocks (e.g. "length" truncated
                # mid-call). Pair them with error results or the next API call is rejected for
                # having unanswered tool_use ids.
                results = _pair_aborted_calls(ctx, message) if tool_calls else []
                yield ev.turn_end(message, results)
                break  # leave L2; L3 decides whether to keep going

            # run the tools (interruptible); feed results back
            results = []
            async for e in cfg.execute_tools(tool_calls, ctx, cfg.interrupt.signal):
                yield e
                if e.type == "tool_end":
                    results.append(e.result)
                    ctx.messages.append(e.result)
            yield ev.turn_end(message, results)

            # if steering interrupted the tools, clear the flag and fold it in next turn
            if cfg.interrupt.is_set():
                cfg.interrupt.clear()
                pending = pending + steering()

        # ── L3: the model stopped. Keep going only if work is queued. ──
        pending = pending + steering()
        if len(pending) == 0:
            pending = pending + follow_ups()
        if len(pending) > 0:
            continue
        break

    yield ev.agent_end()


def _pair_aborted_calls(ctx: Context, message: AssistantMessage) -> list[ToolResultMessage]:
    results: list[ToolResultMessage] = []
    for b in message.content:
        if isinstance(b, ToolCallBlock):
            r = tool_result_text(b.id, b.name, "not executed (run ended)", True)
            ctx.messages.append(r)
            results.append(r)
    return results
