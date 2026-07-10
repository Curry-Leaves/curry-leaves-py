"""The Runner — holds one live conversation and drives the streaming loop.

An Agent is a stateless definition; the Runner composes an Agent with conversation
state (messages, steering/follow-up queues, the interrupt) and runs it against the
pure `agent_loop` engine. It also builds the Context each turn (system prompt + live
tool list), wires subagent delegation (`task`) and handoff (`transfer`), sizes
reasoning effort (auto-thinking), and validates structured output.

    agent  = Agent(model="claude-sonnet-4-5", instructions="…")
    result = await Runner(agent).run("Summarize README.md in three bullets.")
    print(result.output_text)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

import pydantic

from curry_leaves.catalog import compute_cost, resolve_model
from curry_leaves.compaction import Compactor, CompactionConfig, CompactionOutcome, estimate_tokens
from curry_leaves.core.agent import Agent
from curry_leaves.core.blobs import BlobStore
from curry_leaves.core.events import AgentEvent, CompactionEvent, ev
from curry_leaves.core.host import DefaultHost, Host, SubagentHost
from curry_leaves.core.loop import Interrupt, LoopConfig, agent_loop
from curry_leaves.core.messages import (
    AssistantMessage,
    Message,
    StopReason,
    Usage,
    add_usage,
    empty_usage,
    text_of,
    user_text,
)
from curry_leaves.core.tools import Permission, ToolExecutor, ToolRegistry, make_executor
from curry_leaves.elision import Elider, ElisionConfig
from curry_leaves.permission import AuthorizeContext, PermissionEngine
from curry_leaves.prompt import BuildPromptOptions, ContextFile, Environment, build_system_prompt, resolve_identity
from curry_leaves.providers.base import Context, Model, StreamOpts, settings_to_opts
from curry_leaves.session import SessionStore
from curry_leaves.skills import SkillRegistry
from curry_leaves.thinking import AutoThinking, Classifier, Effort, ThinkingConfig, thinking_budget
from curry_leaves.tools.search_tools import SearchToolsTool
from curry_leaves.tools.task import TaskTool
from curry_leaves.tools.transfer import TransferTool
from curry_leaves.util.retry import DefaultRetryPolicy, RetryPolicy

MAX_AGENT_DEPTH = 4

RETRY_PROMPT = (
    "Your previous reply was not valid JSON matching the required schema. Reply again with "
    "ONLY the JSON object — no prose, no markdown fences."
)

# Injected as a top system layer when autonomous mode is on — makes the agent self-drive.
AUTONOMOUS_PROMPT = (
    "AUTONOMOUS MODE. Operate on your own with minimal interruption:\n"
    "- First, establish clarity. If the goal is ambiguous, ask a FEW sharp clarifying questions UP FRONT.\n"
    "- Once you have clarity, choose the best approach and EXECUTE it end-to-end without pausing to ask.\n"
    "- Prefer acting and verifying over asking. Do not seek approval for routine steps.\n"
    "- Stop only when the task is complete (and verified) or you are truly blocked and cannot proceed."
)


@dataclass
class RunConfig:
    """Options for constructing a Runner. A dataclass (not pydantic) — it carries live
    object references (a Host, a BlobStore, a PermissionEngine, ...), not pure data.
    """

    # Named model tiers (e.g. "fast" -> "gpt-4o-mini") the agent's model string resolves via.
    model_preferences: Optional[dict[str, str]] = None
    skills: Optional[SkillRegistry] = None
    # Per-result context budget before offloading to the blob store.
    max_result_chars: Optional[int] = None
    retry: Optional[RetryPolicy] = None
    # Per-tool deadline in seconds (a tool's own `timeout` wins).
    tool_timeout: Optional[float] = None
    # Project convention files to layer into the system prompt (AGENTS.md, etc.).
    context_files: Optional[list[ContextFile]] = None
    # Frontend seam for events + `ask`. Defaults to a headless DefaultHost.
    host: Optional[Host] = None
    blobs: Optional[BlobStore] = None
    # The single customization seam for the framework's two opinions — the persona `system`
    # prompt (used as the identity layer) and the difficulty `classify` function. Anything
    # omitted falls back to the built-in default (neutral identity + generic AutoThinking).
    # See ThinkingConfig. Example: `ThinkingConfig(system=CODING_IDENTITY)`.
    thinking: Optional[ThinkingConfig] = None
    # A session store to record this run to. The Runner attaches it (so it observes every event)
    # and records each user turn passed to `stream`/`run`. The CALLER owns its lifecycle and must
    # call `store.close()` itself — the Runner never closes it (mirroring host/blobs/skills, which
    # are also caller-owned). Ignored for subagents; the root store already captures their activity.
    store: Optional[SessionStore] = None
    # Conversation compaction — summarize old history when it nears the context window so long
    # sessions don't overflow. Automatic (threshold-gated) by default; also available manually via
    # `compact()`. Omit for sensible defaults; pass `CompactionConfig(auto=False)` to disable.
    compaction: Optional[CompactionConfig] = None
    # Tool-result elision — stub out stale tool output (originals kept in the blob store)
    # when the window fills, so lossy compaction rarely fires. OFF by default; pass
    # `ElisionConfig(enabled=True)` to opt in. See elision.py for the full policy.
    elision: Optional[ElisionConfig] = None
    # Permission gate for tool calls. Omitted → NO gating (every tool runs — unchanged, headless-safe).
    # Provided → each call is authorized against the active agent's `permissions` + this engine's
    # approvals; `ask` verdicts prompt through the host. Caller-owned; shared with subagents.
    permission: Optional[PermissionEngine] = None
    # Autonomous mode — prepend a prompt that tells the model to get clarity (a few initial
    # questions if needed), then choose the best path and execute on its own without asking mid-run.
    # Purely a prompt layer; independent of the permission gate. Toggle live via set_autonomous.
    autonomous: Optional[bool] = None
    # Seed conversation history (e.g. replayed from a forked session) instead of starting empty.
    # The Runner copies this list; mutating the one passed in afterward has no effect.
    initial_messages: Optional[list[Message]] = None


@dataclass
class _InternalOpts:
    depth: Optional[int] = None
    root_host: Optional[Host] = None


class RunResult(pydantic.BaseModel):
    """Final assistant text (a JSON string when the agent has an output_type)."""

    model_config = {"arbitrary_types_allowed": True}

    output_text: str
    messages: list[Message]
    stop_reason: Optional[StopReason]
    # Validated structured result when output_type is set (else None).
    output: Optional[object]
    usage: Usage
    events: list[AgentEvent]
    status: str = "completed"


class EventChannel:
    """A tiny push-queue async iterator. Lets `stream()` merge two live sources — the parent
    loop's events and subagent activity arriving out-of-band via the host — into one ordered
    stream. Non-blocking `push`; `close` ends the iteration once drained.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._closed = False
        self._error: BaseException | None = None
        _sentinel: object = object()
        self._sentinel = _sentinel

    def push(self, e: AgentEvent) -> None:
        if self._closed:
            return
        self._queue.put_nowait(e)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put_nowait(self._sentinel)

    def error(self, exc: BaseException) -> None:
        """Like close(), but the exception is re-raised out of __aiter__ once the queue
        drains — so a producer's failure surfaces to the consumer instead of silently
        looking like the run finished. Without this, an exception inside pump() (see
        Runner.stream) only reaches the orphaned, never-awaited pump_task and vanishes."""
        if self._closed:
            return
        self._error = exc
        self.close()

    async def __aiter__(self) -> AsyncIterator[AgentEvent]:
        while True:
            item = await self._queue.get()
            if item is self._sentinel:
                if self._error is not None:
                    raise self._error
                return
            yield item


def _build_exec_registry(base: ToolRegistry, extras: list[Any]) -> ToolRegistry:
    # A fresh registry so the agent's (possibly shared) registry is never mutated.
    reg = ToolRegistry()
    for tool in base.tools():
        reg.register(tool, deferred=base.is_deferred(tool.name))
    for tool in extras:
        reg.register(tool)
    return reg


def _parse_structured(schema: type[pydantic.BaseModel], text: str) -> Optional[object]:
    def try_parse(s: str) -> Optional[object]:
        try:
            return schema.model_validate(json.loads(s))
        except Exception:
            return None

    whole = try_parse(text.strip())
    if whole is not None:
        return whole
    m = re.search(r"\{[\s\S]*\}", text)  # survive ```json fences / prose
    if m:
        return try_parse(m.group(0))
    return None


class Runner:
    agent: Agent
    model: Model
    usage: Usage
    messages: list[Message]

    def __init__(
        self,
        agent: Agent,
        config: Optional[RunConfig] = None,
        internal: Optional[_InternalOpts] = None,
    ) -> None:
        config = config if config is not None else RunConfig()
        internal = internal if internal is not None else _InternalOpts()

        self.usage = empty_usage()
        self.messages = list(config.initial_messages) if config.initial_messages is not None else []

        # (caller-given id, message) — the id lets a caller cancel_pending() an item
        # before it's folded into the loop; None if the caller doesn't need that.
        self._steering: list[tuple[Optional[str], Message]] = []
        self._follow_ups: list[tuple[Optional[str], Message]] = []
        self._interrupt = Interrupt()
        self._running = False

        self._active_tools: set[str] = set()
        self._exec_registry: ToolRegistry
        self._execute_tools: ToolExecutor
        self._auto_thinking: Optional[Classifier] = None
        self._pending_transfer: Optional[Agent] = None

        self._model_preferences: dict[str, str] = dict(config.model_preferences or {})
        self._skills: SkillRegistry = config.skills if config.skills is not None else SkillRegistry(discover=True)
        self._max_result_chars = config.max_result_chars
        self._retry: RetryPolicy = config.retry if config.retry is not None else DefaultRetryPolicy()
        self._tool_timeout = config.tool_timeout
        self._context_files: list[ContextFile] = list(config.context_files or [])
        self._depth = internal.depth if internal.depth is not None else 0
        self._host: Host = config.host if config.host is not None else DefaultHost()
        self._root_host: Host = internal.root_host if internal.root_host is not None else self._host
        self._blobs: BlobStore = config.blobs if config.blobs is not None else BlobStore()
        self._session_store: Optional[SessionStore] = config.store
        self._compactor = Compactor(config.compaction)
        self._elider = Elider(config.elision)
        self._permission: Optional[PermissionEngine] = config.permission
        self._autonomous: bool = config.autonomous if config.autonomous is not None else False
        # Approx tokens the conversation currently occupies, from the last turn's usage. Drives
        # auto-compaction.
        self._context_tokens = 0
        # The persona/identity layer: thinking.system > identity.md override > neutral default.
        self._identity: str = (
            config.thinking.system if config.thinking is not None and config.thinking.system is not None
            else resolve_identity(os.getcwd())
        )
        # A user-supplied difficulty classifier (from thinking.classify), if any.
        self._custom_classify: Optional[Callable[[str], Awaitable[Effort]]] = (
            config.thinking.classify if config.thinking is not None else None
        )

        self.agent = agent
        self._bind_agent(agent)
        # Record to the session store from the root runner only — subagent activity reaches it via
        # the host as subagent_activity events, so a single attach captures the whole tree.
        if self._depth == 0 and self._session_store is not None:
            self._session_store.attach(self)

    # ── binding the active agent ───────────────────────────────────────────────

    def _bind_agent(self, agent: Agent) -> None:
        self.agent = agent
        if isinstance(agent.model, str):
            self.model = resolve_model(agent.model, self._model_preferences)
        else:
            self.model = agent.model
        # A user-supplied classify fn wins (and implies thinking is on); otherwise the built-in
        # AutoThinking is used only when the agent opted in.
        if self._custom_classify is not None:
            self._auto_thinking = _CallableClassifier(self._custom_classify)
        elif agent.auto_thinking:
            self._auto_thinking = AutoThinking(agent.provider, self._tier_model("fast"))
        else:
            self._auto_thinking = None

        self._active_tools.clear()
        extras: list[Any] = [SearchToolsTool(agent.tools, lambda n: self._active_tools.add(n))]
        roster: dict[str, Agent] = {s.name: s for s in agent.subagents}
        if len(roster) > 0 and self._depth < MAX_AGENT_DEPTH:
            extras.append(TaskTool(roster, self._spawn_subagent))
            extras.append(TransferTool(roster, self._transfer_to))
        self._exec_registry = _build_exec_registry(agent.tools, extras)
        self._execute_tools = make_executor(
            self._exec_registry,
            max_result_chars=self._max_result_chars,
            tool_timeout=self._tool_timeout,
            permission=self._permission_gate(),
        )

    def _permission_gate(self) -> Optional[Permission]:
        """A per-binding view of the shared permission engine that carries THIS agent's verdicts
        and host into each authorize call — so a subagent's map/host never clobbers the parent's
        on the one shared engine. None when no gating is configured (executor skips the pass
        entirely).
        """
        engine = self._permission
        if engine is None:
            return None
        return _PermissionGateAdapter(engine, self)

    def _tier_model(self, tier: str) -> Model:
        """Resolve a preference tier to a Model on the main provider, falling back to this.model."""
        ref = self._model_preferences.get(tier)
        if ref:
            provider = self.model.provider if hasattr(self, "model") else None
            return resolve_model(ref, self._model_preferences, provider)
        return self.model

    # ── observation / live control ─────────────────────────────────────────────

    def subscribe(self, fn: Callable[[AgentEvent], None]) -> Callable[[], None]:
        """Observe the event stream (fire-and-forget). Returns an unsubscribe function."""
        subscribe_fn = getattr(self._host, "subscribe", None)
        if callable(subscribe_fn):
            result: Callable[[], None] = subscribe_fn(fn)
            return result
        return lambda: None

    def _emit(self, e: AgentEvent) -> None:
        self._host.emit(e)

    def set_autonomous(self, on: bool) -> None:
        """Toggle autonomous mode live (affects the next turn's system prompt)."""
        self._autonomous = on

    def steer(self, text: str, id: Optional[str] = None) -> None:
        """Inject a message mid-run (folded in before the next model turn). `id`, if
        given, lets a caller cancel_pending(id) it before that happens."""
        self._steering.append((id, user_text(text, origin="steering")))
        if self._running:
            self._interrupt.set()

    def follow_up(self, text: str, id: Optional[str] = None) -> None:
        """Queue a message to run after the current turn settles. `id`, if given, lets
        a caller cancel_pending(id) it before that happens."""
        self._follow_ups.append((id, user_text(text, origin="follow_up")))

    def cancel_pending(self, id: str) -> bool:
        """Remove a not-yet-folded-in steer()/follow_up() item by its `id`. Returns False
        if it was never queued, already folded into the loop, or given no id."""
        for lst in (self._steering, self._follow_ups):
            for i, (item_id, _msg) in enumerate(lst):
                if item_id == id:
                    del lst[i]
                    return True
        return False

    def _drain_steering(self) -> list[Message]:
        out = [msg for _id, msg in self._steering]
        self._steering = []
        return out

    def _drain_follow_ups(self) -> list[Message]:
        out = [msg for _id, msg in self._follow_ups]
        self._follow_ups = []
        return out

    # ── subagents ──────────────────────────────────────────────────────────────

    async def _spawn_subagent(self, agent: Agent, prompt: str) -> str:
        depth = self._depth + 1
        child = Runner(
            agent,
            RunConfig(
                model_preferences=self._model_preferences,
                skills=self._skills,
                max_result_chars=self._max_result_chars,
                retry=self._retry,
                tool_timeout=self._tool_timeout,
                blobs=self._blobs,
                host=SubagentHost(self._root_host, depth, agent.name),
                # inherit the parent's resolved persona + any custom classifier
                thinking=ThinkingConfig(system=self._identity, classify=self._custom_classify),
                # share the ONE permission engine so approvals/allowlist are unified across the tree
                permission=self._permission,
            ),
            _InternalOpts(depth=depth, root_host=self._root_host),
        )
        try:
            result = await child.run(prompt)
            return result.output_text
        finally:
            await child.close()

    def _transfer_to(self, name: str) -> str:
        """`transfer` tool callback: queue a one-way handoff to subagent `name`."""
        target = next((s for s in self.agent.subagents if s.name == name), None)
        if target is None:
            avail = ", ".join(s.name for s in self.agent.subagents) or "(none)"
            return f"Unknown agent '{name}'. Available: {avail}"
        self._pending_transfer = target
        return f"Transferring to '{name}'. It takes over the conversation from here."

    # ── the Context + loop config, rebuilt each turn ───────────────────────────

    def _sync_context(self, ctx: Context) -> None:
        advertised = self._exec_registry.advertised_schemas(self._active_tools)
        ctx.tools = advertised
        output_schema: Optional[str] = None
        if self.agent.output_type is not None:
            output_schema = json.dumps(self.agent.output_type.model_json_schema(), indent=2)
        ctx.system_prompt = build_system_prompt(
            self.agent.instructions,
            BuildPromptOptions(
                identity=self._identity,
                tools={s.name for s in advertised},
                deferred_tools=self._exec_registry.deferred_teasers(self._active_tools),
                context_files=self._context_files,
                skills=self._skills.teasers(),
                subagents=(
                    [(s.name, s.description) for s in self.agent.subagents]
                    if self.agent.subagents
                    else []
                ),
                environment=Environment(
                    cwd=os.getcwd(),
                    date=datetime.now(timezone.utc).date().isoformat(),
                    platform=os.name,
                ),
                output_schema=output_schema,
            ),
        )
        if self._autonomous:
            ctx.system_prompt.append(AUTONOMOUS_PROMPT)

    def _build_context(self) -> Context:
        ctx = Context(
            system_prompt=[],
            messages=self.messages,
            tools=[],
            blobs=self._blobs,
            resolve_skill=lambda n: self._skills.read(n),
            host=self._host,
            spawn=(self._spawn_subagent if self._depth < MAX_AGENT_DEPTH else None),
        )
        self._sync_context(ctx)
        return ctx

    def _make_loop_config(self, opts: StreamOpts) -> LoopConfig:
        return LoopConfig(
            provider=self.agent.provider,
            model=self.model,
            execute_tools=self._execute_tools,
            opts=opts,
            max_turns=self.agent.max_turns,
            interrupt=self._interrupt,
            drain_steering=self._drain_steering,
            drain_follow_ups=self._drain_follow_ups,
            sync_context=self._sync_context,
            should_stop=lambda: self._pending_transfer is not None,
            retry=self._retry,
        )

    async def _loop_with_handoff(self, ctx: Context, opts: StreamOpts) -> AsyncIterator[AgentEvent]:
        """Drive the loop, re-entering under a new agent whenever a `transfer` is queued."""
        while True:
            cfg = self._make_loop_config(opts)
            saw_end = False
            async for e in agent_loop(ctx, cfg):
                if e.type == "agent_end":
                    saw_end = True
                    continue  # hold the end until we know whether a handoff follows
                yield e
            if self._pending_transfer is not None:
                target = self._pending_transfer
                self._pending_transfer = None
                from_name = self.agent.name
                self._bind_agent(target)
                self._sync_context(ctx)
                yield ev.handoff(from_name, target.name)
                continue
            if saw_end:
                yield ev.agent_end()
            return

    # ── entry points ───────────────────────────────────────────────────────────

    async def stream(self, text: str) -> AsyncIterator[AgentEvent]:
        """Stream the events of one run (appends `text` as a user turn first)."""
        # Elide stale tool results first — cheap and lossless, and it may reclaim enough
        # that compaction (the lossy step below) never fires.
        elided = self._elider.maybe_sweep(
            self.messages, self._blobs, self._context_tokens, self.model.context_window
        )
        if elided is not None:
            self._context_tokens = max(0, self._context_tokens - elided.tokens_reclaimed)
            eev = ev.elision(elided.results_elided, elided.tokens_before, elided.tokens_reclaimed)
            self._emit(eev)
            yield eev
        # Compact the prior history if it's still near the window, so the new turn fits.
        if self._compactor.should_auto(len(self.messages), self._context_tokens, self.model.context_window):
            cev = await self._compact_once("auto")
            if cev is not None:
                yield cev
        self.messages.append(user_text(text))
        if self._session_store is not None:
            self._session_store.user(text)  # the event stream carries no user event; record it here
        self._interrupt.clear()

        opts: StreamOpts = settings_to_opts(self.agent.model_settings)
        # Native JSON mode for a pure-extraction agent (output_type + no own tools).
        if self.agent.output_type is not None and len(self.agent.tools.tools()) == 0:
            opts.response_format = {"type": "json_object"}
        if self._auto_thinking is not None:
            effort = await self._auto_thinking.classify(text)
            if effort != Effort.MINIMAL:
                opts.reasoning_effort = effort.value
                opts.thinking_budget = thinking_budget(effort)
            te = ev.thinking(effort.value)
            self._emit(te)
            yield te

        ctx = self._build_context()

        # Unified stream: parent loop events + subagent activity (which arrives out-of-band via
        # the host while a `task` tool is awaited) merged into one ordered channel. A consumer
        # iterating stream() therefore sees the FULL event set from subagents too, tagged with
        # their agent name/depth. (Custom hosts without `subscribe` still get subagent activity
        # via host.emit; it just won't be folded into this pull stream.)
        channel = EventChannel()

        def on_event(e: AgentEvent) -> None:
            if e.type == "subagent_activity":
                channel.push(e)

        off = self.subscribe(on_event)

        async def pump() -> None:
            try:
                async for e in self._run_events(self._loop_with_handoff(ctx, opts)):
                    channel.push(e)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Surface it to stream()'s consumer via the channel (see EventChannel.error)
                # instead of letting it vanish into this Task, which nothing awaits.
                channel.error(exc)
                return
            finally:
                off()
                channel.close()

        pump_task = asyncio.ensure_future(pump())
        try:
            async for e in channel:
                yield e
        finally:
            # A consumer that breaks (or is cancelled) must stop the run, not detach from
            # it — otherwise the provider stream keeps billing into an unread queue.
            pump_task.cancel()
            try:
                await pump_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 — teardown must not mask the exit
                pass

    async def _run_events(self, inner: AsyncIterator[AgentEvent]) -> AsyncIterator[AgentEvent]:
        self._running = True
        try:
            async for e in inner:
                if e.type == "message_end" and isinstance(e.message, AssistantMessage) and e.message.usage:
                    self.usage = add_usage(self.usage, self._priced(e.message.usage))
                    # Occupancy going into the next turn ≈ this call's whole prompt (cached +
                    # uncached) + its output.
                    u = e.message.usage
                    self._context_tokens = u.input + u.cache_read + u.cache_write + u.output
                self._emit(e)
                yield e
        finally:
            self._running = False

    async def run(self, text: str) -> RunResult:
        """Run to completion and return a RunResult."""
        outputresult = await self._drive(text)
        output_text, stop_reason, events = (
            outputresult.output_text,
            outputresult.stop_reason,
            outputresult.events,
        )
        output: Optional[object] = None

        if self.agent.output_type is not None:
            output = _parse_structured(self.agent.output_type, output_text)
            i = 0
            while i < 2 and output is None:
                more = await self._drive(RETRY_PROMPT)
                events = events + more.events
                output_text = more.output_text
                stop_reason = more.stop_reason
                output = _parse_structured(self.agent.output_type, output_text)
                i += 1
        return self._result(output_text, stop_reason, events, output)

    @dataclass
    class _DriveResult:
        output_text: str
        stop_reason: Optional[StopReason]
        events: list[AgentEvent]

    async def _drive(self, text: str) -> "Runner._DriveResult":
        events: list[AgentEvent] = []
        output_text = ""
        stop_reason: Optional[StopReason] = None
        async for e in self.stream(text):
            events.append(e)
            if e.type == "message_end" and isinstance(e.message, AssistantMessage):
                output_text = text_of(e.message.content)  # last assistant message wins
                stop_reason = e.message.stop_reason
        return Runner._DriveResult(output_text=output_text, stop_reason=stop_reason, events=events)

    def _result(
        self,
        output_text: str,
        stop_reason: Optional[StopReason],
        events: list[AgentEvent],
        output: Optional[object],
    ) -> RunResult:
        run_usage = empty_usage()
        for e in events:
            if e.type == "message_end" and isinstance(e.message, AssistantMessage) and e.message.usage:
                run_usage = add_usage(run_usage, self._priced(e.message.usage))
        return RunResult(
            output_text=output_text,
            messages=list(self.messages),
            stop_reason=stop_reason,
            output=output,
            usage=run_usage,
            events=events,
            status="completed",
        )

    def _priced(self, usage: Usage) -> Usage:
        return usage.model_copy(update={"cost": compute_cost(usage, self.model.id)})

    # ── compaction ─────────────────────────────────────────────────────────────

    async def compact(self, instructions: Optional[str] = None) -> CompactionOutcome:
        """Compact the conversation now: summarize older history into one message so the window
        frees up. Safe to call between turns. Returns what happened (a no-op result when there's
        nothing to do). `instructions` focuses the summary (e.g. "keep the auth details"). Also
        wired to `/compact`.
        """
        before = len(self.messages)
        tokens_before = self._context_tokens
        cev = await self._compact_once("manual", instructions)
        if cev is not None:
            return CompactionOutcome(
                compacted=True,
                reason="manual",
                messages_before=cev.messages_before,
                messages_after=cev.messages_after,
                tokens_before=cev.tokens_before,
                summary=cev.summary,
            )
        return CompactionOutcome(
            compacted=False,
            reason="manual",
            messages_before=before,
            messages_after=before,
            tokens_before=tokens_before,
            summary="",
        )

    def _compact_model(self) -> Model:
        """The model used to summarize: a config override (id/tier), else the active model."""
        ref = self._compactor.model_ref
        if ref:
            return resolve_model(ref, self._model_preferences, self.model.provider)
        return self.model

    async def _compact_once(
        self, reason: str, instructions: Optional[str] = None
    ) -> Optional[CompactionEvent]:
        """Run one compaction: summarize, swap `self.messages` contents in place (preserving the
        list reference the loop/ctx hold), reset the occupancy estimate, and emit a compaction
        event. Returns the event, or None if there was nothing to compact or the summary call
        failed.
        """
        if not self._compactor.worth_compacting(self.messages):
            return None  # too little history to be worth it
        before = len(self.messages)
        tokens_before = self._context_tokens
        try:
            new_messages, summary, _kept_tail = await self._compactor.compact(
                self.agent.provider, self._compact_model(), list(self.messages), instructions
            )
        except Exception as e:
            er = ev.error(f"compaction failed: {e}", False)
            self._emit(er)
            return None
        if not summary:
            return None  # empty summary → don't destroy the real history
        self.messages[:] = new_messages
        self._context_tokens = estimate_tokens(self.messages)
        cev = ev.compaction("auto" if reason == "auto" else "manual", before, len(self.messages), tokens_before, summary)
        self._emit(cev)
        return cev

    async def close(self) -> None:
        """Tear down any stateful tools."""
        for tool in self._exec_registry.tools():
            close_fn = getattr(tool, "close", None)
            if close_fn is not None:
                try:
                    await close_fn()
                except Exception:
                    pass


class _CallableClassifier:
    """Wraps a plain `classify` callable as a Classifier (the custom-classify seam)."""

    def __init__(self, fn: Callable[[str], Awaitable[Effort]]) -> None:
        self._fn = fn

    async def classify(self, prompt_text: str) -> Effort:
        return await self._fn(prompt_text)


class _PermissionGateAdapter:
    """Adapts a shared PermissionEngine + a Runner into the `Permission` protocol the tool
    executor consults, carrying THIS Runner's active-agent verdicts and host per call.
    """

    def __init__(self, engine: PermissionEngine, runner: Runner) -> None:
        self._engine = engine
        self._runner = runner

    async def authorize(self, tool: str, risk: Any, args: dict[str, object]) -> Any:
        return await self._engine.authorize(
            tool,
            risk,
            args,
            AuthorizeContext(permissions=self._runner.agent.permissions, host=self._runner._host),
        )
