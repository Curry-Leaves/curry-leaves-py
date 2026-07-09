"""Elision — reclaim context by stubbing out stale tool results.

Long agent sessions are dominated by tool output (file reads, search results, command
logs) that goes stale within a few turns — the file was re-read, the work moved on. Elision
replaces such results with a short self-describing stub; the original is preserved WHOLE in
the blob store and stays one `read artifact://<id>` away. The window is reclaimed
losslessly, well before lossy compaction would fire. Three rules, all deterministic — no
model call, no extra cost:

  eligibility — a result may be elided once it is STALE (superseded by an identical later
                call, or untouched for `age_turns` user turns) and big enough to matter
                (`min_result_tokens`). The model's own text is NEVER touched — by the time
                a result is stale, its distilled insight lives in the assistant's messages.
  batching    — eligible results are not elided one at a time. Editing history invalidates
                the provider's prompt cache from that point forward, so per-turn edits cost
                more than they save. Elisions apply in SWEEPS: only when occupancy crosses
                `occupancy_threshold` AND the reclaimable total clears `min_sweep_savings`
                — then biggest-first until occupancy is back at `reclaim_target`.
  recovery    — every stub says what it was, why it went, shows the first lines, and points
                at the preserved original — so the model recalls instead of guessing.

Sits BELOW compaction on the occupancy ladder (default 50% vs 85%): elision reclaims
recoverable tokens early so the lossy summary rarely fires at all. The Runner owns WHEN
(each user-turn boundary); this module owns WHAT and HOW.

A SEAM, like compaction: `Runner(agent, RunConfig(elision=ElisionConfig(enabled=True)))`.
OFF by default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel

from curry_leaves.compaction import estimate_tokens
from curry_leaves.core.blobs import BlobStore
from curry_leaves.core.messages import (
    AssistantMessage,
    Message,
    TextBlock,
    ToolCallBlock,
    ToolResultMessage,
)


class ElisionConfig(BaseModel):
    """Knobs a caller may pass via `RunConfig.elision`. Everything is optional; most users
    should only ever touch `enabled` (and maybe `age_turns`). The rest guard the
    cache economics — the defaults sweep late, sweep big, and never sweep for scraps.
    """

    # Master switch. Default false — existing sessions see zero behavior change.
    enabled: Optional[bool] = None
    # User turns a result must survive untouched before it counts as stale. Default 5.
    age_turns: Optional[int] = None
    # Results smaller than this (estimated tokens) are never elided — a stub replacing a
    # tiny result saves nothing. Default 400.
    min_result_tokens: Optional[int] = None
    # Fraction of the context window (0-1) below which a sweep never fires. Default 0.50.
    occupancy_threshold: Optional[float] = None
    # Occupancy fraction a sweep tries to bring the window back down to. Default 0.35.
    # The threshold->target gap sets the rhythm: wide gap = few, big, cache-friendly sweeps.
    reclaim_target: Optional[float] = None
    # Don't sweep unless at least this many tokens are reclaimable — the hysteresis that
    # stops us re-invalidating the prompt cache for scraps. Default 5000.
    min_sweep_savings: Optional[int] = None
    # First lines of the original kept in the stub as a preview (0 disables). Default 8.
    preview_lines: Optional[int] = None


class ElisionOutcome(BaseModel):
    """What a sweep did — carried on the elision event."""

    results_elided: int
    tokens_before: int
    tokens_reclaimed: int


_DEFAULT_ENABLED = False
_DEFAULT_AGE_TURNS = 5
_DEFAULT_MIN_RESULT_TOKENS = 400
_DEFAULT_OCCUPANCY_THRESHOLD = 0.50
_DEFAULT_RECLAIM_TARGET = 0.35
_DEFAULT_MIN_SWEEP_SAVINGS = 5000
_DEFAULT_PREVIEW_LINES = 8

# Every stub starts with this — it's how a stub self-identifies to the model, and how a
# later sweep knows to skip an already-elided result.
STUB_PREFIX = "[elided]"

# Rough token cost of a stub (marker + metadata + preview) — used to estimate net savings.
_STUB_EST_TOKENS = 90


@dataclass
class _Candidate:
    """A tool result a sweep may elide: where it sits, what it costs, why it's stale."""

    index: int
    message: ToolResultMessage
    tokens: int
    reason: str  # "superseded" | "stale"


def _is_stub(m: ToolResultMessage) -> bool:
    return (
        len(m.content) > 0
        and isinstance(m.content[0], TextBlock)
        and m.content[0].text.startswith(STUB_PREFIX)
    )


def _text_only(m: ToolResultMessage) -> bool:
    return all(isinstance(b, TextBlock) for b in m.content)


def _call_signature(name: str, arguments: dict[str, object]) -> str:
    return name + ":" + json.dumps(arguments, sort_keys=True, default=str)


class Elider:
    def __init__(self, config: Optional[ElisionConfig] = None) -> None:
        config = config or ElisionConfig()
        self.enabled: bool = config.enabled if config.enabled is not None else _DEFAULT_ENABLED
        self.age_turns: int = max(
            1, config.age_turns if config.age_turns is not None else _DEFAULT_AGE_TURNS
        )
        self.min_result_tokens: int = max(
            0,
            config.min_result_tokens
            if config.min_result_tokens is not None
            else _DEFAULT_MIN_RESULT_TOKENS,
        )
        self.occupancy_threshold: float = (
            config.occupancy_threshold
            if config.occupancy_threshold is not None
            else _DEFAULT_OCCUPANCY_THRESHOLD
        )
        self.reclaim_target: float = (
            config.reclaim_target if config.reclaim_target is not None else _DEFAULT_RECLAIM_TARGET
        )
        self.min_sweep_savings: int = max(
            0,
            config.min_sweep_savings
            if config.min_sweep_savings is not None
            else _DEFAULT_MIN_SWEEP_SAVINGS,
        )
        self.preview_lines: int = max(
            0, config.preview_lines if config.preview_lines is not None else _DEFAULT_PREVIEW_LINES
        )

    # ── the one entry point ────────────────────────────────────────────────────

    def maybe_sweep(
        self,
        messages: list[Message],
        blobs: Optional[BlobStore],
        context_tokens: int,
        context_window: int,
    ) -> Optional[ElisionOutcome]:
        """Run one sweep if it's worth it, else do nothing. Mutates eligible
        ToolResultMessage contents in place (the message LIST is untouched, so references
        held by the loop/ctx stay valid — same contract as compaction). Returns what
        happened, or None when no sweep fired. Deterministic; never raises on well-formed
        history.
        """
        if not self.enabled or context_window <= 0:
            return None
        if context_tokens < context_window * self.occupancy_threshold:
            return None  # the room isn't messy yet

        candidates = self._candidates(messages)
        reclaimable = sum(max(0, c.tokens - _STUB_EST_TOKENS) for c in candidates)
        if reclaimable < self.min_sweep_savings:
            return None  # not worth breaking the prompt cache for

        # Biggest-first until occupancy would be back at the target — heavy-tailed sizes
        # mean this is usually a few large elisions, not many small ones.
        need = context_tokens - int(context_window * self.reclaim_target)
        candidates.sort(key=lambda c: c.tokens, reverse=True)
        reclaimed = 0
        elided = 0
        for c in candidates:
            if reclaimed >= need:
                break
            self._elide(c, blobs)
            reclaimed += max(0, c.tokens - _STUB_EST_TOKENS)
            elided += 1
        if elided == 0:
            return None
        return ElisionOutcome(
            results_elided=elided, tokens_before=context_tokens, tokens_reclaimed=reclaimed
        )

    # ── eligibility ────────────────────────────────────────────────────────────

    def _candidates(self, messages: list[Message]) -> list[_Candidate]:
        """Every tool result that MAY be elided right now: stale (superseded or aged out),
        big enough to matter, text-only, and not already a stub.
        """
        # One walk collects everything eligibility needs: each tool call's signature (to
        # spot identical later calls) and each message's age in user turns.
        call_sig: dict[str, str] = {}  # tool_call_id -> signature
        sig_last_index: dict[str, int] = {}  # signature -> index of its LAST call
        users_after = [0] * len(messages)  # user messages strictly after index i
        seen_users = 0
        for i in range(len(messages) - 1, -1, -1):
            users_after[i] = seen_users
            m = messages[i]
            if m.role == "user":
                seen_users += 1
            elif isinstance(m, AssistantMessage):
                for b in m.content:
                    if isinstance(b, ToolCallBlock):
                        sig = _call_signature(b.name, b.arguments)
                        call_sig[b.id] = sig
                        sig_last_index.setdefault(sig, i)  # reverse walk -> first seen is last

        out: list[_Candidate] = []
        for i, m in enumerate(messages):
            if not isinstance(m, ToolResultMessage):
                continue
            if _is_stub(m) or not _text_only(m):
                continue
            tokens = estimate_tokens([m])
            if tokens < self.min_result_tokens:
                continue
            result_sig = call_sig.get(m.tool_call_id)
            if result_sig is not None and sig_last_index.get(result_sig, i) > i:
                out.append(_Candidate(index=i, message=m, tokens=tokens, reason="superseded"))
            elif users_after[i] >= self.age_turns:
                out.append(_Candidate(index=i, message=m, tokens=tokens, reason="stale"))
        return out

    # ── the stub ───────────────────────────────────────────────────────────────

    def _elide(self, c: _Candidate, blobs: Optional[BlobStore]) -> None:
        """Swap a result's content for its stub, preserving the original in the blob store.
        Only the CONTENT changes — the message keeps its tool_call_id, so the call/result
        pairing the provider API requires is untouched.
        """
        original = "".join(b.text for b in c.message.content if isinstance(b, TextBlock))
        why = (
            "superseded by a later identical call"
            if c.reason == "superseded"
            else f"not referenced for {self.age_turns}+ turns"
        )
        parts = [f"{STUB_PREFIX} Stale `{c.message.tool_name}` result (~{c.tokens} tokens, {why}) removed to reclaim context."]
        if self.preview_lines > 0:
            preview = original.split("\n")[: self.preview_lines]
            parts.append("First lines:\n" + "\n".join(f"  {ln}" for ln in preview))
        if blobs is not None:
            bid = blobs.put_text(original)
            parts.append(
                f"Full result preserved at artifact://{bid} — `read` it (offset/limit supported) if needed."
            )
        else:
            parts.append("Original not preserved (no artifact store) — re-run the tool if needed.")
        c.message.content = [TextBlock(text="\n".join(parts))]
