"""Elision — the deterministic sweep policy in elision.py.

Everything here is pure and synchronous (no model, no I/O): conversations are built by
hand, `Elider.maybe_sweep` runs against them, and we assert on the three rules —
eligibility (stale + big enough), batching (occupancy + savings gates, biggest-first to
the reclaim target), and recovery (stub content, original preserved in the blob store).
"""

from __future__ import annotations

from curry_leaves.core.blobs import BlobStore
from curry_leaves.core.messages import (
    AssistantMessage,
    Message,
    ToolCallBlock,
    text_of,
    tool_result_text,
    user_text,
)
from curry_leaves.elision import STUB_PREFIX, Elider, ElisionConfig

WINDOW = 100_000


def config(**overrides: object) -> ElisionConfig:
    """Enabled, with gates set low enough that a test controls exactly which fire."""
    base: dict[str, object] = {
        "enabled": True,
        "age_turns": 3,
        "min_result_tokens": 100,
        "occupancy_threshold": 0.50,
        "reclaim_target": 0.35,
        "min_sweep_savings": 200,
        "preview_lines": 2,
    }
    base.update(overrides)
    return ElisionConfig.model_validate(base)


def call_turn(call_id: str, tool: str, args: dict[str, object], result: str) -> list[Message]:
    """One assistant tool call + its result."""
    return [
        AssistantMessage(content=[ToolCallBlock(id=call_id, name=tool, arguments=args)]),
        tool_result_text(call_id, tool, result),
    ]


def big(chars: int = 4000) -> str:
    """~chars/4 estimated tokens of multi-line text."""
    line = "x" * 79
    return "\n".join(line for _ in range(chars // 80))


def conversation(turns: int, result_chars: int = 4000) -> list[Message]:
    """`turns` user turns, each followed by one big tool call/result pair."""
    msgs: list[Message] = []
    for i in range(turns):
        msgs.append(user_text(f"turn {i}"))
        msgs.extend(call_turn(f"c{i}", "read", {"path": f"f{i}.py"}, big(result_chars)))
    return msgs


def stubs_in(msgs: list[Message]) -> list[str]:
    return [
        text_of(m.content)
        for m in msgs
        if m.role == "tool_result" and text_of(m.content).startswith(STUB_PREFIX)
    ]


# ── gates: when a sweep does NOT fire ────────────────────────────────────────


def test_disabled_by_default() -> None:
    msgs = conversation(8)
    outcome = Elider(None).maybe_sweep(msgs, BlobStore(), WINDOW, WINDOW)
    assert outcome is None
    assert stubs_in(msgs) == []


def test_no_sweep_below_occupancy_threshold() -> None:
    msgs = conversation(8)
    outcome = Elider(config()).maybe_sweep(msgs, BlobStore(), int(WINDOW * 0.4), WINDOW)
    assert outcome is None


def test_no_sweep_when_savings_below_floor() -> None:
    # Plenty of occupancy, but only one small-ish stale result — under the savings floor.
    msgs = conversation(6, result_chars=800)  # ~200 tokens each
    outcome = Elider(config(min_sweep_savings=5000)).maybe_sweep(
        msgs, BlobStore(), int(WINDOW * 0.9), WINDOW
    )
    assert outcome is None
    assert stubs_in(msgs) == []


def test_no_sweep_when_nothing_is_stale() -> None:
    # Two turns only — nothing has aged past age_turns=3 and nothing is superseded.
    msgs = conversation(2)
    outcome = Elider(config()).maybe_sweep(msgs, BlobStore(), int(WINDOW * 0.9), WINDOW)
    assert outcome is None


# ── eligibility ──────────────────────────────────────────────────────────────


def test_sweep_elides_aged_results_and_keeps_recent_ones() -> None:
    msgs = conversation(8)
    blobs = BlobStore()
    outcome = Elider(config(min_sweep_savings=100)).maybe_sweep(
        msgs, blobs, int(WINDOW * 0.9), WINDOW
    )
    assert outcome is not None and outcome.results_elided > 0
    # The last age_turns=3 turns are untouchable: their results stay verbatim.
    recent = [m for m in msgs[-9:] if m.role == "tool_result"]
    assert all(not text_of(m.content).startswith(STUB_PREFIX) for m in recent)


def test_superseded_result_is_eligible_immediately() -> None:
    # The same read twice in the SAME turn window — too young to age out, but superseded.
    msgs: list[Message] = [user_text("go")]
    msgs.extend(call_turn("c1", "read", {"path": "a.py"}, big()))
    msgs.extend(call_turn("c2", "read", {"path": "a.py"}, big()))
    outcome = Elider(config(min_sweep_savings=100)).maybe_sweep(
        msgs, BlobStore(), int(WINDOW * 0.9), WINDOW
    )
    assert outcome is not None and outcome.results_elided == 1
    first, second = (m for m in msgs if m.role == "tool_result")
    assert text_of(first.content).startswith(STUB_PREFIX)
    assert "superseded" in text_of(first.content)
    assert not text_of(second.content).startswith(STUB_PREFIX)  # the fresh copy survives


def test_small_results_are_never_elided() -> None:
    msgs = conversation(8, result_chars=200)  # ~50 tokens, under min_result_tokens=100
    outcome = Elider(config(min_sweep_savings=0)).maybe_sweep(
        msgs, BlobStore(), int(WINDOW * 0.9), WINDOW
    )
    assert outcome is None


def test_only_tool_results_are_touched() -> None:
    msgs = conversation(8)
    before = [m.model_dump() for m in msgs if m.role in ("user", "assistant")]
    Elider(config(min_sweep_savings=100)).maybe_sweep(msgs, BlobStore(), int(WINDOW * 0.9), WINDOW)
    after = [m.model_dump() for m in msgs if m.role in ("user", "assistant")]
    assert before == after  # user text, assistant text/tool_calls all verbatim


def test_sweep_is_idempotent_on_stubs() -> None:
    msgs = conversation(8)
    blobs = BlobStore()
    elider = Elider(config(min_sweep_savings=100))
    first = elider.maybe_sweep(msgs, blobs, int(WINDOW * 0.9), WINDOW)
    assert first is not None
    # Same occupancy again: everything eligible is already a stub -> nothing to do.
    second = elider.maybe_sweep(msgs, blobs, int(WINDOW * 0.9), WINDOW)
    assert second is None


# ── batching: biggest-first to the reclaim target ────────────────────────────


def test_sweep_stops_at_reclaim_target_biggest_first() -> None:
    # Ten stale results (~1000 tokens each), but only ~1500 tokens over the target:
    # a sweep should elide the minimum needed (2 of them), not everything eligible.
    msgs = conversation(10)
    context_tokens = int(WINDOW * 0.365)  # target is 0.35 -> need ~1500 tokens back
    outcome = Elider(config(occupancy_threshold=0.36, min_sweep_savings=100)).maybe_sweep(
        msgs, BlobStore(), context_tokens, WINDOW
    )
    assert outcome is not None
    assert outcome.results_elided == 2
    assert outcome.tokens_reclaimed >= context_tokens - int(WINDOW * 0.35)


# ── recovery: the stub and the preserved original ────────────────────────────


def test_stub_preserves_original_in_blob_store() -> None:
    msgs = conversation(8)
    blobs = BlobStore()
    original_texts = {text_of(m.content) for m in msgs if m.role == "tool_result"}
    Elider(config(min_sweep_savings=100)).maybe_sweep(msgs, blobs, int(WINDOW * 0.9), WINDOW)
    for stub in stubs_in(msgs):
        # Self-describing: what it was, a preview, and where the original lives.
        assert "read" in stub and "artifact://" in stub
        assert "First lines:" in stub
        bid = stub.split("artifact://")[1].split(" ")[0]
        assert blobs.get_text(bid) in original_texts


def test_pairing_survives_elision() -> None:
    msgs = conversation(8)
    ids_before = [m.tool_call_id for m in msgs if m.role == "tool_result"]
    Elider(config(min_sweep_savings=100)).maybe_sweep(msgs, BlobStore(), int(WINDOW * 0.9), WINDOW)
    ids_after = [m.tool_call_id for m in msgs if m.role == "tool_result"]
    assert ids_before == ids_after  # every tool_call still has its result, same order
