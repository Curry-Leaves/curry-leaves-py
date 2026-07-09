from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from curry_leaves.core.agent import Agent
from curry_leaves.core.messages import TextBlock, ToolCallBlock
from curry_leaves.runner import RunConfig, Runner
from curry_leaves.session import SessionMeta, fork_session, load_transcript, open_session, user_turn_offsets
from curry_leaves.util import paths


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path) -> Iterator[None]:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    paths.set_home(str(fake_home))
    yield
    paths.set_home(None)


def _record_conversation(session_id: str) -> None:
    """A two-turn conversation: turn 1 is plain text, turn 2 involves a tool call."""
    store = open_session(session_id, SessionMeta(model="m", provider="p", cwd="/x"))
    store.user("hello")
    store._write(
        {"kind": "assistant", "content": [TextBlock(text="hi there").model_dump()], "stop_reason": "stop", "usage": None}
    )
    store.user("do a thing")
    store._write(
        {
            "kind": "assistant",
            "content": [ToolCallBlock(id="t1", name="run", arguments={"x": 1}).model_dump()],
            "stop_reason": "tool_use",
            "usage": None,
        }
    )
    store._write({"kind": "tool_end", "id": "t1", "name": "run", "result": "done", "is_error": False})
    store._write(
        {"kind": "assistant", "content": [TextBlock(text="did it").model_dump()], "stop_reason": "stop", "usage": None}
    )
    store.persist_meta(store.metadata)


async def test_fork_full_history_replays_every_root_turn() -> None:
    _record_conversation("src")
    new_store, messages = fork_session("src", "fork-full", SessionMeta(model="m", provider="p", cwd="/x"))
    try:
        assert [m.role for m in messages] == ["user", "assistant", "user", "assistant", "tool_result", "assistant"]
    finally:
        await new_store.close()


async def test_fork_upto_turn_truncates_after_that_users_reply() -> None:
    _record_conversation("src")
    new_store, messages = fork_session("src", "fork-0", SessionMeta(model="m", provider="p", cwd="/x"), upto_turn=0)
    try:
        assert [m.role for m in messages] == ["user", "assistant"]
    finally:
        await new_store.close()


async def test_fork_stamps_forked_from_in_new_meta() -> None:
    _record_conversation("src")
    new_store, _messages = fork_session("src", "fork-meta", SessionMeta(model="m", provider="p", cwd="/x"), upto_turn=0)
    try:
        assert new_store.metadata["forked_from"] == {"session": "src", "upto_turn": 0}
    finally:
        await new_store.close()


async def test_fork_does_not_mutate_source_transcript() -> None:
    _record_conversation("src")
    before = load_transcript("src")
    new_store, _messages = fork_session("src", "fork-x", SessionMeta(model="m", provider="p", cwd="/x"), upto_turn=0)
    try:
        after = load_transcript("src")
        assert before == after
        # The fork's own transcript is the truncated copy, not the full source.
        forked = load_transcript("fork-x")
        assert len(forked) < len(before)
    finally:
        await new_store.close()


def test_user_turn_offsets_ignores_subagent_activity() -> None:
    records = [
        {"kind": "user", "text": "a"},
        {"kind": "assistant", "content": []},
        {"kind": "user", "text": "nested", "depth": 1, "agent": "explore"},  # subagent turn — not root
        {"kind": "user", "text": "b"},
    ]
    assert user_turn_offsets(records) == [0, 3]


async def test_fork_of_unknown_session_yields_empty_history() -> None:
    new_store, messages = fork_session("does-not-exist", "fork-empty", SessionMeta(model="m", provider="p", cwd="/x"))
    try:
        assert messages == []
    finally:
        await new_store.close()


async def test_runner_seeds_from_initial_messages_and_copies() -> None:
    _record_conversation("src")
    new_store, messages = fork_session("src", "fork-runner", SessionMeta(model="m", provider="p", cwd="/x"), upto_turn=0)
    try:
        agent = Agent("claude-sonnet-4-5", instructions="test")
        runner = Runner(agent, RunConfig(initial_messages=messages))
        assert [m.role for m in runner.messages] == ["user", "assistant"]
        messages.clear()
        assert len(runner.messages) == 2  # Runner copied the list; source mutation has no effect
    finally:
        await new_store.close()
