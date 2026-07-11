"""Multimodal content: block -> wire translation per provider, the Runner front door,
and session record/replay fidelity for attachment turns."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pytest

from curry_leaves.core.messages import (
    AudioBlock,
    FileBlock,
    ImageBlock,
    TextBlock,
    UserMessage,
    user_audio,
    user_file,
    user_image,
)
from curry_leaves.providers.anthropic import build_anthropic_request
from curry_leaves.providers.base import Context, StreamOpts, make_model
from curry_leaves.providers.openai import build_openai_request
from curry_leaves.session import SessionMeta, fork_session, open_session
from curry_leaves.util import paths


def _ctx(message: UserMessage) -> Context:
    return Context(system_prompt=[], messages=[message], tools=[])


def _anthropic_content(message: UserMessage) -> list[dict[str, Any]]:
    body = build_anthropic_request(_ctx(message), make_model("claude-x", "anthropic"), StreamOpts())
    content: list[dict[str, Any]] = body["messages"][0]["content"]
    return content


def _openai_content(message: UserMessage) -> Any:
    body = build_openai_request(_ctx(message), make_model("gpt-x", "openai"), StreamOpts())
    return body["messages"][0]["content"]


# ── Anthropic wire ────────────────────────────────────────────────────────────


def test_anthropic_image_base64() -> None:
    content = _anthropic_content(user_image("QUJD", media_type="image/jpeg", text="what is this?"))
    assert content[0] == {"type": "text", "text": "what is this?"}
    assert content[1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"},
    }


def test_anthropic_image_url() -> None:
    content = _anthropic_content(user_image("https://x.test/a.png", kind="url"))
    assert content[0] == {"type": "image", "source": {"type": "url", "url": "https://x.test/a.png"}}


def test_anthropic_file_base64_becomes_document() -> None:
    content = _anthropic_content(user_file("UERG", text="summarize"))
    assert content[1] == {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf", "data": "UERG"},
    }


def test_anthropic_file_url_becomes_document() -> None:
    content = _anthropic_content(user_file("https://x.test/a.pdf", kind="url"))
    assert content[0] == {"type": "document", "source": {"type": "url", "url": "https://x.test/a.pdf"}}


def test_anthropic_rejects_audio() -> None:
    with pytest.raises(ValueError, match="audio"):
        _anthropic_content(user_audio("QUJD"))


# ── OpenAI wire ───────────────────────────────────────────────────────────────


def test_openai_text_only_stays_flat_string() -> None:
    assert _openai_content(UserMessage(content=[TextBlock(text="hi")])) == "hi"


def test_openai_image_base64_becomes_data_uri_part() -> None:
    content = _openai_content(user_image("QUJD", media_type="image/jpeg", text="what is this?"))
    assert content == [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}},
    ]


def test_openai_image_url_passes_through() -> None:
    content = _openai_content(user_image("https://x.test/a.png", kind="url"))
    assert content == [{"type": "image_url", "image_url": {"url": "https://x.test/a.png"}}]


def test_openai_audio_part() -> None:
    content = _openai_content(user_audio("QUJD", format="mp3"))
    assert content == [{"type": "input_audio", "input_audio": {"data": "QUJD", "format": "mp3"}}]


def test_openai_file_base64_part() -> None:
    content = _openai_content(user_file("UERG", filename="report.pdf"))
    assert content == [
        {
            "type": "file",
            "file": {"filename": "report.pdf", "file_data": "data:application/pdf;base64,UERG"},
        }
    ]


def test_openai_rejects_file_url() -> None:
    with pytest.raises(ValueError, match="base64"):
        _openai_content(user_file("https://x.test/a.pdf", kind="url"))


# ── Session record + replay ───────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path) -> Iterator[None]:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    paths.set_home(str(fake_home))
    yield
    paths.set_home(None)


async def test_multimodal_user_turn_survives_fork() -> None:
    store = open_session("mm-src", SessionMeta(model="m", provider="p", cwd="/x"))
    msg = user_image("QUJD", text="look at this")
    store.user("look at this", content=msg.content)
    store.persist_meta(store.metadata)

    new_store, messages = fork_session("mm-src", "mm-fork", SessionMeta(model="m", provider="p", cwd="/x"))
    try:
        assert len(messages) == 1
        replayed = messages[0]
        assert replayed.role == "user"
        assert isinstance(replayed.content[0], TextBlock)
        assert isinstance(replayed.content[1], ImageBlock)
        assert replayed.content[1].source == "QUJD"
    finally:
        await new_store.close()


async def test_text_only_user_turn_keeps_compact_record() -> None:
    store = open_session("txt-src", SessionMeta(model="m", provider="p", cwd="/x"))
    store.user("plain", content=[TextBlock(text="plain")])
    store.persist_meta(store.metadata)

    from curry_leaves.session import load_transcript

    records = load_transcript("txt-src")
    user_records = [r for r in records if r.get("kind") == "user"]
    assert user_records[0]["text"] == "plain"
    assert "content" not in user_records[0]


# ── Block defaults / constructors ─────────────────────────────────────────────


def test_constructors_shape() -> None:
    m = user_audio("QUJD")
    assert isinstance(m.content[0], AudioBlock) and m.content[0].format == "wav"
    f = user_file("UERG")
    assert isinstance(f.content[0], FileBlock)
    assert f.content[0].media_type == "application/pdf" and f.content[0].filename is None
