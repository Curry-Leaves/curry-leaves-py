#!/usr/bin/env python3
"""08_multimodal.py — send an image (or a PDF) as part of a user turn.

`Runner.run()`/`.stream()` accept either plain text or a full `UserMessage`. The
`user_image` / `user_file` / `user_audio` constructors build multimodal turns; each
provider translates the blocks to its own wire format at its edge:

    ImageBlock  -> Anthropic `image`     / OpenAI `image_url`  (base64 data URI or URL)
    FileBlock   -> Anthropic `document`  / OpenAI `file`       (PDF; URL kind is Anthropic-only)
    AudioBlock  ->            —          / OpenAI `input_audio` (wav/mp3; Anthropic rejects)

Run (after `pip install -e .`):

    export ANTHROPIC_API_KEY=sk-ant-...        # or OPENAI_API_KEY / a local vision model
    python3 examples/08_multimodal.py path/to/picture.png "What is in this image?"

Set CURRY_LEAVES_MODEL to pick a model (default: claude-sonnet-4-5). For a fully local
run try a tool-capable Ollama vision model, e.g. CURRY_LEAVES_MODEL=qwen3.5:27b.
"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import os
import sys

from curry_leaves import Agent, Runner, user_file, user_image

MODEL = os.environ.get("CURRY_LEAVES_MODEL", "claude-sonnet-4-5")


async def main() -> None:
    if len(sys.argv) < 2:
        print("usage: 08_multimodal.py <image-or-pdf> [prompt]", file=sys.stderr)
        sys.exit(2)
    path = sys.argv[1]
    prompt = " ".join(sys.argv[2:]) or "Describe this attachment in one paragraph."

    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    media_type = mimetypes.guess_type(path)[0] or "application/octet-stream"

    if media_type == "application/pdf":
        message = user_file(data, filename=os.path.basename(path), text=prompt)
    else:
        message = user_image(data, media_type=media_type, text=prompt)

    agent = Agent(model=MODEL, instructions="You are a concise assistant.", tools=[])
    result = await Runner(agent).run(message)

    print(result.output_text)
    print(f"\n— {result.usage.output} output tokens, cost ${result.usage.cost.total:.4f}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:  # noqa: BLE001 — top-level CLI error boundary, mirrors the TS catch
        print(e, file=sys.stderr)
        sys.exit(1)
