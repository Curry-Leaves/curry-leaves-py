"""A tiny 5-row block font — just the glyphs needed for the "CURRY LEAVES" splash title.
Each glyph is 5 rows of `█`/space; `big_text` composes a word into 5 strings (one per
row) so the banner can paint each row a different color for a top-down gradient.
"""

from __future__ import annotations

Glyph = tuple[str, str, str, str, str]

FONT: dict[str, Glyph] = {
    "C": ("█████", "█    ", "█    ", "█    ", "█████"),
    "U": ("█   █", "█   █", "█   █", "█   █", "█████"),
    "R": ("████ ", "█   █", "████ ", "█ █  ", "█  █ "),
    "Y": ("█   █", "█   █", " ███ ", "  █  ", "  █  "),
    "L": ("█    ", "█    ", "█    ", "█    ", "█████"),
    "E": ("█████", "█    ", "█████", "█    ", "█████"),
    "A": ("█████", "█   █", "█████", "█   █", "█   █"),
    "V": ("█   █", "█   █", "█   █", " █ █ ", "  █  "),
    "S": ("█████", "█    ", "█████", "    █", "█████"),
    " ": ("  ", "  ", "  ", "  ", "  "),
}


def big_text(text: str) -> tuple[str, str, str, str, str]:
    """Render `text` as 5 rows of block characters. Unknown chars are skipped."""
    rows: list[str] = ["", "", "", "", ""]
    for ch in text.upper():
        g = FONT.get(ch)
        if g is None:
            continue
        for r in range(5):
            rows[r] += f"{g[r]} "
    return (rows[0], rows[1], rows[2], rows[3], rows[4])
