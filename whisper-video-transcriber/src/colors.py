"""Color parsing utilities for subtitle styling.

Accepts ``#RRGGBB``, ``RRGGBB`` or a handful of named colors and converts to the
ASS ``&HAABBGGRR`` primary-colour format (alpha first, then blue/green/red).
"""

from __future__ import annotations

NAMED_COLORS = {
    "white": "#FFFFFF",
    "black": "#000000",
    "red": "#FF0000",
    "green": "#00FF00",
    "blue": "#0000FF",
    "yellow": "#FFEB3B",
    "orange": "#FF9800",
    "cyan": "#00E5FF",
    "magenta": "#E040FB",
    "pink": "#FF4081",
    "lime": "#C6FF00",
    "gray": "#9E9E9E",
    "grey": "#9E9E9E",
}


def normalize_hex(value: str) -> str:
    """Return an uppercase ``#RRGGBB`` string from user input."""
    raw = value.strip().lower()
    if raw in NAMED_COLORS:
        raw = NAMED_COLORS[raw].lower()
    raw = raw.lstrip("#")
    if len(raw) == 3:  # allow shorthand like "fa0"
        raw = "".join(ch * 2 for ch in raw)
    if len(raw) != 6 or any(c not in "0123456789abcdef" for c in raw):
        raise ValueError(
            f"色 '{value}' を解釈できません。'#RRGGBB' か色名 "
            f"({', '.join(sorted(NAMED_COLORS))}) を指定してください。"
        )
    return "#" + raw.upper()


def hex_to_ass(value: str, alpha: int = 0) -> str:
    """Convert a color to ASS ``&HAABBGGRR``.

    ``alpha`` is 0 (opaque) .. 255 (transparent) to match ASS semantics.
    """
    hex_rgb = normalize_hex(value).lstrip("#")
    rr, gg, bb = hex_rgb[0:2], hex_rgb[2:4], hex_rgb[4:6]
    aa = f"{max(0, min(255, alpha)):02X}"
    return f"&H{aa}{bb}{gg}{rr}"
