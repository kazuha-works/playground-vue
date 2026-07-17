"""Subtitle rendering: SRT / WebVTT / styled ASS, plus optional burn-in."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .colors import hex_to_ass, normalize_hex
from .media import require_ffmpeg


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class SubtitleStyle:
    font: str = "Arial"
    font_size: int | None = None          # None -> derived from video height
    primary_color: str = "#FFEB3B"        # subtitle fill color
    outline_color: str = "#000000"        # border color for legibility
    outline_width: float = 2.0
    play_res_x: int = 1920
    play_res_y: int = 1080

    def resolved_font_size(self) -> int:
        """Scale the font to the video height when not explicitly set."""
        if self.font_size is not None:
            return self.font_size
        # ~4.5% of the frame height reads well for burned-in captions.
        return max(16, round(self.play_res_y * 0.045))


def _fmt_srt_ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_vtt_ts(seconds: float) -> str:
    return _fmt_srt_ts(seconds).replace(",", ".")


def _fmt_ass_ts(seconds: float) -> str:
    cs = int(round(seconds * 100))  # centiseconds
    h, cs = divmod(cs, 360_000)
    m, cs = divmod(cs, 6_000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def write_srt(segments: Sequence[Segment], path: Path) -> Path:
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{_fmt_srt_ts(seg.start)} --> {_fmt_srt_ts(seg.end)}")
        lines.append(seg.text.strip())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_vtt(segments: Sequence[Segment], path: Path) -> Path:
    lines = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{_fmt_vtt_ts(seg.start)} --> {_fmt_vtt_ts(seg.end)}")
        lines.append(seg.text.strip())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _ass_escape(text: str) -> str:
    # Collapse hard newlines into ASS soft breaks and neutralize braces.
    return text.strip().replace("{", "(").replace("}", ")").replace("\n", "\\N")


def write_ass(segments: Sequence[Segment], path: Path, style: SubtitleStyle) -> Path:
    # Validate colors up front so a typo fails before we write a file.
    primary = hex_to_ass(style.primary_color)
    outline = hex_to_ass(style.outline_color)
    normalize_hex(style.primary_color)

    font_size = style.resolved_font_size()

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {style.play_res_x}
PlayResY: {style.play_res_y}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{style.font},{font_size},{primary},&H000000FF,{outline},&H64000000,0,0,0,0,100,100,0,0,1,{style.outline_width},1,2,40,40,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events = []
    for seg in segments:
        events.append(
            "Dialogue: 0,"
            f"{_fmt_ass_ts(seg.start)},{_fmt_ass_ts(seg.end)},"
            f"Default,,0,0,0,,{_ass_escape(seg.text)}"
        )

    path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return path


def burn_subtitles(video: Path, ass_path: Path, out_path: Path) -> Path:
    """Hard-burn the styled ASS subtitles into a new video via ffmpeg."""
    require_ffmpeg()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Escape characters that the ffmpeg filter parser treats specially.
    escaped = str(ass_path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-vf",
            f"ass='{escaped}'",
            "-c:a",
            "copy",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return out_path


def write_all(
    segments: Iterable[Segment],
    formats: Sequence[str],
    base_path: Path,
    style: SubtitleStyle,
) -> dict[str, Path]:
    """Write each requested subtitle format next to ``base_path``."""
    seg_list = list(segments)
    written: dict[str, Path] = {}
    writers = {
        "srt": lambda p: write_srt(seg_list, p),
        "vtt": lambda p: write_vtt(seg_list, p),
        "ass": lambda p: write_ass(seg_list, p, style),
    }
    for fmt in formats:
        fmt = fmt.strip().lower()
        if fmt not in writers:
            raise ValueError(f"未対応の字幕形式 '{fmt}'. 対応: srt, vtt, ass")
        written[fmt] = writers[fmt](base_path.with_suffix(f".{fmt}"))
    return written
