"""Unit tests for the pure-Python pieces (no GPU / ffmpeg required)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.colors import hex_to_ass, normalize_hex
from src.model_selector import select_model
from src.subtitles import (
    Segment,
    SubtitleStyle,
    _fmt_ass_ts,
    _fmt_srt_ts,
    write_srt,
)


# --- colors -----------------------------------------------------------------

def test_normalize_named_and_shorthand():
    assert normalize_hex("yellow") == "#FFEB3B"
    assert normalize_hex("#fa0") == "#FFAA00"
    assert normalize_hex("00ff00") == "#00FF00"


def test_hex_to_ass_channel_order():
    # ASS is &HAABBGGRR: pure red -> blue/green 00, red FF.
    assert hex_to_ass("#FF0000") == "&H000000FF"
    assert hex_to_ass("#00FF00") == "&H0000FF00"
    assert hex_to_ass("#0000FF") == "&H00FF0000"


def test_invalid_color_raises():
    with pytest.raises(ValueError):
        normalize_hex("not-a-color")


# --- model selection --------------------------------------------------------

def test_auto_selects_by_duration():
    assert select_model("auto", 5).model == "large-v3"
    assert select_model("auto", 25).model == "medium"
    assert select_model("auto", 60).model == "small"
    assert select_model("auto", 500).model == "base"


def test_override_beats_balance():
    assert select_model("quality", 5, override="tiny").model == "tiny"


def test_fixed_presets():
    assert select_model("speed", 999).model == "base"
    assert select_model("balanced", 999).model == "medium"
    assert select_model("quality", 999).model == "large-v3"


# --- timestamps -------------------------------------------------------------

def test_srt_timestamp():
    assert _fmt_srt_ts(3661.5) == "01:01:01,500"


def test_ass_timestamp():
    assert _fmt_ass_ts(3661.5) == "1:01:01.50"


# --- subtitle writing -------------------------------------------------------

def test_write_srt(tmp_path):
    segs = [Segment(0.0, 1.5, " hello "), Segment(1.5, 3.0, "world")]
    out = write_srt(segs, tmp_path / "x.srt")
    body = out.read_text(encoding="utf-8")
    assert "1\n00:00:00,000 --> 00:00:01,500\nhello" in body
    assert "2\n00:00:01,500 --> 00:00:03,000\nworld" in body


def test_font_size_scales_with_resolution():
    assert SubtitleStyle(play_res_y=1080).resolved_font_size() == 49
    assert SubtitleStyle(play_res_y=2160).resolved_font_size() == 97
    assert SubtitleStyle(font_size=30, play_res_y=1080).resolved_font_size() == 30
