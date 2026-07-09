"""Media inspection and audio extraction helpers (thin wrappers over ffmpeg/ffprobe)."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class FFmpegNotFound(RuntimeError):
    """Raised when ffmpeg/ffprobe are not available on PATH."""


@dataclass
class MediaInfo:
    path: Path
    duration_sec: float
    width: int
    height: int
    size_bytes: int

    @property
    def duration_min(self) -> float:
        return self.duration_sec / 60.0

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


def require_ffmpeg() -> None:
    """Ensure both ffmpeg and ffprobe exist, otherwise raise a clear error."""
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise FFmpegNotFound(
            f"必要なコマンドが見つかりません: {', '.join(missing)}. "
            "ffmpeg をインストールしてください (例: apt install ffmpeg / brew install ffmpeg)."
        )


def probe(path: Path) -> MediaInfo:
    """Return duration/resolution/size for a media file using ffprobe."""
    require_ffmpeg()
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)

    duration = float(data.get("format", {}).get("duration", 0.0) or 0.0)

    width = height = 0
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            width = int(stream.get("width", 0) or 0)
            height = int(stream.get("height", 0) or 0)
            # Prefer stream duration when the container header is missing one.
            if duration <= 0 and stream.get("duration"):
                duration = float(stream["duration"])
            break

    return MediaInfo(
        path=path,
        duration_sec=duration,
        width=width,
        height=height,
        size_bytes=path.stat().st_size,
    )


def extract_audio(path: Path, out_wav: Path) -> Path:
    """Extract a 16kHz mono PCM wav — the format Whisper models expect."""
    require_ffmpeg()
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(out_wav),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return out_wav
