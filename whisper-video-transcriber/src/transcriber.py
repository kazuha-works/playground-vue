"""faster-whisper wrapper that runs transcription on the local GPU (or CPU)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .subtitles import Segment


@dataclass
class TranscribeOptions:
    model: str = "large-v3"
    device: str = "auto"          # "auto" | "cuda" | "cpu"
    compute_type: str = "auto"    # "auto" | "float16" | "int8_float16" | "int8" | "float32"
    language: str | None = None   # None -> auto detect
    beam_size: int = 5
    vad_filter: bool = True
    download_root: str | None = None


@dataclass
class TranscribeResult:
    segments: list[Segment]
    language: str
    language_probability: float
    duration: float
    model: str
    device: str
    compute_type: str
    detected: dict = field(default_factory=dict)


def _resolve_device_and_compute(device: str, compute_type: str) -> tuple[str, str]:
    """Choose a sensible device/precision pair, preferring CUDA when present."""
    resolved_device = device
    if device == "auto":
        try:
            from ctranslate2 import get_cuda_device_count

            resolved_device = "cuda" if get_cuda_device_count() > 0 else "cpu"
        except Exception:
            resolved_device = "cpu"

    if compute_type == "auto":
        # float16 is the fast, accurate default on GPU; int8 keeps CPU usable.
        resolved_compute = "float16" if resolved_device == "cuda" else "int8"
    else:
        resolved_compute = compute_type

    return resolved_device, resolved_compute


def transcribe(
    audio_path: Path,
    options: TranscribeOptions,
    on_segment: Callable[[Segment], None] | None = None,
) -> TranscribeResult:
    """Transcribe an audio file, streaming segments to ``on_segment`` as they land."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "faster-whisper がインストールされていません。"
            "`pip install -r requirements.txt` を実行してください。"
        ) from exc

    device, compute_type = _resolve_device_and_compute(
        options.device, options.compute_type
    )

    model = WhisperModel(
        options.model,
        device=device,
        compute_type=compute_type,
        download_root=options.download_root,
    )

    segments_iter, info = model.transcribe(
        str(audio_path),
        language=options.language,
        beam_size=options.beam_size,
        vad_filter=options.vad_filter,
    )

    collected: list[Segment] = []
    for seg in segments_iter:
        segment = Segment(start=seg.start, end=seg.end, text=seg.text)
        collected.append(segment)
        if on_segment is not None:
            on_segment(segment)

    return TranscribeResult(
        segments=collected,
        language=info.language,
        language_probability=info.language_probability,
        duration=info.duration,
        model=options.model,
        device=device,
        compute_type=compute_type,
    )
