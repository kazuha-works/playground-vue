#!/usr/bin/env python3
"""Local, GPU-accelerated automatic video transcription.

Pipeline:
  1. Probe the video (duration / resolution / size).
  2. Auto-select a Whisper model that balances accuracy vs. runtime by length.
  3. Extract 16kHz mono audio with ffmpeg.
  4. Transcribe with faster-whisper on the GPU (falls back to CPU).
  5. Emit SRT / VTT / colored ASS subtitles, and optionally burn them in.

Example:
    python transcribe.py talk.mp4 --language ja --color yellow --burn
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from src import media
from src.model_selector import select_model
from src.subtitles import SubtitleStyle, burn_subtitles, write_all
from src.transcriber import TranscribeOptions, transcribe


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="transcribe",
        description="ローカルGPUで動画を自動文字起こしし、色付き字幕を生成します。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", type=Path, help="入力動画ファイル")
    p.add_argument("-o", "--output-dir", type=Path, default=Path("output"),
                   help="出力先ディレクトリ")

    # Model / balance
    p.add_argument("--balance", choices=["auto", "speed", "balanced", "quality"],
                   default="auto",
                   help="動画の長さに応じたモデル自動選択、または固定プリセット")
    p.add_argument("--model", default="auto",
                   help="モデルを明示指定 (tiny/base/small/medium/large-v3)。'auto'で--balanceに従う")

    # Compute
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                   help="推論デバイス")
    p.add_argument("--compute-type", default="auto",
                   help="精度 (auto/float16/int8_float16/int8/float32)")
    p.add_argument("--language", default=None,
                   help="言語コード (例: ja, en)。未指定なら自動検出")
    p.add_argument("--beam-size", type=int, default=5, help="ビームサーチ幅")
    p.add_argument("--no-vad", action="store_true", help="VAD(無音除去)を無効化")

    # Subtitle styling
    p.add_argument("--formats", default="srt,vtt,ass",
                   help="生成する字幕形式 (カンマ区切り: srt,vtt,ass)")
    p.add_argument("--color", default="#FFEB3B",
                   help="字幕の文字色 (#RRGGBB か色名)")
    p.add_argument("--outline-color", default="#000000", help="字幕の縁取り色")
    p.add_argument("--font", default="Arial", help="字幕フォント名")
    p.add_argument("--font-size", type=int, default=None,
                   help="字幕サイズ(px)。未指定なら解像度から自動")
    p.add_argument("--burn", action="store_true",
                   help="色付き字幕を動画に焼き込んだmp4も出力 (要ffmpeg)")

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if not args.input.exists():
        print(f"エラー: 入力ファイルが見つかりません: {args.input}", file=sys.stderr)
        return 1

    try:
        media.require_ffmpeg()
    except media.FFmpegNotFound as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1

    # 1. Probe the video.
    info = media.probe(args.input)
    print(
        f"[入力] {args.input.name}  "
        f"長さ={info.duration_min:.1f}分  解像度={info.width}x{info.height}  "
        f"サイズ={info.size_mb:.1f}MB"
    )

    # 2. Decide the model.
    choice = select_model(args.balance, info.duration_min, override=args.model)
    print(f"[モデル] {choice.model}  ({choice.reason})")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem

    # 3. Extract audio to a temp wav.
    with tempfile.TemporaryDirectory() as tmp:
        wav = media.extract_audio(args.input, Path(tmp) / f"{stem}.wav")
        print("[音声] 16kHz mono wav を抽出しました")

        # 4. Transcribe (stream progress to stdout).
        opts = TranscribeOptions(
            model=choice.model,
            device=args.device,
            compute_type=args.compute_type,
            language=args.language,
            beam_size=args.beam_size,
            vad_filter=not args.no_vad,
        )
        print("[文字起こし] 開始...")

        def _progress(seg) -> None:
            print(f"  {seg.start:7.1f}s  {seg.text.strip()}")

        result = transcribe(wav, opts, on_segment=_progress)

    print(
        f"[文字起こし] 完了  言語={result.language} "
        f"(確度 {result.language_probability:.2f})  "
        f"デバイス={result.device}/{result.compute_type}  "
        f"セグメント数={len(result.segments)}"
    )

    if not result.segments:
        print("警告: 文字起こし結果が空でした。", file=sys.stderr)
        return 2

    # 5. Write subtitles. Match ASS canvas to the real video resolution so the
    #    auto font-size and positioning scale correctly.
    style = SubtitleStyle(
        font=args.font,
        font_size=args.font_size,
        primary_color=args.color,
        outline_color=args.outline_color,
        play_res_x=info.width or 1920,
        play_res_y=info.height or 1080,
    )
    formats = [f for f in args.formats.split(",") if f.strip()]
    base = args.output_dir / stem
    written = write_all(result.segments, formats, base, style)
    for fmt, path in written.items():
        print(f"[字幕] {fmt.upper()}: {path}")

    if args.burn:
        ass_path = written.get("ass")
        if ass_path is None:
            # Burning needs the styled ASS; generate it on demand.
            from src.subtitles import write_ass
            ass_path = write_ass(result.segments, base.with_suffix(".ass"), style)
        out_mp4 = args.output_dir / f"{stem}.subtitled.mp4"
        print("[焼き込み] 字幕を動画に合成中...")
        burn_subtitles(args.input, ass_path, out_mp4)
        print(f"[焼き込み] 出力: {out_mp4}")

    print("完了しました。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
