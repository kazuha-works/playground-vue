# whisper-video-transcriber

ローカルのGPUで動画を**自動文字起こし**し、**色付き字幕**を生成するツールです。
[faster-whisper](https://github.com/SYSTRAN/faster-whisper)（CTranslate2実装のWhisper）を利用します。

## 特長

- 🎮 **ローカルGPU実行** — CUDAが使えれば自動でGPU推論（`float16`）、無ければCPUにフォールバック
- ⚖️ **動画の長さでモデルを自動バランス** — 短い動画は高精度モデル、長い動画は高速モデルを自動選択
- 🎨 **字幕の色をカスタマイズ** — 文字色・縁取り色を自由に指定、解像度に応じて文字サイズも自動調整
- 📄 **複数フォーマット出力** — `SRT` / `WebVTT` / 色付き `ASS`、さらに動画への**焼き込み**にも対応
- 🈺 **多言語対応・自動言語検出**（日本語もOK）

## 必要環境

- Python 3.9+
- [ffmpeg](https://ffmpeg.org/)（音声抽出・字幕焼き込みに使用。`ffprobe` も同梱されます）
- NVIDIA GPU + CUDA（任意。無くてもCPUで動作します）

## インストール

```bash
# ffmpeg（システム）
sudo apt install ffmpeg        # Debian/Ubuntu
# brew install ffmpeg         # macOS

# Python 依存
pip install -r requirements.txt

# GPU (CUDA 12) を使う場合、cuDNN/cuBLAS ランタイムも
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

## 使い方

```bash
# もっともシンプル（言語自動検出・モデル自動選択・字幕生成）
python transcribe.py input.mp4

# 日本語指定 + 黄色い字幕 + 動画へ焼き込み
python transcribe.py input.mp4 --language ja --color yellow --burn

# 精度優先で large-v3 固定、縁取りを白に
python transcribe.py input.mp4 --balance quality --outline-color white

# 出力形式を SRT のみに、出力先を指定
python transcribe.py input.mp4 --formats srt -o ./subs
```

出力は既定で `output/` に生成されます（`input.srt`, `input.vtt`, `input.ass`、`--burn` 時は `input.subtitled.mp4`）。

## モデルの自動バランス（`--balance auto`）

動画の**長さ**に応じて、精度と処理時間のバランスをとってモデルを選びます。

| 動画の長さ | 選択モデル  | ねらい          |
| ---------- | ----------- | --------------- |
| 〜10分     | `large-v3`  | 最高精度        |
| 〜30分     | `medium`    | 精度と速度の両立 |
| 〜90分     | `small`     | 速度寄り        |
| 90分〜     | `base`      | 長尺でも高速    |

固定したい場合は `--balance speed|balanced|quality`、または `--model large-v3` のように明示指定できます。

## 主なオプション

| オプション          | 説明                                                    | 既定値      |
| ------------------- | ------------------------------------------------------- | ----------- |
| `--balance`         | `auto` / `speed` / `balanced` / `quality`               | `auto`      |
| `--model`           | `tiny`/`base`/`small`/`medium`/`large-v3`（`auto`で自動）| `auto`      |
| `--device`          | `auto` / `cuda` / `cpu`                                 | `auto`      |
| `--compute-type`    | `auto`/`float16`/`int8_float16`/`int8`/`float32`         | `auto`      |
| `--language`        | 言語コード（例 `ja`, `en`）。未指定なら自動検出         | 自動        |
| `--color`           | 字幕の文字色（`#RRGGBB` か色名）                        | `#FFEB3B`   |
| `--outline-color`   | 字幕の縁取り色                                          | `#000000`   |
| `--font` / `--font-size` | フォント名 / サイズ(px、未指定なら解像度から自動)  | `Arial` / 自動 |
| `--formats`         | 生成する字幕形式（`srt,vtt,ass`）                        | `srt,vtt,ass` |
| `--burn`            | 色付き字幕を動画に焼き込んだmp4も出力                    | 無効        |

色名は `white, black, red, green, blue, yellow, orange, cyan, magenta, pink, lime, gray` に対応。

## 仕組み

```
動画 ──probe──▶ 長さ/解像度/サイズ
   │
   ├─ モデル自動選択（長さベース）
   │
   ├─ ffmpegで16kHz mono wavを抽出
   │
   ├─ faster-whisperでGPU文字起こし
   │
   └─ SRT / VTT / 色付きASS を生成 ─(--burn)─▶ 字幕入りmp4
```

## テスト

```bash
pip install pytest
python -m pytest tests/
```

GPU/ffmpeg 不要の純Python部分（色変換・モデル選択・字幕整形）を検証します。

## ライセンス

MIT
