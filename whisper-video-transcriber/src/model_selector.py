"""Pick a Whisper model size that balances accuracy against runtime.

The core idea the user asked for: adapt the model to the size of the video.
Short clips can afford the largest, most accurate model; multi-hour recordings
lean toward faster models so a transcription still finishes in reasonable time,
even on a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

# Model sizes ordered from fastest/lightest to slowest/most-accurate.
MODEL_LADDER = ["tiny", "base", "small", "medium", "large-v3"]

# Fixed presets that ignore the video length.
FIXED_PRESETS = {
    "speed": "base",
    "balanced": "medium",
    "quality": "large-v3",
}

# Duration thresholds (minutes) -> model, used only when balance == "auto".
# Walk the list top-down and take the first bucket the video fits in.
_AUTO_BUCKETS = [
    (10.0, "large-v3"),   # <= 10 min: go for maximum accuracy
    (30.0, "medium"),     # <= 30 min
    (90.0, "small"),      # <= 90 min
    (float("inf"), "base"),  # anything longer: keep it fast
]


@dataclass
class ModelChoice:
    model: str
    reason: str


def select_model(balance: str, duration_min: float, override: str | None = None) -> ModelChoice:
    """Resolve the effective Whisper model.

    Precedence: explicit ``override`` > fixed preset > auto (by duration).
    """
    if override and override != "auto":
        if override not in MODEL_LADDER:
            raise ValueError(
                f"未知のモデル '{override}'. 選択可能: {', '.join(MODEL_LADDER)}"
            )
        return ModelChoice(override, f"ユーザー指定 (--model {override})")

    if balance in FIXED_PRESETS:
        model = FIXED_PRESETS[balance]
        return ModelChoice(model, f"バランス '{balance}' の固定モデル")

    if balance != "auto":
        raise ValueError(
            f"未知のバランス '{balance}'. 選択可能: auto, {', '.join(FIXED_PRESETS)}"
        )

    for limit, model in _AUTO_BUCKETS:
        if duration_min <= limit:
            return ModelChoice(
                model,
                f"自動選択: 長さ {duration_min:.1f} 分 (<= {limit} 分) -> {model}",
            )

    # Unreachable because the last bucket is +inf, but keep the type checker happy.
    return ModelChoice("base", "フォールバック")
