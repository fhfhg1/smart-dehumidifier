"""水箱储水分层估算模型(纯函数,无依赖)。

不再只按"模式"粗估,而是按特征分层回退,层层放宽直到样本够:
    模式+场景+湿度区间 → 模式+场景 → 模式+湿度区间 → 模式 → 场景 → 全局
并对每层样本做异常值过滤(中位数 ± 3*MAD),减少错误校正污染估算。

样本格式(由 coordinator 的水箱反馈写入):
    {"mode": str, "scene": str, "humidity_bucket": str, "rate": float}
其中 rate 为"每分钟储水增量(L/min)"或任意可平均的储水速率指标。
"""
from __future__ import annotations

from statistics import median
from typing import Any

MIN_SAMPLES = 3


def humidity_bucket(humidity: float) -> str:
    """把湿度分到区间,作为分层键。"""
    if humidity >= 75:
        return "very_high"
    if humidity >= 65:
        return "high"
    if humidity >= 55:
        return "mid"
    return "low"


def _reject_outliers(values: list[float]) -> list[float]:
    if len(values) < 5:
        return values
    med = median(values)
    mad = median([abs(v - med) for v in values]) or 0.0
    if mad == 0:
        return values
    kept = [v for v in values if abs(v - med) <= 3.0 * mad]
    return kept if len(kept) >= 3 else values


def _avg(values: list[float]) -> float | None:
    vals = _reject_outliers([v for v in values if v and v > 0])
    return round(sum(vals) / len(vals), 4) if vals else None


def estimate_rate(samples: list[dict[str, Any]], *, mode: str, scene: str,
                  humidity: float) -> tuple[float | None, str]:
    """按分层回退估算储水速率,返回 (估算值, 命中层名)。无样本则 (None, '无样本')。"""
    bucket = humidity_bucket(humidity)

    def pick(pred) -> list[float]:
        return [float(s.get("rate", 0) or 0) for s in samples if pred(s)]

    layers = [
        ("模式+场景+湿度区间", lambda s: s.get("mode") == mode and s.get("scene") == scene and s.get("humidity_bucket") == bucket),
        ("模式+场景", lambda s: s.get("mode") == mode and s.get("scene") == scene),
        ("模式+湿度区间", lambda s: s.get("mode") == mode and s.get("humidity_bucket") == bucket),
        ("模式", lambda s: s.get("mode") == mode),
        ("场景", lambda s: s.get("scene") == scene),
        ("全局", lambda s: True),
    ]
    for name, pred in layers:
        vals = pick(pred)
        if len([v for v in vals if v > 0]) >= MIN_SAMPLES:
            avg = _avg(vals)
            if avg is not None:
                return avg, name
    # 兜底:全局任意正样本
    avg = _avg([float(s.get("rate", 0) or 0) for s in samples])
    return (avg, "全局") if avg is not None else (None, "无样本")
