#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any


TIME_FORMATS = (
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
)

RUN_EVENTS = {
    "stop_target",
    "stop_low_protect",
    "stop_critical_low_protect",
    "stop_early_drop",
    "stop_clothes_dry_timeout",
    "stop_shoes_dry_timeout",
    "stop_manual",
}

REBOUND_EVENTS = {
    "rebound_30m": 30,
    "rebound_60m": 60,
    "rebound_90m": 90,
    "rebound_120m": 120,
}

# --- 学习核心可调参数 ---------------------------------------------------------
# 从原先散落在函数里的魔法数字抽出,集中在此便于回测/调参。
# 改这些值不影响 HA 控制逻辑的结构,只改变"学习"部分的行为。

# exponential_weighted_average: 衰减跨度(单位:样本)。越小越偏向最近样本。
EWMA_DECAY_SPAN = 5.0

# filter_samples: 各分组(场景+模式/场景/模式)至少要这么多样本才被采用。
MIN_SAMPLES_PER_GROUP = 3

# blend_rate: 当前实测值 vs 历史学习值 的混合权重,按历史样本量分档。
# (sample_count 上界, 给"当前实测值"的权重)；最后一档 None = 兜底。
BLEND_RECENT_WEIGHTS: tuple[tuple[int | None, float], ...] = (
    (3, 0.72),    # 样本 <3:更信当前实测
    (8, 0.62),
    (16, 0.50),
    (None, 0.36),  # 样本 >=16:更信历史学习
)

# classify_confidence: 置信度分级阈值。
CONF_HIGH_MIN_SAMPLES = 8
CONF_HIGH_MAX_CV = 0.35
CONF_MEDIUM_MIN_SAMPLES = 4
CONF_MEDIUM_MAX_CV = 0.75

# backtest: 预测与实际事件配对时,实际事件必须落在预测时间之后这么多分钟内,
# 否则视为"没有对应的真实结果"而跳过(避免跨越长时间间隔的错误配对)。
BACKTEST_MATCH_HORIZON_MIN = 360

# 所有"倒计时/时长"类控制参数。这些当前由 compute_predictions 用启发式算出、
# 经 control_takeover 接管、HA 端有规则默认值兜底。每次决策的取值会被记进
# 预测日志(decided_timers),为后续按"等待时长 -> 实际结果"学习它们做准备。
TIMER_KEYS = (
    "low_protect_confirm_minutes",
    "critical_low_confirm_seconds",
    "lockout_minutes",
    "min_runtime_minutes",
    "start_confirm_minutes",
    "stop_confirm_minutes",
    "early_stop_guard_minutes",
    "lock_release_delta",
    "auto_start_window",
)

# outcome(结果好坏)判定阈值,用于 backtest/stats 评估倒计时选得好不好。
SHORT_CYCLE_MIN = 20    # 停机后 < 这么多分钟又开机 → 短循环(等太短/锁太短)
OVERDRY_MARGIN = 5.0    # 停机时 end_humidity 低于 target 这么多 → 过度除湿(等太长)

# 预测日志轮转上限。每次落盘后只保留最近这么多条,避免无限增长。
# ~每 15 分钟一条 → 8000 条约覆盖 80+ 天。
PREDICTIONS_MAX_LINES = 8000


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in TIME_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def parse_line_fields(row: list[str]) -> tuple[str | None, str, dict[str, str]]:
    if len(row) < 2:
        return None, "", {}
    timestamp = row[0].strip()
    event = row[1].strip()
    data: dict[str, str] = {}
    for part in row[2:]:
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            key, value = part.split("=", 1)
            data[key.strip()] = value.strip()
        else:
            data.setdefault("message", part)
    return timestamp, event, data


def as_float(data: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(data.get(key, default))
    except (TypeError, ValueError):
        return default


def as_int(data: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(data.get(key, default)))
    except (TypeError, ValueError):
        return default


def jsonl_dump(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any], max_lines: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if max_lines is not None:
        _trim_to_last(path, max_lines)


def _trim_to_last(path: Path, max_lines: int) -> None:
    """把文件裁剪到最后 max_lines 行(超出才重写),用于日志轮转。"""
    with path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
    if len(lines) <= max_lines:
        return
    with path.open("w", encoding="utf-8") as handle:
        handle.writelines(lines[-max_lines:])


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def exponential_weighted_average(values: list[float]) -> float:
    if not values:
        return 0.0
    weights = [math.exp((idx - len(values) + 1) / EWMA_DECAY_SPAN) for idx in range(len(values))]
    weighted_total = sum(value * weight for value, weight in zip(values, weights))
    total_weight = sum(weights) or 1.0
    return weighted_total / total_weight


def build_structured_samples(source: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    runs: list[dict[str, Any]] = []
    rebounds: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []

    if not source.exists():
        return runs, rebounds, snapshots

    with source.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header_skipped = False
        for row in reader:
            if not row:
                continue
            if not header_skipped and row[0] == "timestamp":
                header_skipped = True
                continue
            timestamp, event, data = parse_line_fields(row)
            ts = parse_time(timestamp)
            scene = data.get("scene", "normal")
            mode = data.get("mode", data.get("mode_or_state", "comfort"))
            season = data.get("season", "")
            weather = data.get("weather", "")
            period = data.get("period", "")
            target_humidity = as_float(data, "target")
            if target_humidity <= 0:
                target_humidity = as_float(data, "target_humidity", 0.0)
            if target_humidity <= 0:
                target_humidity = 60.0

            if event in RUN_EVENTS and "duration_min" in data:
                sample = {
                    "run_id": f"{timestamp}:{event}",
                    "timestamp": timestamp,
                    "event": event,
                    "scene": scene,
                    "mode": mode,
                    "season": season,
                    "weather_bucket": weather,
                    "period": period,
                    "start_humidity": as_float(data, "start_h"),
                    "end_humidity": as_float(data, "end_h"),
                    "target_humidity": target_humidity,
                    "duration_min": as_int(data, "duration_min"),
                    "drop_amount": as_float(data, "drop"),
                    "drop_rate": as_float(data, "drop_rate"),
                    "alpha": as_float(data, "alpha"),
                    "reason": data.get("reason", ""),
                    "datetime": ts.isoformat() if ts else "",
                }
                runs.append(sample)
                continue

            if event in REBOUND_EVENTS:
                elapsed_min = as_int(data, "elapsed_min", REBOUND_EVENTS[event])
                start_humidity = as_float(data, "start_h")
                end_humidity = as_float(data, "humidity")
                rebound_rate = as_float(data, "rebound_rate")
                if rebound_rate <= 0 and elapsed_min > 0 and end_humidity >= start_humidity:
                    rebound_rate = round((end_humidity - start_humidity) / elapsed_min, 3)
                sample = {
                    "rebound_id": f"{timestamp}:{event}",
                    "timestamp": timestamp,
                    "event": event,
                    "scene": scene,
                    "mode": mode,
                    "season": season,
                    "weather_bucket": weather,
                    "period": period,
                    "start_humidity": start_humidity,
                    "end_humidity": end_humidity,
                    "target_humidity": target_humidity,
                    "elapsed_min": elapsed_min,
                    "sample_window_min": REBOUND_EVENTS[event],
                    "rebound_rate": rebound_rate,
                    "alpha": as_float(data, "alpha"),
                    "datetime": ts.isoformat() if ts else "",
                }
                rebounds.append(sample)
                continue

            if event == "snapshot":
                sample = {
                    "snapshot_id": f"{timestamp}:{event}",
                    "timestamp": timestamp,
                    "scene": scene,
                    "mode": mode,
                    "state": data.get("state", ""),
                    "running": data.get("running", ""),
                    "humidity": as_float(data, "humidity"),
                    "target_humidity": target_humidity,
                    "drop_rate": as_float(data, "drop_rate"),
                    "rebound_rate": as_float(data, "rebound_rate"),
                    "confidence": data.get("confidence", ""),
                    "datetime": ts.isoformat() if ts else "",
                }
                snapshots.append(sample)

    runs.sort(key=lambda item: item["timestamp"])
    rebounds.sort(key=lambda item: item["timestamp"])
    snapshots.sort(key=lambda item: item["timestamp"])
    return runs, rebounds, snapshots


def _reject_outliers(values: list[float]) -> list[float]:
    """用中位数 ± 3*MAD 剔除极端样本,避免一两次异常运行把均值带偏。
    样本 <5 时不过滤(数据太少,宁可全留)。"""
    if len(values) < 5:
        return values
    med = median(values)
    mad = median([abs(v - med) for v in values]) or 0.0
    if mad == 0:
        return values
    kept = [v for v in values if abs(v - med) <= 3.0 * mad]
    return kept if len(kept) >= 3 else values


def filter_samples(
    samples: list[dict[str, Any]],
    *,
    scene: str,
    mode: str,
    key: str,
) -> tuple[list[float], str]:
    exact = [sample for sample in samples if sample.get("scene") == scene and sample.get("mode") == mode and sample.get(key, 0) > 0]
    if len(exact) >= MIN_SAMPLES_PER_GROUP:
        return _reject_outliers([float(sample[key]) for sample in exact]), "场景+模式"
    by_scene = [sample for sample in samples if sample.get("scene") == scene and sample.get(key, 0) > 0]
    if len(by_scene) >= MIN_SAMPLES_PER_GROUP:
        return _reject_outliers([float(sample[key]) for sample in by_scene]), "场景"
    by_mode = [sample for sample in samples if sample.get("mode") == mode and sample.get(key, 0) > 0]
    if len(by_mode) >= MIN_SAMPLES_PER_GROUP:
        return _reject_outliers([float(sample[key]) for sample in by_mode]), "模式"
    fallback = [float(sample[key]) for sample in samples if sample.get(key, 0) > 0]
    return _reject_outliers(fallback), "全局"


def coefficient_of_variation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    if avg <= 0:
        return 0.0
    return pstdev(values) / avg


def clamp(value: float, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(round(value))))


def average_rate(samples: list[dict[str, Any]], key: str) -> float:
    values = [float(sample[key]) for sample in samples if float(sample.get(key, 0) or 0) > 0]
    return round(mean(values), 3) if values else 0.0


def classify_anomaly(
    runs: list[dict[str, Any]],
    rebounds: list[dict[str, Any]],
    context: PredictionContext,
    effective_drop: float,
    effective_rebound: float,
) -> tuple[str, str]:
    recent_runs = [sample for sample in runs if sample.get("drop_rate", 0) > 0][-5:]
    older_runs = [sample for sample in runs[:-5] if sample.get("drop_rate", 0) > 0]
    recent_rebounds = [sample for sample in rebounds if sample.get("rebound_rate", 0) > 0][-5:]
    older_rebounds = [sample for sample in rebounds[:-5] if sample.get("rebound_rate", 0) > 0]

    recent_run_avg = average_rate(recent_runs, "drop_rate")
    older_run_avg = average_rate(older_runs, "drop_rate")
    recent_rebound_avg = average_rate(recent_rebounds, "rebound_rate")
    older_rebound_avg = average_rate(older_rebounds, "rebound_rate")
    recent_duration_avg = average_rate(recent_runs, "duration_min")

    if len(recent_runs) >= 3 and older_run_avg > 0 and recent_run_avg < older_run_avg * 0.6:
        return "warning", "最近几次除湿速度明显低于历史水平，建议检查滤网、摆放位置或门窗状态。"
    if len(recent_rebounds) >= 3 and older_rebound_avg > 0 and recent_rebound_avg > older_rebound_avg * 1.5:
        return "warning", "最近回潮速度明显变快，可能门窗未关严或房间存在持续湿源。"
    if context.scene == "window_open_suspected":
        return "notice", "湿度下降异常缓慢，系统怀疑当前有开窗或持续进湿。"
    if context.scene == "night" and len(recent_runs) >= 3 and recent_duration_avg > 0 and recent_duration_avg < 20:
        return "notice", "夜间近期启动较频繁，建议继续观察是否需要提高夜间启动阈值。"
    if effective_rebound >= 0.10:
        return "notice", "当前环境回潮较快，系统已提高提前开机倾向。"
    return "normal", "当前未发现明显异常，设备与环境表现基本正常。"


# 外部环境开机偏置:正=更不爱开机,负=更早开机。每条规则仅在对应输入可用时生效。
EXTERNAL_BIAS_MIN = -3
EXTERNAL_BIAS_MAX = 10
_AWAY_TOKENS = {"away", "not_home", "离开", "无人", "外出"}
_HOME_TOKENS = {"home", "在家", "有人"}
_COOLING_TOKENS = {"cool", "cooling", "制冷", "制冷中"}


def compute_external_advice(context: "PredictionContext") -> tuple[int, str]:
    """根据外部环境算出开机阈值偏置 + 中文说明。无可用外部信号则返回 (0, 默认语)。"""
    bias = 0
    notes: list[str] = []
    humid = context.humidity > context.target

    if context.window_open is True:
        bias += 10
        notes.append("检测到开窗,已大幅抑制自动开机(此时除湿基本无效)")
    if context.hvac and any(tok in context.hvac.lower() for tok in _COOLING_TOKENS):
        bias += 2
        notes.append("空调制冷中(本身在除湿),延后开机避免重复运行")
    if context.presence and any(tok in context.presence.lower() for tok in _AWAY_TOKENS) and humid:
        bias -= 1
        notes.append("无人在家,允许更早除湿(不打扰)")
    if context.outdoor_humidity is not None and context.outdoor_humidity >= 85:
        bias -= 1
        notes.append("室外高湿,提前除湿以抵御进湿")
    if context.rainy_now is True:
        bias -= 1
        notes.append("当前降雨,提前除湿")

    bias = max(EXTERNAL_BIAS_MIN, min(EXTERNAL_BIAS_MAX, bias))
    return bias, ";".join(notes) if notes else "无外部环境调整(传感器未配置或无触发)"


@dataclass
class PredictionContext:
    humidity: float
    target: float
    scene: str
    mode: str
    state: str
    running: bool
    current_drop_rate: float
    current_rebound_rate: float
    start_threshold: float
    min_runtime_left: int
    now: datetime
    # 外部环境(均可选,None/空 = 该传感器未配置或不可用 → 不参与决策)
    window_open: bool | None = None
    presence: str | None = None      # home / away / unknown
    hvac: str | None = None          # cooling / heating / idle / unknown
    outdoor_humidity: float | None = None
    rainy_now: bool | None = None


def blend_rate(current_rate: float, learned_rate: float, sample_count: int) -> float:
    if current_rate > 0 and learned_rate > 0:
        recent_weight = BLEND_RECENT_WEIGHTS[-1][1]
        for upper, weight in BLEND_RECENT_WEIGHTS:
            if upper is not None and sample_count < upper:
                recent_weight = weight
                break
        return round((current_rate * recent_weight) + (learned_rate * (1 - recent_weight)), 3)
    if current_rate > 0:
        return round(current_rate, 3)
    return round(learned_rate, 3)


def classify_confidence(sample_count: int, cv: float, source_scope: str) -> tuple[str, str]:
    if sample_count >= CONF_HIGH_MIN_SAMPLES and cv <= CONF_HIGH_MAX_CV and source_scope in {"场景+模式", "场景"}:
        return "high", "当前场景样本较充足，且波动较小"
    if sample_count >= CONF_MEDIUM_MIN_SAMPLES and cv <= CONF_MEDIUM_MAX_CV:
        return "medium", "已有可用样本，预测开始稳定"
    if sample_count == 0:
        return "low", "还没有足够样本，系统正在积累数据"
    return "low", "样本还不够集中，系统继续观察环境变化"


def compute_predictions(
    runs: list[dict[str, Any]],
    rebounds: list[dict[str, Any]],
    snapshots: list[dict[str, Any]],
    context: PredictionContext,
    *,
    learned_drop_override: float | None = None,
    learned_rebound_override: float | None = None,
) -> dict[str, Any]:
    drop_values, drop_scope = filter_samples(runs, scene=context.scene, mode=context.mode, key="drop_rate")
    rebound_values, rebound_scope = filter_samples(rebounds, scene=context.scene, mode=context.mode, key="rebound_rate")

    # 若提供模型预测(learned_*_override),用它替代 EWMA 启发式;否则维持原行为。
    predictor = "model" if (learned_drop_override is not None or learned_rebound_override is not None) else "heuristic"
    learned_drop = round(learned_drop_override if learned_drop_override is not None
                         else exponential_weighted_average(drop_values), 3)
    learned_rebound = round(learned_rebound_override if learned_rebound_override is not None
                            else exponential_weighted_average(rebound_values), 3)
    effective_drop = blend_rate(context.current_drop_rate, learned_drop, len(drop_values))
    effective_rebound = blend_rate(context.current_rebound_rate, learned_rebound, len(rebound_values))

    drop_cv = coefficient_of_variation(drop_values)
    rebound_cv = coefficient_of_variation(rebound_values)
    confidence_drop, reason_drop = classify_confidence(len(drop_values), drop_cv, drop_scope)
    confidence_rebound, reason_rebound = classify_confidence(len(rebound_values), rebound_cv, rebound_scope)
    order = {"low": 0, "medium": 1, "high": 2}
    final_confidence = confidence_drop if order[confidence_drop] <= order[confidence_rebound] else confidence_rebound
    confidence_reason = f"下降样本 {len(drop_values)} 条（{drop_scope}），回潮样本 {len(rebound_values)} 条（{rebound_scope}）。{reason_drop}；{reason_rebound}。"

    scene_sample_count = max(len(drop_values), len(rebound_values))
    strong_learning = scene_sample_count >= 5 and final_confidence in {"medium", "high"}
    mature_learning = scene_sample_count >= 8 and final_confidence == "high"
    drop_model_ready = len(drop_values) >= 12 and final_confidence in {"medium", "high"}
    rebound_model_ready = len(rebound_values) >= 18 and final_confidence == "high"

    if rebound_model_ready and drop_model_ready:
        learning_stage = "takeover_ready"
        learning_stage_cn = "可进入主控验证"
        next_milestone = "可开始小范围验证模型接管，重点观察回潮预测和锁定释放。"
    elif strong_learning:
        learning_stage = "parameter_learning"
        learning_stage_cn = "参数学习中"
        next_milestone = "继续积累更多回潮样本，优先验证回潮预测能否反超启发式。"
    else:
        learning_stage = "observing"
        learning_stage_cn = "观察积累中"
        next_milestone = "先积累运行和回潮样本，让模型先学会这个房间的基础节奏。"

    base_window = 10 if context.mode == "energy_saving" else 30 if context.mode == "drying" else 20
    auto_window = base_window
    if effective_rebound >= 0.08:
        auto_window += 8
    elif effective_rebound >= 0.05:
        auto_window += 4
    elif effective_rebound <= 0.02:
        auto_window -= 3
    if final_confidence == "high":
        auto_window += 4
    elif final_confidence == "low":
        auto_window -= 5
    auto_window = max(5, min(45, auto_window))

    day_offset = 4
    night_offset = 7
    if effective_rebound >= 0.08:
        day_offset -= 1
        night_offset -= 1
    elif effective_rebound <= 0.02:
        day_offset += 1
        night_offset += 1
    if context.scene == "shower_spike":
        day_offset -= 1
    elif context.scene == "window_open_suspected":
        day_offset += 1
        night_offset += 1
    elif context.scene == "night":
        night_offset += 1
    if context.mode == "energy_saving":
        day_offset += 1
        night_offset += 1
    elif context.mode == "drying":
        day_offset -= 1
    if final_confidence == "low":
        day_offset = 4
        night_offset = 7
    day_offset = clamp(day_offset, 2, 8)
    night_offset = clamp(night_offset, 4, 12)

    start_confirm = 6 if context.mode == "energy_saving" else 4 if context.mode == "drying" else 5
    if context.scene in {"shower_spike", "rainy_high_humidity"} or effective_rebound >= 0.08:
        start_confirm -= 1
    if final_confidence == "low":
        start_confirm += 1
    if context.scene == "night":
        start_confirm += 1
    if context.scene == "window_open_suspected":
        start_confirm += 1
    start_confirm = clamp(start_confirm, 2, 8)

    stop_confirm = 6 if context.mode == "energy_saving" else 4 if context.mode == "drying" else 5
    if effective_rebound >= 0.08:
        stop_confirm -= 1
    elif effective_rebound <= 0.02:
        stop_confirm += 1
    if final_confidence == "low":
        stop_confirm += 1
    if context.scene == "night":
        stop_confirm += 1
    stop_confirm = clamp(stop_confirm, 2, 8)

    min_runtime_minutes = 32 if context.mode == "energy_saving" else 36 if context.mode == "drying" else 28
    if context.scene in {"drying_clothes", "rainy_high_humidity"}:
        min_runtime_minutes += 6
    elif context.scene == "shower_spike":
        min_runtime_minutes += 3
    elif context.scene == "night":
        min_runtime_minutes += 4
    elif context.scene == "window_open_suspected":
        min_runtime_minutes -= 4
    if effective_rebound >= 0.08:
        min_runtime_minutes += 6
    elif effective_rebound >= 0.05:
        min_runtime_minutes += 3
    elif effective_rebound <= 0.02:
        min_runtime_minutes -= 3
    if final_confidence == "low":
        min_runtime_minutes = clamp((min_runtime_minutes + 30) / 2, 24, 36)
    elif final_confidence == "high" and effective_drop >= 0.18 and effective_rebound <= 0.03:
        min_runtime_minutes -= 3
    min_runtime_minutes = clamp(min_runtime_minutes, 18, 45)

    low_protect_confirm = 5
    if final_confidence == "low":
        low_protect_confirm += 1
    if effective_drop >= 0.20:
        low_protect_confirm -= 1
    if context.target <= 58:
        low_protect_confirm += 1
    low_protect_confirm = clamp(low_protect_confirm, 3, 7)

    critical_low_confirm_seconds = 30
    if effective_drop >= 0.25:
        critical_low_confirm_seconds -= 6
    elif effective_drop >= 0.18:
        critical_low_confirm_seconds -= 3
    elif effective_drop <= 0.06:
        critical_low_confirm_seconds += 6
    if context.target <= 58:
        critical_low_confirm_seconds += 4
    if context.scene == "night":
        critical_low_confirm_seconds += 4
    if context.mode == "drying":
        critical_low_confirm_seconds -= 4
    elif context.mode == "energy_saving":
        critical_low_confirm_seconds += 3
    if final_confidence == "low":
        critical_low_confirm_seconds = clamp((critical_low_confirm_seconds + 30) / 2, 20, 45)
    critical_low_confirm_seconds = clamp(critical_low_confirm_seconds, 15, 60)

    lockout_minutes = 55 if context.mode == "energy_saving" else 35 if context.mode == "drying" else 45
    if effective_rebound >= 0.08 and final_confidence in {"medium", "high"}:
        lockout_minutes -= 12
    elif effective_rebound >= 0.05:
        lockout_minutes -= 6
    elif effective_rebound <= 0.02:
        lockout_minutes += 6
    if context.scene == "window_open_suspected":
        lockout_minutes += 10
    lockout_minutes = clamp(lockout_minutes, 20, 90)

    lock_release_delta = 12 if context.mode == "energy_saving" else 8 if context.mode == "drying" else 10
    if effective_rebound >= 0.08 and final_confidence in {"medium", "high"}:
        lock_release_delta -= 3
    elif effective_rebound >= 0.05:
        lock_release_delta -= 2
    elif effective_rebound <= 0.02:
        lock_release_delta += 1
    if context.scene == "window_open_suspected":
        lock_release_delta += 2
    lock_release_delta = clamp(lock_release_delta, 6, 15)

    clothes_duration = 120
    if context.scene in {"drying_clothes", "rainy_high_humidity"}:
        clothes_duration += 20
    if effective_rebound >= 0.08:
        clothes_duration += 20
    if effective_drop > 0 and effective_drop < 0.10:
        clothes_duration += 20
    elif final_confidence == "high" and effective_drop >= 0.18:
        clothes_duration -= 10
    clothes_duration = clamp(clothes_duration, 60, 240)

    shoes_duration = 60
    if context.scene in {"rainy_high_humidity", "drying_clothes"}:
        shoes_duration += 10
    if effective_rebound >= 0.08:
        shoes_duration += 10
    if effective_drop > 0 and effective_drop < 0.10:
        shoes_duration += 10
    elif final_confidence == "high" and effective_drop >= 0.18:
        shoes_duration -= 5
    shoes_duration = clamp(shoes_duration, 30, 120)

    allow_skip_lock = (
        len(rebound_values) >= 5
        and final_confidence in {"medium", "high"}
        and effective_rebound >= 0.07
        and context.scene not in {"window_open_suspected", "night"}
    )
    anomaly_level, anomaly_summary = classify_anomaly(runs, rebounds, context, effective_drop, effective_rebound)
    if anomaly_level == "warning":
        min_runtime_minutes = clamp(min_runtime_minutes + 4, 18, 45)
    elif anomaly_level == "notice" and context.scene == "window_open_suspected":
        min_runtime_minutes = clamp(min_runtime_minutes - 2, 18, 45)

    early_stop_guard_minutes = 18 if context.mode == "energy_saving" else 12 if context.mode == "drying" else 15
    if effective_drop >= 0.25:
        early_stop_guard_minutes -= 3
    elif effective_drop <= 0.08:
        early_stop_guard_minutes += 3
    if effective_rebound >= 0.08:
        early_stop_guard_minutes += 3
    elif effective_rebound <= 0.02:
        early_stop_guard_minutes -= 2
    if context.scene == "night":
        early_stop_guard_minutes += 2
    elif context.scene == "shower_spike":
        early_stop_guard_minutes -= 2
    if final_confidence == "low":
        early_stop_guard_minutes = clamp((early_stop_guard_minutes + 15) / 2, 10, 24)
    early_stop_guard_minutes = clamp(early_stop_guard_minutes, 8, 30)

    predicted_stop_minutes: int | None = None
    predicted_stop_time = "--"
    if context.running and context.humidity > context.target and effective_drop > 0.01:
        humidity_gap = max(context.humidity - context.target, 0)
        predicted_stop_minutes = max(math.ceil(humidity_gap / effective_drop), context.min_runtime_left)
        predicted_stop_time = (context.now + timedelta(minutes=predicted_stop_minutes)).strftime("%H:%M:%S")
    elif context.running and context.humidity <= context.target:
        predicted_stop_minutes = max(context.min_runtime_left, 0)
        predicted_stop_time = (context.now + timedelta(minutes=predicted_stop_minutes)).strftime("%H:%M:%S")

    predicted_start_minutes: int | None = None
    predicted_start_time = "--"
    if not context.running and context.humidity < context.start_threshold and effective_rebound > 0.01:
        humidity_gap = max(context.start_threshold - context.humidity, 0)
        minutes_to_line = math.ceil(humidity_gap / effective_rebound)
        predicted_start_minutes = max(minutes_to_line - auto_window, 0)
        predicted_start_time = (context.now + timedelta(minutes=predicted_start_minutes)).strftime("%H:%M:%S")

    external_start_bias, external_advice = compute_external_advice(context)

    learning_state = "已建立可训练样本集" if (len(runs) + len(rebounds)) >= 6 else "正在积累样本"
    trend = "除湿中" if context.running else ("回潮较快" if effective_rebound >= 0.08 else "缓慢回潮" if effective_rebound >= 0.03 else "环境稳定")
    control_takeover = {
        "auto_start_window": len(rebound_values) >= 3,
        "day_offset": strong_learning,
        "night_offset": strong_learning,
        "start_confirm_minutes": len(rebound_values) >= 3,
        "stop_confirm_minutes": len(drop_values) >= 3,
        "min_runtime_minutes": len(drop_values) >= 3 and len(rebound_values) >= 2,
        "low_protect_confirm_minutes": len(drop_values) >= 3,
        "critical_low_confirm_seconds": len(drop_values) >= 3,
        "lockout_minutes": len(rebound_values) >= 5,
        "early_stop_guard_minutes": len(drop_values) >= 3,
        "lock_release_delta": len(rebound_values) >= 5 and final_confidence in {"medium", "high"},
        "allow_skip_lock": allow_skip_lock,
        "clothes_duration": scene_sample_count >= 3,
        "shoes_duration": scene_sample_count >= 3,
    }
    control_takeover_summary = "、".join(
        label
        for key, label in [
            ("auto_start_window", "提前开机窗口"),
            ("day_offset", "白天阈值补偿"),
            ("night_offset", "夜间阈值补偿"),
            ("start_confirm_minutes", "开机确认"),
            ("stop_confirm_minutes", "关机确认"),
            ("min_runtime_minutes", "最小运行时长"),
            ("low_protect_confirm_minutes", "低湿确认"),
            ("critical_low_confirm_seconds", "极低湿确认"),
            ("lockout_minutes", "低湿锁定"),
            ("early_stop_guard_minutes", "提前关机观察"),
            ("lock_release_delta", "跳锁定差值"),
            ("clothes_duration", "干衣时长"),
            ("shoes_duration", "干鞋时长"),
        ]
        if control_takeover.get(key)
    ) or "当前仍以规则默认值为主"

    readiness_summary = (
        f"下降样本 {len(drop_values)} 条，回潮样本 {len(rebound_values)} 条；"
        f"当前处于{learning_stage_cn}阶段。"
    )

    return {
        "learning_state": learning_state,
        "dataset_runs": len(runs),
        "dataset_rebounds": len(rebounds),
        "dataset_snapshots": len(snapshots),
        "dataset_total": len(runs) + len(rebounds) + len(snapshots),
        "scene_sample_count": scene_sample_count,
        "learning_stage": learning_stage,
        "learning_stage_cn": learning_stage_cn,
        "drop_model_ready": drop_model_ready,
        "rebound_model_ready": rebound_model_ready,
        "readiness_summary": readiness_summary,
        "next_milestone": next_milestone,
        "scene": context.scene,
        "mode": context.mode,
        "machine_state": context.state,
        "current_trend": trend,
        "effective_drop_rate": round(effective_drop, 3),
        "effective_rebound_rate": round(effective_rebound, 3),
        "learned_drop_rate": round(learned_drop, 3),
        "learned_rebound_rate": round(learned_rebound, 3),
        "drop_source_scope": drop_scope,
        "rebound_source_scope": rebound_scope,
        "prediction_confidence": final_confidence,
        "prediction_confidence_cn": {"high": "高", "medium": "中", "low": "低"}[final_confidence],
        "prediction_confidence_reason": confidence_reason,
        "predicted_stop_time": predicted_stop_time,
        "predicted_stop_minutes": predicted_stop_minutes,
        "predicted_next_start_time": predicted_start_time,
        "predicted_next_start_minutes": predicted_start_minutes,
        "auto_start_window": auto_window,
        "day_start_offset": day_offset,
        "night_start_offset": night_offset,
        "start_confirm_minutes": start_confirm,
        "stop_confirm_minutes": stop_confirm,
        "min_runtime_minutes": min_runtime_minutes,
        "low_protect_confirm_minutes": low_protect_confirm,
        "critical_low_confirm_seconds": critical_low_confirm_seconds,
        "lockout_minutes": lockout_minutes,
        "early_stop_guard_minutes": early_stop_guard_minutes,
        "lock_release_delta": lock_release_delta,
        "allow_skip_lock": allow_skip_lock,
        "clothes_duration_suggested": clothes_duration,
        "shoes_duration_suggested": shoes_duration,
        "control_takeover": control_takeover,
        "control_takeover_summary": control_takeover_summary,
        "anomaly_level": anomaly_level,
        "anomaly_summary": anomaly_summary,
        "external_start_bias": external_start_bias,
        "external_advice": external_advice,
        "predictor": predictor,
        "last_sync": context.now.strftime("%Y-%m-%d %H:%M:%S"),
    }


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


_UNAVAILABLE = {"", "unknown", "unavailable", "none", "null", "未知", "-1"}


def _opt_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    v = value.strip().lower()
    if v in _UNAVAILABLE:
        return None
    if v in {"on", "true", "1", "open", "yes", "打开", "开"}:
        return True
    if v in {"off", "false", "0", "closed", "no", "关闭", "关"}:
        return False
    return None


def _opt_str(value: str | None) -> str | None:
    if value is None:
        return None
    v = value.strip()
    return None if v.lower() in _UNAVAILABLE else v


def _opt_float(value: str | None) -> float | None:
    if value is None:
        return None
    v = value.strip()
    if v.lower() in _UNAVAILABLE:
        return None
    try:
        f = float(v)
    except ValueError:
        return None
    return None if f < 0 else f


def command_sync(args: argparse.Namespace) -> int:
    source = Path(args.source)
    runs_path = Path(args.runs)
    rebounds_path = Path(args.rebounds)
    snapshots_path = Path(args.snapshots)
    output_path = Path(args.output)

    runs, rebounds, snapshots = build_structured_samples(source)
    jsonl_dump(runs_path, runs)
    jsonl_dump(rebounds_path, rebounds)
    jsonl_dump(snapshots_path, snapshots)

    context = PredictionContext(
        humidity=float(args.current_humidity),
        target=float(args.target_humidity),
        scene=args.scene,
        mode=args.mode,
        state=args.machine_state,
        running=bool(int(args.running)),
        current_drop_rate=float(args.current_drop_rate),
        current_rebound_rate=float(args.current_rebound_rate),
        start_threshold=float(args.start_threshold),
        min_runtime_left=max(int(args.min_runtime_left), 0),
        now=datetime.now(),
        window_open=_opt_bool(getattr(args, "window_open", "")),
        presence=_opt_str(getattr(args, "presence", "")),
        hvac=_opt_str(getattr(args, "hvac", "")),
        outdoor_humidity=_opt_float(getattr(args, "outdoor_humidity", "")),
        rainy_now=_opt_bool(getattr(args, "rainy", "")),
    )
    result = compute_predictions(runs, rebounds, snapshots, context)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")

    # 反馈闭环:把这次预测落盘,供之后的 backtest 与实际事件对比。
    # 仅在显式提供 --predictions 时记录;不影响任何控制逻辑。
    if getattr(args, "predictions", ""):
        log_prediction(Path(args.predictions), context, result)
    return 0


def log_prediction(path: Path, context: PredictionContext, result: dict[str, Any]) -> None:
    """记录一条预测,带绝对时间戳,便于事后与真实启停事件配对。"""
    def absolute(minutes: Any) -> str | None:
        if minutes is None:
            return None
        return (context.now + timedelta(minutes=int(minutes))).isoformat()

    record = {
        "prediction_time": context.now.isoformat(),
        "running": context.running,
        "scene": context.scene,
        "mode": context.mode,
        "confidence": result.get("prediction_confidence"),
        "predictor": result.get("predictor"),
        "humidity": context.humidity,
        "target": context.target,
        "start_threshold": context.start_threshold,
        "effective_drop_rate": result.get("effective_drop_rate"),
        "effective_rebound_rate": result.get("effective_rebound_rate"),
        "predicted_stop_minutes": result.get("predicted_stop_minutes"),
        "predicted_stop_at": absolute(result.get("predicted_stop_minutes")),
        "predicted_next_start_minutes": result.get("predicted_next_start_minutes"),
        "predicted_next_start_at": absolute(result.get("predicted_next_start_minutes")),
        # 本次每个倒计时/时长参数的取值。用于日后把"等了多久"与实际结果配对,
        # 让这些参数从手写启发式逐步过渡到可学习。键名与 TIMER_KEYS 对应。
        "decided_timers": {key: result.get(key) for key in TIMER_KEYS},
    }
    append_jsonl(path, record, max_lines=PREDICTIONS_MAX_LINES)


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return parse_time(value)


def actual_events_from_runs(runs: list[dict[str, Any]]) -> tuple[list[datetime], list[datetime]]:
    """从运行段还原真实的启停时间。

    每个 run 记录的是停机事件:停机时刻 = datetime/timestamp,
    开机时刻 = 停机时刻 - duration_min。
    """
    stops: list[datetime] = []
    starts: list[datetime] = []
    for run in runs:
        stop_dt = _parse_iso(run.get("datetime")) or parse_time(run.get("timestamp"))
        if stop_dt is None:
            continue
        stops.append(stop_dt)
        duration = run.get("duration_min") or 0
        if duration > 0:
            starts.append(stop_dt - timedelta(minutes=int(duration)))
    stops.sort()
    starts.sort()
    return starts, stops


def prediction_bias(predictions: list[dict[str, Any]], runs: list[dict[str, Any]],
                    horizon_min: int = 360) -> dict[str, Any]:
    """预测误差反馈:对比历史"预测的开/关机时间 vs 实际发生时间",
    返回有符号的平均偏差(分钟,正=实际比预测晚)。用近 30 条预测做窗口。
    """
    starts, stops = actual_events_from_runs(runs)
    stop_err: list[float] = []
    start_err: list[float] = []
    for pred in predictions[-200:]:
        pt = _parse_iso(pred.get("prediction_time"))
        if pt is None:
            continue
        ps = _parse_iso(pred.get("predicted_stop_at"))
        if ps is not None:
            actual = next((s for s in stops if s >= pt), None)
            if actual is not None and (actual - pt).total_seconds() / 60 <= horizon_min:
                stop_err.append((actual - ps).total_seconds() / 60.0)
        pn = _parse_iso(pred.get("predicted_next_start_at"))
        if pn is not None:
            actual = next((s for s in starts if s >= pt), None)
            if actual is not None and (actual - pt).total_seconds() / 60 <= horizon_min:
                start_err.append((actual - pn).total_seconds() / 60.0)
    recent_stop = stop_err[-30:]
    recent_start = start_err[-30:]
    return {
        "stop_bias_min": round(mean(recent_stop), 1) if recent_stop else None,
        "start_bias_min": round(mean(recent_start), 1) if recent_start else None,
        "n_stop": len(recent_stop),
        "n_start": len(recent_start),
    }


def _first_after(events: list[datetime], after: datetime) -> datetime | None:
    for event in events:
        if event >= after:
            return event
    return None


def _summarize(errors: list[float]) -> dict[str, Any]:
    if not errors:
        return {"n": 0, "mae_min": None, "bias_min": None}
    return {
        "n": len(errors),
        "mae_min": round(mean(abs(e) for e in errors), 2),
        "bias_min": round(mean(errors), 2),
    }


def compute_outcomes(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """从运行段评估"等待时长选得好不好"的结果信号:
    - 短循环:停机后很快又开机(low_protect / lockout 等太短)。
    - 过度除湿:停机时湿度掉破 target - OVERDRY_MARGIN(等太长 / min_runtime 过大)。
    纯靠 runs 即可计算,不依赖预测日志。
    """
    items: list[dict[str, Any]] = []
    for run in runs:
        stop = _parse_iso(run.get("datetime")) or parse_time(run.get("timestamp"))
        if stop is None:
            continue
        duration = run.get("duration_min") or 0
        start = stop - timedelta(minutes=int(duration)) if duration else None
        items.append({
            "start": start,
            "stop": stop,
            "end_h": run.get("end_humidity"),
            "target": run.get("target_humidity"),
            "event": run.get("event"),
        })
    items.sort(key=lambda it: it["stop"])

    gaps: list[float] = []
    short_cycles = 0
    for cur, nxt in zip(items, items[1:]):
        if nxt["start"] is None:
            continue
        gap = (nxt["start"] - cur["stop"]).total_seconds() / 60.0
        if gap < 0:
            continue
        gaps.append(gap)
        if gap < SHORT_CYCLE_MIN:
            short_cycles += 1

    over_dry = sum(
        1 for it in items
        if it["end_h"] and it["target"] and it["end_h"] < it["target"] - OVERDRY_MARGIN
    )
    return {
        "runs": len(items),
        "short_cycle_count": short_cycles,
        "short_cycle_rate": round(short_cycles / len(gaps), 3) if gaps else None,
        "median_cycle_gap_min": round(median(gaps), 1) if gaps else None,
        "over_dry_count": over_dry,
        "over_dry_rate": round(over_dry / len(items), 3) if items else None,
    }


def command_backtest(args: argparse.Namespace) -> int:
    predictions = read_jsonl(Path(args.predictions))
    runs = read_jsonl(Path(args.runs))
    starts, stops = actual_events_from_runs(runs)
    horizon = timedelta(minutes=BACKTEST_MATCH_HORIZON_MIN)

    stop_errors: list[float] = []
    start_errors: list[float] = []
    by_conf: dict[str, list[float]] = {}

    for pred in predictions:
        pt = _parse_iso(pred.get("prediction_time"))
        if pt is None:
            continue
        conf = pred.get("confidence") or "unknown"

        predicted_stop = _parse_iso(pred.get("predicted_stop_at"))
        if predicted_stop is not None:
            actual = _first_after(stops, pt)
            if actual is not None and actual - pt <= horizon:
                err = (actual - predicted_stop).total_seconds() / 60.0
                stop_errors.append(err)
                by_conf.setdefault(conf, []).append(err)

        predicted_start = _parse_iso(pred.get("predicted_next_start_at"))
        if predicted_start is not None:
            actual = _first_after(starts, pt)
            if actual is not None and actual - pt <= horizon:
                err = (actual - predicted_start).total_seconds() / 60.0
                start_errors.append(err)
                by_conf.setdefault(conf, []).append(err)

    report = {
        "predictions_logged": len(predictions),
        "actual_starts": len(starts),
        "actual_stops": len(stops),
        "stop_prediction": _summarize(stop_errors),
        "start_prediction": _summarize(start_errors),
        "by_confidence": {conf: _summarize(errs) for conf, errs in sorted(by_conf.items())},
        "outcomes": compute_outcomes(runs),
    }

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    print(f"已记录预测 {report['predictions_logged']} 条 | 真实开机 {len(starts)} 次 / 关机 {len(stops)} 次")
    for label, key in (("停机预测", "stop_prediction"), ("开机预测", "start_prediction")):
        s = report[key]
        if s["n"] == 0:
            print(f"  {label}: 暂无可配对样本")
        else:
            print(f"  {label}: n={s['n']}  MAE={s['mae_min']} 分钟  偏差={s['bias_min']:+} 分钟(正=实际晚于预测)")
    if report["by_confidence"]:
        print("  按置信度:")
        for conf, s in report["by_confidence"].items():
            if s["n"]:
                print(f"    {conf}: n={s['n']}  MAE={s['mae_min']} 分钟  偏差={s['bias_min']:+} 分钟")
    o = report["outcomes"]
    print("  结果质量(倒计时选得好不好):")
    print(f"    短循环: {o['short_cycle_count']} 次"
          + (f"(占周期 {o['short_cycle_rate']:.0%}, 中位间隔 {o['median_cycle_gap_min']} 分钟)" if o["short_cycle_rate"] is not None else ""))
    print(f"    过度除湿: {o['over_dry_count']} 次"
          + (f"(占运行 {o['over_dry_rate']:.0%})" if o["over_dry_rate"] is not None else ""))
    return 0


def _dist(values: list[float]) -> dict[str, Any]:
    vals = [v for v in values if v and v > 0]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": round(mean(vals), 3),
        "median": round(median(vals), 3),
        "min": round(min(vals), 3),
        "max": round(max(vals), 3),
        "cv": round(coefficient_of_variation(vals), 3),
    }


def command_stats(args: argparse.Namespace) -> int:
    runs = read_jsonl(Path(args.runs))
    rebounds = read_jsonl(Path(args.rebounds))
    snapshots = read_jsonl(Path(args.snapshots))

    per_scene: dict[str, int] = {}
    per_mode: dict[str, int] = {}
    for sample in runs + rebounds:
        per_scene[sample.get("scene", "?")] = per_scene.get(sample.get("scene", "?"), 0) + 1
        per_mode[sample.get("mode", "?")] = per_mode.get(sample.get("mode", "?"), 0) + 1

    report = {
        "dataset": {
            "runs": len(runs),
            "rebounds": len(rebounds),
            "snapshots": len(snapshots),
            "total": len(runs) + len(rebounds) + len(snapshots),
        },
        "by_scene": dict(sorted(per_scene.items(), key=lambda kv: -kv[1])),
        "by_mode": dict(sorted(per_mode.items(), key=lambda kv: -kv[1])),
        "drop_rate": _dist([s.get("drop_rate", 0) for s in runs]),
        "rebound_rate": _dist([s.get("rebound_rate", 0) for s in rebounds]),
        "duration_min": _dist([s.get("duration_min", 0) for s in runs]),
        "outcomes": compute_outcomes(runs),
        "ready_for_model": max((per_scene or {None: 0}).values()) >= 50,
    }

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    d = report["dataset"]
    print(f"样本总量: {d['total']} (运行 {d['runs']} / 回潮 {d['rebounds']} / 快照 {d['snapshots']})")
    print(f"  按场景: {report['by_scene']}")
    print(f"  按模式: {report['by_mode']}")
    for label, key in (("下降速率", "drop_rate"), ("回潮速率", "rebound_rate"), ("运行时长", "duration_min")):
        s = report[key]
        if s["n"]:
            print(f"  {label}: n={s['n']} 均值={s['mean']} 中位={s['median']} 范围[{s['min']},{s['max']}] CV={s['cv']}")
        else:
            print(f"  {label}: 暂无样本")
    o = report["outcomes"]
    print(f"  短循环 {o['short_cycle_count']} 次 / 过度除湿 {o['over_dry_count']} 次")
    print(f"  是否够上回归模型(任一场景≥50): {'是' if report['ready_for_model'] else '否,继续积累'}")
    return 0


def command_dump_json(args: argparse.Namespace) -> int:
    payload = read_json(Path(args.file))
    if not payload:
        payload = {
            "learning_state": "尚未生成预测文件",
            "dataset_runs": 0,
            "dataset_rebounds": 0,
            "dataset_snapshots": 0,
            "dataset_total": 0,
            "scene": "normal",
            "mode": "comfort",
            "machine_state": "off",
            "current_trend": "环境稳定",
            "effective_drop_rate": 0,
            "effective_rebound_rate": 0,
            "learned_drop_rate": 0,
            "learned_rebound_rate": 0,
            "drop_source_scope": "全局",
            "rebound_source_scope": "全局",
            "prediction_confidence": "low",
            "prediction_confidence_cn": "低",
            "prediction_confidence_reason": "学习引擎尚未完成初始化",
            "predicted_stop_time": "--",
            "predicted_stop_minutes": None,
            "predicted_next_start_time": "--",
            "predicted_next_start_minutes": None,
            "auto_start_window": 20,
            "scene_sample_count": 0,
            "day_start_offset": 4,
            "night_start_offset": 7,
            "start_confirm_minutes": 5,
            "stop_confirm_minutes": 5,
            "min_runtime_minutes": 30,
            "low_protect_confirm_minutes": 5,
            "critical_low_confirm_seconds": 30,
            "lockout_minutes": 45,
            "early_stop_guard_minutes": 15,
            "lock_release_delta": 10,
            "allow_skip_lock": False,
            "clothes_duration_suggested": 120,
            "shoes_duration_suggested": 60,
            "control_takeover": {},
            "control_takeover_summary": "当前仍以规则默认值为主",
            "anomaly_level": "normal",
            "anomaly_summary": "学习引擎尚未完成初始化",
            "external_start_bias": 0,
            "external_advice": "无外部环境调整(传感器未配置或无触发)",
            "last_sync": "",
        }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smart dehumidifier lightweight ML engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync = subparsers.add_parser("sync")
    sync.add_argument("--source", required=True)
    sync.add_argument("--runs", required=True)
    sync.add_argument("--rebounds", required=True)
    sync.add_argument("--snapshots", required=True)
    sync.add_argument("--output", required=True)
    sync.add_argument("--current-humidity", required=True)
    sync.add_argument("--target-humidity", required=True)
    sync.add_argument("--scene", required=True)
    sync.add_argument("--mode", required=True)
    sync.add_argument("--machine-state", required=True)
    sync.add_argument("--running", required=True)
    sync.add_argument("--current-drop-rate", required=True)
    sync.add_argument("--current-rebound-rate", required=True)
    sync.add_argument("--start-threshold", required=True)
    sync.add_argument("--min-runtime-left", required=True)
    sync.add_argument("--predictions", default="", help="可选:预测落盘路径(jsonl),供 backtest 使用")
    # 外部环境(均可选,空/unknown = 不参与决策)
    sync.add_argument("--window-open", default="")
    sync.add_argument("--presence", default="")
    sync.add_argument("--hvac", default="")
    sync.add_argument("--outdoor-humidity", default="")
    sync.add_argument("--rainy", default="")
    sync.set_defaults(func=command_sync)

    backtest = subparsers.add_parser("backtest")
    backtest.add_argument("--predictions", required=True)
    backtest.add_argument("--runs", required=True)
    backtest.add_argument("--json", action="store_true", help="输出 JSON 报告而非中文摘要")
    backtest.set_defaults(func=command_backtest)

    stats = subparsers.add_parser("stats")
    stats.add_argument("--runs", required=True)
    stats.add_argument("--rebounds", required=True)
    stats.add_argument("--snapshots", required=True)
    stats.add_argument("--json", action="store_true", help="输出 JSON 报告而非中文摘要")
    stats.set_defaults(func=command_stats)

    dump_json = subparsers.add_parser("dump-json")
    dump_json.add_argument("--file", required=True)
    dump_json.set_defaults(func=command_dump_json)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
