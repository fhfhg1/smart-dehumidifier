#!/usr/bin/env python3
"""智能除湿机物理仿真器 —— 在没有足够真实数据前,生成逼真的学习日志,
用来验证整条 ML 管线(sync / backtest / stats),甚至预训练/对照实验。

物理模型(逐分钟):
- 开机时湿度按 drop_rate(%/min, 随场景/模式 + 噪声)下降;
- 关机时湿度按 rebound_rate(%/min, 随场景 + 噪声)回潮;
- 控制采用与线上类似的阈值规则:高于启动阈值确认后开机;达到目标且满足最小
  运行时长后停机(stop_target);掉破 target - low_protect 则低湿保护停机。
输出 CSV 格式与 smart_dehumidifier_learning.csv 完全一致,可直接喂给
`smart_dehumidifier_ml.py sync`。

用法:
  simulate_dehumidifier.py --out sim.csv --days 14
  simulate_dehumidifier.py --out sim.csv --days 14 --replay-predictions pred.jsonl
"""
from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

# 每个场景的物理参数:开机下降速率、关机回潮速率(%/min)的基准值。
SCENES = {
    "normal": {"drop": 0.40, "rebound": 0.05, "weight": 5},
    "drying_clothes": {"drop": 0.30, "rebound": 0.12, "weight": 2},
    "rainy_high_humidity": {"drop": 0.28, "rebound": 0.10, "weight": 2},
    "shower_spike": {"drop": 0.45, "rebound": 0.18, "weight": 1},
    "night": {"drop": 0.38, "rebound": 0.04, "weight": 3},
}
MODES = {
    "comfort": {"target": 60, "start_offset": 4, "min_runtime": 28},
    "energy_saving": {"target": 60, "start_offset": 6, "min_runtime": 32},
    "drying": {"target": 55, "start_offset": 3, "min_runtime": 36},
}
LOW_PROTECT_DELTA = 5      # 掉到 target - 该值触发低湿保护停机
SNAPSHOT_EVERY_MIN = 15
HEADER = "timestamp,event,scene,mode_or_state,humidity,target,extra"


def pick_scene(rng: random.Random) -> str:
    names = list(SCENES)
    weights = [SCENES[n]["weight"] for n in names]
    return rng.choices(names, weights=weights)[0]


def fmt(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%S")


def simulate(days: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    start = datetime.now().replace(second=0, microsecond=0) - timedelta(days=days)
    lines: list[str] = [HEADER]

    humidity = 62.0
    running = False
    run_start_time: datetime | None = None
    run_start_h = 0.0
    above_streak = 0
    last_stop_time: datetime | None = None
    last_stop_h = 0.0
    rebound_marks = {30: False, 60: False, 90: False}

    minutes = days * 24 * 60
    for i in range(minutes):
        now = start + timedelta(minutes=i)
        # 每天换一次场景/模式,夜间强制 night
        scene = "night" if (now.hour >= 23 or now.hour < 7) else pick_scene(rng)
        mode = rng.choices(list(MODES), weights=[5, 3, 2])[0]
        m = MODES[mode]
        sc = SCENES[scene]
        target = m["target"]
        start_threshold = target + m["start_offset"]

        if running:
            drop = max(sc["drop"] * rng.uniform(0.7, 1.3), 0.02)
            humidity -= drop
            dur = int((now - run_start_time).total_seconds() / 60)
            reached = humidity <= target and dur >= m["min_runtime"]
            low_protect = humidity <= target - LOW_PROTECT_DELTA
            if reached or low_protect:
                event = "stop_low_protect" if low_protect and not reached else "stop_target"
                drop_amt = round(run_start_h - humidity, 1)
                drop_rate = round(drop_amt / dur, 3) if dur > 0 else 0
                period = "night" if scene == "night" else ("day" if now.hour < 18 else "evening")
                lines.append(",".join([
                    fmt(now), event, f"scene={scene}", "season=winter", "weather=normal",
                    f"period={period}", f"start_h={round(run_start_h, 1)}",
                    f"end_h={round(humidity, 1)}", f"duration_min={dur}",
                    f"drop={drop_amt}", f"drop_rate={drop_rate}", "alpha=0.6", f"mode={mode}",
                ]))
                running = False
                last_stop_time = now
                last_stop_h = humidity
                rebound_marks = {30: False, 60: False, 90: False}
        else:
            humidity += max(sc["rebound"] * rng.uniform(0.6, 1.4), 0.0)
            # 回潮采样
            if last_stop_time is not None:
                elapsed = int((now - last_stop_time).total_seconds() / 60)
                for window in (30, 60, 90):
                    if not rebound_marks[window] and elapsed >= window:
                        rebound_marks[window] = True
                        rate = round((humidity - last_stop_h) / elapsed, 3) if elapsed > 0 else 0
                        lines.append(",".join([
                            fmt(now), f"rebound_{window}m", f"scene={scene}",
                            f"start_h={round(last_stop_h, 1)}",
                            f"humidity={round(humidity, 1)}", f"elapsed_min={elapsed}",
                            f"rebound_rate={max(rate, 0)}",
                        ]))
            # 开机判定
            if humidity > start_threshold:
                above_streak += 1
            else:
                above_streak = 0
            if above_streak >= 5:
                lines.append(",".join([
                    fmt(now), "start_auto", f"scene={scene}", f"mode={mode}",
                    f"humidity={round(humidity, 1)}", f"target={target}", "reason=sim",
                ]))
                running = True
                run_start_time = now
                run_start_h = humidity
                above_streak = 0

        if i % SNAPSHOT_EVERY_MIN == 0:
            state = "running" if running else "off"
            lines.append(",".join([
                fmt(now), "snapshot", f"scene={scene}", f"mode={mode}", f"state={state}",
                f"running={'on' if running else 'off'}", f"humidity={round(humidity, 1)}",
                f"target={target}", "drop_rate=0", "rebound_rate=0", "confidence=low",
            ]))
        humidity = min(max(humidity, 40.0), 85.0)
    return lines


def replay_predictions(csv_path: Path, out_path: Path) -> int:
    """在生成的 CSV 上重放:逐快照点调用引擎的 compute_predictions 并落盘,
    使 backtest 能立刻端到端验证。复用 smart_dehumidifier_ml 的引擎逻辑。"""
    import smart_dehumidifier_ml as engine

    runs, rebounds, snapshots = engine.build_structured_samples(csv_path)
    count = 0
    for snap in snapshots:
        now = engine._parse_iso(snap.get("datetime")) or engine.parse_time(snap.get("timestamp"))
        if now is None:
            continue
        runs_so_far = [r for r in runs if r["timestamp"] <= snap["timestamp"]]
        reb_so_far = [r for r in rebounds if r["timestamp"] <= snap["timestamp"]]
        snap_so_far = [s for s in snapshots if s["timestamp"] <= snap["timestamp"]]
        target = snap.get("target_humidity") or 60
        context = engine.PredictionContext(
            humidity=snap.get("humidity", 0.0),
            target=target,
            scene=snap.get("scene", "normal"),
            mode=snap.get("mode", "comfort"),
            state=snap.get("state", ""),
            running=snap.get("running") == "on",
            current_drop_rate=0.0,
            current_rebound_rate=0.0,
            start_threshold=target + 5,
            min_runtime_left=0,
            now=now,
        )
        result = engine.compute_predictions(runs_so_far, reb_so_far, snap_so_far, context)
        engine.log_prediction(out_path, context, result)
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Dehumidifier physics simulator")
    parser.add_argument("--out", required=True, help="输出 CSV 路径(学习日志格式)")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replay-predictions", default="",
                        help="可选:在生成的 CSV 上重放并产出 predictions.jsonl,供 backtest")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = simulate(args.days, args.seed)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已生成 {len(lines) - 1} 条事件 -> {out}")

    if args.replay_predictions:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        pred = Path(args.replay_predictions)
        if pred.exists():
            pred.unlink()
        n = replay_predictions(out, pred)
        print(f"已重放 {n} 条预测 -> {pred}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
