#!/usr/bin/env python3
"""智能除湿:回归模型"就绪"监控。

判定"门槛2"是否达成,并附带最新 backtest 摘要。输出一行 JSON:
  {"status": "ready"|"waiting", "summary": "<给手机看的中文一行>"}

- ready  : model.json 已生成(回归模型真训出来了)——这才是"开始变智能"的硬信号。
- waiting: 还没训出来;summary 给出"下降样本还差几条到 40"。

被 Home Assistant 的 command_line 传感器定时调用;automation 在状态由
waiting→ready 跳变时把 summary 推到手机。纯只读,不改任何状态。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

CONFIG = Path("/Users/zhenghaowei/homeassistant-run/config")
ML_TOOL = Path("/Users/zhenghaowei/Documents/homeassistant/tools/smart_dehumidifier_ml.py")
PYTHON = "/Users/zhenghaowei/homeassistant-run/.venv/bin/python"

MODEL = CONFIG / "smart_dehumidifier_model.json"
RUNS = CONFIG / "smart_dehumidifier_runs.jsonl"
REBOUNDS = CONFIG / "smart_dehumidifier_rebounds.jsonl"
PREDICTIONS = CONFIG / "smart_dehumidifier_predictions.jsonl"

DROP_TRAIN_THRESHOLD = 40  # MIN_SAMPLES_TO_TRAIN


def _count_drop_samples() -> int:
    if not RUNS.exists():
        return 0
    n = 0
    for line in RUNS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            if float(json.loads(line).get("drop_rate", 0) or 0) > 0:
                n += 1
        except (ValueError, json.JSONDecodeError):
            continue
    return n


def _backtest_brief() -> str:
    """跑一次 backtest，取关机/开机 MAE，失败则返回空串。"""
    try:
        out = subprocess.run(
            [PYTHON, str(ML_TOOL), "backtest", "--predictions", str(PREDICTIONS),
             "--runs", str(RUNS), "--json"],
            capture_output=True, text=True, timeout=120,
        )
        data = json.loads(out.stdout)
        stop = (data.get("stop_prediction") or {}).get("mae_min")
        start = (data.get("start_prediction") or {}).get("mae_min")
        if stop is None and start is None:
            return ""
        return f"关机MAE {stop} 分 / 开机MAE {start} 分"
    except Exception:
        return ""


def _model_verdict() -> str:
    """model.json 在时，读交叉验证看模型是否赢过经验公式。"""
    try:
        d = json.loads(MODEL.read_text(encoding="utf-8"))
    except Exception:
        return "模型已生成(读取细节失败)"
    parts = []
    for key, label in (("drop_rate", "下降"), ("rebound_rate", "回潮")):
        cv = (d.get(key) or {}).get("cv") or {}
        ml, heu = cv.get("ml_mae"), cv.get("heuristic_mae")
        if ml is not None and heu is not None:
            parts.append(f"{label}模型{'赢' if ml < heu else '没赢'}经验({ml}vs{heu})")
    return "；".join(parts) if parts else "模型已生成"


def main() -> int:
    if MODEL.exists():
        verdict = _model_verdict()
        brief = _backtest_brief()
        summary = f"🎉 回归模型已训练!{verdict}。" + (f" 当前{brief}。" if brief else "")
        print(json.dumps({"status": "ready", "summary": summary}, ensure_ascii=False))
    else:
        drop = _count_drop_samples()
        gap = max(DROP_TRAIN_THRESHOLD - drop, 0)
        summary = f"回归模型还没训出来:下降样本 {drop}/{DROP_TRAIN_THRESHOLD}(还差约 {gap} 次除湿)。"
        print(json.dumps({"status": "waiting", "summary": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
