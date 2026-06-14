#!/usr/bin/env python3
"""智能除湿:模型"全面接管"监控。

读集成实际服役的模型 config/smart_dehumidifier/model_latest.json,按 predict_rate
的同款判据(CV 上 ml_mae <= heuristic_mae 才算"赢、在服役")判断两条速率:

- 下降模型(关机预测):一般已在服役。
- 回潮模型(开机预测):噪声大、最难——它什么时候开始赢过经验,才是"完全模型驱动"的真信号。

输出一行 JSON {"status","summary"}:
- status=ready : 回潮模型也开始赢了(两条都模型驱动)——这才发手机提醒。
- status=waiting: 回潮仍回退经验;summary 给出两条当前的 模型vs经验 对比。

被 HA 的 command_line 传感器定时调用;automation 在 waiting→ready 跳变时推送。只读。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

CONFIG = Path("/Users/zhenghaowei/homeassistant-run/config")
ML_TOOL = Path("/Users/zhenghaowei/Documents/homeassistant/tools/smart_dehumidifier_ml.py")
PYTHON = "/Users/zhenghaowei/homeassistant-run/.venv/bin/python"

# 集成实际读写的模型文件(注意:在 smart_dehumidifier/ 子目录,文件名 model_latest.json)
MODEL = CONFIG / "smart_dehumidifier" / "model_latest.json"
RUNS = CONFIG / "smart_dehumidifier_runs.jsonl"
PREDICTIONS = CONFIG / "smart_dehumidifier_predictions.jsonl"


def _rate_verdict(entry: dict) -> tuple[bool, str]:
    """返回 (是否在服役/赢, 中文一句)。判据与 model.predict_rate 一致。"""
    if not entry or not entry.get("trained"):
        return False, "未训练"
    cv = entry.get("cv") or {}
    ml, heu = cv.get("ml_mae"), cv.get("heuristic_mae")
    if ml is None or heu is None:
        return False, "无CV"
    if ml <= heu:
        pct = round((heu - ml) / heu * 100) if heu else 0
        return True, f"服役中(赢经验 {pct}%)"
    pct = round((ml - heu) / heu * 100) if heu else 0
    return False, f"回退经验(输 {pct}%)"


def _backtest_brief() -> str:
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


def main() -> int:
    if not MODEL.exists():
        print(json.dumps({"status": "waiting", "summary": "模型尚未生成(集成可能还没训练)。"},
                         ensure_ascii=False))
        return 0
    try:
        bundle = json.loads(MODEL.read_text(encoding="utf-8"))
    except Exception:
        print(json.dumps({"status": "waiting", "summary": "模型文件读取失败。"}, ensure_ascii=False))
        return 0

    models = bundle.get("models") or {}
    drop_win, drop_txt = _rate_verdict(models.get("drop_rate"))
    reb_win, reb_txt = _rate_verdict(models.get("rebound_rate"))
    brief = _backtest_brief()
    tail = f" 当前{brief}。" if brief else ""

    if reb_win:
        summary = (f"🎉 回潮模型开始赢经验了!现在关机+开机预测都由本地模型驱动。"
                   f"下降:{drop_txt};回潮:{reb_txt}。{tail}")
        print(json.dumps({"status": "ready", "summary": summary}, ensure_ascii=False))
    else:
        summary = (f"下降模型{drop_txt};回潮模型{reb_txt}(还在回退经验,这是最后的缺口)。{tail}")
        print(json.dumps({"status": "waiting", "summary": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
