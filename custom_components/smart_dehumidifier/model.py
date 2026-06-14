"""本地自学习模型(差异化卖点)。

纯 numpy 岭回归预测 drop_rate / rebound_rate;在本机训练、本机存储(数据不出户),
版本化保存,且只有在交叉验证证明比启发式更准时才被启用。numpy 缺失时整体降级
(返回 None),引擎自动回退到启发式 —— 不影响可用性。
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy 缺失时全程降级
    np = None

from . import ml_core

ALPHA = 1.0


def available() -> bool:
    return np is not None


def _hour_of(sample: dict[str, Any]) -> float:
    dt = ml_core._parse_iso(sample.get("datetime")) or ml_core.parse_time(sample.get("timestamp"))
    return dt.hour + dt.minute / 60.0 if dt else 12.0


def _month_of(sample: dict[str, Any]) -> int:
    dt = ml_core._parse_iso(sample.get("datetime")) or ml_core.parse_time(sample.get("timestamp"))
    return dt.month if dt else 6


def _features(sample: dict[str, Any], spec: dict[str, Any]) -> list[float]:
    hour = _hour_of(sample)
    month = _month_of(sample)
    start_h = float(sample.get("start_humidity", 0) or 0)
    target = float(sample.get("target_humidity", 60) or 60)
    feats = [
        start_h,
        target,
        start_h - target,                       # humidity_gap:离目标多远,直接影响降湿/回潮
        math.sin(2 * math.pi * hour / 24.0),
        math.cos(2 * math.pi * hour / 24.0),
        math.sin(2 * math.pi * month / 12.0),   # 月份周期:季节性(回南天/梅雨 vs 干燥季)
        math.cos(2 * math.pi * month / 12.0),
    ]
    feats += [1.0 if str(sample.get("scene", "")) == sc else 0.0 for sc in spec.get("scenes", [])]
    feats += [1.0 if str(sample.get("mode", "")) == md else 0.0 for md in spec.get("modes", [])]
    feats += [1.0 if str(sample.get("season", "")) == ss else 0.0 for ss in spec.get("seasons", [])]
    feats += [1.0 if str(sample.get("period", "")) == pp else 0.0 for pp in spec.get("periods", [])]
    return feats


def _matrix(samples: list[dict[str, Any]], key: str, spec: dict[str, Any]):
    rows, ys, used = [], [], []
    for s in samples:
        val = s.get(key, 0) or 0
        if val <= 0:
            continue
        rows.append(_features(s, spec))
        ys.append(float(val))
        used.append(s)
    return np.array(rows, dtype=float), np.array(ys, dtype=float), used


def _fit(X, y, alpha: float) -> dict[str, Any]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    A = Xs.T @ Xs + alpha * np.eye(Xs.shape[1])
    ybar = float(y.mean())
    w = np.linalg.solve(A, Xs.T @ (y - ybar))
    return {"mu": mu.tolist(), "sd": sd.tolist(), "w": w.tolist(), "intercept": ybar}


def _predict(model: dict[str, Any], feats: list[float]) -> float:
    mu = np.array(model["mu"]); sd = np.array(model["sd"]); w = np.array(model["w"])
    xs = (np.array(feats) - mu) / sd
    return float(model["intercept"] + xs @ w)


def _cv(samples: list[dict[str, Any]], key: str, spec: dict[str, Any], alpha: float) -> dict[str, Any]:
    X, y, used = _matrix(samples, key, spec)
    n = len(y)
    if n < 4:
        return {"n": n, "ml_mae": None, "heuristic_mae": None}
    k = 5 if n >= 10 else n
    idx = np.arange(n)
    np.random.RandomState(0).shuffle(idx)
    folds = [idx[i::k] for i in range(k)]
    ml_err, base_err = [], []
    for test in folds:
        if len(test) == 0:
            continue
        mask = np.ones(n, dtype=bool); mask[test] = False
        if mask.sum() < 2:
            continue
        m = _fit(X[mask], y[mask], alpha)
        for ti in test:
            ml_err.append(abs(_predict(m, list(X[ti])) - y[ti]))
        train = [used[i] for i in range(n) if mask[i]]
        for ti in test:
            vals, _ = ml_core.filter_samples(train, scene=used[ti].get("scene", ""),
                                             mode=used[ti].get("mode", ""), key=key)
            base_err.append(abs(ml_core.exponential_weighted_average(vals) - y[ti]))
    return {
        "n": n,
        "ml_mae": round(float(np.mean(ml_err)), 5) if ml_err else None,
        "heuristic_mae": round(float(np.mean(base_err)), 5) if base_err else None,
    }


def train(runs: list[dict[str, Any]], rebounds: list[dict[str, Any]], alpha: float = ALPHA) -> dict[str, Any] | None:
    """训练并返回版本化模型 dict;numpy 不可用则返回 None。"""
    if np is None:
        return None
    out: dict[str, Any] = {"version": time.strftime("%Y%m%d-%H%M%S"), "alpha": alpha, "models": {}}
    for name, samples, key in (("drop_rate", runs, "drop_rate"), ("rebound_rate", rebounds, "rebound_rate")):
        spec = {
            "scenes": sorted({str(s.get("scene", "")) for s in samples}),
            "modes": sorted({str(s.get("mode", "")) for s in samples}),
            "seasons": sorted({str(s.get("season", "")) for s in samples if s.get("season")}),
            "periods": sorted({str(s.get("period", "")) for s in samples if s.get("period")}),
        }
        X, y, _ = _matrix(samples, key, spec)
        cv = _cv(samples, key, spec, alpha)
        if len(y) < 4:
            out["models"][name] = {"trained": False, "n": len(y), "cv": cv}
            continue
        out["models"][name] = {"trained": True, "n": len(y), "spec": spec,
                               "model": _fit(X, y, alpha), "cv": cv}
    return out


def predict_rate(bundle: dict[str, Any], name: str, sample: dict[str, Any]) -> float | None:
    """推理:仅当该速率模型已训练 *且* CV 上优于启发式时才给值,否则 None(回退启发式)。"""
    if np is None or not bundle:
        return None
    entry = bundle.get("models", {}).get(name)
    if not entry or not entry.get("trained"):
        return None
    cv = entry.get("cv") or {}
    ml_mae, h_mae = cv.get("ml_mae"), cv.get("heuristic_mae")
    if ml_mae is None or h_mae is None or ml_mae > h_mae:
        return None  # 模型没更准 → 不接管
    pred = _predict(entry["model"], _features(sample, entry["spec"]))
    return max(round(pred, 4), 0.0)


def status(bundle: dict[str, Any] | None) -> dict[str, Any]:
    """供前端展示的模型状态。"""
    if not available():
        return {"available": False, "reason": "numpy 不可用"}
    if not bundle:
        return {"available": True, "trained": False, "version": None}
    info: dict[str, Any] = {"available": True, "version": bundle.get("version"), "models": {}}
    active_any = False
    for name, e in bundle.get("models", {}).items():
        cv = e.get("cv") or {}
        better = (cv.get("ml_mae") is not None and cv.get("heuristic_mae") is not None
                  and cv["ml_mae"] <= cv["heuristic_mae"])
        active = bool(e.get("trained") and better)
        active_any = active_any or active
        info["models"][name] = {"trained": e.get("trained"), "n": e.get("n"),
                                "ml_mae": cv.get("ml_mae"), "heuristic_mae": cv.get("heuristic_mae"),
                                "active": active}
    info["trained"] = any(e.get("trained") for e in bundle.get("models", {}).values())
    info["active"] = active_any
    return info


def save(path: Path, bundle: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
    # 版本化副本(本地保留历史,便于回滚/审计)
    versioned = path.with_name(f"model_{bundle.get('version', 'unknown')}.json")
    versioned.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")


def load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
