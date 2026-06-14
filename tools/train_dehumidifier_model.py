#!/usr/bin/env python3
"""除湿机速率回归模型(真 ML 原型)。

用 runs/rebounds 样本,以特征(场景/模式/起始湿度/目标/时段)回归预测
drop_rate 与 rebound_rate。纯 numpy 实现岭回归(项目无 sklearn),交叉验证
对照现有启发式(EWMA + 场景分桶),证明 ML 是否更准。

子命令:
  compare --runs r.jsonl --rebounds rb.jsonl [--kfold 5]
      交叉验证对比 "岭回归" vs "启发式 EWMA" 预测速率的 MAE。
  train   --runs r.jsonl --rebounds rb.jsonl --out model.json [--alpha 1.0]
      在全量数据上拟合并保存模型(系数+特征规格+CV 指标),供推理使用。

模型文件可被推理端加载(见 predict_rate),无需 numpy 之外的依赖。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smart_dehumidifier_ml as ml  # 复用样本读取与启发式基线


def _hour_of(sample: dict[str, Any]) -> float:
    dt = ml._parse_iso(sample.get("datetime")) or ml.parse_time(sample.get("timestamp"))
    return dt.hour + dt.minute / 60.0 if dt else 12.0


def _categories(samples: list[dict[str, Any]], key: str) -> list[str]:
    return sorted({str(s.get(key, "")) for s in samples})


def build_matrix(samples: list[dict[str, Any]], target_key: str, spec: dict[str, Any]):
    """把样本编码成 (X, y, used_samples)。spec 含 scenes/modes 类别列表。"""
    scenes, modes = spec["scenes"], spec["modes"]
    rows, ys, used = [], [], []
    for s in samples:
        val = s.get(target_key, 0) or 0
        if val <= 0:
            continue
        hour = _hour_of(s)
        feats = [
            float(s.get("start_humidity", 0) or 0),
            float(s.get("target_humidity", 60) or 60),
            math.sin(2 * math.pi * hour / 24.0),
            math.cos(2 * math.pi * hour / 24.0),
        ]
        feats += [1.0 if str(s.get("scene", "")) == sc else 0.0 for sc in scenes]
        feats += [1.0 if str(s.get("mode", "")) == md else 0.0 for md in modes]
        rows.append(feats)
        ys.append(float(val))
        used.append(s)
    return np.array(rows, dtype=float), np.array(ys, dtype=float), used


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, Any]:
    """标准化特征 + 不惩罚截距的岭回归。返回可序列化的模型参数。"""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    n_features = Xs.shape[1]
    A = Xs.T @ Xs + alpha * np.eye(n_features)
    ybar = float(y.mean())
    w = np.linalg.solve(A, Xs.T @ (y - ybar))
    return {"mu": mu.tolist(), "sd": sd.tolist(), "w": w.tolist(), "intercept": ybar}


def ridge_predict(model: dict[str, Any], X: np.ndarray) -> np.ndarray:
    mu = np.array(model["mu"]); sd = np.array(model["sd"]); w = np.array(model["w"])
    Xs = (X - mu) / sd
    return model["intercept"] + Xs @ w


def heuristic_baseline(train: list[dict[str, Any]], test_sample: dict[str, Any], key: str) -> float:
    """复刻现有引擎:按 场景+模式→场景→模式→全局 取样本,做 EWMA。"""
    values, _ = ml.filter_samples(
        train, scene=test_sample.get("scene", ""), mode=test_sample.get("mode", ""), key=key
    )
    return ml.exponential_weighted_average(values)


def kfold_indices(n: int, k: int, seed: int = 0):
    idx = np.arange(n)
    np.random.RandomState(seed).shuffle(idx)
    if n < 2 * k:  # 样本太少 → 留一法
        k = n
    return [idx[i::k] for i in range(k)]


def cross_validate(samples: list[dict[str, Any]], target_key: str, alpha: float, kfold: int) -> dict[str, Any]:
    spec = {"scenes": _categories(samples, "scene"), "modes": _categories(samples, "mode")}
    X, y, used = build_matrix(samples, target_key, spec)
    n = len(y)
    if n < 4:
        return {"n": n, "note": "样本不足,无法交叉验证(需≥4)"}
    folds = kfold_indices(n, kfold)
    ml_err, base_err = [], []
    for test_idx in folds:
        if len(test_idx) == 0:
            continue
        mask = np.ones(n, dtype=bool); mask[test_idx] = False
        if mask.sum() < 2:
            continue
        model = ridge_fit(X[mask], y[mask], alpha)
        preds = ridge_predict(model, X[test_idx])
        ml_err += list(np.abs(preds - y[test_idx]))
        train_samples = [used[i] for i in range(n) if mask[i]]
        for ti in test_idx:
            base = heuristic_baseline(train_samples, used[ti], target_key)
            base_err.append(abs(base - y[ti]))
    return {
        "n": n,
        "ml_mae": round(float(np.mean(ml_err)), 4) if ml_err else None,
        "heuristic_mae": round(float(np.mean(base_err)), 4) if base_err else None,
    }


def command_compare(args: argparse.Namespace) -> int:
    runs = ml.read_jsonl(Path(args.runs))
    rebounds = ml.read_jsonl(Path(args.rebounds))
    print("交叉验证对比(MAE 越小越好):")
    for label, samples, key in (("下降速率 drop_rate", runs, "drop_rate"),
                                ("回潮速率 rebound_rate", rebounds, "rebound_rate")):
        r = cross_validate(samples, key, args.alpha, args.kfold)
        if r.get("note"):
            print(f"  {label}: {r['note']}(n={r['n']})")
            continue
        ml_mae, h_mae = r["ml_mae"], r["heuristic_mae"]
        verdict = "—"
        if ml_mae is not None and h_mae is not None:
            if h_mae == 0:
                verdict = "持平"
            else:
                imp = (h_mae - ml_mae) / h_mae * 100
                verdict = f"ML {'更优' if imp > 0 else '更差'} {abs(imp):.0f}%"
        print(f"  {label}: n={r['n']}  岭回归 MAE={ml_mae}  启发式 MAE={h_mae}  → {verdict}")
    return 0


def command_train(args: argparse.Namespace) -> int:
    runs = ml.read_jsonl(Path(args.runs))
    rebounds = ml.read_jsonl(Path(args.rebounds))
    out: dict[str, Any] = {"alpha": args.alpha, "models": {}}
    for name, samples, key in (("drop_rate", runs, "drop_rate"),
                               ("rebound_rate", rebounds, "rebound_rate")):
        spec = {"scenes": _categories(samples, "scene"), "modes": _categories(samples, "mode")}
        X, y, _ = build_matrix(samples, key, spec)
        cv = cross_validate(samples, key, args.alpha, args.kfold)
        if len(y) < 4:
            out["models"][name] = {"trained": False, "n": len(y), "cv": cv}
            continue
        model = ridge_fit(X, y, args.alpha)
        out["models"][name] = {"trained": True, "n": len(y), "spec": spec, "model": model, "cv": cv}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"模型已保存 -> {args.out}")
    for name, m in out["models"].items():
        print(f"  {name}: trained={m['trained']} n={m['n']} cv={m.get('cv')}")
    return 0


def predict_rate(out: dict[str, Any], name: str, sample: dict[str, Any]) -> float | None:
    """推理端可调用:用已训练模型预测某样本的速率。模型未训练则返回 None。"""
    entry = out.get("models", {}).get(name)
    if not entry or not entry.get("trained"):
        return None
    # 手动构造单样本特征(build_matrix 会按 rate>0 过滤,这里不能用它)
    spec = entry["spec"]
    hour = _hour_of(sample)
    feats = [
        float(sample.get("start_humidity", 0) or 0),
        float(sample.get("target_humidity", 60) or 60),
        math.sin(2 * math.pi * hour / 24.0),
        math.cos(2 * math.pi * hour / 24.0),
    ]
    feats += [1.0 if str(sample.get("scene", "")) == sc else 0.0 for sc in spec["scenes"]]
    feats += [1.0 if str(sample.get("mode", "")) == md else 0.0 for md in spec["modes"]]
    pred = ridge_predict(entry["model"], np.array([feats], dtype=float))[0]
    return max(round(float(pred), 4), 0.0)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dehumidifier rate regression (ridge) prototype")
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--runs", required=True)
    c.add_argument("--rebounds", required=True)
    c.add_argument("--alpha", type=float, default=1.0)
    c.add_argument("--kfold", type=int, default=5)
    c.set_defaults(func=command_compare)
    t = sub.add_parser("train")
    t.add_argument("--runs", required=True)
    t.add_argument("--rebounds", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--alpha", type=float, default=1.0)
    t.add_argument("--kfold", type=int, default=5)
    t.set_defaults(func=command_train)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
