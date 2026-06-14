#!/usr/bin/env python3
"""锁住 smart_dehumidifier_ml 引擎当前行为的单元测试。
运行: python -m unittest tests.test_smart_dehumidifier_ml
(或在 tests/ 上层目录: <venv>/bin/python -m unittest discover tests)
这些测试不依赖真实数据,供后续重构/换模型时做回归保护。
"""
from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import smart_dehumidifier_ml as ml  # noqa: E402


class TestParsing(unittest.TestCase):
    def test_parse_line_fields_kv(self):
        ts, ev, data = ml.parse_line_fields(
            ["2026-06-12T16:42:00", "stop_target", "scene=normal", "duration_min=30"]
        )
        self.assertEqual(ts, "2026-06-12T16:42:00")
        self.assertEqual(ev, "stop_target")
        self.assertEqual(data["scene"], "normal")
        self.assertEqual(data["duration_min"], "30")

    def test_as_float_and_int_fallback(self):
        self.assertEqual(ml.as_float({"x": "1.5"}, "x"), 1.5)
        self.assertEqual(ml.as_float({}, "x", 9.0), 9.0)
        self.assertEqual(ml.as_int({"x": "3.9"}, "x"), 3)
        self.assertEqual(ml.as_int({"x": "bad"}, "x", 7), 7)


class TestMath(unittest.TestCase):
    def test_ewma_weights_recent_more(self):
        # 后面的值权重更大
        self.assertGreater(ml.exponential_weighted_average([0.0, 1.0]), 0.5)
        self.assertEqual(ml.exponential_weighted_average([]), 0.0)

    def test_blend_rate_thresholds(self):
        # 样本越少越偏向当前实测(权重越大)
        few = ml.blend_rate(0.3, 0.1, sample_count=1)
        many = ml.blend_rate(0.3, 0.1, sample_count=20)
        self.assertGreater(few, many)
        # 只有一边有值时直接返回该值
        self.assertEqual(ml.blend_rate(0.3, 0.0, 5), 0.3)
        self.assertEqual(ml.blend_rate(0.0, 0.1, 5), 0.1)

    def test_clamp(self):
        self.assertEqual(ml.clamp(100, 2, 8), 8)
        self.assertEqual(ml.clamp(-5, 2, 8), 2)
        self.assertEqual(ml.clamp(5.6, 2, 8), 6)


class TestFilterAndConfidence(unittest.TestCase):
    def _samples(self, n, scene="normal", mode="comfort", rate=0.2):
        return [{"scene": scene, "mode": mode, "drop_rate": rate} for _ in range(n)]

    def test_filter_falls_back_through_scopes(self):
        vals, scope = ml.filter_samples(self._samples(5), scene="normal", mode="comfort", key="drop_rate")
        self.assertEqual(scope, "场景+模式")
        self.assertEqual(len(vals), 5)
        # 不够阈值时回退到全局
        vals, scope = ml.filter_samples(
            self._samples(2, scene="x", mode="y"), scene="normal", mode="comfort", key="drop_rate"
        )
        self.assertEqual(scope, "全局")

    def test_confidence_levels(self):
        self.assertEqual(ml.classify_confidence(0, 0.0, "全局")[0], "low")
        self.assertEqual(ml.classify_confidence(10, 0.2, "场景")[0], "high")
        self.assertEqual(ml.classify_confidence(5, 0.5, "模式")[0], "medium")


class TestOutcomesAndEvents(unittest.TestCase):
    def _run(self, stop_iso, duration, end_h, target=60):
        return {
            "timestamp": stop_iso, "datetime": stop_iso, "duration_min": duration,
            "end_humidity": end_h, "target_humidity": target, "event": "stop_target",
        }

    def test_actual_events_from_runs(self):
        runs = [self._run("2026-06-12T16:42:00", 30, 54)]
        starts, stops = ml.actual_events_from_runs(runs)
        self.assertEqual(stops[0], datetime(2026, 6, 12, 16, 42))
        self.assertEqual(starts[0], datetime(2026, 6, 12, 16, 12))  # stop - 30min

    def test_short_cycle_and_over_dry(self):
        runs = [
            self._run("2026-06-12T16:00:00", 30, 54),   # end_h 54 < 60-5 -> over_dry
            self._run("2026-06-12T16:10:00", 5, 59),    # start=16:05, gap from 16:00 = 5min -> short cycle
        ]
        o = ml.compute_outcomes(runs)
        self.assertEqual(o["runs"], 2)
        self.assertEqual(o["short_cycle_count"], 1)
        self.assertEqual(o["over_dry_count"], 1)


class TestComputePredictions(unittest.TestCase):
    def test_returns_safe_defaults_with_no_data(self):
        ctx = ml.PredictionContext(
            humidity=68, target=60, scene="normal", mode="comfort", state="on",
            running=True, current_drop_rate=0.0, current_rebound_rate=0.0,
            start_threshold=67, min_runtime_left=10, now=datetime(2026, 6, 12, 19, 0),
        )
        result = ml.compute_predictions([], [], [], ctx)
        self.assertEqual(result["prediction_confidence"], "low")
        # 所有倒计时键都存在且为整数,可被 HA 安全消费
        for key in ml.TIMER_KEYS:
            self.assertIn(key, result)
            self.assertIsInstance(result[key], int)


if __name__ == "__main__":
    unittest.main()
