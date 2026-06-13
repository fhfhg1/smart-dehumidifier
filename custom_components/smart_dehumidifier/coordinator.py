"""DataUpdateCoordinator —— 进程内引擎 + 事件检测(自动积累训练样本)+ 本地模型。"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from . import appliances, climate_math, ml_core, model, tank_model
from .const import (
    APPLIANCES_LOG,
    AUTO_TRAIN_EVERY,
    CONF_DEHUMIDIFIER,
    CONF_ENABLE_MODEL,
    CONF_EXTRA_APPLIANCES,
    CONF_HUMIDITY_SENSOR,
    CONF_MODE,
    CONF_TANK_CAPACITY,
    CONF_TARGET,
    CONF_TARGET_MODE,
    CONF_TEMP_SENSOR,
    CONF_UPDATE_INTERVAL,
    DATA_DIRNAME,
    DEFAULT_ENABLE_MODEL,
    DEFAULT_MODE,
    DEFAULT_START_OFFSET,
    DEFAULT_TANK_CAPACITY,
    DEFAULT_TARGET,
    DEFAULT_TARGET_MODE,
    DEFAULT_UPDATE_INTERVAL,
    DEFAULT_WATER_RATE_LPM,
    LEARNING_CSV,
    LOW_PROTECT_DELTA,
    MIN_SAMPLES_TO_TRAIN,
    MODEL_LATEST,
    PREDICTIONS_FILE,
    STATE_FILE,
    TARGET_MODE_MOLD,
    UNAVAILABLE_STATES,
    WATER_CALIB_FRACTIONS,
    WATER_SAMPLES_FILE,
)

_LOGGER = logging.getLogger(__name__)
_CSV_HEADER = "timestamp,event,scene,mode_or_state,humidity,target,extra"
_REBOUND_WINDOWS = (30, 60, 90)
_NIGHT_START_HOUR = 23
_NIGHT_START_MINUTE = 30
_NIGHT_END_HOUR = 8


class SmartDehumidifierCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        interval = entry.options.get(CONF_UPDATE_INTERVAL, DEFAULT_UPDATE_INTERVAL)
        super().__init__(
            hass, _LOGGER, name=DATA_DIRNAME, update_interval=timedelta(seconds=interval)
        )
        self.entry = entry
        self._csv_path = Path(hass.config.path(DATA_DIRNAME, LEARNING_CSV))
        self._model_path = Path(hass.config.path(DATA_DIRNAME, MODEL_LATEST))
        self._state_path = Path(hass.config.path(DATA_DIRNAME, STATE_FILE))
        self._predictions_path = Path(hass.config.path(DATA_DIRNAME, PREDICTIONS_FILE))
        self._water_samples_path = Path(hass.config.path(DATA_DIRNAME, WATER_SAMPLES_FILE))
        self._water_run_minutes = 0.0  # 自上次倒水校准以来的累计运行分钟(持久化)
        self._appliances_log_path = Path(hass.config.path(DATA_DIRNAME, APPLIANCES_LOG))
        self._appliance_state: dict[str, dict[str, Any]] = {}  # 额外设备的在途运行状态(持久化)
        # 模型与检测状态在首个 executor 周期里懒加载,避免在事件循环里阻塞读文件
        self._model: dict[str, Any] | None = None
        self._model_loaded = False
        self._state_loaded = False
        # 事件检测状态(跨更新保留;HA 重启后重置,历史样本仍在 CSV)
        self._prev_running: bool | None = None
        self._run_start_time: datetime | None = None
        self._run_start_h: float = 0.0
        self._last_stop_time: datetime | None = None
        self._last_stop_h: float = 0.0
        self._rebound_done: set[int] = set()
        self._update_count = 0
        # 接管控制(由 switch 实体置位;默认关 = 仅建议,装好不会突然抢控制)
        self.control_enabled = False
        self.last_control_action: str | None = None

    async def async_initialize(self) -> None:
        """Preload persisted model/state off the event loop before first refresh."""
        await self.hass.async_add_executor_job(self._initialize_sync)

    def _initialize_sync(self) -> None:
        """Load persisted files in a worker thread."""
        if not self._model_loaded:
            self._model = model.load(self._model_path)
            self._model_loaded = True
        if not self._state_loaded:
            self._load_state()
            self._state_loaded = True

    # ---- 更新主流程 -----------------------------------------------------------
    async def _async_update_data(self) -> dict[str, Any]:
        cfg = self.entry.data
        opts = self.entry.options
        # 湿度:优先用配置的湿度实体;没配或读不到则回退读除湿机自身的 current_humidity 属性
        hum_entity = cfg.get(CONF_HUMIDITY_SENSOR)
        humidity = self._read_float(hum_entity) if hum_entity else None
        if humidity is None:
            humidity = self._read_attr_float(cfg[CONF_DEHUMIDIFIER], "current_humidity")
        if humidity is None:
            raise UpdateFailed("humidity unavailable (no sensor and no current_humidity attribute)")
        target = float(opts.get(CONF_TARGET, cfg.get(CONF_TARGET, DEFAULT_TARGET)))
        running = self.hass.states.is_state(cfg[CONF_DEHUMIDIFIER], "on")
        enable_model = opts.get(CONF_ENABLE_MODEL, DEFAULT_ENABLE_MODEL)
        target_mode = opts.get(CONF_TARGET_MODE, DEFAULT_TARGET_MODE)
        operating_mode = opts.get(CONF_MODE, DEFAULT_MODE)
        temp_entity = opts.get(CONF_TEMP_SENSOR, cfg.get(CONF_TEMP_SENSOR))
        temp_c = self._read_temp_c(temp_entity)
        # 额外设备:在事件循环里读它们当前是否运行,交给 _compute 做循环检测+学习
        extra = opts.get(CONF_EXTRA_APPLIANCES, []) or []
        appliance_running = {eid: self._is_running_state(eid) for eid in extra}

        result = await self.hass.async_add_executor_job(
            self._compute, humidity, target, running, enable_model, target_mode, temp_c,
            operating_mode, appliance_running
        )

        # 接管控制(仅当总开关打开;湿度可用已在上面校验)。动作在事件循环里执行。
        action = result.get("control_action")
        if self.control_enabled and action in ("start", "stop"):
            await self._actuate(cfg[CONF_DEHUMIDIFIER], action)
            self.last_control_action = f"{action}@{result.get('last_sync')}"
        result["control_enabled"] = self.control_enabled
        result["last_control_action"] = self.last_control_action
        return result

    async def _actuate(self, entity_id: str, action: str) -> None:
        service = "turn_on" if action == "start" else "turn_off"
        await self.hass.services.async_call(
            "homeassistant", service, {"entity_id": entity_id}, blocking=False
        )

    def _read_attr_float(self, entity_id: str, attr: str) -> float | None:
        state = self.hass.states.get(entity_id)
        if state is None:
            return None
        try:
            return float(state.attributes.get(attr))
        except (TypeError, ValueError):
            return None

    def _read_temp_c(self, entity_id: str | None) -> float | None:
        if not entity_id:
            return None
        state = self.hass.states.get(entity_id)
        if state is None or state.state in UNAVAILABLE_STATES:
            return None
        try:
            value = float(state.state)
        except (TypeError, ValueError):
            return None
        celsius = climate_math.to_celsius(value, state.attributes.get("unit_of_measurement"))
        # 合理性护栏:超出室内合理范围(如电磁炉 78℃)视为无效 → 防霉模式回退固定目标
        if celsius < -10 or celsius > 45:
            return None
        return celsius

    def _read_float(self, entity_id: str) -> float | None:
        state = self.hass.states.get(entity_id)
        if state is None or state.state in UNAVAILABLE_STATES:
            return None
        try:
            return float(state.state)
        except (TypeError, ValueError):
            return None

    # ---- executor 线程:文件 IO + 纯计算 --------------------------------------
    def _compute(self, humidity: float, target: float, running: bool, enable_model: bool,
                 target_mode: str, temp_c: float | None, operating_mode: str = DEFAULT_MODE,
                 appliance_running: dict[str, bool] | None = None) -> dict[str, Any]:
        now = datetime.now()
        scene = self._scene_for_now(now)

        if not self._model_loaded or not self._state_loaded:
            self._initialize_sync()

        # 露点 / 霉菌风险,以及"防霉模式"下的有效目标(取更严者)
        dew_point = climate_math.dew_point_c(temp_c, humidity) if temp_c is not None else None
        mold_risk = climate_math.mold_risk_level(humidity, temp_c)
        user_target = target
        if target_mode == TARGET_MODE_MOLD:
            target = float(min(user_target, climate_math.mold_safe_target(temp_c)))

        self._detect_events(now, humidity, target, running)
        self._append(now, self._snapshot_line(now, humidity, target, running))

        runs, rebounds, snapshots = ml_core.build_structured_samples(self._csv_path)

        drop_override = rebound_override = None
        if enable_model and self._model:
            sample = {"scene": scene, "mode": operating_mode, "start_humidity": humidity,
                      "target_humidity": target, "datetime": now.isoformat()}
            drop_override = model.predict_rate(self._model, "drop_rate", sample)
            rebound_override = model.predict_rate(self._model, "rebound_rate", sample)

        initial_start_offset = 7 if scene == "night" else DEFAULT_START_OFFSET

        context = ml_core.PredictionContext(
            humidity=humidity, target=target, scene=scene, mode=operating_mode,
            state="running" if running else "off", running=running,
            current_drop_rate=0.0, current_rebound_rate=0.0,
            start_threshold=target + initial_start_offset, min_runtime_left=0, now=now,
        )
        result = ml_core.compute_predictions(
            runs, rebounds, snapshots, context,
            learned_drop_override=drop_override, learned_rebound_override=rebound_override,
        )

        result["dew_point_c"] = dew_point
        result["mold_risk_level"] = mold_risk
        result["effective_target"] = target
        result["current_humidity"] = round(humidity, 1)
        result["target_humidity"] = round(target, 1)
        result["configured_target_humidity"] = round(user_target, 1)
        result["is_running"] = running
        result["control_action"] = self._decide_action(now, humidity, target, running, result)
        result["start_threshold_applied"] = round(self._start_threshold_for_now(now, target, result), 1)
        result["is_night_window"] = scene == "night"

        # ---- 预测误差反馈闭环 ----
        # 1) 用历史预测(原始引擎输出)对比实际启停,得到时间偏差;
        # 2) 把本次"原始预测"落盘(供下次度量);3) 用偏差修正展示/决策用的预测时间与若干倒计时。
        bias = ml_core.prediction_bias(ml_core.read_jsonl(self._predictions_path), runs)
        ml_core.log_prediction(self._predictions_path, context, result)
        result["prediction_bias"] = bias
        self._apply_prediction_feedback(result, bias, now)

        # ---- 水箱预测:运行时累计"做功" → 学到的换算估当前水量 ----
        self._update_water(running, humidity, operating_mode, scene, result)

        # ---- 多设备学习(只观察):额外设备的启停循环 → 学典型运行时长 ----
        result["appliances"] = self._observe_appliances(now, appliance_running or {}, scene)

        self._update_count += 1
        if self._update_count % AUTO_TRAIN_EVERY == 0:
            self._maybe_train(runs, rebounds)
        result["model_status"] = model.status(self._model)
        return result

    # ---- 多设备学习(只观察,不控制)---------------------------------------------
    # 视为"未运行/空闲"的状态(含烹饪类设备的待机/保温/完成等),其余视为运行中
    _APPLIANCE_OFF = {
        "off", "unavailable", "unknown", "none", "",
        "idle", "standby", "shutdown", "shut_down", "waiting", "appointment", "scheduled",
        "paused", "pause", "keepwarm", "keep_warm", "warm",
        "finished", "finish", "done", "complete", "completed", "stopped", "stop", "cancel",
        "待机", "空闲", "关机", "预约", "暂停", "保温", "完成", "已完成", "停止",
    }

    def _is_running_state(self, entity_id: str) -> bool:
        state = self.hass.states.get(entity_id)
        if state is None or state.state is None:
            return False
        return state.state.strip().lower() not in self._APPLIANCE_OFF  # 大小写不敏感

    def _observe_appliances(self, now: datetime, running_map: dict[str, bool],
                            scene: str) -> dict[str, Any]:
        """对每台声明的额外设备做循环检测:on→off 记一条运行循环并学典型时长。"""
        out: dict[str, Any] = {}
        period = "night" if scene == "night" else ("day" if now.hour < 18 else "evening")
        samples_all = ml_core.read_jsonl(self._appliances_log_path)
        for eid, running in running_map.items():
            st = self._appliance_state.setdefault(eid, {"running": running, "start_iso": None})
            if running and not st["running"]:           # 开机
                st["start_iso"] = now.isoformat()
            elif not running and st["running"] and st.get("start_iso"):  # 关机 → 记一次循环
                start = ml_core._parse_iso(st["start_iso"])
                dur = max(int((now - start).total_seconds() / 60), 0) if start else 0
                if dur > 0:
                    ml_core.append_jsonl(self._appliances_log_path, {
                        "entity": eid, "datetime": now.isoformat(),
                        "duration_min": dur, "period": period, "scene": scene,
                    }, max_lines=5000)
                st["start_iso"] = None
            st["running"] = running
            mine = [s for s in samples_all if s.get("entity") == eid]
            typical, layer = appliances.layered_estimate(
                mine, key="duration_min", context={"period": period, "scene": scene},
                layer_keys=("period", "scene"))
            out[eid] = {
                "is_running": running,
                "cycles": len(mine),
                "typical_duration_min": typical,
                "estimate_source": layer,
            }
        self._save_state()
        return out

    # ---- 水箱预测 ---------------------------------------------------------------
    def _tank_capacity(self) -> float:
        return float(self.entry.options.get(
            CONF_TANK_CAPACITY, self.entry.data.get(CONF_TANK_CAPACITY, DEFAULT_TANK_CAPACITY)))

    def _water_rate_lpm(self, mode: str, scene: str, humidity: float) -> tuple[float, str]:
        """每运行 1 分钟积水多少升:优先用校准样本分层学习,无样本用兜底。"""
        samples = ml_core.read_jsonl(self._water_samples_path)
        rate, layer = tank_model.estimate_rate(samples, mode=mode, scene=scene, humidity=humidity)
        if rate is None or rate <= 0:
            return DEFAULT_WATER_RATE_LPM, "兜底估算"
        return rate, layer

    def _update_water(self, running: bool, humidity: float, mode: str, scene: str,
                      result: dict[str, Any]) -> None:
        if running:  # 只在运行时累计做功
            self._water_run_minutes += self.update_interval.total_seconds() / 60.0
            self._save_state()
        rate, layer = self._water_rate_lpm(mode, scene, humidity)
        capacity = self._tank_capacity()
        liters = round(self._water_run_minutes * rate, 2)
        fill = round(min(liters / capacity * 100, 100), 1) if capacity > 0 else 0.0
        result["water_estimated_liters"] = liters
        result["water_tank_capacity"] = capacity
        result["water_fill_percent"] = fill
        result["water_remaining_liters"] = round(max(capacity - liters, 0), 2)
        result["water_rate_lpm"] = round(rate, 4)
        result["water_rate_source"] = layer
        result["water_level_text"] = (
            "水箱接近满,建议尽快倒水" if fill >= 90 else
            "水箱过半" if fill >= 50 else "水箱容量正常")

    def record_water_calibration(self, level_label: str) -> None:
        """倒水反馈:报"倒水前大概水位"→ 学一条"做功→升水"换算,并把累计清零。"""
        frac = WATER_CALIB_FRACTIONS.get(level_label)
        if frac is None:
            return
        liters_before = frac * self._tank_capacity()
        if self._water_run_minutes > 1 and liters_before > 0:
            rate = liters_before / self._water_run_minutes
            now = datetime.now()
            mode = self.entry.options.get(CONF_MODE, DEFAULT_MODE)
            sample = {
                "datetime": now.isoformat(),
                "mode": mode,
                "scene": self._scene_for_now(now),
                "humidity_bucket": tank_model.humidity_bucket(self._read_float(self.entry.data.get(CONF_HUMIDITY_SENSOR) or "") or 60),
                "rate": round(rate, 5),
            }
            ml_core.append_jsonl(self._water_samples_path, sample, max_lines=2000)
        self._water_run_minutes = 0.0  # 倒水后清零
        self._save_state()

    def _apply_prediction_feedback(self, result: dict[str, Any], bias: dict[str, Any],
                                   now: datetime) -> None:
        """用历史预测偏差动态修正本次输出(就地改 result)。
        正偏差=实际比预测晚 → 顺延对应预计时间;并据此轻推确认/最小运行/提前窗口。"""
        sb = bias.get("stop_bias_min")
        stb = bias.get("start_bias_min")
        if sb is not None and result.get("predicted_stop_minutes") is not None:
            m = max(int(round(result["predicted_stop_minutes"] + sb)), 0)
            result["predicted_stop_minutes"] = m
            result["predicted_stop_time"] = (now + timedelta(minutes=m)).strftime("%H:%M:%S")
            # 关机总体偏晚(sb>0):说明降湿比预期慢 → 关机确认略放宽、最小运行略增
            if sb > 5:
                result["stop_confirm_minutes"] = ml_core.clamp(result.get("stop_confirm_minutes", 5) + 1, 2, 8)
                result["min_runtime_minutes"] = ml_core.clamp(result.get("min_runtime_minutes", 30) + 2, 18, 45)
        if stb is not None and result.get("predicted_next_start_minutes") is not None:
            m = max(int(round(result["predicted_next_start_minutes"] + stb)), 0)
            result["predicted_next_start_minutes"] = m
            result["predicted_next_start_time"] = (now + timedelta(minutes=m)).strftime("%H:%M:%S")
            # 开机偏晚(stb>0=实际比预测晚→我们开早了):缩短提前窗口、开机确认略增
            if stb > 5:
                result["auto_start_window"] = ml_core.clamp(result.get("auto_start_window", 20) - 3, 5, 45)
                result["start_confirm_minutes"] = ml_core.clamp(result.get("start_confirm_minutes", 5) + 1, 2, 8)
            elif stb < -5:  # 开机偏早不足(实际比预测早→我们开晚了):加大提前窗口
                result["auto_start_window"] = ml_core.clamp(result.get("auto_start_window", 20) + 3, 5, 45)

    def _decide_action(self, now: datetime, humidity: float, target: float,
                       running: bool, result: dict[str, Any]) -> str | None:
        """带硬安全限位的启停判定。返回 'start'/'stop'/None(建议;是否执行由总开关决定)。"""
        start_threshold = self._start_threshold_for_now(now, target, result)
        min_runtime = result.get("min_runtime_minutes", 30) or 30
        lockout = result.get("lockout_minutes", 45) or 45
        if not running:
            if self._last_stop_time is not None:
                idle_min = (now - self._last_stop_time).total_seconds() / 60
                if idle_min < lockout:  # 锁定期内不重启,防短循环
                    return None
            return "start" if humidity >= start_threshold else None
        # 运行中
        if humidity <= target - LOW_PROTECT_DELTA:
            return "stop"  # 低湿保护:无视最小运行时长立即停
        run_min = (now - self._run_start_time).total_seconds() / 60 if self._run_start_time else min_runtime
        if humidity <= target and run_min >= min_runtime:
            return "stop"
        return None

    @staticmethod
    def _scene_for_now(now: datetime) -> str:
        if now.hour > _NIGHT_START_HOUR or (now.hour == _NIGHT_START_HOUR and now.minute >= _NIGHT_START_MINUTE):
            return "night"
        if now.hour < _NIGHT_END_HOUR:
            return "night"
        return "normal"

    def _start_threshold_for_now(self, now: datetime, target: float, result: dict[str, Any]) -> float:
        bias = result.get("external_start_bias", 0) or 0
        is_night = self._scene_for_now(now) == "night"
        offset_key = "night_start_offset" if is_night else "day_start_offset"
        fallback = 7 if is_night else DEFAULT_START_OFFSET
        offset = result.get(offset_key, fallback) or fallback
        return target + float(offset) + float(bias)

    # ---- 事件检测:把启停/回潮转成训练样本 ------------------------------------
    def _detect_events(self, now: datetime, humidity: float, target: float, running: bool) -> None:
        if self._prev_running is None:
            self._prev_running = running
            return
        if running and not self._prev_running:  # 开机
            self._run_start_time, self._run_start_h = now, humidity
            scene = self._scene_for_now(now)
            self._append(now, f"{self._ts(now)},start_auto,scene={scene},mode=comfort,"
                              f"humidity={round(humidity,1)},target={int(target)},reason=auto")
        elif not running and self._prev_running:  # 关机 → 记录一次运行
            if self._run_start_time:
                dur = max(int((now - self._run_start_time).total_seconds() / 60), 1)
                drop = round(self._run_start_h - humidity, 1)
                rate = round(drop / dur, 3) if dur > 0 else 0
                event = "stop_low_protect" if humidity <= target - LOW_PROTECT_DELTA else "stop_target"
                scene = self._scene_for_now(self._run_start_time)
                self._append(now, f"{self._ts(now)},{event},scene={scene},start_h={round(self._run_start_h,1)},"
                                  f"end_h={round(humidity,1)},duration_min={dur},drop={drop},"
                                  f"drop_rate={max(rate,0)},mode=comfort")
            self._last_stop_time, self._last_stop_h = now, humidity
            self._rebound_done = set()
        elif not running and self._last_stop_time is not None:  # 回潮采样
            elapsed = int((now - self._last_stop_time).total_seconds() / 60)
            for w in _REBOUND_WINDOWS:
                if w not in self._rebound_done and elapsed >= w:
                    self._rebound_done.add(w)
                    rate = round((humidity - self._last_stop_h) / elapsed, 3) if elapsed > 0 else 0
                    scene = self._scene_for_now(self._last_stop_time)
                    self._append(now, f"{self._ts(now)},rebound_{w}m,scene={scene},"
                                      f"start_h={round(self._last_stop_h,1)},humidity={round(humidity,1)},"
                                      f"elapsed_min={elapsed},rebound_rate={max(rate,0)}")
        self._prev_running = running
        self._save_state()

    # ---- 在途运行状态持久化(跨重启不丢正在进行的循环)----------------------
    def _load_state(self) -> None:
        try:
            d = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return
        self._prev_running = d.get("prev_running")
        rs, ls = d.get("run_start_time"), d.get("last_stop_time")
        self._run_start_time = datetime.fromisoformat(rs) if rs else None
        self._last_stop_time = datetime.fromisoformat(ls) if ls else None
        self._run_start_h = d.get("run_start_h", 0.0)
        self._last_stop_h = d.get("last_stop_h", 0.0)
        self._rebound_done = set(d.get("rebound_done", []))
        self._water_run_minutes = d.get("water_run_minutes", 0.0)
        self._appliance_state = d.get("appliance_state", {})

    def _save_state(self) -> None:
        data = {
            "prev_running": self._prev_running,
            "run_start_time": self._run_start_time.isoformat() if self._run_start_time else None,
            "last_stop_time": self._last_stop_time.isoformat() if self._last_stop_time else None,
            "run_start_h": self._run_start_h,
            "last_stop_h": self._last_stop_h,
            "rebound_done": sorted(self._rebound_done),
            "water_run_minutes": round(self._water_run_minutes, 2),
            "appliance_state": self._appliance_state,
        }
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(json.dumps(data), encoding="utf-8")
        except OSError:
            pass

    # ---- 训练(本地)---------------------------------------------------------
    def _maybe_train(self, runs: list, rebounds: list) -> None:
        if max(len(runs), len(rebounds)) < MIN_SAMPLES_TO_TRAIN:
            return
        self.train_now(runs, rebounds)

    def train_now(self, runs: list | None = None, rebounds: list | None = None) -> dict[str, Any]:
        """立即训练并落盘(服务 train_model 调用)。返回模型状态。"""
        if runs is None or rebounds is None:
            runs, rebounds, _ = ml_core.build_structured_samples(self._csv_path)
        bundle = model.train(runs, rebounds)
        if bundle:
            model.save(self._model_path, bundle)
            self._model = bundle
            _LOGGER.info("smart_dehumidifier model trained: %s", model.status(bundle))
        return model.status(self._model)

    # ---- 文件辅助 -------------------------------------------------------------
    @staticmethod
    def _ts(now: datetime) -> str:
        return now.strftime("%Y-%m-%dT%H:%M:%S")

    def _snapshot_line(self, now: datetime, humidity: float, target: float, running: bool) -> str:
        scene = self._scene_for_now(now)
        return (f"{self._ts(now)},snapshot,scene={scene},mode=comfort,"
                f"state={'running' if running else 'off'},running={'on' if running else 'off'},"
                f"humidity={round(humidity,1)},target={int(target)},drop_rate=0,rebound_rate=0,confidence=low")

    def _append(self, now: datetime, line: str) -> None:
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self._csv_path.exists() or self._csv_path.stat().st_size == 0
        with self._csv_path.open("a", encoding="utf-8") as handle:
            if new_file:
                handle.write(_CSV_HEADER + "\n")
            handle.write(line + "\n")
