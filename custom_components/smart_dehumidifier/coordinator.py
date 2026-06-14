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
        self._run_start_target: float = 0.0
        self._run_start_mode: str = DEFAULT_MODE
        self._run_start_scene: str = "normal"
        self._last_stop_time: datetime | None = None
        self._last_stop_h: float = 0.0
        self._last_stop_target: float = 0.0
        self._last_stop_mode: str = DEFAULT_MODE
        self._last_stop_scene: str = "normal"
        self._rebound_done: set[int] = set()
        self._update_count = 0
        self._humidity_trace: list[dict[str, Any]] = []
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
        # 室外湿度/温度:插件自带、零配置 —— 直接在事件循环里读 weather 实体,无需用户
        # 在 configuration.yaml 里贴模板传感器。室外高湿时引擎会提前开机抵御进湿;读不到则
        # 为 None,不参与决策。温度暂作记录/未来特征,不进当前决策。
        outdoor_humidity, outdoor_temperature = self._read_outdoor()

        result = await self.hass.async_add_executor_job(
            self._compute, humidity, target, running, enable_model, target_mode, temp_c,
            operating_mode, appliance_running, outdoor_humidity, outdoor_temperature
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

    def _read_outdoor(self) -> tuple[float | None, float | None]:
        """室外湿度/温度,插件自带、零配置。

        湿度:优先读约定覆盖实体 sensor.smart_dehumidifier_outdoor_humidity(老用户/手动指
        定时仍生效),否则自动发现任一带 humidity 属性的 weather 实体。温度:直接读该 weather
        实体的 temperature 属性(室外温度本就没有独立传感器,天气是最现成的来源)。
        """
        humidity = self._read_float("sensor.smart_dehumidifier_outdoor_humidity")
        temperature: float | None = None
        for state in self.hass.states.async_all("weather"):
            attrs = state.attributes
            if humidity is None and isinstance(attrs.get("humidity"), (int, float)):
                humidity = float(attrs["humidity"])
            if temperature is None and isinstance(attrs.get("temperature"), (int, float)):
                temperature = float(attrs["temperature"])
            if humidity is not None and temperature is not None:
                break
        return humidity, temperature

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
                 appliance_running: dict[str, bool] | None = None,
                 outdoor_humidity: float | None = None,
                 outdoor_temperature: float | None = None) -> dict[str, Any]:
        now = datetime.now()

        if not self._model_loaded or not self._state_loaded:
            self._initialize_sync()

        # 露点 / 霉菌风险,以及"防霉模式"下的有效目标(取更严者)
        dew_point = climate_math.dew_point_c(temp_c, humidity) if temp_c is not None else None
        mold_risk = climate_math.mold_risk_level(humidity, temp_c)
        user_target = target
        if target_mode == TARGET_MODE_MOLD:
            target = float(min(user_target, climate_math.mold_safe_target(temp_c)))

        self._record_humidity_trace(now, humidity, target, running, operating_mode)
        scene = self._classify_scene(now, humidity, target, running, operating_mode)
        current_drop_rate = self._current_drop_rate(now, humidity)
        current_rebound_rate = self._current_rebound_rate(now, humidity)
        rebound_age_minutes = None
        if not running and self._last_stop_time is not None:
            rebound_age_minutes = round(max((now - self._last_stop_time).total_seconds() / 60.0, 0.0), 2)

        self._detect_events(now, humidity, target, running, operating_mode, scene,
                            outdoor_humidity, outdoor_temperature)
        runs, rebounds, snapshots = ml_core.build_structured_samples(self._csv_path)

        drop_override = rebound_override = None
        if enable_model and self._model:
            sample = {"scene": scene, "mode": operating_mode, "start_humidity": humidity,
                      "target_humidity": target, "datetime": now.isoformat(),
                      "season": self._season_for_now(now), "period": self._period_for_now(now),
                      "outdoor_humidity": outdoor_humidity if outdoor_humidity is not None else -999,
                      "outdoor_temp": outdoor_temperature if outdoor_temperature is not None else -999}
            drop_override = model.predict_rate(self._model, "drop_rate", sample)
            rebound_override = model.predict_rate(self._model, "rebound_rate", sample)

        initial_start_offset = 7 if scene == "night" else DEFAULT_START_OFFSET

        context = ml_core.PredictionContext(
            humidity=humidity, target=target, scene=scene, mode=operating_mode,
            state="running" if running else "off", running=running,
            current_drop_rate=current_drop_rate, current_rebound_rate=current_rebound_rate,
            start_threshold=target + initial_start_offset, min_runtime_left=0, now=now,
            rebound_age_minutes=rebound_age_minutes,
            outdoor_humidity=outdoor_humidity,
            outdoor_temperature=outdoor_temperature,
        )
        result = ml_core.compute_predictions(
            runs, rebounds, snapshots, context,
            learned_drop_override=drop_override, learned_rebound_override=rebound_override,
        )

        result["dew_point_c"] = dew_point
        result["mold_risk_level"] = mold_risk
        result["outdoor_humidity"] = round(outdoor_humidity, 1) if outdoor_humidity is not None else None
        result["outdoor_temperature"] = round(outdoor_temperature, 1) if outdoor_temperature is not None else None
        result["effective_target"] = target
        result["current_humidity"] = round(humidity, 1)
        result["target_humidity"] = round(target, 1)
        result["configured_target_humidity"] = round(user_target, 1)
        result["is_running"] = running
        result["current_drop_rate_live"] = round(current_drop_rate, 3)
        result["current_rebound_rate_live"] = round(current_rebound_rate, 3)
        result["rebound_age_minutes"] = rebound_age_minutes
        result["is_night_window"] = scene == "night"
        self._append(
            now,
            self._snapshot_line(
                now,
                humidity,
                target,
                running,
                operating_mode,
                scene,
                current_drop_rate,
                current_rebound_rate,
                result.get("prediction_confidence", "low"),
            ),
        )

        # ---- 预测误差反馈闭环 ----
        # 1) 用历史预测(原始引擎输出)对比实际启停,得到时间偏差;
        # 2) 用偏差修正展示/决策用的预测时间与若干倒计时; 3) 把本次"有效预测"落盘供下次度量。
        bias = ml_core.prediction_bias(ml_core.read_jsonl(self._predictions_path), runs)
        result["prediction_bias"] = bias
        result["recent_stop_bias_min"] = bias.get("stop_bias_min")
        result["recent_start_bias_min"] = bias.get("start_bias_min")
        result["recent_stop_bias_samples"] = bias.get("n_stop", 0)
        result["recent_start_bias_samples"] = bias.get("n_start", 0)
        self._apply_prediction_feedback(result, bias, now)

        # ---- 水箱预测:运行时累计"做功" → 学到的换算估当前水量 ----
        self._update_water(running, humidity, operating_mode, scene, result)

        # ---- 多设备学习(只观察):额外设备的启停循环 → 学典型运行时长 ----
        result["appliances"] = self._observe_appliances(now, appliance_running or {}, scene)

        outcomes = ml_core.compute_outcomes(runs)
        result["short_cycle_rate"] = outcomes.get("short_cycle_rate")
        result["over_dry_rate"] = outcomes.get("over_dry_rate")

        self._update_count += 1
        if self._update_count % AUTO_TRAIN_EVERY == 0:
            self._maybe_train(runs, rebounds)
        model_status = model.status(self._model)
        result["model_status"] = model_status
        drop_model = (model_status.get("models") or {}).get("drop_rate") or {}
        rebound_model = (model_status.get("models") or {}).get("rebound_rate") or {}
        result["model_drop_active"] = bool(drop_model.get("active"))
        result["model_rebound_active"] = bool(rebound_model.get("active"))
        result["model_drop_samples"] = drop_model.get("n", 0)
        result["model_rebound_samples"] = rebound_model.get("n", 0)

        required_rebound_samples = 18
        required_start_bias_samples = 12
        start_bias_limit = 12.0
        short_cycle_limit = 0.15
        rebound_takeover_rule = (
            f"回潮接管条件：回潮样本≥{required_rebound_samples}，模型误差优于启发式，"
            f"最近启动偏差≤{int(start_bias_limit)}分钟，短循环率≤{int(short_cycle_limit * 100)}%。"
        )
        blockers: list[str] = []
        if not result["model_rebound_active"]:
            blockers.append("回潮模型尚未优于启发式")
        rebound_samples = int(result.get("model_rebound_samples") or 0)
        if rebound_samples < required_rebound_samples:
            blockers.append("回潮样本还不够")
        start_bias = result.get("recent_start_bias_min")
        start_bias_n = int(result.get("recent_start_bias_samples") or 0)
        if start_bias is None or start_bias_n < required_start_bias_samples:
            blockers.append("启动偏差样本还不足")
        elif abs(float(start_bias)) > start_bias_limit:
            blockers.append("最近启动预测偏差仍偏大")
        short_cycle_rate = result.get("short_cycle_rate")
        if short_cycle_rate is not None and float(short_cycle_rate) > short_cycle_limit:
            blockers.append("短循环率偏高")
        result["rebound_takeover_ready"] = not blockers
        result["rebound_takeover_rule"] = rebound_takeover_rule
        result["rebound_takeover_blockers"] = "；".join(blockers) if blockers else "已满足回潮模型接管条件，可进入小范围验证。"
        result["rebound_takeover_required_rebound_samples"] = required_rebound_samples
        result["rebound_takeover_current_rebound_samples"] = rebound_samples
        result["rebound_takeover_missing_rebound_samples"] = max(required_rebound_samples - rebound_samples, 0)
        result["rebound_takeover_required_start_bias_samples"] = required_start_bias_samples
        result["rebound_takeover_current_start_bias_samples"] = start_bias_n
        result["rebound_takeover_missing_start_bias_samples"] = max(required_start_bias_samples - start_bias_n, 0)
        result["rebound_takeover_start_bias_limit"] = start_bias_limit
        result["rebound_takeover_current_start_bias"] = round(float(start_bias), 2) if start_bias is not None else None
        result["rebound_takeover_start_bias_gap"] = (
            round(max(abs(float(start_bias)) - start_bias_limit, 0.0), 2) if start_bias is not None else None
        )
        result["rebound_takeover_short_cycle_limit"] = short_cycle_limit
        result["rebound_takeover_current_short_cycle_rate"] = round(float(short_cycle_rate), 3) if short_cycle_rate is not None else None
        result["rebound_takeover_short_cycle_gap"] = (
            round(max(float(short_cycle_rate) - short_cycle_limit, 0.0), 3) if short_cycle_rate is not None else None
        )

        gap_parts: list[str] = []
        if not result["model_rebound_active"]:
            gap_parts.append("回潮模型还没稳定优于启发式")
        if result["rebound_takeover_missing_rebound_samples"] > 0:
            gap_parts.append(f"回潮样本还差 {result['rebound_takeover_missing_rebound_samples']} 条")
        if result["rebound_takeover_missing_start_bias_samples"] > 0:
            gap_parts.append(f"启动偏差样本还差 {result['rebound_takeover_missing_start_bias_samples']} 条")
        elif result["rebound_takeover_start_bias_gap"] not in (None, 0):
            gap_parts.append(f"启动偏差还需收敛 {result['rebound_takeover_start_bias_gap']} 分钟")
        if result["rebound_takeover_short_cycle_gap"] not in (None, 0):
            gap_parts.append(
                f"短循环率还需下降 {round(result['rebound_takeover_short_cycle_gap'] * 100, 1)}%"
            )
        result["rebound_takeover_gap_summary"] = "；".join(gap_parts) if gap_parts else "已经满足全部接管门槛。"
        met_items: list[str] = []
        missing_items: list[str] = []
        if result["model_rebound_active"]:
            met_items.append("回潮模型已可用")
        else:
            missing_items.append("回潮模型尚未优于启发式")
        if rebound_samples >= required_rebound_samples:
            met_items.append("回潮样本达标")
        else:
            missing_items.append(f"回潮样本还差 {result['rebound_takeover_missing_rebound_samples']} 条")
        if start_bias is not None and start_bias_n >= required_start_bias_samples and abs(float(start_bias)) <= start_bias_limit:
            met_items.append("启动偏差已收敛")
        elif start_bias_n < required_start_bias_samples:
            missing_items.append(f"启动偏差样本还差 {result['rebound_takeover_missing_start_bias_samples']} 条")
        else:
            missing_items.append(f"启动偏差还需收敛 {result['rebound_takeover_start_bias_gap']} 分钟")
        if short_cycle_rate is None or float(short_cycle_rate) <= short_cycle_limit:
            met_items.append("短循环率安全")
        else:
            missing_items.append(
                f"短循环率还需下降 {round(result['rebound_takeover_short_cycle_gap'] * 100, 1)}%"
            )
        result["rebound_takeover_total_checks"] = 4
        result["rebound_takeover_met_checks"] = len(met_items)
        result["rebound_takeover_missing_checks"] = max(4 - len(met_items), 0)
        result["rebound_takeover_met_items"] = "、".join(met_items) if met_items else "暂未满足"
        result["rebound_takeover_missing_items"] = "；".join(missing_items) if missing_items else "无"
        result["rebound_takeover_allow_now"] = bool(
            result["rebound_takeover_ready"]
            and result["model_rebound_active"]
            and result.get("prediction_confidence") in {"medium", "high"}
        )

        takeover_active = bool(result["rebound_takeover_allow_now"])
        if takeover_active and self.control_enabled:
            takeover_state = "active"
            takeover_state_cn = "回潮模型已参与开机"
            takeover_summary = "当前已启用回潮模型参与开机时机判断，仍保留阈值与锁定期兜底。"
        elif takeover_active:
            takeover_state = "ready"
            takeover_state_cn = "可进入接管验证"
            takeover_summary = "回潮模型已达到验证门槛；打开自主控制后，会先在开机侧小范围接管。"
        elif result["model_rebound_active"]:
            takeover_state = "validating"
            takeover_state_cn = "回潮模型验证中"
            takeover_summary = "回潮模型已开始优于启发式，但仍需更多样本和更小偏差才能进入接管。"
        else:
            takeover_state = "fallback"
            takeover_state_cn = "固定规则主控"
            takeover_summary = "当前仍以固定阈值和最小运行时间为主，模型继续在后台学习。"
        result["rebound_takeover_state"] = takeover_state
        result["rebound_takeover_state_cn"] = takeover_state_cn
        result["rebound_takeover_summary"] = takeover_summary

        anomaly = ml_core.build_anomaly_report(
            runs,
            rebounds,
            context,
            result.get("effective_drop_rate") or 0.0,
            result.get("effective_rebound_rate") or 0.0,
            outcomes=outcomes,
            bias=bias,
        )
        result.update(anomaly)

        self._apply_start_gate_feedback(now, humidity, target, result)
        ml_core.log_prediction(self._predictions_path, context, result)

        result["control_action"] = self._decide_action(now, humidity, target, running, result)
        result["start_threshold_applied"] = round(self._start_threshold_for_now(now, target, result), 1)
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
            humidity = self._read_float(self.entry.data.get(CONF_HUMIDITY_SENSOR) or "")
            if humidity is None:
                humidity = self._read_attr_float(self.entry.data[CONF_DEHUMIDIFIER], "current_humidity") or 60.0
            sample = {
                "datetime": now.isoformat(),
                "mode": mode,
                "scene": self._classify_scene(
                    now,
                    float(humidity),
                    float(self.entry.options.get(CONF_TARGET, self.entry.data.get(CONF_TARGET, DEFAULT_TARGET))),
                    bool(self._prev_running),
                    mode,
                ),
                "humidity_bucket": tank_model.humidity_bucket(float(humidity)),
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

    def _apply_start_gate_feedback(self, now: datetime, humidity: float, target: float,
                                   result: dict[str, Any]) -> None:
        """把锁定期和跳锁定条件合并进"实际可执行"的开机时间展示。"""
        raw_minutes = result.get("predicted_next_start_minutes")
        raw_time = result.get("predicted_next_start_time")
        result["predicted_next_start_raw_minutes"] = raw_minutes
        result["predicted_next_start_raw_time"] = raw_time
        result["lockout_remaining_minutes"] = 0
        result["skip_lock_ready"] = False
        result["start_gate_status_cn"] = "按阈值等待"

        if result.get("is_running"):
            result["start_gate_status_cn"] = "运行中"
            return

        lockout = result.get("lockout_minutes", 45) or 45
        idle_min = None
        if self._last_stop_time is not None:
            idle_min = max((now - self._last_stop_time).total_seconds() / 60.0, 0.0)
        if idle_min is None:
            return

        remaining = max(lockout - idle_min, 0.0)
        result["lockout_remaining_minutes"] = round(remaining, 1)

        lock_release_delta = result.get("lock_release_delta", 10) or 10
        confidence = result.get("prediction_confidence", "low")
        takeover_ready = bool(result.get("rebound_takeover_ready") and result.get("model_rebound_active"))
        skip_lock_ready = bool(
            result.get("allow_skip_lock")
            and takeover_ready
            and confidence in {"medium", "high"}
            and humidity >= target + float(lock_release_delta)
        )
        result["skip_lock_ready"] = skip_lock_ready

        if remaining <= 0:
            result["start_gate_status_cn"] = "锁定已结束"
            return
        if skip_lock_ready:
            result["start_gate_status_cn"] = "满足跳锁定条件"
            return

        effective_minutes = max(int(round(raw_minutes or 0)), int(round(remaining)))
        result["predicted_next_start_minutes"] = effective_minutes
        result["predicted_next_start_time"] = (now + timedelta(minutes=effective_minutes)).strftime("%H:%M:%S")
        result["start_gate_status_cn"] = f"锁定中，还需约 {int(round(remaining))} 分钟"

    def _decide_action(self, now: datetime, humidity: float, target: float,
                       running: bool, result: dict[str, Any]) -> str | None:
        """带硬安全限位的启停判定。返回 'start'/'stop'/None(建议;是否执行由总开关决定)。"""
        start_threshold = self._start_threshold_for_now(now, target, result)
        min_runtime = result.get("min_runtime_minutes", 30) or 30
        lockout = result.get("lockout_minutes", 45) or 45
        start_confirm = result.get("start_confirm_minutes", 5) or 5
        lock_release_delta = result.get("lock_release_delta", 10) or 10
        confidence = result.get("prediction_confidence", "low")
        takeover_ready = bool(result.get("rebound_takeover_ready") and result.get("model_rebound_active"))
        predicted_start_minutes = result.get("predicted_next_start_minutes")
        if not running:
            if self._last_stop_time is not None:
                idle_min = (now - self._last_stop_time).total_seconds() / 60
                if idle_min < lockout:  # 锁定期内不重启,防短循环
                    can_skip_lock = bool(
                        result.get("allow_skip_lock")
                        and takeover_ready
                        and confidence in {"medium", "high"}
                        and predicted_start_minutes is not None
                        and predicted_start_minutes <= 0
                        and humidity >= target + float(lock_release_delta)
                    )
                    if not can_skip_lock:
                        return None
            model_start_floor = target + max(2.0, float((result.get("night_start_offset") if result.get("is_night_window") else result.get("day_start_offset")) or 4) - 2.0)
            if (
                takeover_ready
                and confidence in {"medium", "high"}
                and predicted_start_minutes is not None
                and predicted_start_minutes <= start_confirm
                and humidity >= model_start_floor
                and result.get("scene") != "window_open_suspected"
            ):
                return "start"
            return "start" if humidity >= start_threshold else None
        # 运行中
        if humidity <= target - LOW_PROTECT_DELTA:
            return "stop"  # 低湿保护:无视最小运行时长立即停
        run_min = (now - self._run_start_time).total_seconds() / 60 if self._run_start_time else min_runtime
        predicted_stop_minutes = result.get("predicted_stop_minutes")
        early_stop_guard = result.get("early_stop_guard_minutes", 15) or 15
        overdry_risk = result.get("overdry_risk_level", "low")
        # 模型已有一定把握时,允许在接近目标且已运行足够久时提前收手,降低过度除湿概率。
        if (
            predicted_stop_minutes is not None
            and predicted_stop_minutes <= 0
            and humidity <= target + 0.5
            and run_min >= early_stop_guard
            and confidence in {"medium", "high"}
        ):
            return "stop"
        if (
            overdry_risk in {"medium", "high"}
            and run_min >= early_stop_guard
            and humidity <= target + (1.0 if overdry_risk == "high" else 0.6)
            and confidence in {"medium", "high"}
        ):
            return "stop"
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

    @staticmethod
    def _period_for_now(now: datetime) -> str:
        if now.hour < _NIGHT_END_HOUR:
            return "night"
        if now.hour < 18:
            return "day"
        return "evening"

    @staticmethod
    def _season_for_now(now: datetime) -> str:
        if now.month in (12, 1, 2):
            return "summer"
        if now.month in (3, 4, 5):
            return "autumn"
        if now.month in (6, 7, 8):
            return "winter"
        return "spring"

    def _record_humidity_trace(self, now: datetime, humidity: float, target: float,
                               running: bool, mode: str) -> None:
        self._humidity_trace.append(
            {
                "time": now,
                "humidity": float(humidity),
                "target": float(target),
                "running": bool(running),
                "mode": mode,
            }
        )
        cutoff = now - timedelta(hours=2)
        self._humidity_trace = [row for row in self._humidity_trace if row["time"] >= cutoff]

    def _trace_rate(self, now: datetime, minutes: int) -> float | None:
        if not self._humidity_trace:
            return None
        cutoff = now - timedelta(minutes=minutes)
        candidate = next((row for row in self._humidity_trace if row["time"] >= cutoff), None)
        if candidate is None:
            candidate = self._humidity_trace[0]
        current = self._humidity_trace[-1]
        elapsed = max((current["time"] - candidate["time"]).total_seconds() / 60.0, 1.0)
        return (float(current["humidity"]) - float(candidate["humidity"])) / elapsed

    def _current_drop_rate(self, now: datetime, humidity: float) -> float:
        if self._run_start_time is None or not self._prev_running:
            return 0.0
        elapsed = max((now - self._run_start_time).total_seconds() / 60.0, 1.0)
        return round(max(self._run_start_h - humidity, 0.0) / elapsed, 3)

    def _current_rebound_rate(self, now: datetime, humidity: float) -> float:
        if self._last_stop_time is None or self._prev_running:
            return 0.0
        elapsed = max((now - self._last_stop_time).total_seconds() / 60.0, 1.0)
        return round(max(humidity - self._last_stop_h, 0.0) / elapsed, 3)

    def _classify_scene(self, now: datetime, humidity: float, target: float,
                        running: bool, mode: str) -> str:
        baseline = self._scene_for_now(now)
        rise_10m = self._trace_rate(now, 10)
        rise_15m = self._trace_rate(now, 15)
        rise_60m = self._trace_rate(now, 60)
        live_drop = self._current_drop_rate(now, humidity)

        if rise_10m is not None and rise_10m >= 0.8:
            return "shower_spike"
        if rise_15m is not None and rise_15m >= (10.0 / 15.0):
            return "shower_spike"
        if mode == "drying":
            return "drying_clothes"
        if running and self._run_start_time is not None:
            run_minutes = (now - self._run_start_time).total_seconds() / 60.0
            if run_minutes >= 30 and humidity >= max(target + 5, 65) and live_drop < 0.08:
                return "drying_clothes"
            if run_minutes >= 15 and humidity >= target + 2 and live_drop < 0.05:
                return "window_open_suspected"
        if rise_60m is not None and rise_60m >= 0.12 and humidity >= max(target + 5, 65):
            return "rainy_high_humidity"
        return baseline

    def _start_threshold_for_now(self, now: datetime, target: float, result: dict[str, Any]) -> float:
        bias = result.get("external_start_bias", 0) or 0
        is_night = self._scene_for_now(now) == "night"
        offset_key = "night_start_offset" if is_night else "day_start_offset"
        fallback = 7 if is_night else DEFAULT_START_OFFSET
        offset = result.get(offset_key, fallback) or fallback
        return target + float(offset) + float(bias)

    # ---- 事件检测:把启停/回潮转成训练样本 ------------------------------------
    def _detect_events(self, now: datetime, humidity: float, target: float, running: bool,
                       mode: str, scene: str,
                       outdoor_humidity: float | None = None,
                       outdoor_temperature: float | None = None) -> None:
        # 室外温湿度作为 run/rebound 样本字段记录,供回潮/下降模型当特征(缺则留空,解析回退占位)。
        oh = "" if outdoor_humidity is None else round(outdoor_humidity, 1)
        ot = "" if outdoor_temperature is None else round(outdoor_temperature, 1)
        outdoor = f",outdoor_humidity={oh},outdoor_temp={ot}"
        if self._prev_running is None:
            self._prev_running = running
            return
        if running and not self._prev_running:  # 开机
            self._run_start_time, self._run_start_h = now, humidity
            self._run_start_target = target
            self._run_start_mode = mode
            self._run_start_scene = scene
            self._append(
                now,
                f"{self._ts(now)},start_auto,scene={scene},mode={mode},"
                f"season={self._season_for_now(now)},period={self._period_for_now(now)},"
                f"humidity={round(humidity,1)},target={round(target,1)},reason=auto",
            )
        elif not running and self._prev_running:  # 关机 → 记录一次运行
            if self._run_start_time:
                dur = max(int((now - self._run_start_time).total_seconds() / 60), 1)
                drop = round(self._run_start_h - humidity, 1)
                rate = round(drop / dur, 3) if dur > 0 else 0
                event = "stop_low_protect" if humidity <= target - LOW_PROTECT_DELTA else "stop_target"
                run_scene = self._run_start_scene or scene
                run_mode = self._run_start_mode or mode
                run_target = self._run_start_target or target
                self._append(
                    now,
                    f"{self._ts(now)},{event},scene={run_scene},mode={run_mode},"
                    f"season={self._season_for_now(self._run_start_time)},period={self._period_for_now(self._run_start_time)},"
                    f"start_h={round(self._run_start_h,1)},end_h={round(humidity,1)},target={round(run_target,1)},"
                    f"duration_min={dur},drop={drop},drop_rate={max(rate,0)}{outdoor},"
                    f"reason={'low_protect' if event == 'stop_low_protect' else 'target_reached'}",
                )
            self._last_stop_time, self._last_stop_h = now, humidity
            self._last_stop_target = self._run_start_target or target
            self._last_stop_mode = self._run_start_mode or mode
            self._last_stop_scene = self._run_start_scene or scene
            self._rebound_done = set()
        elif not running and self._last_stop_time is not None:  # 回潮采样
            elapsed = int((now - self._last_stop_time).total_seconds() / 60)
            for w in _REBOUND_WINDOWS:
                if w not in self._rebound_done and elapsed >= w:
                    self._rebound_done.add(w)
                    rate = round((humidity - self._last_stop_h) / elapsed, 3) if elapsed > 0 else 0
                    rebound_scene = self._last_stop_scene or scene
                    rebound_mode = self._last_stop_mode or mode
                    rebound_target = self._last_stop_target or target
                    self._append(
                        now,
                        f"{self._ts(now)},rebound_{w}m,scene={rebound_scene},mode={rebound_mode},"
                        f"season={self._season_for_now(self._last_stop_time)},period={self._period_for_now(self._last_stop_time)},"
                        f"start_h={round(self._last_stop_h,1)},humidity={round(humidity,1)},target={round(rebound_target,1)},"
                        f"elapsed_min={elapsed},rebound_rate={max(rate,0)}{outdoor},reason=rebound_tracking",
                    )
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
        self._run_start_target = d.get("run_start_target", 0.0)
        self._run_start_mode = d.get("run_start_mode", DEFAULT_MODE)
        self._run_start_scene = d.get("run_start_scene", "normal")
        self._last_stop_h = d.get("last_stop_h", 0.0)
        self._last_stop_target = d.get("last_stop_target", 0.0)
        self._last_stop_mode = d.get("last_stop_mode", DEFAULT_MODE)
        self._last_stop_scene = d.get("last_stop_scene", "normal")
        self._rebound_done = set(d.get("rebound_done", []))
        self._water_run_minutes = d.get("water_run_minutes", 0.0)
        self._appliance_state = d.get("appliance_state", {})

    def _save_state(self) -> None:
        data = {
            "prev_running": self._prev_running,
            "run_start_time": self._run_start_time.isoformat() if self._run_start_time else None,
            "last_stop_time": self._last_stop_time.isoformat() if self._last_stop_time else None,
            "run_start_h": self._run_start_h,
            "run_start_target": self._run_start_target,
            "run_start_mode": self._run_start_mode,
            "run_start_scene": self._run_start_scene,
            "last_stop_h": self._last_stop_h,
            "last_stop_target": self._last_stop_target,
            "last_stop_mode": self._last_stop_mode,
            "last_stop_scene": self._last_stop_scene,
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

    def _snapshot_line(self, now: datetime, humidity: float, target: float, running: bool,
                       mode: str, scene: str, drop_rate: float, rebound_rate: float,
                       confidence: str) -> str:
        return (f"{self._ts(now)},snapshot,scene={scene},mode={mode},"
                f"season={self._season_for_now(now)},period={self._period_for_now(now)},"
                f"state={'running' if running else 'off'},running={'on' if running else 'off'},"
                f"humidity={round(humidity,1)},target={round(target,1)},"
                f"drop_rate={round(drop_rate,3)},rebound_rate={round(rebound_rate,3)},confidence={confidence}")

    def _append(self, now: datetime, line: str) -> None:
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self._csv_path.exists() or self._csv_path.stat().st_size == 0
        with self._csv_path.open("a", encoding="utf-8") as handle:
            if new_file:
                handle.write(_CSV_HEADER + "\n")
            handle.write(line + "\n")
