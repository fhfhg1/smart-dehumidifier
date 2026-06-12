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

from . import climate_math, ml_core, model
from .const import (
    AUTO_TRAIN_EVERY,
    CONF_DEHUMIDIFIER,
    CONF_ENABLE_MODEL,
    CONF_HUMIDITY_SENSOR,
    CONF_MODE,
    CONF_TARGET,
    CONF_TARGET_MODE,
    CONF_TEMP_SENSOR,
    CONF_UPDATE_INTERVAL,
    DATA_DIRNAME,
    DEFAULT_ENABLE_MODEL,
    DEFAULT_MODE,
    DEFAULT_START_OFFSET,
    DEFAULT_TARGET,
    DEFAULT_TARGET_MODE,
    DEFAULT_UPDATE_INTERVAL,
    LEARNING_CSV,
    LOW_PROTECT_DELTA,
    MIN_SAMPLES_TO_TRAIN,
    MODEL_LATEST,
    STATE_FILE,
    TARGET_MODE_MOLD,
    UNAVAILABLE_STATES,
)

_LOGGER = logging.getLogger(__name__)
_CSV_HEADER = "timestamp,event,scene,mode_or_state,humidity,target,extra"
_REBOUND_WINDOWS = (30, 60, 90)


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

        result = await self.hass.async_add_executor_job(
            self._compute, humidity, target, running, enable_model, target_mode, temp_c, operating_mode
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
                 target_mode: str, temp_c: float | None, operating_mode: str = DEFAULT_MODE) -> dict[str, Any]:
        now = datetime.now()

        if not self._model_loaded:  # executor 线程里懒加载模型(阻塞读放这里)
            self._model = model.load(self._model_path)
            self._model_loaded = True
        if not self._state_loaded:  # 恢复跨重启的在途运行状态
            self._load_state()
            self._state_loaded = True

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
            sample = {"scene": "normal", "mode": operating_mode, "start_humidity": humidity,
                      "target_humidity": target, "datetime": now.isoformat()}
            drop_override = model.predict_rate(self._model, "drop_rate", sample)
            rebound_override = model.predict_rate(self._model, "rebound_rate", sample)

        context = ml_core.PredictionContext(
            humidity=humidity, target=target, scene="normal", mode=operating_mode,
            state="running" if running else "off", running=running,
            current_drop_rate=0.0, current_rebound_rate=0.0,
            start_threshold=target + DEFAULT_START_OFFSET, min_runtime_left=0, now=now,
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

        self._update_count += 1
        if self._update_count % AUTO_TRAIN_EVERY == 0:
            self._maybe_train(runs, rebounds)
        result["model_status"] = model.status(self._model)
        return result

    def _decide_action(self, now: datetime, humidity: float, target: float,
                       running: bool, result: dict[str, Any]) -> str | None:
        """带硬安全限位的启停判定。返回 'start'/'stop'/None(建议;是否执行由总开关决定)。"""
        bias = result.get("external_start_bias", 0) or 0
        start_threshold = target + DEFAULT_START_OFFSET + bias
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

    # ---- 事件检测:把启停/回潮转成训练样本 ------------------------------------
    def _detect_events(self, now: datetime, humidity: float, target: float, running: bool) -> None:
        if self._prev_running is None:
            self._prev_running = running
            return
        if running and not self._prev_running:  # 开机
            self._run_start_time, self._run_start_h = now, humidity
            self._append(now, f"{self._ts(now)},start_auto,scene=normal,mode=comfort,"
                              f"humidity={round(humidity,1)},target={int(target)},reason=auto")
        elif not running and self._prev_running:  # 关机 → 记录一次运行
            if self._run_start_time:
                dur = max(int((now - self._run_start_time).total_seconds() / 60), 1)
                drop = round(self._run_start_h - humidity, 1)
                rate = round(drop / dur, 3) if dur > 0 else 0
                event = "stop_low_protect" if humidity <= target - LOW_PROTECT_DELTA else "stop_target"
                self._append(now, f"{self._ts(now)},{event},scene=normal,start_h={round(self._run_start_h,1)},"
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
                    self._append(now, f"{self._ts(now)},rebound_{w}m,scene=normal,"
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

    def _save_state(self) -> None:
        data = {
            "prev_running": self._prev_running,
            "run_start_time": self._run_start_time.isoformat() if self._run_start_time else None,
            "last_stop_time": self._last_stop_time.isoformat() if self._last_stop_time else None,
            "run_start_h": self._run_start_h,
            "last_stop_h": self._last_stop_h,
            "rebound_done": sorted(self._rebound_done),
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
        return (f"{self._ts(now)},snapshot,scene=normal,mode=comfort,"
                f"state={'running' if running else 'off'},running={'on' if running else 'off'},"
                f"humidity={round(humidity,1)},target={int(target)},drop_rate=0,rebound_rate=0,confidence=low")

    def _append(self, now: datetime, line: str) -> None:
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self._csv_path.exists() or self._csv_path.stat().st_size == 0
        with self._csv_path.open("a", encoding="utf-8") as handle:
            if new_file:
                handle.write(_CSV_HEADER + "\n")
            handle.write(line + "\n")
