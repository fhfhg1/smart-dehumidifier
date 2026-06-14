"""Sensor platform —— 引擎主传感器 + 模型状态传感器(设备分组 + 可翻译名)。"""
from __future__ import annotations

from typing import Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .coordinator import SmartDehumidifierCoordinator

_ENGINE_ATTRS = (
    "current_humidity", "target_humidity", "configured_target_humidity", "is_running",
    "dataset_total", "dataset_runs", "dataset_rebounds", "dataset_snapshots", "scene", "mode",
    "prediction_confidence", "prediction_confidence_cn",
    "stop_confidence", "stop_confidence_cn", "start_confidence", "start_confidence_cn",
    "effective_drop_rate", "effective_rebound_rate",
    "predicted_stop_time", "predicted_next_start_time", "predicted_next_start_raw_time",
    "predicted_next_start_raw_minutes", "lockout_remaining_minutes", "skip_lock_ready", "start_gate_status_cn",
    "min_runtime_minutes", "lockout_minutes", "low_protect_confirm_minutes",
    "control_takeover_summary", "anomaly_level", "anomaly_summary",
    "external_advice", "external_start_bias", "predictor",
    "learning_stage", "learning_stage_cn", "drop_model_ready", "rebound_model_ready",
    "readiness_summary", "next_milestone",
    "model_drop_active", "model_rebound_active", "model_drop_samples", "model_rebound_samples",
    "recent_stop_bias_min", "recent_start_bias_min", "recent_stop_bias_samples", "recent_start_bias_samples",
    "short_cycle_rate", "over_dry_rate",
    "rebound_takeover_ready", "rebound_takeover_rule", "rebound_takeover_blockers",
    "rebound_takeover_state", "rebound_takeover_state_cn", "rebound_takeover_summary",
    "rebound_takeover_required_rebound_samples", "rebound_takeover_current_rebound_samples",
    "rebound_takeover_missing_rebound_samples", "rebound_takeover_required_start_bias_samples",
    "rebound_takeover_current_start_bias_samples", "rebound_takeover_missing_start_bias_samples",
    "rebound_takeover_start_bias_limit", "rebound_takeover_current_start_bias",
    "rebound_takeover_start_bias_gap", "rebound_takeover_short_cycle_limit",
    "rebound_takeover_current_short_cycle_rate", "rebound_takeover_short_cycle_gap",
    "rebound_takeover_gap_summary", "rebound_takeover_total_checks",
    "rebound_takeover_met_checks", "rebound_takeover_missing_checks",
    "rebound_takeover_met_items", "rebound_takeover_missing_items",
    "rebound_takeover_allow_now",
    "overdry_risk_level", "overdry_risk_summary", "recent_over_dry_rate", "short_cycle_rate_learning",
    "anomaly_code", "anomaly_title", "anomaly_actions",
    "dew_point_c", "mold_risk_level", "effective_target",
    "outdoor_humidity", "outdoor_temperature",
    "control_enabled", "control_action", "last_control_action", "prediction_bias",
    "water_estimated_liters", "water_tank_capacity", "water_fill_percent",
    "water_remaining_liters", "water_level_text", "water_rate_lpm", "water_rate_source",
    "appliances", "last_sync", "rebound_age_minutes", "rebound_warmup_active",
    "rebound_warmup_minutes", "rebound_warmup_progress",
    "effective_rebound_rate_raw_blend", "effective_rebound_rate_damped",
)


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    coordinator: SmartDehumidifierCoordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([
        SmartDehumidifierMLSensor(coordinator, entry),
        SmartDehumidifierModelSensor(coordinator, entry),
        SmartDehumidifierHumiditySensor(coordinator, entry),
    ])


class _BaseEntity(CoordinatorEntity[SmartDehumidifierCoordinator], SensorEntity):
    # 固定 ASCII 命名 → entity_id 确定、与设备所在区域无关(打包仪表盘才能通用)
    _attr_has_entity_name = False

    def __init__(self, coordinator: SmartDehumidifierCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._entry = entry
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name="Smart Dehumidifier",
            manufacturer="Smart Dehumidifier",
            model="Self-learning controller",
        )


class SmartDehumidifierMLSensor(_BaseEntity):
    _attr_name = "Smart Dehumidifier Learning Engine"
    _attr_icon = "mdi:brain"

    def __init__(self, coordinator, entry) -> None:
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_ml_engine"

    @property
    def native_value(self) -> str | None:
        return (self.coordinator.data or {}).get("learning_state")

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        data = self.coordinator.data or {}
        return {k: data.get(k) for k in _ENGINE_ATTRS}


class SmartDehumidifierModelSensor(_BaseEntity):
    _attr_name = "Smart Dehumidifier Model Status"
    _attr_icon = "mdi:chart-bell-curve-cumulative"

    def __init__(self, coordinator, entry) -> None:
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_model_status"

    @property
    def native_value(self) -> str | None:
        status = (self.coordinator.data or {}).get("model_status") or {}
        if not status.get("available"):
            return "unavailable"
        if status.get("active"):
            return "active"
        return "trained" if status.get("trained") else "collecting"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        return (self.coordinator.data or {}).get("model_status") or {}


class SmartDehumidifierHumiditySensor(_BaseEntity):
    """干净的数值湿度实体:供仪表盘 gauge / 历史曲线 / 长期统计使用。"""

    _attr_name = "Smart Dehumidifier Humidity"
    _attr_device_class = SensorDeviceClass.HUMIDITY
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = "%"

    def __init__(self, coordinator, entry) -> None:
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_humidity"

    @property
    def native_value(self) -> float | None:
        return (self.coordinator.data or {}).get("current_humidity")
