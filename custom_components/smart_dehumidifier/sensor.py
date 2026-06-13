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
    "effective_drop_rate", "effective_rebound_rate",
    "predicted_stop_time", "predicted_next_start_time",
    "min_runtime_minutes", "lockout_minutes", "low_protect_confirm_minutes",
    "control_takeover_summary", "anomaly_level", "anomaly_summary",
    "external_advice", "external_start_bias", "predictor",
    "dew_point_c", "mold_risk_level", "effective_target",
    "control_enabled", "control_action", "last_control_action",
    "prediction_bias", "last_sync",
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
