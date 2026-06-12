"""Diagnostics —— 一键导出排障信息(不含敏感数据)。"""
from __future__ import annotations

from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import DOMAIN
from .coordinator import SmartDehumidifierCoordinator


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: ConfigEntry
) -> dict[str, Any]:
    coordinator: SmartDehumidifierCoordinator = hass.data[DOMAIN][entry.entry_id]
    data = coordinator.data or {}
    return {
        "options": dict(entry.options),
        "update_interval_s": coordinator.update_interval.total_seconds()
        if coordinator.update_interval else None,
        "dataset": {
            "runs": data.get("dataset_runs"),
            "rebounds": data.get("dataset_rebounds"),
            "total": data.get("dataset_total"),
        },
        "predictor": data.get("predictor"),
        "prediction_confidence": data.get("prediction_confidence"),
        "model_status": data.get("model_status"),
        "external_advice": data.get("external_advice"),
        "anomaly_level": data.get("anomaly_level"),
    }
