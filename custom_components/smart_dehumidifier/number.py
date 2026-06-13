"""Number platform —— 目标湿度(可在仪表盘直接调,写入配置选项)。"""
from __future__ import annotations

from homeassistant.components.number import NumberEntity, NumberMode
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    CONF_TANK_CAPACITY,
    CONF_TARGET,
    DEFAULT_TANK_CAPACITY,
    DEFAULT_TARGET,
    DOMAIN,
)


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    async_add_entities([TargetHumidityNumber(entry), TankCapacityNumber(entry)])


class TargetHumidityNumber(NumberEntity):
    _attr_has_entity_name = False
    _attr_name = "Smart Dehumidifier Target Humidity"
    _attr_icon = "mdi:water-percent"
    _attr_native_min_value = 30
    _attr_native_max_value = 80
    _attr_native_step = 1
    _attr_native_unit_of_measurement = "%"
    _attr_mode = NumberMode.SLIDER

    def __init__(self, entry: ConfigEntry) -> None:
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_target"
        self._attr_device_info = DeviceInfo(identifiers={(DOMAIN, entry.entry_id)})

    @property
    def native_value(self) -> float:
        return float(
            self._entry.options.get(CONF_TARGET, self._entry.data.get(CONF_TARGET, DEFAULT_TARGET))
        )

    async def async_set_native_value(self, value: float) -> None:
        opts = dict(self._entry.options)
        opts[CONF_TARGET] = value
        self.hass.config_entries.async_update_entry(self._entry, options=opts)


class TankCapacityNumber(NumberEntity):
    _attr_has_entity_name = False
    _attr_name = "Smart Dehumidifier Tank Capacity"
    _attr_icon = "mdi:cup-water"
    _attr_native_min_value = 0.5
    _attr_native_max_value = 20.0
    _attr_native_step = 0.5
    _attr_native_unit_of_measurement = "L"
    _attr_mode = NumberMode.BOX

    def __init__(self, entry: ConfigEntry) -> None:
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_tank_capacity"
        self._attr_device_info = DeviceInfo(identifiers={(DOMAIN, entry.entry_id)})

    @property
    def native_value(self) -> float:
        return float(
            self._entry.options.get(CONF_TANK_CAPACITY, self._entry.data.get(CONF_TANK_CAPACITY, DEFAULT_TANK_CAPACITY))
        )

    async def async_set_native_value(self, value: float) -> None:
        opts = dict(self._entry.options)
        opts[CONF_TANK_CAPACITY] = value
        self.hass.config_entries.async_update_entry(self._entry, options=opts)
