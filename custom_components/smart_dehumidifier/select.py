"""Select platform —— 运行模式(节能 / 舒适 / 干衣,写入配置选项)。"""
from __future__ import annotations

from homeassistant.components.select import SelectEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import CONF_MODE, DEFAULT_MODE, DOMAIN, OPERATING_MODES, WATER_CALIB_FRACTIONS


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    coordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([OperatingModeSelect(entry), WaterCalibrationSelect(coordinator, entry)])


class OperatingModeSelect(SelectEntity):
    _attr_has_entity_name = False
    _attr_name = "Smart Dehumidifier Operating Mode"
    _attr_translation_key = "operating_mode"
    _attr_icon = "mdi:tune-variant"
    _attr_options = OPERATING_MODES

    def __init__(self, entry: ConfigEntry) -> None:
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_mode"
        self._attr_device_info = DeviceInfo(identifiers={(DOMAIN, entry.entry_id)})

    @property
    def current_option(self) -> str:
        return self._entry.options.get(CONF_MODE, DEFAULT_MODE)

    async def async_select_option(self, option: str) -> None:
        opts = dict(self._entry.options)
        opts[CONF_MODE] = option
        self.hass.config_entries.async_update_entry(self._entry, options=opts)


class WaterCalibrationSelect(SelectEntity):
    """倒水前水位校准:选一次=告诉系统"我刚倒水、倒前大概是这个水位",
    系统据此学习"运行做功→升水"并把累计清零。"""

    _attr_has_entity_name = False
    _attr_name = "Smart Dehumidifier Water Calibration"
    _attr_icon = "mdi:cup-water"
    _attr_options = list(WATER_CALIB_FRACTIONS.keys())

    def __init__(self, coordinator, entry: ConfigEntry) -> None:
        self._coordinator = coordinator
        self._attr_unique_id = f"{entry.entry_id}_water_calibration"
        self._attr_device_info = DeviceInfo(identifiers={(DOMAIN, entry.entry_id)})
        self._last: str | None = None

    @property
    def current_option(self) -> str | None:
        return self._last

    async def async_select_option(self, option: str) -> None:
        self._last = option
        await self.hass.async_add_executor_job(self._coordinator.record_water_calibration, option)
        self.async_write_ha_state()
        await self._coordinator.async_request_refresh()
