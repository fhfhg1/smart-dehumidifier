"""Select platform —— 运行模式(节能 / 舒适 / 干衣,写入配置选项)。"""
from __future__ import annotations

from homeassistant.components.select import SelectEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import CONF_MODE, DEFAULT_MODE, DOMAIN, OPERATING_MODES


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    async_add_entities([OperatingModeSelect(entry)])


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
