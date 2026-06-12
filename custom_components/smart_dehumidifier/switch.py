"""Switch platform —— 自主控制总开关。

打开后,集成才会按引擎决策(带硬安全限位)真正开关除湿机;默认关闭 = 仅建议,
保证装好不会突然抢走对硬件的控制权。开关状态跨重启保留。
"""
from __future__ import annotations

from typing import Any

from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity

from .const import DOMAIN
from .coordinator import SmartDehumidifierCoordinator


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    coordinator: SmartDehumidifierCoordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([AutonomousControlSwitch(coordinator, entry)])


class AutonomousControlSwitch(SwitchEntity, RestoreEntity):
    _attr_has_entity_name = False
    _attr_name = "Smart Dehumidifier Autonomous Control"
    _attr_icon = "mdi:robot"
    _attr_entity_category = EntityCategory.CONFIG

    def __init__(self, coordinator: SmartDehumidifierCoordinator, entry: ConfigEntry) -> None:
        self._coordinator = coordinator
        self._attr_unique_id = f"{entry.entry_id}_control"
        self._attr_device_info = DeviceInfo(identifiers={(DOMAIN, entry.entry_id)})

    @property
    def is_on(self) -> bool:
        return self._coordinator.control_enabled

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        last = await self.async_get_last_state()
        if last is not None and last.state == "on":
            self._coordinator.control_enabled = True

    async def async_turn_on(self, **kwargs: Any) -> None:
        self._coordinator.control_enabled = True
        self.async_write_ha_state()

    async def async_turn_off(self, **kwargs: Any) -> None:
        self._coordinator.control_enabled = False
        self.async_write_ha_state()
