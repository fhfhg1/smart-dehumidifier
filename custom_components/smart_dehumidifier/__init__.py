"""The Smart Dehumidifier integration.

把原本散在 YAML + shell_command 里的自学习引擎,收进一个进程内运行的
HA 自定义集成:UI 配置、路径相对化、无 subprocess、传感器掉线兜底。
"""
from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import issue_registry as ir

from .const import DOMAIN, PLATFORMS, SERVICE_TRAIN
from .coordinator import SmartDehumidifierCoordinator

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Smart Dehumidifier from a config entry."""
    coordinator = SmartDehumidifierCoordinator(hass, entry)
    await coordinator.async_initialize()
    await coordinator.async_config_entry_first_refresh()

    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = coordinator
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(_async_reload_entry))
    _async_register_services(hass)
    _async_recommend_frontend_cards(hass)
    return True


def _async_recommend_frontend_cards(hass: HomeAssistant) -> None:
    """安装后在「设置 → 修复」里提示:打包仪表盘需要的 HACS 前端卡片。"""
    ir.async_create_issue(
        hass,
        DOMAIN,
        "frontend_cards",
        is_fixable=False,
        is_persistent=False,
        severity=ir.IssueSeverity.WARNING,
        translation_key="frontend_cards",
        learn_more_url="https://hacs.xyz/",
    )


def _async_register_services(hass: HomeAssistant) -> None:
    """注册 train_model 服务(在本机按需训练所有已配置实例)。"""
    if hass.services.has_service(DOMAIN, SERVICE_TRAIN):
        return

    async def _handle_train(call: ServiceCall) -> None:
        for coordinator in hass.data.get(DOMAIN, {}).values():
            await hass.async_add_executor_job(coordinator.train_now)
            await coordinator.async_request_refresh()

    hass.services.async_register(DOMAIN, SERVICE_TRAIN, _handle_train)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id, None)
    return unload_ok


async def _async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload when options change."""
    await hass.config_entries.async_reload(entry.entry_id)
