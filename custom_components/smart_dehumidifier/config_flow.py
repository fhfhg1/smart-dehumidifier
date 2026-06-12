"""Config flow for Smart Dehumidifier —— UI 配置(选湿度传感器 + 除湿机开关 + 目标)。"""
from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.core import callback
from homeassistant.helpers.selector import (
    BooleanSelector,
    EntitySelector,
    EntitySelectorConfig,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
)

from .const import (
    CONF_DEHUMIDIFIER,
    CONF_ENABLE_MODEL,
    CONF_HUMIDITY_SENSOR,
    CONF_TARGET,
    CONF_TARGET_MODE,
    CONF_TEMP_SENSOR,
    CONF_UPDATE_INTERVAL,
    DEFAULT_ENABLE_MODEL,
    DEFAULT_TARGET,
    DEFAULT_TARGET_MODE,
    DEFAULT_UPDATE_INTERVAL,
    DOMAIN,
    TARGET_MODE_FIXED,
    TARGET_MODE_MOLD,
)


def _schema(defaults: dict[str, Any] | None = None) -> vol.Schema:
    defaults = defaults or {}

    def _opt(key: str):
        """有猜测值时把字段标记为 Optional 并带默认;无则 Optional 无默认。"""
        val = defaults.get(key)
        return vol.Optional(key, default=val) if val else vol.Optional(key)

    return vol.Schema(
        {
            _opt(CONF_HUMIDITY_SENSOR): EntitySelector(
                EntitySelectorConfig(domain=["sensor", "number", "input_number"])
            ),
            (vol.Required(CONF_DEHUMIDIFIER, default=defaults[CONF_DEHUMIDIFIER])
             if defaults.get(CONF_DEHUMIDIFIER) else vol.Required(CONF_DEHUMIDIFIER)): EntitySelector(
                EntitySelectorConfig(domain=["switch", "fan", "humidifier", "climate"])
            ),
            _opt(CONF_TEMP_SENSOR): EntitySelector(
                EntitySelectorConfig(domain=["sensor", "number", "input_number"])
            ),
            vol.Required(
                CONF_TARGET, default=defaults.get(CONF_TARGET, DEFAULT_TARGET)
            ): NumberSelector(
                NumberSelectorConfig(min=30, max=80, step=1, mode=NumberSelectorMode.SLIDER, unit_of_measurement="%")
            ),
        }
    )


def _options_schema(defaults: dict[str, Any] | None = None) -> vol.Schema:
    defaults = defaults or {}

    def _opt_entity(key: str):
        val = defaults.get(key)
        return vol.Optional(key, default=val) if val else vol.Optional(key)

    return vol.Schema(
        {
            vol.Required(
                CONF_TARGET, default=defaults.get(CONF_TARGET, DEFAULT_TARGET)
            ): NumberSelector(
                NumberSelectorConfig(min=30, max=80, step=1, mode=NumberSelectorMode.SLIDER, unit_of_measurement="%")
            ),
            vol.Required(
                CONF_UPDATE_INTERVAL, default=defaults.get(CONF_UPDATE_INTERVAL, DEFAULT_UPDATE_INTERVAL)
            ): NumberSelector(
                NumberSelectorConfig(min=30, max=600, step=10, mode=NumberSelectorMode.BOX, unit_of_measurement="s")
            ),
            vol.Required(
                CONF_TARGET_MODE, default=defaults.get(CONF_TARGET_MODE, DEFAULT_TARGET_MODE)
            ): SelectSelector(
                SelectSelectorConfig(
                    options=[TARGET_MODE_FIXED, TARGET_MODE_MOLD],
                    translation_key="target_mode",
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Required(
                CONF_ENABLE_MODEL, default=defaults.get(CONF_ENABLE_MODEL, DEFAULT_ENABLE_MODEL)
            ): BooleanSelector(),
            _opt_entity(CONF_TEMP_SENSOR): EntitySelector(
                EntitySelectorConfig(domain=["sensor", "number", "input_number"])
            ),
        }
    )


def _friendly(state) -> str:
    return (state.attributes.get("friendly_name") or "").lower()


def guess_humidity(hass) -> str | None:
    """自动匹配最像"当前室内湿度"的实体。"""
    best, best_score = None, 0
    for st in hass.states.async_all("sensor"):
        eid = st.entity_id
        score = 0
        if st.attributes.get("device_class") == "humidity":
            score += 100
        if st.attributes.get("unit_of_measurement") == "%":
            score += 10
        if "current_humidity" in eid:
            score += 50
        if "humid" in eid or "湿" in _friendly(st):
            score += 20
        for bad in ("change", "mean", "max", "min", "rate", "drop", "rebound",
                    "trend", "outdoor", "alert", "target", "threshold"):
            if bad in eid:
                score -= 40
        if score > best_score:
            best, best_score = eid, score
    return best if best_score > 0 else None


# 明显不是室温的设备/语义关键词(电磁炉、炸锅、烤箱、探针、设定/目标温度等)
_APPLIANCE_HINTS = (
    "airfryer", "fryer", "cooker", "oven", "probe", "kettle", "grill", "stove",
    "induction", "chunmi", "电磁炉", "炸", "烤", "锅", "壶", "cook",
    "change", "mean", "max", "min", "feels", "dew", "target", "move", "set",
)

_AMBIENT_TEMP_HINTS = (
    "room", "bedroom", "living", "hall", "indoor", "ambient", "thermo",
    "温度", "室温", "房间", "卧室", "客厅", "温湿度",
)


def guess_temperature(hass) -> str | None:
    """自动匹配室温传感器:device_class=temperature + 读数落在室内合理范围,
    并排除电磁炉/炸锅等设备温度。找不到合适的就返回 None(让用户留空)。"""
    best, best_score = None, 0
    for st in hass.states.async_all("sensor"):
        eid = st.entity_id
        name = _friendly(st)
        if any(h in eid or h in name for h in _APPLIANCE_HINTS):
            continue
        if st.attributes.get("device_class") != "temperature":
            continue
        # 读数必须像室温(换算成摄氏后 0~40℃),否则跳过
        try:
            value = float(st.state)
        except (TypeError, ValueError):
            continue
        unit = st.attributes.get("unit_of_measurement")
        celsius = (value - 32) * 5.0 / 9.0 if unit and "f" in str(unit).lower() else value
        if not (0 <= celsius <= 40):
            continue
        score = 100 + (10 if ("temp" in eid or "温" in name) else 0)
        if any(h in eid or h in name for h in _AMBIENT_TEMP_HINTS):
            score += 25
        else:
            score -= 35
        if score > best_score:
            best, best_score = eid, score
    return best if best_score >= 110 else None


def guess_dehumidifier(hass) -> str | None:
    for domain in ("humidifier", "switch", "fan", "climate"):
        for st in hass.states.async_all(domain):
            if "dehumid" in st.entity_id or "除湿" in _friendly(st) or "抽湿" in _friendly(st):
                return st.entity_id
    return None


class SmartDehumidifierConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle the UI setup flow."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        errors: dict[str, str] = {}
        if user_input is not None:
            await self.async_set_unique_id(user_input[CONF_DEHUMIDIFIER])
            self._abort_if_unique_id_configured()
            return self.async_create_entry(title="Smart Dehumidifier", data=user_input)
        # 自动匹配:预填最可能的湿度/温度/除湿机实体,用户确认即可
        defaults = {
            CONF_HUMIDITY_SENSOR: guess_humidity(self.hass),
            CONF_TEMP_SENSOR: guess_temperature(self.hass),
            CONF_DEHUMIDIFIER: guess_dehumidifier(self.hass),
        }
        return self.async_show_form(
            step_id="user", data_schema=_schema(defaults), errors=errors
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> "SmartDehumidifierOptionsFlow":
        return SmartDehumidifierOptionsFlow()


class SmartDehumidifierOptionsFlow(OptionsFlow):
    """改目标湿度、更新周期、是否启用本地模型 —— 无需删重建。"""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        if user_input is not None:
            user_input = dict(user_input)
            user_input[CONF_TEMP_SENSOR] = user_input.get(CONF_TEMP_SENSOR) or ""
            return self.async_create_entry(title="", data=user_input)
        opts = self.config_entry.options
        data = self.config_entry.data
        defaults = {
            CONF_TARGET: opts.get(CONF_TARGET, data.get(CONF_TARGET, DEFAULT_TARGET)),
            CONF_UPDATE_INTERVAL: opts.get(CONF_UPDATE_INTERVAL, DEFAULT_UPDATE_INTERVAL),
            CONF_TARGET_MODE: opts.get(CONF_TARGET_MODE, DEFAULT_TARGET_MODE),
            CONF_ENABLE_MODEL: opts.get(CONF_ENABLE_MODEL, DEFAULT_ENABLE_MODEL),
            CONF_TEMP_SENSOR: opts.get(CONF_TEMP_SENSOR, data.get(CONF_TEMP_SENSOR)),
        }
        return self.async_show_form(step_id="init", data_schema=_options_schema(defaults))
