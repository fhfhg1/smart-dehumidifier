"""露点 / 霉菌风险计算(纯函数,无依赖,便于单测)。

把"固定湿度 60%"升级为按温度的健康阈值:冷房间相对湿度更易凝结/发霉,
因此目标更严;温暖时可放宽。仅做软约束(取 min(用户目标, 防霉目标))。
"""
from __future__ import annotations

import math

_A = 17.27
_B = 237.7


def dew_point_c(temp_c: float, rh: float) -> float | None:
    """Magnus 公式算露点(℃)。rh 为百分比。"""
    if rh <= 0 or rh > 100:
        return None
    gamma = (_A * temp_c) / (_B + temp_c) + math.log(rh / 100.0)
    return round((_B * gamma) / (_A - gamma), 1)


def mold_safe_target(temp_c: float | None) -> int:
    """该温度下建议的最高相对湿度(防霉/防凝结)。temp 未知则用通用 60%。"""
    if temp_c is None:
        return 60
    if temp_c < 16:
        return 55
    if temp_c < 20:
        return 58
    return 60


def mold_risk_level(rh: float, temp_c: float | None) -> str:
    """霉菌风险等级:high / medium / low / minimal。"""
    level = "minimal"
    if rh >= 62:
        level = "low"
    if rh >= 70:
        level = "medium"
    if rh >= 80:
        level = "high"
    # 低温会拉高风险(更接近凝结)
    if temp_c is not None and temp_c < 16 and rh >= 65 and level in ("low", "minimal"):
        level = "medium"
    return level


def to_celsius(value: float, unit: str | None) -> float:
    """把温度统一成摄氏。unit 形如 '°F'/'°C'/None。"""
    if unit and "f" in unit.lower():
        return round((value - 32) * 5.0 / 9.0, 1)
    return value
