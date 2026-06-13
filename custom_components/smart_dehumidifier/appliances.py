"""可扩展家居设备学习接口(预留扩展点)。

设计原则——因为每个用户家里设备/实体/语义都不同:
  * 声明式:用户声明"哪个实体属于哪种设备类型",插件不硬编码任何家庭;
  * 只观察:只看每台设备真实的"运行/停止"与可选信号(温度/功率),据此学习,
    所以学到的永远是该用户家里的规律(天然按家自适应,无全局假设);
  * 按类型 Profile:每种设备一个 Profile,声明"该学什么";
  * 通用底座:循环检测 + 分层回退估算对所有设备通用,新增设备=注册一个 Profile,
    无需改除湿机控制内核。

除湿机是第一个内置类型;电磁炉/空气炸锅/通用开关设备已登记 Profile(待 config flow
让用户挑实体 + coordinator 通用观察循环接上后即可学习)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Any, Callable


@dataclass(frozen=True)
class ApplianceProfile:
    appliance_type: str                       # dehumidifier / induction_cooker / air_fryer / generic
    name: str                                 # 展示名
    state_on: tuple[str, ...] = ("on",)        # 视为"运行中"的状态值
    signals: tuple[str, ...] = ()              # 额外观察的信号(属性/实体名),如 temperature/power
    learns: tuple[str, ...] = ("duration_min",)  # 这种设备值得学习/预测的目标(给 UI/文档)
    # 分层维度:从粗到细的特征键,层层放宽回退(各设备可不同)
    layer_keys: tuple[str, ...] = ("period", "scene")


def cycle_sample(cycle: dict[str, Any], profile: ApplianceProfile) -> dict[str, Any]:
    """把一次完整运行循环(start/end/duration/context)提炼成一条学习样本。
    通用、设备无关;具体设备可在 profile.learns 里加自己的派生量。"""
    return {
        "appliance": profile.appliance_type,
        "datetime": cycle.get("start_iso", ""),
        "duration_min": cycle.get("duration_min", 0),
        "period": cycle.get("period", ""),
        "scene": cycle.get("scene", "normal"),
        **{k: cycle.get(k) for k in profile.signals if k in cycle},
    }


PROFILE_REGISTRY: dict[str, ApplianceProfile] = {}


def register(profile: ApplianceProfile) -> None:
    PROFILE_REGISTRY[profile.appliance_type] = profile


def get_profile(appliance_type: str) -> ApplianceProfile:
    return PROFILE_REGISTRY.get(appliance_type, PROFILE_REGISTRY["generic"])


# ---- 内置 Profile(预留;除湿机已有完整实现,其余待接观察循环)----------------
register(ApplianceProfile(
    "dehumidifier", "除湿机", state_on=("on",),
    learns=("drop_rate", "rebound_rate", "run_duration_min"),
    layer_keys=("scene", "period"),
))
register(ApplianceProfile(
    "induction_cooker", "电磁炉", state_on=("on", "heating", "cooking"),
    signals=("temperature", "power"),
    learns=("preheat_min", "cook_duration_min", "time_to_temp_min"),
    layer_keys=("period",),
))
register(ApplianceProfile(
    "air_fryer", "空气炸锅", state_on=("on", "cooking", "heating"),
    signals=("temperature", "target_temperature"),
    learns=("preheat_min", "cook_duration_min"),
    layer_keys=("period",),
))
register(ApplianceProfile("generic", "通用开关设备", state_on=("on",)))


def layered_estimate(samples: list[dict[str, Any]], *, key: str,
                     context: dict[str, Any], layer_keys: tuple[str, ...]) -> tuple[float | None, str]:
    """通用分层回退估算(任何设备都能用):按 layer_keys 从全特征逐层放宽求均值,
    每层做中位数±3MAD 异常过滤。返回 (估算值, 命中层)。"""
    def avg(vals: list[float]) -> float | None:
        vals = [v for v in vals if v and v > 0]
        if len(vals) >= 5:
            med = median(vals)
            mad = median([abs(v - med) for v in vals]) or 0.0
            if mad:
                vals = [v for v in vals if abs(v - med) <= 3.0 * mad] or vals
        return round(sum(vals) / len(vals), 3) if vals else None

    # 从"全部 layer_keys 都匹配"逐步去掉最后一个键 → 最后全局
    for depth in range(len(layer_keys), -1, -1):
        keys = layer_keys[:depth]
        def match(s, keys=keys):
            return all(s.get(k) == context.get(k) for k in keys)
        vals = [float(s.get(key, 0) or 0) for s in samples if match(s)]
        if len([v for v in vals if v > 0]) >= 3:
            a = avg(vals)
            if a is not None:
                layer = "+".join(keys) if keys else "全局"
                return a, layer
    return None, "无样本"
