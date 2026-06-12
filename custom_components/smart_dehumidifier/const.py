"""Constants for the Smart Dehumidifier integration."""
from __future__ import annotations

DOMAIN = "smart_dehumidifier"
PLATFORMS = ["sensor", "switch", "number", "select"]

# 运行模式(引擎 mode):节能 / 舒适 / 干衣
CONF_MODE = "operating_mode"
OPERATING_MODES = ["energy_saving", "comfort", "drying"]
DEFAULT_MODE = "comfort"

# 配置项(config flow)
CONF_HUMIDITY_SENSOR = "humidity_sensor"
CONF_DEHUMIDIFIER = "dehumidifier_switch"
CONF_TEMP_SENSOR = "temperature_sensor"
CONF_TARGET = "target_humidity"

# 选项(options flow,可后改)
CONF_UPDATE_INTERVAL = "update_interval"
CONF_ENABLE_MODEL = "enable_model"
CONF_TARGET_MODE = "target_mode"

# 目标模式:固定湿度 vs 按温度算的防霉目标
TARGET_MODE_FIXED = "fixed"
TARGET_MODE_MOLD = "mold_risk"
DEFAULT_TARGET_MODE = TARGET_MODE_FIXED

DEFAULT_TARGET = 60
DEFAULT_START_OFFSET = 4
DEFAULT_UPDATE_INTERVAL = 60
DEFAULT_ENABLE_MODEL = True

# 低湿保护:停机时湿度低于 target - 该值视为低湿保护停机
LOW_PROTECT_DELTA = 5

# 数据/模型文件均在 HA config dir 下(路径相对化)
DATA_DIRNAME = "smart_dehumidifier"
LEARNING_CSV = "learning.csv"
MODEL_LATEST = "model_latest.json"
STATE_FILE = "state.json"  # 事件检测的在途运行状态,跨重启持久化

# ML 生命周期
MIN_SAMPLES_TO_TRAIN = 40       # 任一速率样本达到才训练
AUTO_TRAIN_EVERY = 30           # 每 N 次更新尝试自动训练一次
SERVICE_TRAIN = "train_model"

UNAVAILABLE_STATES = {"unknown", "unavailable", "none", "", None}
