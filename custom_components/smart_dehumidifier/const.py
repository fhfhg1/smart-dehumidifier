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
PREDICTIONS_FILE = "predictions.jsonl"  # live 预测日志,供预测误差反馈闭环
WATER_SAMPLES_FILE = "water_samples.jsonl"  # 水箱校准样本(倒水反馈),供 tank_model 学习
APPLIANCES_LOG = "appliances_log.jsonl"     # 其它家电的运行循环样本(只观察学习)
CONF_EXTRA_APPLIANCES = "extra_appliances"  # 用户声明的额外要学习的设备实体(列表)

# 水箱预测:容量(升)+ 倒水前水位校准选项
CONF_TANK_CAPACITY = "tank_capacity"
DEFAULT_TANK_CAPACITY = 3.0
WATER_CALIB_FRACTIONS = {"空了": 0.0, "四分之一": 0.25, "二分之一": 0.5, "四分之三": 0.75, "已满": 1.0}
DEFAULT_WATER_RATE_LPM = 0.004  # 无校准样本时的兜底:每运行 1 分钟约积水 0.004L

# ML 生命周期
MIN_SAMPLES_TO_TRAIN = 40       # 任一速率样本达到才训练
AUTO_TRAIN_EVERY = 30           # 每 N 次更新尝试自动训练一次
SERVICE_TRAIN = "train_model"

UNAVAILABLE_STATES = {"unknown", "unavailable", "none", "", None}
