# Smart Dehumidifier (Home Assistant 自定义集成)

把原本散在 `configuration.yaml` + `shell_command` 里的除湿机自学习引擎,产品化为一个
**进程内运行、UI 配置、可在任意 HA 实例安装**的自定义集成。开发于 HA 2026.6。

## 相比旧 YAML 方案解决了什么(核心阻塞)

| 旧方案 | 本集成 |
|---|---|
| 32 处硬编码 `/Users/zhenghaowei/...` | 路径相对化(`hass.config.path(...)`) |
| 3 条 shell_command 调外部 python | 引擎进程内运行(`ml_core.py`),无 subprocess、无注入面 |
| 手改 YAML + 复制文件 + 重启 HA | UI 配置流程(config flow),装好即用 |
| `migrate_entity_registry.py` 改注册表修拼音 id | 集成自分配 `unique_id`,无需 hack |
| 湿度传感器掉线无兜底 | 掉线时 `UpdateFailed` → 实体转不可用,不写脏值 |
| 全中文写死 | i18n(`strings.json` + `translations/en、zh-Hans`) |

## 安装(试用,与现有 YAML 系统并存)

1. 把整个 `custom_components/smart_dehumidifier/` 拷到目标 HA 的 `config/custom_components/` 下。
2. 重启 HA。
3. 设置 → 设备与服务 → 添加集成 → 搜索 “Smart Dehumidifier”。
4. 选湿度传感器 + 除湿机(switch/fan/humidifier)+ 目标湿度 → 完成。
5. 生成实体 `sensor.*_learning_engine`,state=学习状态,属性含全部预测/控制参数。

> 本集成独立运行,不影响你现有的 YAML 抽湿机系统;可同时存在做对照。

## 文件

- `manifest.json` 集成清单(domain/version/config_flow/iot_class)
- `config_flow.py` UI 配置流程(EntitySelector / NumberSelector)
- `coordinator.py` DataUpdateCoordinator:读状态(事件循环)→ 记样本+跑引擎(executor)
- `ml_core.py` 自学习引擎(从 `tools/smart_dehumidifier_ml.py` 打包,纯标准库)
- `sensor.py` 暴露引擎结果的实体
- `const.py` / `strings.json` / `translations/`
- `brand/` 本地图标与 logo（Home Assistant 2026.3+ 会通过本地 brands API 自动代理）

## 0.2.0 新增(通用性/健壮性 · UX/i18n · ML 生命周期)

- **事件检测**:协程内识别启停转换,自动记录 run/rebound 样本到 `learning.csv`
  —— 引擎从此真正积累训练数据,而非停在默认值。
- **options flow**:随时改目标湿度 / 更新周期 / 是否启用本地模型(无需删重建)。
- **本地自学习模型(差异化)** `model.py`:纯 numpy 岭回归,**本机训练、本机存储**
  (数据不出户),版本化保存(`model_<version>.json`),**仅当交叉验证证明比启发式更准时
  才接管预测**(`predict_rate` 门控)。`train_model` 服务可手动触发,亦每 N 次更新自动训练。
- **模型状态传感器** + `external_advice`/`predictor` 属性:可见当前用模型还是启发式、CV 误差。
- **UX/i18n**:设备分组(device_info)、实体名翻译(en + zh-Hans)、`diagnostics.py` 一键排障。
- numpy 缺失时模型整体降级、自动回退启发式,不影响可用性。

> 注:`ml_core.py` 相比 `tools/smart_dehumidifier_ml.py` 多了 `compute_predictions` 的
> `learned_*_override` 参数与 `predictor` 输出键(供模型接管),其余一致。

## 0.3.0 新增(实际接管控制 · 露点/防霉目标)

- **自主控制开关** `switch.*_autonomous_control`(`switch.py`):打开后集成才按引擎决策
  **真正开关除湿机**,带硬安全限位(开机阈值 / 锁定防短循环 / 最小运行时长 / 低湿保护
  越级停机)。**默认关闭 = 仅建议**,装好不会突然抢控制;开关状态跨重启保留。
  执行走通用 `homeassistant.turn_on/off`,兼容 switch/fan/humidifier。
- **露点 / 防霉目标**(`climate_math.py` + 可选温度传感器):options 里选"防霉模式"后,
  有效目标 = min(用户目标, 按温度算的防霉上限)(冷房更严:<16℃→55%、<20℃→58%、否则 60%)。
  暴露 `dew_point_c` / `mold_risk_level` / `effective_target` 属性。temp 缺失则回退固定目标。

## 打包仪表盘

`lovelace/dashboard.yaml` 提供了一份可直接粘贴的插件界面卡片，引用集成自动创建的固定实体
（含数值湿度 `sensor.smart_dehumidifier_humidity`）。

当前这套界面依赖两张 HACS 前端卡：

- Mushroom
- button-card

用法：仪表盘 → 编辑 → 添加卡片 → 手动 → 粘贴 `lovelace/dashboard.yaml` 全部内容 → 保存。

如果你想使用完整页面版本，请改用仓库根目录下的 `ui-smart-dehumidifier-plugin.yaml`。

## 路线(对标 HA 集成质量等级)

- **已达成(0.3.0)**:config + options flow、进程内引擎、失效安全、事件检测自学习、
  本地版本化模型 + 门控、**闭环接管控制 + 硬安全限位**、**露点/防霉目标**、设备分组、
  i18n、diagnostics、服务。
- **下一步(→ silver/gold)**:多设备充分测试、外部环境并入集成、打包仪表盘卡片、
  `tests/` + GitHub CI、HACS 发布元数据(`hacs.json`)、能耗/费用统计。
