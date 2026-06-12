# 交接文档 —— 智能除湿机自学习系统

面向接手者(codex / 后续 Claude)。冷启动读这一份即可,细节指向各子文档。
最后更新:2026-06-12。

## 0. 一句话现状

存在**两套并行系统**;新的自定义集成已装到 live 并运行,**自主控制默认关(仅观测/建议,不碰硬件)**。

## 1. 两套系统(别混)

### A. 线上 YAML 系统(当前生产)
- `staging/configuration.yaml`、`staging/automations.yaml`、`staging/ui-bedroom-control.yaml`
- ML 走 `shell_command` 调 `tools/smart_dehumidifier_ml.py`(进程外)。
- 部署:`./deploy-staging.sh`(check_config → 停 HA → 备份 → 拷贝 → `tools/migrate_entity_registry.py` → 起 HA)。
- 仪表盘改动可只 `cp staging/ui-bedroom-control.yaml <LIVE>/config/` + 浏览器硬刷新,免重启。
- LIVE = `/Users/zhenghaowei/homeassistant-run/config/`。

### B. 自定义集成(产品化,已装 live、已配置、运行中)
- `custom_components/smart_dehumidifier/`(v0.3.0,HA 2026.6)。**进程内**运行,无 shell_command。
- 已配置实例:湿度=`sensor.smart_dehumidifier_current_humidity`、除湿机=`humidifier.150633095553996_humidifier`、温度=小米温度传感器、目标 60。
- 生成实体:`sensor.smart_dehumidifier_learning_engine`、`sensor.smart_dehumidifier_model_status`、`switch.smart_dehumidifier_autonomous_control`(默认 off)。
- 更新集成:`rsync -a --delete --exclude __pycache__ custom_components/smart_dehumidifier/ <LIVE>/config/custom_components/smart_dehumidifier/` 然后 `launchctl kickstart -k gui/$(id -u)/local.homeassistant.core`。
- 安装/路线细节见 `custom_components/smart_dehumidifier/README.md`。

## 2. 集成文件地图

| 文件 | 职责 |
|---|---|
| `manifest.json` | 清单(config_flow、requirements: numpy) |
| `config_flow.py` | UI 配置 + options flow + **自动匹配**(guess_humidity/temperature/dehumidifier) |
| `coordinator.py` | 进程内引擎:读状态→事件检测(记 run/rebound 样本)→ 跑引擎 → 接管控制(`_decide_action` 带硬限位)→ 自动训练 |
| `ml_core.py` | 启发式引擎(=`tools/smart_dehumidifier_ml.py` 的副本 + `learned_*_override`/`predictor`) |
| `model.py` | 本地岭回归(numpy):训练/版本化/CV 门控(CV 优于启发式才接管) |
| `climate_math.py` | 露点 / 防霉目标(Magnus、随温度的防霉上限) |
| `sensor.py` / `switch.py` | 实体(学习引擎、模型状态;自主控制总开关) |
| `diagnostics.py` / `services.yaml` | 排障导出;`train_model` 服务 |
| `strings.json` / `translations/` | i18n(en + zh-Hans) |

## 3. 离线工具(tools/)

- `smart_dehumidifier_ml.py`:引擎本体 + `sync`/`backtest`/`stats`/`dump-json` 子命令。
- `train_dehumidifier_model.py`:`compare`(岭回归 vs 启发式 CV 对照)/`train`。
- `simulate_dehumidifier.py`:物理仿真器,`--replay-predictions` 可端到端验证管线。
- `migrate_entity_registry.py`:重命名 entity_id;**含 `EXTRA_MAP`**(见 §5)。
- 测试:`tests/test_smart_dehumidifier_ml.py`(`python -m unittest tests.test_smart_dehumidifier_ml`)。

常用验证:
```
PY=/Users/zhenghaowei/homeassistant-run/.venv/bin/python
$PY tools/simulate_dehumidifier.py --out /tmp/sim.csv --days 30 --replay-predictions /tmp/pred.jsonl
$PY tools/smart_dehumidifier_ml.py sync --source /tmp/sim.csv --runs /tmp/r.jsonl --rebounds /tmp/rb.jsonl --snapshots /tmp/s.jsonl --output /tmp/o.json --current-humidity 65 --target-humidity 60 --scene normal --mode comfort --machine-state off --running 0 --current-drop-rate 0 --current-rebound-rate 0 --start-threshold 64 --min-runtime-left 0
$PY tools/smart_dehumidifier_ml.py stats --runs /tmp/r.jsonl --rebounds /tmp/rb.jsonl --snapshots /tmp/s.jsonl
$PY tools/train_dehumidifier_model.py compare --runs /tmp/r.jsonl --rebounds /tmp/rb.jsonl
```

## 4. ML 怎么运作(决策 + 兜底)

- **倒计时/时长参数**(low_protect/lockout/min_runtime/confirm…):基准(按模式)→ 按场景+学到的 drop/rebound 速率+置信度加减 → clamp 安全区间 → `control_takeover` 样本够才接管,否则规则默认。
- **速率**=blend(当前实测, 历史 EWMA);集成里若本地模型 CV 更准 → 用模型预测替代(`learned_*_override`)。
- **外部环境**(YAML 系统已接):开窗/空调/室外湿度/降雨 → `external_start_bias` 调启动阈值;仅在对应传感器可用时生效。集成端尚未并入(见 §6)。
- 反馈闭环:`predictions.jsonl` 记每次预测 + `decided_timers`;`backtest` 算 MAE + 短循环/过度除湿 outcome。

## 5. 关键 gotcha(踩过的坑)

1. **command_line 传感器 entity_id 是拼音**:无法在 YAML 声明 default_entity_id,`migrate_entity_registry.py` 只改 template 平台 → 曾导致 `sensor.smart_dehumidifier_ml_engine` 不存在、47 处引用失效、takeover 长期空转。已加 `EXTRA_MAP` 修复。**新增 command_line 传感器要被固定 id 引用时,必须加进 EXTRA_MAP。**
2. `ml_core.py` 与 `tools/smart_dehumidifier_ml.py` **几乎相同但有分叉**:前者多 `compute_predictions` 的 `learned_*_override` 参数 + `predictor` 键。改引擎逻辑要同步两处。
3. 预测时间展示传感器要用正则守卫只认 `HH:MM`,否则中文状态文案会让 `today_at()` 崩(已修 next_start_time)。
4. 仿真里 drop_rate 方差小 → 模型对 drop 仅持平、对 rebound 提升明显(~24%)。这是数据特性,非 bug。
5. **entity_id 稳定性**:集成实体用 `has_entity_name=True`,entity_id 由设备名(及所在区域)派生。本机设备在区域「奥克兰」→ 后加的数值湿度实体被命名为 `sensor.ao_ke_lan_smart_dehumidifier_humidity`(早建的 3 个核心实体无前缀)。**打包仪表盘 `lovelace/dashboard.yaml` 因此只依赖 3 个稳定核心实体**(湿度走 gauge 的 `attribute: current_humidity`,不引用易变的数值湿度实体)。若要彻底稳:把实体改成 `has_entity_name=False` + 显式 `_attr_name`(object_id 不含设备/区域前缀)——属待办,改动会影响命名,需评估。

## 5b. 插件可控实体(2026-06-13 新增)+ 仪表盘现状

插件现暴露:`sensor.*_learning_engine`/`*_model_status`/`*_humidity`、`switch.*_autonomous_control`、
**`number.*_target_humidity`(目标湿度)**、**`select.*_operating_mode`(节能/舒适/干衣)**、服务 `train_model`。
number/select 写入 config entry options → 触发重载;运行模式已接进引擎(`PredictionContext.mode`)。

仪表盘 `ui-smart-dehumidifier-plugin.yaml`(= 侧栏「智能除湿引擎」,**不归 deploy 脚本管**,改完
`cp` 到 live + 浏览器硬刷新)目前是**卧室"抽湿机"视图的逐行复制**,仍引用 YAML 系统实体
(`input_number.bedroom_humidity_comfort_target`、`input_select.smart_dehumidifier_mode`、
`sensor.smart_dehumidifier_water_*`、众多 `sensor.humidity_*` 模板)。在本机全可用、外观=卧室;
**对外分发需把这些引用改到插件实体**(目标→number、模式→select、湿度→learning_engine 属性、
水箱无插件等价物需新增 sensor 或删卡)。

✅ **entity_id 已确定化(2026-06-13)**:所有插件实体改为 `has_entity_name=False` + 固定 ASCII
`_attr_name`,entity_id 与设备/区域无关、对所有用户一致:
`sensor.smart_dehumidifier_{learning_engine,model_status,humidity}`、
`switch.smart_dehumidifier_autonomous_control`、`number.smart_dehumidifier_target_humidity`、
`select.smart_dehumidifier_operating_mode`。本机原带 `ao_ke_lan_` 前缀的 3 个已在注册表改回干净 id。

✅ **「智能除湿引擎」仪表盘已改为全插件实体 + 零配置兜底**:`ui-smart-dehumidifier-plugin.yaml`
现只引用上述 6 个插件实体(0 处 YAML 系统实体),每字段带 fallback,仅装插件即可正常显示/控制
(目标湿度 number、运行模式 select、自主控制 switch、训练服务、湿度变色大屏 + 趋势)。
依赖 HACS:Mushroom + button-card(repair 提示)。

## 6. 剩余路线(→ HA 质量等级 silver/gold)

- [ ] `tests/` + GitHub CI(pytest + HA 测试框架,覆盖 config flow / coordinator / model);`hassfest`、`ruff`/`mypy`。
- [ ] `hacs.json` + 仓库元数据 → HACS 一键安装。
- [ ] 把外部环境(天气/室外湿度/在家/HVAC)**并入集成**(目前只在 YAML 系统)。
- [ ] 能耗 / 电费统计(kWh + 电价 + 每日除水量)。
- [ ] 打包仪表盘卡片 / Lovelace strategy(免手抄 YAML)。
- [ ] 多设备充分测试;跨重启恢复 coordinator 事件检测状态;config entry 迁移。
- [ ] 从设备属性自动取湿度已做(`current_humidity` 兜底);可扩展到从 humidifier 实体读 target。

## 7. 记忆/文档索引

- 本文件(总入口)。
- `custom_components/smart_dehumidifier/README.md`(集成安装 + 版本特性)。
- `tools/ML_ROADMAP.md`(离线工具、outcome 定义、真 ML 接入路径)。
- Claude 记忆:`memory/dehumidifier-ml-feedback-loop.md`、`memory/ha-config-file-layout.md`。
