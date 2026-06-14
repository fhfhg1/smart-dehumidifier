# 智能除湿机:从启发式到机器学习 的路线 + 交接说明

面向接手者(codex / 后续的 Claude）。当前系统**不是真 ML**,是自适应启发式
(EWMA + 手写 if/else),已具备"动态 + 规则兜底"。本文件记录现状、已完成的
Step 0、以及把"倒计时类参数"学出来的设计。

## 现状架构

```
HA 事件 → learning_log.py → smart_dehumidifier_learning.csv
   → ml.py sync → runs/rebounds/snapshots.jsonl + ml_output.json (+ predictions.jsonl)
   → sensor.smart_dehumidifier_ml_engine → 自动化用 control_takeover 决定是否采用
```

- 所有可调参数集中在 `smart_dehumidifier_ml.py` 顶部常量:`EWMA_DECAY_SPAN`、
  `MIN_SAMPLES_PER_GROUP`、`BLEND_RECENT_WEIGHTS`、`CONF_*`、`TIMER_KEYS`、
  `BACKTEST_MATCH_HORIZON_MIN`。
- shell_command 直接跑仓库里的 `tools/*.py`,**改 .py 即时生效,无需 deploy**;
  只有改 `staging/*.yaml` 才需 `./deploy-staging.sh`。

## 9 个倒计时/时长参数(TIMER_KEYS)

均已"引擎计算 + control_takeover 接管 + HA 端 `| int(默认)` 兜底"。目前是启发式:

| 参数 | 含义 | 兜底默认 |
|---|---|---|
| low_protect_confirm_minutes | 低湿保护:连续低于阈值多久才停 | 5 min |
| critical_low_confirm_seconds | 极低湿:多久强停 | 30 s |
| lockout_minutes | 低湿停机后锁定多久不重启 | 45 min |
| min_runtime_minutes | 最小运行时长 | 30 min |
| start_confirm_minutes / stop_confirm_minutes | 开/关机确认 | 5 / 5 min |
| early_stop_guard_minutes | 提前关机观察期 | 15 min |
| lock_release_delta | 跳过锁定所需的湿度差 | 10 |
| auto_start_window | 提前开机窗口 | 20 min |

## Step 0 已完成(反馈闭环,2026-06-12)

1. 魔法数字抽成常量(见上)。
2. `sync --predictions <jsonl>`:每次决策追加一条记录,含绝对时间戳的
   `predicted_stop_at` / `predicted_next_start_at`,以及 `decided_timers`
   (本次每个 TIMER_KEYS 的取值)。已在 staging 配置接好,落到
   `config/smart_dehumidifier_predictions.jsonl`。
3. `backtest --predictions <jsonl> --runs <jsonl> [--json]`:把预测停/开机时间与
   真实事件(runs 的 stop 事件,开机时刻 = stop - duration_min)配对,算 MAE / 偏差,
   按置信度分组。

**当前数据量极小(个位数 runs),先靠这套攒带标签的数据,别急着上模型。**

## ⚠️ 重要修复(2026-06-12):control_takeover 曾长期失效

`sensor.smart_dehumidifier_ml_engine` 与新加的 `sensor.smart_dehumidifier_learning_stats`
都是 **command_line 平台**,而 command_line 传感器**无法在 YAML 声明 default_entity_id**
(HA 不接受该键),`migrate_entity_registry.py` 又只重命名 template 平台 → 这两个实体的
entity_id 一直是中文名生成的拼音(如 `sensor.zhi_neng_chu_shi_qing_liang_xue_xi_yin_qing`)。
但 automations/模板里有 47 处写的是 `sensor.smart_dehumidifier_ml_engine`,指向不存在的
实体 → `state_attr(...)` 恒为 None → `control_takeover` 恒为 {} → **所有 ML 参数一直在
吃规则默认值,引擎算的值从未真正接管控制**。

修复:`migrate_entity_registry.py` 增加 `EXTRA_MAP`(command_line unique_id → 目标
entity_id),deploy 时(HA 停机)一并改名。**以后新增任何 command_line 传感器,若要被
automations 用固定 id 引用,必须把它的 unique_id→entity_id 加进 EXTRA_MAP。**
副作用:此前休眠的 takeover 现已激活(值有 clamp/gate/规则兜底,安全)。

## 第三批(2026-06-12):真 ML 原型 + 外部环境控制

### ① 真 ML 原型 `tools/train_dehumidifier_model.py`(纯 numpy 岭回归)
- `compare --runs --rebounds [--alpha --kfold]`:交叉验证对比 岭回归 vs 启发式EWMA 的
  速率预测 MAE。30 天仿真上结果:**rebound_rate 岭回归比启发式准 ~24%**,drop_rate 持平
  (仿真里 drop 方差小)。rebound 正是驱动"提前开机"时机的量,改进点对得上之前 +31min 偏差。
- `train --runs --rebounds --out model.json`:全量拟合并保存(系数+特征规格+CV)。
- 推理:`predict_rate(model_json, name, sample)` 可被引擎加载(无需 sklearn)。
- **未来接入路径**(待真实数据 ~50+/场景):在 `compute_predictions` 里,当 model.json 存在
  且该 rate 已训练时,用 `predict_rate` 替代 `learned_drop/learned_rebound` 的 EWMA 值;
  `control_takeover` 改由模型 CV 误差/不确定度驱动(替代纯样本量门槛)。先 compare 验证再接。

### ② 外部环境智能控制(已上线,可降级)
- `compute_external_advice(context)` 产出 `external_start_bias`(开机阈值偏置,正=更不爱开机)
  + `external_advice`(中文)。规则**仅在对应传感器可用时生效**:开窗 +10(基本抑制开机)、
  空调制冷 +2、室外湿度≥85 −1、当前降雨 −1、无人在家且高湿 −1。clamp [−3,+10]。
- 输入经 sync 新增参数 `--window-open/--presence/--hvac/--outdoor-humidity/--rainy`,由
  `shell_command.smart_dehumidifier_ml_sync` 从对应传感器注入(没配 override → unknown → 不生效)。
- 消费点:`sensor.smart_dehumidifier_day/night_start_threshold` = target + offset + **bias**
  (bias 默认 0,无外部数据零影响)。`external_advice`/`external_start_bias` 已加进
  `sensor.smart_dehumidifier_ml_engine` 属性,可在 more-info 查看。
- 待增强(codex):真窗磁→暂停正在运行的机器(目前只影响开机阈值);天气**预报**前瞻
  (现仅用当前降雨);presence/HVAC 的更细策略;把 external_advice 显示到仪表盘卡片。

## 已完成的工具(2026-06-12 第二批)

- **物理仿真器 `tools/simulate_dehumidifier.py`**:逐分钟湿度物理模型 + 阈值控制,
  生成与学习日志同格式的 CSV。`--replay-predictions <jsonl>` 会在生成数据上重放
  引擎,产出 predictions.jsonl,使 backtest 能**端到端验证**。
  例:`simulate_dehumidifier.py --out sim.csv --days 14 --replay-predictions pred.jsonl`
  → `ml.py sync --source sim.csv ...` → `ml.py backtest` / `stats`。
- **`ml.py stats --runs --rebounds --snapshots [--json]`**:数据健康度(每场景/模式
  样本数、速率/时长分布、CV、短循环/过度除湿次数、`ready_for_model`=任一场景≥50)。
  已接成只读 command_line 传感器 `sensor.smart_dehumidifier_learning_stats`(scan 600s)。
- **`ml.py backtest` 新增 outcome 段**:`compute_outcomes(runs)` 算短循环
  (`SHORT_CYCLE_MIN`)、过度除湿(`OVERDRY_MARGIN`)、中位周期间隔。
- **单测 `tests/test_smart_dehumidifier_ml.py`**(stdlib unittest,无依赖):
  `<venv>/bin/python -m unittest tests.test_smart_dehumidifier_ml`。重构前必须保持绿。

### 用合成数据已暴露的现象(14 天仿真,供 codex 参考方向)

- 启发式**开机预测系统性偏早 ~+31 分钟**(stop 预测 MAE≈8.7,start MAE≈37.6)。
- **置信度与准确度不相关**(high 档 MAE 反而比 low 档差)——`classify_confidence`
  现在只看样本量+CV,应改为由 backtest 实测误差驱动。这是换模型的首要动机。

## 下一步:让 TIMER_KEYS 可学习(待 codex / 数据足够后做)

核心缺口:每个倒计时需要一个**结果好坏(outcome)**信号,否则无法学。建议:

- **low_protect_confirm_minutes / lockout_minutes**:停机后若湿度在 `lockout_minutes`
  内重新越过启动阈值 → 短循环(等太短/锁太短,惩罚);若停机时 `end_h` 明显低于
  `target`(如 < target-5) → 过度除湿(等太长,惩罚)。理想值最小化两类惩罚之和。
- **min_runtime_minutes**:运行时长 < 它却已达标 → 可缩短;频繁因 early_stop_guard
  被打断 → 可调。
- **auto_start_window / start_confirm_minutes**:预测开机 vs 实际开机的 MAE(已可由
  backtest 得到)即其 outcome;偏差系统性为正/负 → 调窗口。

落地方式(推荐,数据 ~50+/场景 后):
1. 在 backtest 里为每个 timer 增加 outcome 计算(用 predictions.decided_timers +
   后续 runs/rebounds 还原结果),先**离线评估**当前启发式选值好不好。
2. 再把 `compute_predictions` 里某个 timer 的启发式块,替换为"按场景/模式分桶的
   贝叶斯/岭回归预测最优等待时长 + 不确定度",`control_takeover[该key]` 改为由
   **模型不确定度**驱动(不确定就回退现有启发式 → 再回退 HA 规则默认)。
3. 一次只换一个 timer,用 backtest 对比"换之前/之后"的 outcome,确认变好再上下一个。

**安全底线**:无论怎么学,HA 端的 `| int(默认值)` 和 clamp 边界必须保留,模型只在
栏杆内活动;严禁让模型直接控制开关,只产出参数供规则引擎消费。
