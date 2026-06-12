# Smart Dehumidifier 智能除湿机自学习集成

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://github.com/hacs/integration)

一个 Home Assistant 自定义集成:**进程内运行的自学习除湿引擎**——观察湿度与除湿机的开停,
学习你家这间屋子的「降湿速率 / 回潮速率」,并据此动态决定开机阈值、最小运行时长、低湿保护、
锁定时长等一整套倒计时参数;可选**接管设备开关**(带硬安全限位),也支持**露点/防霉目标**。
**数据全部本地训练与存储,不出户。**

## 特性
- 进程内引擎(无 shell_command)、传感器掉线安全回退、跨重启不丢学习进度。
- 本地 numpy 岭回归模型,版本化保存,**仅当交叉验证优于规则启发式时才接管预测**。
- UI 配置流程(config flow)+ 选项(目标湿度 / 更新周期 / 防霉模式 / 是否启用模型)。
- 自动匹配湿度/温度/除湿机实体;湿度也可从除湿机 `current_humidity` 属性读取。
- 实体:学习引擎、模型状态、湿度(数值)、自主控制开关、目标湿度(number)、运行模式(select)。
- 服务 `smart_dehumidifier.train_model` 手动训练;i18n(中/英)。

## 安装(HACS 自定义仓库)
1. HACS → 右上角菜单 → **Custom repositories**,地址填 `https://github.com/fhfhg1/smart-dehumidifier`,类别选 **Integration**。
2. 在 HACS 里搜索 **Smart Dehumidifier** 安装 → 重启 Home Assistant。
3. **设置 → 设备与服务 → 添加集成** → 搜 “Smart Dehumidifier” → 选湿度传感器 + 除湿机(switch/fan/humidifier/climate)+(可选)温度 + 目标湿度。

## 仪表盘(可选,开箱即用)
`custom_components/smart_dehumidifier/lovelace/dashboard.yaml` 是一张配套界面(湿度变色大屏 + 控制 +
预测 + 趋势),只引用本集成实体。需要 HACS 前端卡 **Mushroom** 与 **button-card**(集成会在
「设置 → 修复」里提示)。用法:仪表盘 → 编辑 → 添加卡片 → 手动 → 粘贴该文件内容。

## 工作原理(简述)
- **倒计时参数**:基准(按模式)→ 按场景 + 学到的速率 + 置信度增减 → clamp 安全区间 →
  样本足够才接管,否则用规则默认。
- **学习素材**来自完整的「开机→关机」循环(降湿率)与关机后的回潮采样(回潮率);
  环境干燥、设备不运行时自然无新样本,这是正常的。

## 许可
MIT
