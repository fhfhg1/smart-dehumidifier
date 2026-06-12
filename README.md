# Smart Dehumidifier

一个面向 Home Assistant 的智能除湿插件，包含：

- 自定义集成：`custom_components/smart_dehumidifier/`
- 对应前端界面：`ui-smart-dehumidifier-plugin.yaml`

当前这套方案可以只依赖：

- 插件本体
- `Mushroom`
- `button-card`
- `ui-smart-dehumidifier-plugin.yaml` 这份 Lovelace 界面

不再依赖原先卧室专用的 YAML 自动化、helper 和模板实体。

## 安装

1. 将 `custom_components/smart_dehumidifier/` 复制到你的 Home Assistant `config/custom_components/` 目录。
2. 重启 Home Assistant。
3. 在 `设置 → 设备与服务` 中添加 `Smart Dehumidifier`。
4. 选择湿度传感器、除湿机实体、目标湿度。
5. 如果没有合适的温度传感器，可以留空。

## 前端界面

主界面文件：

- `ui-smart-dehumidifier-plugin.yaml`

这是当前推荐使用的“智能除湿引擎”界面。

另外插件内也提供了一个可复用卡片模板：

- `custom_components/smart_dehumidifier/lovelace/dashboard.yaml`

## 前端依赖

当前界面依赖以下 HACS 前端卡片：

- Mushroom
- button-card

## 当前控制基线

在没有积累足够样本时，插件开启“自主控制”后会先按保守规则运行：

- 白天：目标湿度 +4%
- 夜间（23:30-08:00）：目标湿度 +7%
- 低湿保护：湿度低于目标 -5% 时立即停机

也就是说，它是：

**规则兜底 + 学习增强**

而不是一上来就完全交给模型。

## 仓库说明

这个仓库当前重点包含：

- 插件代码
- 插件前端界面
- 交接说明 `HANDOFF.md`

更完整的卧室旧系统、历史 YAML 试验文件和本地运行环境文件不属于当前发布范围。
