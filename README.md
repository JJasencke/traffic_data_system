# Traffic Data System

交通与天气流式处理系统（单轨重构版）。

- 采集：百度天气/路况 API
- 消息：Kafka
- 计算：Spark Structured Streaming
- 存储：HDFS（可映射 Hive）
- 预测：SARIMA run-once 任务已实现

## 快速开始（WSL）

1. 初始化配置

```bash
cp .env.example .env
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

3. 在 `.env` 至少填写：
- `BAIDU_AK`
- `PROJECT_ROOT`
- `ROAD_LIST_FILE`
- `KAFKA_BOOTSTRAP_SERVERS`

4. 一键启动全链路

```bash
bash scripts/start_stack.sh
```

说明：该脚本会拉起 `tmux` 会话并自动附着（进入会话界面）。

5. 一键链路检查

```bash
bash scripts/check_stack.sh
```

6. 一键停止

```bash
bash scripts/stop_stack.sh
```

## 脚本入口

```text
scripts/
  start_stack.sh   # 启动全栈（ZK/Kafka + collector + 3个流任务 + 可选监控窗）
  start_stack_ide.sh # IDE 观测模式启动（不刷日志，显示状态面板）
  attach_stack.sh  # 重新连接会话
  check_stack.sh   # 全链路冒烟检查
  run_prediction_once.sh # 单次执行SARIMA预测作业
  stack_dashboard.py # 全链路状态仪表盘（单次/持续刷新）
  stop_stack.sh    # 停止会话和进程
```

## 常用附加命令

```bash
# IDE 一键启动 + 可视化状态面板
bash scripts/start_stack_ide.sh

# 如果你手动 detach 了 tmux 会话
bash scripts/attach_stack.sh

# 仅告警不失败（适合刚启动、尚未产出数据时）
STRICT=false bash scripts/check_stack.sh

# 单独运行状态面板
python scripts/stack_dashboard.py --watch

# 单次执行SARIMA预测作业（run-once）
bash scripts/run_prediction_once.sh
```

## 项目结构

```text
src/
  core/
  ingestion/
  messaging/
  streaming/
  prediction/
```

详细运行说明见 `docs/README.md`。
