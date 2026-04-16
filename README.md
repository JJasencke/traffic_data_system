# Traffic Data System

交通与天气流式处理系统（单轨重构版）。

- 采集：百度天气/路况 API
- 消息：Kafka
- 计算：Spark Structured Streaming
- 存储：HDFS（可映射 Hive）
- 预测：SARIMA 任务入口已预留

## 快速开始（WSL）

1. 初始化配置

```bash
cp .env.example .env
```

2. 关键字段（至少）
- `BAIDU_AK`
- `PROJECT_ROOT`
- `ROAD_LIST_FILE`
- `KAFKA_BOOTSTRAP_SERVERS`

3. 一键拉起全流程（ZK/Kafka + 5 运行终端）

```bash
bash scripts/start_stack.sh
```

4. 一键停止

```bash
bash scripts/stop_stack.sh
```

详细说明见：`docs/README.md`

## 项目结构

```text
src/
  core/
  ingestion/
  messaging/
  streaming/
  prediction/
scripts/
  start_stack.sh
  stop_stack.sh
  attach_stack.sh
```
