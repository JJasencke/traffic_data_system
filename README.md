# Traffic Data System

交通与天气流式处理系统（单轨重构版）。

- 采集：百度天气/路况 API
- 消息：Kafka
- 计算：Spark Structured Streaming
- 存储：HDFS（可映射 Hive）
- 预测：SARIMA 任务入口已预留

## 快速开始

1. 配置环境变量

```bash
cp .env.example .env
```

2. 填写 `.env` 中至少以下字段：
- `BAIDU_AK`
- `PROJECT_ROOT`
- `ROAD_LIST_FILE`
- `KAFKA_BOOTSTRAP_SERVERS`

3. 详细运行说明见：`docs/README.md`

## 项目结构

```text
src/
  core/
    config.py
  ingestion/
    adapters/baidu_adapters.py
    clients/baidu_api_client.py
    repositories/road_repository.py
    services/collector_service.py
    jobs/run_collector_job.py
  messaging/
    kafka_producer.py
  streaming/
    runtime.py
    schemas.py
    jobs/
      traffic_detail_stream_job.py
      weather_detail_stream_job.py
      traffic_avg_speed_stream_job.py
  prediction/
    jobs/sarima_predict_job.py
```
