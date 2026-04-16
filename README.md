## 推荐关闭顺序

```
kafka-server-stop.sh
zkServer.sh stop
hive --service metastore >/dev/null 2>&1
stop-yarn.sh
stop-dfs.sh
```

HiveServer2 如果你是 `nohup` 起的，可以这样找进程再停：

```
jps
kill -9 进程号
```

### 第一次初始化时才需要做

- 创建 topic
- 调整配置
- 测试 Kafka 命令行

**topic 不用每次重建。**
 Kafka 会把 topic 元数据和消息保存在它的数据目录里，只要你没删数据目录，topic 会一直在。Kafka 的事件和 topic 是持久化保存的，不是每次开机都清空。

# 启动流程

## 1. 开 WSL

```
wsl
###  先启动 SSH
sudo service ssh start
```

## 2. 启动 Hadoop

```
cd /mnt/d/bigdata/apps/hadoop
sbin/start-dfs.sh
sbin/start-yarn.sh
mapred --daemon start historyserver
```

## 3. 检查进程

```
jps
```

关闭hadoop

```
sbin/stop-yarn.sh 
sbin/stop-dfs.sh
```



## 4. 启动 Kafka

终端1：

```
cd /mnt/d/bigdata/apps/kafka
bin/zookeeper-server-start.sh config/zookeeper.properties
```

终端2：

```
cd /mnt/d/bigdata/apps/kafka
bin/kafka-server-start.sh config/server.properties
```

## 5. 启动 Spark 或 Hive

Spark：

```
cd /mnt/d/bigdata/apps/spark
bin/pyspark
```

Hive：

```
cd /mnt/d/bigdata/apps/hive
```

然后再启动：

```
nohup hive --service metastore > logs/metastore.log 2>&1 &
nohup hiveserver2 > logs/hiveserver2.log 2>&1 &
```

然后等 3 到 5 秒，检查：

```
jps
ss -lntp | grep 10000
```

再看日志尾部：

```
tail -n 50 logs/metastore.log
tail -n 50 logs/hiveserver2.log
```

## 终端 3

自动采集程序

```
cd /mnt/d/bigdata/myproject
source .venv/bin/activate
export PYTHONPATH=src
python src/run_collector.py
```

然后分别再开 3 个终端启动 Spark 流任务。

------

## 终端 4：交通明细流

```
cd /mnt/d/bigdata/apps/spark
bin/spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
  /mnt/d/bigdata/myproject/src/streaming/traffic_detail_stream_job.py
```

## 终端 5：天气明细流

```
cd /mnt/d/bigdata/apps/spark
bin/spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
  /mnt/d/bigdata/myproject/src/streaming/weather_detail_stream_job.py
```

## 终端 6：平均速度聚合流

```
cd /mnt/d/bigdata/apps/spark
bin/spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
  /mnt/d/bigdata/myproject/src/streaming/traffic_avg_speed_stream_job.py
```







# 一、先定最顶层目标：你的毕设最后到底要产出什么

你的题目是：

**基于流批一体化的哈尔滨市道路交通分析与预测系统**

这意味着最终成果不能只是“能采数据”，也不能只是“做个预测模型”，而应该是一个完整闭环：

## 最终要展示的成果

1. **实时交通状态**
   - 当前重点道路的路况情况
   - 当前天气情况
   - 当前拥堵程度
2. **历史交通分析**
   - 某条道路在一段时间内的平均速度变化
   - 不同时间段拥堵情况统计
   - 天气与交通状态之间的关系
3. **短时交通预测**
   - 对未来 10 分钟某条重点道路平均速度进行预测
   - 给出预测结果和误差分析
4. **地图可视化**
   - 在哈尔滨地图上展示重点道路当前状态
   - 展示历史统计结果或预测结果

所以你的系统最终不是为了“存数据”，而是为了服务这 4 类结果。

------

# 二、从最终结果往前倒推：每一部分需要什么数据

------

## 1. 地图可视化最后要展示什么

地图页面后面最可能要展示的是：

### 实时图层

- 道路名称
- 道路当前速度
- 拥堵等级
- 拥堵趋势
- 天气情况
- 采集时间

### 历史图层/历史页面

- 某条道路一段时间内平均速度折线图
- 某时段拥堵等级分布
- 天气变化与速度变化关系

### 预测图层/预测页面

- 某条道路未来 10 分钟预测速度
- 历史真实值与预测值对比

------

## 地图可视化因此需要什么字段

至少需要：

### 道路身份字段

- `road_name`
- 最好还有 `road_id`（后面可自定义）
- 所属城市 `city`
- 可能还需要 `district`

### 空间位置字段

这是后面地图一定会需要的：

- `longitude`
- `latitude`
- 或者 `polyline`
- 或者至少一个你自己维护的“道路坐标表”

注意：
**百度实时路况接口返回里不一定直接给你整条道路的地图几何信息。**
所以后面地图部分大概率要额外维护一张：

**重点道路基础信息表**

例如：

| road_name | city | district | center_lng | center_lat |
| --------- | ---- | -------- | ---------- | ---------- |
|           |      |          |            |            |

这张表可以人工整理，毕设完全可行。

### 实时交通字段

- `speed`
- `section_status`
- `congestion_distance`
- `congestion_trend`
- `description`
- `collect_time`

### 天气字段

- `weather_text`
- `temperature`
- `humidity`
- `wind_dir`
- `visibility`

------

# 三、预测模块到底需要什么数据

现在倒推 SARIMA。

你前面说得对，SARIMA 不是凭空跑出来的，它一定需要一个**稳定的时间序列**。

------

## 1. SARIMA 最适合预测什么

对你这个项目，最适合预测的目标不是太多，而是选一个最核心的：

### 推荐预测目标

**某条重点道路未来 10 分钟的平均速度**

为什么选它：

- 你已经能从路况数据中拿到 `speed`
- 速度是连续变量，适合时间序列建模
- 比“拥堵等级”更适合 SARIMA
- 论文里容易解释，也容易评估误差

------

## 2. SARIMA 需要的数据形式

SARIMA 需要的不是杂乱 JSON，而是这种结构：

| timestamp        | road_name | avg_speed |
| ---------------- | --------- | --------- |
| 2026-03-08 10:00 | 中山路    | 15.2      |
| 2026-03-08 10:01 | 中山路    | 14.8      |
| 2026-03-08 10:02 | 中山路    | 13.6      |

也就是说：

### 预测模块真正需要的是

- **按固定时间间隔组织的单路段速度序列**

所以前面实时流处理、HDFS、Hive，都必须围绕这个结果服务。

------

## 3. SARIMA 因此倒逼前面必须准备哪些字段

为了支持 SARIMA，历史数据中至少要有：

- `collect_time`
- `road_name`
- `speed` 或 `avg_speed`

更理想一点，可以额外保留：

- `weather_text`
- `temperature`
- `section_status`

虽然 SARIMA 本身主要用时间序列值，但这些辅助字段可以帮助你做后续分析，比如：

- 雪天时模型表现是否变差
- 拥堵等级和速度是否一致

------

# 四、历史分析模块需要什么

你论文里不可能只展示预测，你还需要展示“分析”。

所以历史分析部分要回答的问题通常是：

- 哪些路平时最拥堵
- 哪些时间段最慢
- 天气对速度有没有影响
- 某条道路一天内速度如何变化

------

## 历史分析因此需要哪些指标

### 路段基础统计

- 平均速度
- 最小速度
- 最大速度
- 速度标准差

### 拥堵统计

- 不同拥堵等级出现次数
- 拥堵趋势统计
- 平均拥堵距离

### 时间统计

- 按小时平均速度
- 按天平均速度
- 工作日/周末对比（后面可扩展）

### 天气关联统计

- 不同天气下的平均速度
- 温度与速度关系（可选）

------

## 历史分析因此倒逼前面要保留哪些字段

- `collect_time`
- `road_name`
- `speed`
- `section_status`
- `congestion_distance`
- `congestion_trend`
- `weather_text`
- `temperature`

这几个字段就是你后面 HDFS / Hive 里最值得保留的核心字段。

------

# 五、再往前倒推：Hive 到底要承接什么

Hive 不是单纯为了“建个表”，而是为了支持：

1. 历史查询
2. 统计分析
3. 预测数据提取

所以 Hive 中的数据结构一定要围绕后面要查什么来设计。

------

## Hive 至少要有哪几类表

我建议后面最终形成 3 类表，而不是一张表包打天下。

### 1. 交通明细历史表

记录每次采集/每条路的原始核心交通状态

用途：

- 最基础的历史数据仓库
- 后面做各种统计和预测的原材料

建议字段：

- `collect_time`
- `dt`
- `road_name`
- `city`
- `speed`
- `section_status`
- `congestion_distance`
- `congestion_trend`
- `description`

### 2. 天气历史表

记录每次采集时的天气状态

用途：

- 天气分析
- 与交通数据关联分析

建议字段：

- `collect_time`
- `dt`
- `city`
- `district`
- `weather_text`
- `temperature`
- `humidity`
- `wind_dir`
- `visibility`

### 3. 交通聚合统计表

记录 Spark 实时计算出的聚合结果，比如平均速度

用途：

- 直接供可视化和预测提取使用
- 减少后面再聚合成本

建议字段：

- `collect_time`
- `dt`
- `road_name`
- `avg_speed`

------

# 六、再往前倒推：Spark 实时处理到底要做什么

现在倒推到 Spark。

Spark 不是为了“把 Kafka 读出来就结束”，而是要把数据变成后面能用的历史结果。

------

## Spark 模块的职责应该分成两层

### 第一层：明细清洗层

从 Kafka 里读出统一 JSON 后：

- 解析 schema
- 去空值
- 过滤异常速度
- 形成标准交通明细流
- 形成标准天气明细流

### 第二层：实时聚合层

在明细流基础上计算：

- 每条道路平均速度
- 拥堵等级数量
- 趋势数量

------

## Spark 最终应输出什么

### 输出 1：交通明细历史数据

给 HDFS/Hive 使用
字段示例：

- `collect_time`
- `road_name`
- `speed`
- `section_status`
- `congestion_distance`
- `congestion_trend`

### 输出 2：天气历史数据

给 HDFS/Hive 使用
字段示例：

- `collect_time`
- `weather_text`
- `temperature`
- `humidity`

### 输出 3：实时聚合速度数据

给 HDFS/Hive / 预测 / 可视化使用
字段示例：

- `collect_time`
- `road_name`
- `avg_speed`

------

## 所以 Spark 的核心不是一个脚本，而是三个输出方向

你现在已经做了一个“平均速度聚合输出”，后面完善后可以变成：

1. `traffic_detail_stream`
2. `weather_detail_stream`
3. `traffic_avg_speed_stream`

------

# 七、再往前倒推：Kafka 应该承接什么

Kafka 是实时数据接入层，不是长期分析层。
它的作用是“承上启下”。

------

## Kafka 的职责

### 对上游

承接 Python 采集程序发来的统一结构消息

### 对下游

给 Spark 提供稳定、可缓冲、可解耦的实时输入

------

## Kafka topic 最好怎么设计

你现在的数据源只有两个，所以 topic 非常清晰：

### 1. `traffic_raw`

存统一后的路况消息

### 2. `weather_raw`

存统一后的天气消息

后面如果想扩展，也可以加：

- `traffic_avg_speed`
- 但当前阶段没必要

------

## Kafka 中消息里需要哪些字段

这里只保留后续真正要用的字段，不要把太多无用原始数据塞进去。

### `traffic_raw` 建议字段

- `source`
- `data_type`
- `collect_time`
- `city`
- `road_name`
- `description`
- `overall_status`
- `overall_status_desc`
- `section_desc`
- `road_type`
- `congestion_distance`
- `speed`
- `section_status`
- `congestion_trend`

### `weather_raw` 建议字段

- `source`
- `data_type`
- `collect_time`
- `country`
- `province`
- `city`
- `district`
- `district_id`
- `weather_text`
- `temperature`
- `humidity`
- `wind_class`
- `wind_dir`
- `visibility`
- `aqi`

------

# 八、再往前倒推：统一格式层为什么存在

这就是 adapters 的意义。

因为百度返回的是两种完全不同的 JSON：

- 天气 JSON
- 路况 JSON

如果不先做适配层，下游会出现问题：

- Kafka 发不同结构，难以消费
- Spark schema 很混乱
- Hive 字段难设计
- 后面换数据源很麻烦

------

## 统一格式层的职责

### 1. 提取真正有用的字段

不是全量搬运百度 JSON，而是提取后续要分析的字段。

### 2. 规范字段命名

比如：

- 天气统一叫 `weather_text`
- 速度统一叫 `speed`
- 时间统一叫 `collect_time`

### 3. 附加系统语义字段

比如：

- `source`
- `data_type`

------

## 统一格式层应当怎么设计

### 天气统一格式

```json
{
  "source": "baidu_weather",
  "data_type": "weather",
  "collect_time": "...",
  "city": "哈尔滨市",
  "district": "南岗区",
  "weather_text": "晴",
  "temperature": -7,
  "humidity": 47,
  "wind_dir": "东北风",
  "visibility": 30000
}
```

### 路况统一格式

```json
{
  "source": "baidu_traffic",
  "data_type": "traffic",
  "collect_time": "...",
  "city": "哈尔滨市",
  "road_name": "中山路",
  "speed": 13.6,
  "section_status": 3,
  "congestion_distance": 210,
  "congestion_trend": "加重",
  "description": "中山路：畅通；北向南,中山路附近拥堵。"
}
```

------

# 九、再往前倒推：原始采集层到底该采什么

现在回到最前面。

你已经确定只用两个 API：

- 天气
- 实时路况

并且只采哈尔滨重点道路。

------

## 原始采集层的职责

### 1. 定时调用 API

- 天气每轮采一次
- 路况遍历重点道路逐条采

### 2. 保留原始 JSON（适量）

为了调试和论文样例

### 3. 把原始 JSON 交给适配层处理

采集层本身不要掺杂太多业务逻辑

------

## 原始采集层现在要采哪些指标

### 天气原始 JSON 中建议提取

- `location.city`
- `location.name`
- `now.text`
- `now.temp`
- `now.rh`
- `now.wind_dir`
- `now.wind_class`
- `now.vis`
- `now.aqi`
- `now.uptime`

### 路况原始 JSON 中建议提取

- 顶层 `description`
- 顶层 `evaluation.status`
- 顶层 `evaluation.status_desc`
- `road_traffic[].road_name`
- `congestion_sections[].congestion_distance`
- `congestion_sections[].speed`
- `congestion_sections[].status`
- `congestion_sections[].congestion_trend`
- `congestion_sections[].section_desc`

------

# 十、现在把全流程串起来

现在我们把整个系统按“结果倒推”整理成一条完整逻辑。

------

## 目标层

### 最终输出

- 实时交通态势
- 历史统计分析
- 短时速度预测
- 地图可视化展示

------

## 分析层

### 需要的核心结果

- 每条重点道路的实时平均速度
- 拥堵等级统计
- 趋势统计
- 历史速度序列
- 天气与交通关联分析结果

------

## 存储层

### 需要的数据表

- `traffic_detail_history`
- `weather_history`
- `traffic_avg_speed_history`

这些表要能供 Hive 查询，供 SARIMA 取数，供地图可视化读数。

------

## 流处理层

### Spark 要做的事

- 消费 `traffic_raw` 和 `weather_raw`
- 解析 JSON
- 清洗
- 输出交通明细
- 输出天气明细
- 输出平均速度聚合结果

------

## 接入层

### Kafka 要做的事

- 承接采集程序发送的统一格式消息
- 解耦采集和 Spark
- 缓冲实时数据

------

## 适配层

### adapters 要做的事

- 百度天气 JSON → 统一天气结构
- 百度路况 JSON → 统一路况结构

------

## 采集层

### Python 自动采集要做的事

- 读取重点道路列表
- 调用天气 API
- 调用实时路况 API
- 保存必要原始 JSON
- 调用适配层
- 发 Kafka
- 周期性自动运行

------

# 十一、我们接下来每一步应该怎么推进

现在整个逻辑已经清楚了，后面每一步都应该服务这个总目标，而不是散着做。

------

## 第一步：规范数据字段和目录结构

要完成：

- 最终统一字段清单
- HDFS 历史目录规范
- Hive 表结构设计

## 第二步：自动化采集

要完成：

- 重点道路列表
- 自动采集循环
- 稳定写 Kafka

## 第三步：持续流处理和历史沉淀

要完成：

- Spark 自动消费
- 写 HDFS
- Hive 查询

## 第四步：积累历史数据

先跑 6 小时，再跑 24 小时

## 第五步：做 SARIMA 预测

从 Hive 或 HDFS 中抽取某条路的速度序列

## 第六步：地图可视化

读实时结果 + 历史结果 + 预测结果

------

# 十二、为了以后和我同步方便，我们应该统一几个“讨论对象”

后面你和我同步时，建议都围绕下面这几个固定对象说，这样不会乱。

### 1. 采集层对象

- road_list
- weather_raw
- traffic_raw

### 2. 统一结构对象

- weather_record
- traffic_record

### 3. 流处理对象

- traffic_detail_stream
- weather_detail_stream
- traffic_avg_speed_stream

### 4. HDFS/Hive对象

- traffic_detail_history
- weather_history
- traffic_avg_speed_history

### 5. 预测对象

- 单路段分钟级平均速度序列

### 6. 可视化对象

- 重点道路基础信息表
- 实时状态结果
- 预测结果

这样后面我们讨论每一步时，会非常清楚“我们现在在系统哪一层”。

------

# 十三、我给你的一个结论

你现在最正确的做法，不是再立即继续写新代码，而是先把下面三样东西固定下来：

1. **最终统一字段设计**
2. **HDFS/Hive 历史表设计**
3. **预测目标与可视化目标倒推出来的数据需求**

这三样定死后，后面每一步代码都不会偏。

------

接下来我建议我们就按这个顺序继续，不再跳着做：

**下一步先做：最终统一字段设计总表 + 各层输入输出对照表。**

你只要回我一句：

**继续做统一字段设计总表**

我就直接把整个系统里
“原始 JSON字段 → 统一字段 → Kafka字段 → Spark字段 → Hive字段 → 预测字段 → 可视化字段”
整理成一张完整的对照说明给你。







## 第一步：重启自动采集程序

因为旧 Kafka 里的消息还是空 `road_id` 的旧结构，先停掉旧的 `run_collector.py`，再重新启动：

```
cd /mnt/d/bigdata/myproject
source .venv/bin/activate
export PYTHONPATH=src
python src/run_collector.py
```

然后用 Kafka consumer 看一条新消息，确认：

```
"road_id": "RB001"
```

------

## 第二步：清理今天这批不完整的历史数据

因为你 HDFS 里已经写进去一批 `road_id=""` 的旧数据了。为了让今天这批正式数据干净，建议把今天的旧分区删掉再重跑。

先停掉对应 Spark 流任务，然后删目录：

```
cd /mnt/d/bigdata/apps/hadoop/bin
hdfs dfs -rm -r /traffic/history/traffic_detail
```

如果天气流也已经写过旧数据，同样删：

```
hdfs dfs -rm -r /traffic/history/weather
```

如果 avg_speed 有旧目录，也删掉：

```
hdfs dfs -rm -r /traffic/history/avg_speed
```

------

## 第三步：删除对应 Spark checkpoint

因为 checkpoint 会记住处理进度。你想让今天这批数据按新结构重新落盘，就把正式版 checkpoint 也删掉。

```
rm -rf /mnt/d/bigdata/myproject/checkpoints/traffic_detail
rm -rf /mnt/d/bigdata/myproject/checkpoints/weather
rm -rf /mnt/d/bigdata/myproject/checkpoints/avg_speed
```

------

## 第四步：重新启动正式版 Spark 流任务

先从最稳的 `traffic_detail_stream_job.py` 开始：

```
cd /mnt/d/bigdata/spark
bin/spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
  /mnt/d/bigdata/myproject/src/streaming/traffic_detail_stream_job.py
```

让自动采集程序跑一两轮后，检查：

```
cd /mnt/d/bigdata/hadoop/bin
hdfs dfs -ls /traffic/history/traffic_detail
hdfs dfs -cat /traffic/history/traffic_detail/dt=2026-03-18/part-*.csv
```



查看hdfs

```
 hdfs dfs -ls /
```



# 一、怎么看 Kafka / 明细数据（你现在最该会的）

你现在要看两类东西：

1. **Kafka 里的实时数据（源头）**
2. **HDFS 里的明细数据（Spark 处理后的结果）**

------

## ✅ 1. 看 Kafka 里的数据（最原始）

这是最重要的调试手段。

### 看交通数据

```
cd /mnt/d/bigdata/apps/kafka

bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic traffic_raw \
  --from-beginning
```

### 看天气数据

```
bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic weather_raw \
  --from-beginning
```

# 二、测试数据怎么删除（非常重要）

你后面一定会反复删数据，这里我给你**最标准、安全的删法**。

------

## ✅ 1. 删除 HDFS 数据（最常用）

### 删除某一天

```
cd /mnt/d/bigdata/apps/hadoop/bin

hdfs dfs -rm -r /traffic/history/avg_speed/dt=2026-03-19
hdfs dfs -rm -r /traffic/history/traffic_detail/dt=2026-03-19
hdfs dfs -rm -r /traffic/history/weather/dt=2026-03-19
```

------

### 删除整个表（测试阶段常用）

```
hdfs dfs -rm -r /traffic/history/avg_speed
hdfs dfs -rm -r /traffic/history/traffic_detail
hdfs dfs -rm -r /traffic/history/weather
```

------

## ✅ 2. 删除 Spark checkpoint（必须做！）

否则 Spark 会“记住进度”，导致：

👉 数据不重新写

```
rm -rf /mnt/d/bigdata/myproject/checkpoints/traffic_detail
rm -rf /mnt/d/bigdata/myproject/checkpoints/weather
rm -rf /mnt/d/bigdata/myproject/checkpoints/avg_speed
```





## 1. 交通明细字段说明表

例如：

- `road_id`：道路唯一标识
- `road_name`：道路名称
- `collect_time`：采集时间
- `overall_status`：接口整体路况等级
- `section_status`：拥堵片段等级
- `speed`：接口原始速度
- `congestion_distance`：拥堵距离
- `congestion_trend`：拥堵趋势

------

## 2. 天气字段说明表

例如：

- `weather_text`
- `temperature`
- `humidity`
- `wind_dir`
- `visibility`
- `aqi`

------

## 3. 聚合速度字段说明表

例如：

- `minute_bucket`：分钟级时间桶
- `avg_speed`：该道路该分钟最终平均速度
- `record_count`：本分钟记录数
- `api_count`：真实 API 速度记录数
- `estimated_count`：规则估算速度记录数

## 更准确的版本

- 采集程序负责调用实时路况和天气 API。
- 原始 API 返回的 JSON 先在 `adapters.py` 中完成字段统一，再在 `collector.py` 中补充采集时间、道路元数据等系统字段。
- 标准化后的记录被作为 Kafka producer 的消息发送到 `traffic_raw` 和 `weather_raw` 两个 topic 中。
- Kafka 作为消息中间件，负责缓存和解耦采集端与计算端。
- Spark Streaming 从 Kafka 中实时消费这些 JSON 消息，先通过 schema 将其解析为结构化 DataFrame，再分别生成交通明细、天气明细和分钟级平均速度聚合结果。
- 这些结构化结果被写入 HDFS 历史目录。
- Hive 再将 HDFS 上的文件映射成表，支持后续 SQL 查询、历史分析和模型取数。
- SARIMA 则基于某条道路的分钟级 `avg_speed` 序列进行短时交通速度预测。

好，我们继续。
下面我直接给你一版**可以拿去写论文 / 做答辩的系统架构说明**，我会分成 4 部分：

1. **系统总体架构说明**
2. **各模块职责表**
3. **数据流转过程说明**
4. **SARIMA 预测模块该怎么接入整个系统**

你可以把这版当成你毕设的“系统设计主线”。

------

# 一、系统总体架构说明

你的项目可以概括为一个典型的**流批一体化交通分析与预测系统**。

## 1. 系统目标

系统面向哈尔滨市重点道路，围绕实时交通路况和天气数据，完成以下任务：

- 实时采集道路交通与天气信息
- 对实时数据进行标准化与流式处理
- 将处理结果沉淀为历史数据
- 支持 Hive 查询与统计分析
- 基于分钟级速度序列进行短时交通预测
- 为后续地图可视化与图表展示提供数据支撑

------

## 2. 系统总体分层

你的系统可以分成 7 层：

### （1）数据源层

包括外部 API：

- 百度实时路况 API
- 百度天气 API

这一层负责提供原始实时数据。

------

### （2）采集与适配层

包括：

- `run_collector.py`
- `collector.py`
- `adapters.py`
- `road_list.csv`

这一层负责：

- 定时调用 API
- 对原始 JSON 做第一次字段统一
- 补充道路 ID、行政区、坐标、基准速度等元数据
- 形成系统内部统一记录

------

### （3）消息队列层

包括：

- Zookeeper
- Kafka
- Topic：`traffic_raw`、`weather_raw`

这一层负责：

- 接收采集程序发送的实时消息
- 对采集端和计算端进行解耦
- 缓冲实时流数据
- 支持多个下游 Spark 任务并行消费

------

### （4）流处理层

包括 3 个 Spark Streaming 作业：

- `traffic_detail_stream_job.py`
- `weather_detail_stream_job.py`
- `traffic_avg_speed_stream_job.py`

这一层负责：

- 解析 Kafka 中的 JSON 消息
- 转换为结构化 DataFrame
- 对数据进行清洗、筛选和聚合
- 输出交通明细、天气明细和分钟级平均速度结果

------

### （5）存储层

包括：

- HDFS 历史目录

例如：

- `/traffic/history/traffic_detail`
- `/traffic/history/weather`
- `/traffic/history/avg_speed`

这一层负责：

- 持久化存储流处理结果
- 形成后续历史分析与模型训练的数据基础

------

### （6）数据仓库与查询层

包括：

- Hive
- `traffic_detail_history`
- `weather_history`
- `traffic_avg_speed_history`

这一层负责：

- 将 HDFS 上的历史文件映射为结构化表
- 支持 SQL 查询、统计分析和模型取数

------

### （7）分析预测与展示层

包括：

- Python + pandas
- SARIMA 模型
- 后续地图可视化 / 图表展示

这一层负责：

- 从 Hive 中提取单路段分钟级速度序列
- 建立短时交通预测模型
- 输出未来若干分钟的预测结果
- 为论文图表和系统展示提供结果支撑

------

# 二、各模块职责表

下面这张表你后面可以直接整理进论文。

| 模块               | 主要文件/组件                      | 作用                                                         |
| ------------------ | ---------------------------------- | ------------------------------------------------------------ |
| 数据源层           | 路况 API、天气 API                 | 提供原始实时交通和天气数据                                   |
| 道路配置模块       | `road_list.csv`                    | 保存重点道路基础信息，如 `road_id`、`road_name`、`district`、坐标、`free_flow_speed` |
| 适配模块           | `adapters.py`                      | 将外部 API 原始 JSON 转换为系统统一字段结构                  |
| 采集调度模块       | `run_collector.py`、`collector.py` | 定时调用 API，补充采集时间和道路元数据，并发送到 Kafka       |
| 消息队列模块       | Kafka、Zookeeper                   | 缓冲实时消息，实现采集端与处理端解耦                         |
| 交通明细流处理模块 | `traffic_detail_stream_job.py`     | 消费 `traffic_raw`，生成交通明细历史数据                     |
| 天气明细流处理模块 | `weather_detail_stream_job.py`     | 消费 `weather_raw`，生成天气明细历史数据                     |
| 平均速度聚合模块   | `traffic_avg_speed_stream_job.py`  | 对交通速度进行补值和分钟级聚合，形成预测输入数据             |
| 历史存储模块       | HDFS                               | 存储 Spark 输出的历史 CSV 文件                               |
| 数据仓库模块       | Hive                               | 对 HDFS 历史数据建立表映射，支持 SQL 查询                    |
| 预测分析模块       | Python、SARIMA                     | 对单路段分钟级 `avg_speed` 序列进行短时预测                  |
| 可视化模块         | 地图 / 图表                        | 展示实时路况、历史统计和预测结果                             |

------

# 三、数据流转过程说明

这一部分非常重要，你后面在答辩时几乎就是照着这个顺序讲。

------

## 第一步：采集外部实时数据

系统通过 Python 采集程序按固定周期访问外部 API。

### 当前采样策略

- 交通路况：**每分钟采集一次**
- 天气数据：**每 10 分钟采集一次**

交通采集会遍历 `road_list.csv` 中配置的重点道路，逐条请求对应道路的实时路况。

------

## 第二步：适配外部 JSON 为统一格式

外部 API 返回的是原始 JSON，字段结构复杂且不同数据源之间格式不一致。

因此系统在 `adapters.py` 中完成第一轮统一：

- 百度天气 JSON → `weather_record`
- 百度路况 JSON → `traffic_record`

这一阶段主要完成：

- 选择真正有价值的字段
- 统一字段命名
- 屏蔽不同 API 的结构差异

例如：

- 天气中的温度统一命名为 `temperature`
- 路况中的速度统一命名为 `speed`
- 路况片段状态统一命名为 `section_status`

------

## 第三步：补充系统元数据

在 `collector.py` 中，系统对适配后的记录继续补充系统内部所需字段，包括：

- `collect_time`
- `dt`
- `road_id`
- `district`
- `center_lng`
- `center_lat`
- `free_flow_speed`

其中：

- `collect_time` 用于标识采集时刻
- `dt` 用于 HDFS / Hive 按天分区
- `free_flow_speed` 用于缺失速度值时的规则补值

到这一步，数据已经从“外部原始接口格式”转变成“系统内部标准记录格式”。

------

## 第四步：发送至 Kafka Topic

采集模块将标准化后的记录发送到 Kafka：

- 交通数据进入 `traffic_raw`
- 天气数据进入 `weather_raw`

Kafka 的作用主要有三个：

1. 解耦采集端与流处理端
2. 缓冲短时高频消息
3. 支持多个 Spark 作业同时独立消费同一份数据

例如，同一份 `traffic_raw` 消息可以同时被：

- 交通明细流任务消费
- 平均速度聚合流任务消费

------

## 第五步：Spark Streaming 实时消费 Kafka 数据

Spark Streaming 从 Kafka 中读取消息时，最开始拿到的是 JSON 字符串。
随后通过 `schema + from_json` 将其解析为结构化 DataFrame。

这一步不是业务字段命名统一，而是**数据类型和表结构的结构化处理**。

例如：

- `speed` 解析为数值型
- `road_id` 解析为字符串型
- `section_status` 解析为整数型

------

## 第六步：生成交通明细与天气明细历史表

### 1. 交通明细流任务

`traffic_detail_stream_job.py` 负责从 `traffic_raw` 中提取交通字段，并写入 HDFS。

输出内容主要包括：

- 道路 ID
- 道路名称
- 采集时间
- 路况描述
- 路段状态
- 拥堵距离
- 速度等

这部分数据适合做历史事实回溯和统计分析。

------

### 2. 天气明细流任务

`weather_detail_stream_job.py` 负责从 `weather_raw` 中提取天气字段，并写入 HDFS。

输出内容主要包括：

- 城市
- 行政区
- 天气现象
- 温度
- 湿度
- 风向
- 能见度等

这部分数据适合做天气变化分析以及与交通数据的关联分析。

------

## 第七步：生成分钟级平均速度聚合表

`traffic_avg_speed_stream_job.py` 是目前最关键的分析层流任务。

它的逻辑是：

### （1）保留原始速度

如果 API 返回了速度值，则直接作为 `raw_speed` 使用。

### （2）对缺失速度进行规则补值

如果某条道路当前没有返回速度，则基于：

- `free_flow_speed`
- `section_status`

推算最终速度 `final_speed`。

这样做的原因是：
部分道路在畅通或特定状态下，接口不会直接返回速度值，但预测模型必须要有连续的速度序列。

------

### （3）按分钟桶聚合

系统将 `collect_time` 规整为 `minute_bucket`，例如：

- `2026-03-19 10:59:00`

然后按：

- `road_id`
- `minute_bucket`

对该分钟内的速度进行聚合，生成：

- `avg_speed`
- `record_count`
- `api_count`
- `estimated_count`

这一步的意义不是简单“重复平均”，而是把实时采样结果规整成**标准时间粒度的分钟级序列**，为后续建模准备数据。

------

## 第八步：写入 HDFS 历史目录

Spark 将三类结果写入 HDFS：

- 交通明细
- 天气明细
- 平均速度聚合结果

这些数据虽然在逻辑上存储在 HDFS 中，但由于你当前运行的是本地伪分布式 Hadoop 环境，因此底层物理文件仍位于本机磁盘中。

也就是说：

- 逻辑层面：这是 HDFS 历史存储
- 物理层面：文件最终落在你电脑本地磁盘路径对应的位置

------

## 第九步：Hive 建表映射

Hive 不直接存储原始文件，而是对 HDFS 上的数据文件建立结构化映射。

通过 Hive，你可以把 HDFS 目录中的 CSV 文件映射为：

- `traffic_detail_history`
- `weather_history`
- `traffic_avg_speed_history`

这样就可以使用 SQL 来查询数据，例如：

```sql
SELECT minute_bucket, avg_speed
FROM traffic_avg_speed_history
WHERE road_id = 'RB001'
ORDER BY minute_bucket;
```

这一步使系统具备了标准化的数据仓库查询能力。

------

# 四、SARIMA 预测模块如何接入系统

这一部分你现在不用怕，我先按“项目逻辑”讲清楚，不先讲复杂公式。

------

## 1. SARIMA 的输入是什么

SARIMA 不能直接吃整张交通明细表，它要吃的是：

**某一条道路的连续时间序列**

在你的系统里，这个序列来自：

- `traffic_avg_speed_history`

更具体地说，就是某条路的：

- `minute_bucket`
- `avg_speed`

例如：

| minute_bucket | avg_speed |
| ------------- | --------- |
| 10:00         | 18.6      |
| 10:01         | 17.9      |
| 10:02         | 19.2      |

这就是模型输入。

------

## 2. 预测流程是什么

你后面做 SARIMA 时，整个过程可以分成 6 步：

### 第一步：从 Hive 抽取单条路的分钟级序列

例如提取：

- 中山路 `RB001`
- 最近 6 小时
- 每分钟平均速度

------

### 第二步：导入 Python 做预处理

主要包括：

- 时间排序
- 缺失值检查
- 是否连续
- 是否存在异常值
- 画折线图观察趋势

------

### 第三步：划分训练集和测试集

例如：

- 前 80% 作为训练集
- 后 20% 作为测试集

------

### 第四步：建立 SARIMA 模型

SARIMA 会从历史序列中学习：

- 最近值对未来的影响
- 趋势变化
- 周期波动

你的项目里，它要学的是：

**道路平均速度随时间变化的规律**

------

### 第五步：预测未来若干分钟

例如预测未来：

- 5 分钟
- 10 分钟

输出未来各分钟的速度预测值。

------

### 第六步：误差评估

把预测结果和真实值做比较，常用指标包括：

- MAE
- RMSE
- MAPE

这样你论文里就可以评价：

- 模型是否有效
- 哪些时段预测更准
- 哪些道路更适合预测

------

# 五、你这个系统里，SARIMA 的位置到底是什么

你一定要记住：

## SARIMA 不是整个系统的中心

它只是系统最上层的一个分析模块。

真正支撑 SARIMA 的，是前面那整条链路：

- 采集
- Kafka
- Spark
- HDFS
- Hive
- 分钟级聚合

如果没有前面的这些，SARIMA 根本拿不到稳定的时间序列输入。

所以你后面在论文里要体现的是：

> 本文工作的重点并不只是构建预测模型，而是构建一套从实时采集、流式处理、历史沉淀到短时预测的完整交通分析与预测系统。

这个定位会比“我只训练了一个 SARIMA”更完整，也更像一个毕设系统。

------

# 六、给你一版可以直接放论文里的“系统运行逻辑描述”

你可以直接参考下面这段：

> 系统首先通过 Python 采集程序按固定时间间隔访问哈尔滨市重点道路的实时路况接口和天气接口，并在适配层对原始 JSON 数据进行统一字段转换。随后，采集程序结合道路基础信息表补充道路标识、行政区、坐标、采集时间及自由流速度等元数据，并将标准化后的记录发送至 Kafka 的 `traffic_raw` 与 `weather_raw` 两个主题。
>
> Spark Streaming 作为流处理层，从 Kafka 中实时消费消息，利用预定义 schema 将 JSON 字符串解析为结构化 DataFrame，分别生成交通明细流、天气明细流和分钟级道路平均速度聚合流。其中，平均速度聚合模块针对接口缺失速度值的问题，结合道路自由流速度和路况状态进行规则化补值，并按道路和分钟粒度完成速度聚合，形成后续预测所需的标准时间序列。
>
> 经过流处理后的结果最终写入 HDFS 历史目录，并由 Hive 建立表结构映射，为历史查询、统计分析及模型训练提供统一的数据支撑。最后，系统基于分钟级道路速度序列构建 SARIMA 模型，实现对重点道路未来短时交通速度的预测，并为后续的图表和地图可视化展示提供结果支撑。

------

# 七、答辩时你可以怎么口头讲

如果老师问你“整个项目怎么跑起来的”，你可以这样说：

> 我们的系统先通过 Python 定时采集道路路况和天气数据，然后在采集阶段就先把外部接口的 JSON 转换成项目统一格式，再补充道路 ID、行政区、坐标和采集时间等字段。处理好的消息被发送到 Kafka 中，Kafka 作为消息队列负责缓存和解耦。
>
> 然后 Spark Streaming 从 Kafka 实时消费这些消息，分别处理成交通明细、天气明细和分钟级平均速度数据，并将这些历史结果存储到 HDFS 中。接着 Hive 对 HDFS 上的历史文件建立表映射，使得后续可以通过 SQL 进行查询和分析。最后，我们从分钟级平均速度表中提取某条道路的时间序列，使用 SARIMA 模型进行短时交通速度预测。

这段口语化程度比较合适。

------



## 1. 交通明细表 `traffic_detail_history`



| 序号 | 字段名                | 类型   | 说明             |
| ---- | --------------------- | ------ | ---------------- |
| 1    | `collect_time`        | STRING | 采集时间         |
| 2    | `city`                | STRING | 城市名称         |
| 3    | `district`            | STRING | 行政区名称       |
| 4    | `road_id`             | STRING | 道路唯一标识     |
| 5    | `road_name`           | STRING | 道路名称         |
| 6    | `description`         | STRING | 路况描述文本     |
| 7    | `overall_status`      | INT    | 整体路况等级     |
| 8    | `overall_status_desc` | STRING | 整体路况文字描述 |
| 9    | `section_desc`        | STRING | 拥堵路段描述     |
| 10   | `road_type`           | STRING | 道路类型         |
| 11   | `congestion_distance` | DOUBLE | 拥堵距离         |
| 12   | `speed`               | DOUBLE | API 原始速度     |
| 13   | `section_status`      | INT    | 路段拥堵状态等级 |
| 14   | `congestion_trend`    | STRING | 拥堵趋势         |
| 15   | `dt`                  | STRING | 日期分区字段     |

------

### 对应 Hive 建表语句

```
DROP TABLE IF EXISTS traffic_detail_history;

CREATE EXTERNAL TABLE traffic_detail_history (
  collect_time         STRING,
  city                 STRING,
  district             STRING,
  road_id              STRING,
  road_name            STRING,
  description          STRING,
  overall_status       INT,
  overall_status_desc  STRING,
  section_desc         STRING,
  road_type            STRING,
  congestion_distance  DOUBLE,
  speed                DOUBLE,
  section_status       INT,
  congestion_trend     STRING
)
PARTITIONED BY (dt STRING)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS PARQUET
LOCATION '/traffic/history/traffic_detail'
TBLPROPERTIES ("skip.header.line.count"="1");

MSCK REPAIR TABLE traffic_detail_history;
```

------

## 2. 天气明细表 `weather_history`



| 序号 | 字段名             | 类型   | 说明              |
| ---- | ------------------ | ------ | ----------------- |
| 1    | `collect_time`     | STRING | 采集时间          |
| 2    | `country`          | STRING | 国家              |
| 3    | `province`         | STRING | 省份              |
| 4    | `city`             | STRING | 城市              |
| 5    | `district`         | STRING | 行政区            |
| 6    | `district_id`      | STRING | 行政区 ID         |
| 7    | `weather_text`     | STRING | 天气现象          |
| 8    | `temperature`      | DOUBLE | 当前温度          |
| 9    | `feels_like`       | DOUBLE | 体感温度          |
| 10   | `humidity`         | DOUBLE | 相对湿度          |
| 11   | `wind_class`       | STRING | 风力等级          |
| 12   | `wind_dir`         | STRING | 风向              |
| 13   | `precipitation_1h` | DOUBLE | 过去 1 小时降水量 |
| 14   | `clouds`           | DOUBLE | 云量              |
| 15   | `visibility`       | DOUBLE | 能见度            |
| 16   | `aqi`              | DOUBLE | 空气质量指数      |
| 17   | `pm25`             | DOUBLE | PM2.5             |
| 18   | `pm10`             | DOUBLE | PM10              |
| 19   | `pressure`         | DOUBLE | 气压              |
| 20   | `dt`               | STRING | 日期分区字段      |

------

### 对应 Hive 建表语句

```
DROP TABLE IF EXISTS weather_history;

CREATE EXTERNAL TABLE weather_history (
  collect_time        STRING,
  country             STRING,
  province            STRING,
  city                STRING,
  district            STRING,
  district_id         STRING,
  weather_text        STRING,
  temperature         DOUBLE,
  feels_like          DOUBLE,
  humidity            DOUBLE,
  wind_class          STRING,
  wind_dir            STRING,
  precipitation_1h    DOUBLE,
  clouds              DOUBLE,
  visibility          DOUBLE,
  aqi                 DOUBLE,
  pm25                DOUBLE,
  pm10                DOUBLE,
  pressure            DOUBLE
)
PARTITIONED BY (dt STRING)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/traffic/history/weather';

MSCK REPAIR TABLE weather_history;
```

------

## 3. 平均速度聚合表 `traffic_avg_speed_history`



| 序号 | 字段名            | 类型         | 说明                |
| ---- | ----------------- | ------------ | ------------------- |
| 1    | `minute_bucket`   | STRING       | 分钟级时间桶        |
| 2    | `city`            | STRING       | 城市                |
| 3    | `district`        | STRING       | 行政区              |
| 4    | `road_id`         | STRING       | 道路唯一标识        |
| 5    | `road_name`       | STRING       | 道路名称            |
| 6    | `avg_speed`       | DOUBLE       | 最终分钟平均速度    |
| 7    | `record_count`    | BIGINT / INT | 该分钟记录数        |
| 8    | `api_count`       | BIGINT / INT | 真实 API 速度记录数 |
| 9    | `estimated_count` | BIGINT / INT | 规则补值记录数      |
| 10   | `dt`              | STRING       | 日期分区字段        |

> 在 Hive 里，`count()` 通常映射成 `BIGINT` 更稳。

------

### 对应 Hive 建表语句

```
DROP TABLE IF EXISTS traffic_avg_speed_history;

CREATE EXTERNAL TABLE traffic_avg_speed_history (
  minute_bucket   STRING,
  city            STRING,
  district        STRING,
  road_id         STRING,
  road_name       STRING,
  avg_speed       DOUBLE,
  record_count    BIGINT,
  api_count       BIGINT,
  estimated_count BIGINT
)
PARTITIONED BY (dt STRING)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/traffic/history/avg_speed'
TBLPROPERTIES ("skip.header.line.count"="1");

MSCK REPAIR TABLE traffic_avg_speed_history;
```

# 我帮你把页面设计压缩成最简版（你可以直接用）

你可以把系统就做成一个页面，结构如下👇

------

# 🌐 页面结构（简化版）

## 1️⃣ 顶部

- 标题：哈尔滨交通分析与预测平台
- 当前时间
- 数据更新时间

------

## 2️⃣ 左侧（控制区）

- 模式切换：
  - 实时模式
  - 预测模式

------

## 3️⃣ 中间（地图）

👉 百度地图

显示：

### 实时模式

- 12 条道路当前状态（颜色）

### 预测模式

- 12 条道路 10 分钟后状态（颜色）

------

## 4️⃣ 右侧（点击道路后显示）

### 基本信息

- 道路名称
- 区域

### 实时

- 当前速度
- 当前拥堵等级

### 预测

- 10 分钟后速度
- 预测拥堵等级

### 图表

- 过去 2 小时速度折线图

------

## 5️⃣ 可选（底部）

- 当前各道路速度对比图（简单柱状图）

# sarima给你一个“现实可行”的方案（重点）

👉 你不需要做“严格实时在线训练模型”，
 👉 你需要做的是：

# ✅ **准实时预测（Near Real-Time Prediction）**

也就是：

- 数据是实时流入的 ✅
- 预测是**定时更新的（比如每1分钟）** ✅
- 页面展示是实时刷新的 ✅

------

## 👉 最稳的实现方式

### 后端做一个“预测任务”（循环跑）

比如：

- 每 1 分钟执行一次
- 从 Hive / HDFS 取最近 2 小时数据
- 对 12 条路分别做预测
- 预测未来 10 分钟
- 写入一张表：

👉 `traffic_prediction_result`

------

### 前端怎么用？

- 实时模式 → 读 `traffic_avg_speed_history`
- 预测模式 → 读 `traffic_prediction_result`

👉 页面切换是“实时的”，用户感知是实时预测

------

## ❗关键一句话

> 本系统采用准实时滚动预测机制，每分钟基于最新历史数据更新短时交通预测结果。



# 标准大数据环境版本（必须照这个来）

这是我给你定的**最稳组合（面试也认可）**：

| 组件      | 版本  |
| --------- | ----- |
| Java      | 1.8   |
| Hadoop    | 3.3.6 |
| Hive      | 3.1.3 |
| Zookeeper | 3.8.x |
| Kafka     | 3.5.x |
| Spark     | 3.5.x |

👉 这是官方兼容链：

- Hive 3.1.3 ✅ Hadoop 3.x
- Kafka 3.5 ✅ Zookeeper 3.8
- Spark 3.5 ✅ Hadoop 3.x

------

# 标准目录结构（以后必须这样）

```
/mnt/d/bigdata/
├── jdk1.8
├── hadoop-3.3.6
├── hive-3.1.3
├── zookeeper-3.8.4
├── kafka_2.13-3.5.0
├── spark-3.5.0
```







可以，我给你把整个项目按“**每个模块是干什么的**”和“**数据怎么流动**”两条线彻底讲清楚。你这一步很重要，因为后面不管是写论文、做答辩，还是自己继续改代码，都要先把这条主线讲顺。

你现在的理解已经有 70% 是对的，我帮你修正和补全剩下的 30%。

------

# 一、先用一句话说清你的项目到底在干什么

你的项目本质上是在做一条完整的数据链路：

**外部实时 API → Python 采集 → Kafka 缓冲与解耦 → Spark Streaming 实时处理 → HDFS 历史存储 → Hive 结构化查询 → SARIMA 预测分析 → 地图/图表展示**

也就是说，你的系统不是只有“采集”，也不是只有“预测”，而是一个完整闭环。

------

# 二、你现在开的这些终端，各自到底在做什么

你现在开着：

- 一个 Zookeeper
- 一个 Kafka
- 一个采集终端
- 一个交通明细 Spark 终端
- 一个天气明细 Spark 终端
- 一个聚合速度 Spark 终端

这 6 个东西其实可以分成 4 层。

------

## 第一层：消息中间层

### 1. Zookeeper

它主要是给 Kafka 做协调服务的。
你现在可以先把它理解成：

- 帮 Kafka 管理一些运行状态和协调信息
- 让 Kafka 能正常启动和工作

对你这个毕设来说，**你不用把 Zookeeper 讲得特别深**，知道它是 Kafka 的依赖协调组件就够了。

------

### 2. Kafka

Kafka 不是“计算引擎”，它是**消息队列 / 实时数据通道**。

它的作用不是分析数据，而是：

- 接住采集程序送来的数据
- 暂时缓存这些数据
- 让多个下游程序都能独立消费这些数据

你可以把它理解成一个“实时数据中转站”。

比如你采集到一条交通数据，不是直接写 HDFS，而是先发到 Kafka 里的 `traffic_raw` 这个 topic。
然后 Spark 再从这个 topic 去读。

------

## 第二层：采集与标准化层

### 3. 采集程序

也就是你现在运行的：

```bash
python src/run_collector.py
```

它本质上是整个系统的**数据入口**。

它负责两件事：

1. 调用外部 API
   - 天气 API
   - 实时路况 API
2. 把拿到的数据整理后发到 Kafka

------

# 三、你问得最关键的一个点：标准化到底在哪里发生

这个问题非常重要。

答案是：

## 标准化其实分成两次

------

## 第一次标准化：在 `adapters.py`

这是**字段语义标准化**。

外部 API 返回的是百度原始 JSON，这个 JSON 很乱，而且天气和路况长得完全不一样。
所以你在 `adapters.py` 里做了第一轮转换：

- `adapt_weather(raw)`：把百度天气 JSON 转成你的 `weather_record`
- `adapt_traffic(raw, city)`：把百度路况 JSON 转成你的 `traffic_record`

这一层做的事是：

- 提取有用字段
- 统一字段命名
- 去掉很多后面不用的接口杂项
- 让天气和交通都变成“项目内部统一对象”

比如：

百度天气原始字段可能叫：

- `result.location.city`
- `result.now.text`
- `result.now.temp`

经过 adapter 后，会变成你项目统一字段：

- `city`
- `weather_text`
- `temperature`

路况也是一样。

所以：

### 你可以这样理解

`adapters.py` 做的是：

**“外部接口格式” → “项目统一业务格式”**

------

## 第二次标准化：在 `collector.py`

这是**补项目运行字段和道路元数据**。

adapter 做完之后，数据虽然已经统一了，但还不完整。
所以 `collector.py` 里又做了一次 enrich：

### 对天气：

补：

- `collect_time`
- `dt`

### 对交通：

补：

- `collect_time`
- `dt`
- `road_id`
- `district`
- `center_lng`
- `center_lat`
- `free_flow_speed`（你后来加的）

所以你现在可以把 `collector.py` 理解成：

**把统一业务格式再补成“可以进入系统的数据记录”**

------

## 第三次“结构化”：在 Spark 里

这一步不是“业务字段标准化”，而是**类型结构化**。

Kafka 里存的是 JSON 字符串。
Spark 读 Kafka 的时候，先拿到的是：

- 一列字符串 `json_str`

然后你在 Spark 里通过：

- `from_json(...)`
- 配合 `common.py` 里定义好的 schema

把 JSON 字符串解析成真正有字段、有类型的 Spark DataFrame。

例如：

- `speed` 变成 `DoubleType`
- `section_status` 变成 `IntegerType`
- `collect_time` 是 `StringType`

所以这一步可以理解成：

**“统一 JSON 字符串” → “可计算的结构化表”**

------

# 四、把整个数据流按顺序串起来

下面我按实际顺序，给你讲整个项目的数据是怎么流动的。

------

## 第 1 步：采集程序调用 API

`run_collector.py` 每 60 秒跑一轮。

每轮中：

- 天气按 10 分钟采一次
- 交通每分钟遍历所有重点道路采一次

这一步得到的是：

- 百度天气原始 JSON
- 百度路况原始 JSON

------

## 第 2 步：adapter 把原始 JSON 变成项目统一格式

这一步在 `adapters.py`。

作用是：

- 百度原始天气 JSON → `weather_record`
- 百度原始路况 JSON → `traffic_record`

这里最重要的是：
**你已经把外部 API 的混乱结构变成了自己系统认可的字段。**

------

## 第 3 步：collector 给记录补充系统字段

这一步在 `collector.py`。

它补进去：

- 采集时间 `collect_time`
- 分区日期 `dt`
- 道路元数据 `road_id/district`
- 地图点位 `center_lng/center_lat`
- 速度基准 `free_flow_speed`

到这里，一条记录就已经是“系统内部标准记录”了。

------

## 第 4 步：发送到 Kafka topic

collector 通过 Kafka producer 把数据发出去。

发送到：

- `weather_raw`
- `traffic_raw`

这里你要注意一件事：

### Kafka 里现在放的不是百度原始 JSON

而是：

**你已经标准化过的项目 JSON**

这一点非常重要。

也就是说，现在 topic 里的 `raw`，严格来说是“采集后的原始流”，不是外部 API 最原始格式。

------

## 第 5 步：Spark Streaming 从 Kafka 消费

你现在有 3 个 Spark Streaming 任务，它们都在独立消费 Kafka。

这就是 Kafka 解耦最核心的意义：

### 同一份 `traffic_raw`，可以被不同 Spark 任务分别消费

比如：

- `traffic_detail_stream_job.py`
- `traffic_avg_speed_stream_job.py`

都可以同时读 `traffic_raw`，互不影响。

------

# 五、三个 Spark 任务分别是干什么的

这是你后面答辩时必须讲清楚的重点。

------

## 1. `traffic_detail_stream_job.py`

它负责生成：

**交通明细历史表**

它做的事是：

1. 从 Kafka 读 `traffic_raw`
2. 用 schema 把 JSON 解析成 DataFrame
3. 选出交通明细字段
4. 过滤明显无效数据
5. 写入 HDFS

它产出的数据适合做：

- 历史明细查询
- 拥堵状态分析
- 字段回溯
- 后续与天气做关联分析

也就是说，这张表更偏“原始事实层”。

------

## 2. `weather_detail_stream_job.py`

它负责生成：

**天气明细历史表**

流程和交通明细类似：

1. 从 Kafka 读 `weather_raw`
2. JSON 解析
3. 选天气字段
4. 写 HDFS

这张表后面用来做：

- 天气变化查询
- 天气与交通关联分析
- 给预测做辅助解释

------

## 3. `traffic_avg_speed_stream_job.py`

这是你现在最关键的一张计算表。

它负责生成：

**分钟级道路平均速度表**

它不是简单保存原始速度，而是做了进一步处理：

1. 从 `traffic_raw` 读交通数据
2. 解析字段
3. 保留 `raw_speed`
4. 对缺失速度做补值，生成 `final_speed`
5. 按分钟桶 `minute_bucket` 聚合
6. 生成每条路每分钟的 `avg_speed`

它的用途是：

- 给 SARIMA 做输入
- 给历史速度折线图做数据源
- 给后面可视化和预测结果展示做基础

所以这张表更偏“分析建模层”。

------

# 六、你说“Spark 中把 JSON 变成 stringtype，再存 HDFS”——这里我帮你纠正一下

这个说法差一点点。

正确理解应该是：

## Kafka 里最开始拿到的是 JSON 字符串

Spark 读 Kafka 时，拿到的是：

- 原始 `value`
- 然后 `CAST(value AS STRING)` 变成 `json_str`

这时候它确实只是字符串。

------

## 但随后 Spark 又把这个字符串解析成结构化字段

通过：

- `from_json(col("json_str"), schema)`

它又把字符串解析成 DataFrame 的结构化字段了。

所以最终写入 HDFS 的，不是“一整段 JSON 字符串”，而是：

**按列展开后的结构化 CSV 数据**

例如会变成：

- `collect_time`
- `road_id`
- `road_name`
- `speed`
- `section_status`

这样的列式数据。

所以更准确的说法是：

**Kafka 中消息是 JSON 字符串，Spark 先读字符串，再按 schema 解析成结构化表，最后写入 HDFS。**

------

# 七、HDFS 是你电脑本地吗

这个问题你一定要讲清楚，因为很多人会混淆。

答案是：

## 逻辑上：它是 HDFS 文件系统

## 物理上：它底层还是存在你本机磁盘上

因为你现在不是在多台服务器上跑 Hadoop 集群，而是在自己电脑 / WSL 里跑的 Hadoop。
所以：

- 从系统逻辑上看，你是在往 HDFS 写数据
- 从物理存储上看，这些文件最后还是落在你本机的磁盘目录里

所以你可以这样理解：

### 你现在跑的是“本地伪分布式 HDFS”

不是云上的大集群，但仍然保留了 HDFS 的访问方式和组织方式。

你后面答辩可以这么讲：

> 本系统采用本地伪分布式 Hadoop 环境，历史数据以 HDFS 路径形式进行存储与管理，虽然底层物理存储仍位于本机磁盘，但逻辑上按照 HDFS 方式组织，便于模拟大数据平台中的分布式存储流程。

这个说法很稳。

------

# 八、HDFS 和 Hive 的关系到底是什么

这个也很重要。

很多人会以为 Hive 把数据“存到 Hive 里”，其实不是。

------

## HDFS：负责真正存文件

例如：

- `/traffic/history/traffic_detail`
- `/traffic/history/weather`
- `/traffic/history/avg_speed`

这些目录里是真正的数据文件，比如 CSV part 文件。

------

## Hive：负责把 HDFS 上的数据映射成表

Hive 更像一个“表结构层”。

它做的是：

- 告诉系统这些 HDFS 文件对应什么表
- 这些列叫什么
- 每列是什么类型
- 分区字段是什么

所以 Hive 本质上是：

**对 HDFS 文件做结构化映射，方便你用 SQL 查**

比如你建了：

- `traffic_detail_history`
- `weather_history`
- `traffic_avg_speed_history`

那么你就可以：

```sql
SELECT road_name, avg_speed
FROM traffic_avg_speed_history
WHERE road_id = 'RB001';
```

------

## 一句话总结 HDFS 和 Hive

- **HDFS 存数据**
- **Hive 管表结构和 SQL 查询**

------

# 九、你说“可以做一张明细表解释字段含义”，这个想法非常对

而且这不是“可做可不做”，而是**很应该做**。

因为你后面写论文和答辩时，老师一定会问：

- 这个 `section_status` 是什么
- `overall_status` 和 `section_status` 有什么区别
- `speed_source` 是什么意思
- `api_count` / `estimated_count` 是怎么来的

所以你很适合整理一张“核心指标说明表”。

比如可以分成三张。

------

## 1. 交通明细字段说明表

例如：

- `road_id`：道路唯一标识
- `road_name`：道路名称
- `collect_time`：采集时间
- `overall_status`：接口整体路况等级
- `section_status`：拥堵片段等级
- `speed`：接口原始速度
- `congestion_distance`：拥堵距离
- `congestion_trend`：拥堵趋势

------

## 2. 天气字段说明表

例如：

- `weather_text`
- `temperature`
- `humidity`
- `wind_dir`
- `visibility`
- `aqi`

------

## 3. 聚合速度字段说明表

例如：

- `minute_bucket`：分钟级时间桶
- `avg_speed`：该道路该分钟最终平均速度
- `record_count`：本分钟记录数
- `api_count`：真实 API 速度记录数
- `estimated_count`：规则估算速度记录数

这张表后面会非常有用。

------

# 十、SARIMA 预测的时候，到底在做什么

你说你对算法不熟，这很正常。
我先不讲太复杂的公式，只讲你项目里它到底在干什么。

------

## 1. 你的预测目标是什么

你现在最适合预测的是：

**某条道路未来若干分钟的平均速度**

例如：

- 未来 10 分钟中山路的平均速度

------

## 2. SARIMA 需要什么输入

SARIMA 需要的是一条单变量时间序列。

在你项目里，就是这样一张序列：

| minute_bucket | avg_speed |
| ------------- | --------- |
| 10:00         | 15.2      |
| 10:01         | 14.8      |
| 10:02         | 13.9      |

也就是说：

### 你不能直接拿交通明细表喂给 SARIMA

你要先从 `traffic_avg_speed_history` 里，抽出某一条路的分钟级速度序列。

例如：

```sql
SELECT minute_bucket, avg_speed
FROM traffic_avg_speed_history
WHERE road_id = 'RB001'
ORDER BY minute_bucket;
```

这才是模型输入。

------

## 3. SARIMA 为什么适合你这个项目

因为它适合：

- 按时间顺序排列的数据
- 连续数值预测
- 存在趋势和周期的序列

你的 `avg_speed` 正好是：

- 分钟级
- 连续变量
- 随时间变化
- 可能存在高峰/低峰规律

所以很合适。

------

# 十一、SARIMA 在你项目中的实际流程

你可以把它理解成下面这 6 步。

------

## 第 1 步：从 Hive 抽取单条道路序列

例如抽：

- `RB001`
- 最近 6 小时
- 每分钟 `avg_speed`

得到一列时间序列。

------

## 第 2 步：导出到 Python

你可以用：

- Hive 查询后保存 CSV
- 或者直接用 pandas 读取导出的数据

形成：

- 时间索引
- 速度值序列

------

## 第 3 步：做预处理

比如：

- 时间排序
- 缺失值检查
- 看是否有断点
- 画折线图

这一步主要是检查数据能不能建模。

------

## 第 4 步：训练 SARIMA

SARIMA 会根据过去一段时间的速度变化规律，去拟合一个时间序列模型。

它学的是：

- 最近值对未来的影响
- 差分后的趋势
- 周期性波动

你不用一开始就把数学讲得很深，只要知道它是在“从历史速度规律里学习未来走势”。

------

## 第 5 步：预测未来几步

比如预测未来：

- 5 分钟
- 10 分钟

输出未来时刻的速度值。

------

## 第 6 步：评估误差

把预测值和真实值对比，算：

- MAE
- RMSE
- MAPE

这样你论文里就能说：

- 模型效果怎么样
- 误差多大
- 哪些时段预测更准

------

# 十二、把整个项目再总结成一句“答辩版”表述

你后面可以这样讲：

> 本系统首先通过 Python 采集程序定时调用哈尔滨市重点道路实时路况 API 和天气 API，并在 adapter 层将外部原始 JSON 转换为系统统一数据结构，补充采集时间、道路标识和位置等元数据后发送到 Kafka。随后，Spark Streaming 作为流处理层从 Kafka 实时消费数据，分别生成交通明细、天气明细以及分钟级道路平均速度聚合结果，并将这些结果持久化到 HDFS。Hive 对 HDFS 中的历史文件进行表结构映射，从而支持 SQL 查询、历史分析与后续预测建模。最终，系统基于分钟级道路速度序列，利用 SARIMA 模型实现短时交通速度预测，并可进一步支撑图表或地图可视化展示。

这段话已经很接近你毕设里系统架构描述了。

------

# 十三、我最后帮你纠正一下你原来的那段理解

你原来的理解可以修正成下面这个版本：

## 更准确的版本

- 采集程序负责调用实时路况和天气 API。
- 原始 API 返回的 JSON 先在 `adapters.py` 中完成字段统一，再在 `collector.py` 中补充采集时间、道路元数据等系统字段。
- 标准化后的记录被作为 Kafka producer 的消息发送到 `traffic_raw` 和 `weather_raw` 两个 topic 中。
- Kafka 作为消息中间件，负责缓存和解耦采集端与计算端。
- Spark Streaming 从 Kafka 中实时消费这些 JSON 消息，先通过 schema 将其解析为结构化 DataFrame，再分别生成交通明细、天气明细和分钟级平均速度聚合结果。
- 这些结构化结果被写入 HDFS 历史目录。
- Hive 再将 HDFS 上的文件映射成表，支持后续 SQL 查询、历史分析和模型取数。
- SARIMA 则基于某条道路的分钟级 `avg_speed` 序列进行短时交通速度预测。

这个版本就非常完整了。

------

你现在最适合继续做的，不是马上碰 SARIMA 代码，而是先把这两样东西定出来：

**1. 系统模块说明图**
**2. 核心字段说明表**

你要是愿意，我下一步可以直接帮你整理成：

**“毕设论文可直接用的系统架构说明 + 模块职责表 + 数据流说明图文字版”**