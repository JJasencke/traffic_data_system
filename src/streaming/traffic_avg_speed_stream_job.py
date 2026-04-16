from pyspark.sql.functions import (
    avg, col, count, date_format, date_trunc, from_json, lit, to_timestamp, when
)
from common import create_spark_session, get_traffic_schema


def write_avg_speed_batch(batch_df, batch_id):
    """
    将每个微批次的分钟级道路速度聚合结果写入 HDFS 历史目录
    """
    if batch_df.rdd.isEmpty():
        print(f"Batch {batch_id} 为空，跳过")
        return

    (
        batch_df.write
        .mode("append")
        .option("header", True)
        .partitionBy("dt")
        .csv("hdfs://localhost:9000/traffic/history/avg_speed")
    )

    print(f"Batch {batch_id} 已写入 HDFS /traffic/history/avg_speed")


def build_final_speed_expr():
    status_desc = col("overall_status_desc")
    status_code = when(col("section_status").isNotNull(), col("section_status")).otherwise(col("overall_status"))
    free_flow_speed = col("free_flow_speed")

    return (
        when(col("raw_speed").isNotNull(), col("raw_speed"))
        .when(
            free_flow_speed.isNotNull() & (
                status_desc.contains("严重") | (status_code == lit(4))
            ),
            free_flow_speed * lit(0.30)
        )
        .when(
            free_flow_speed.isNotNull() & (
                status_desc.contains("拥堵") | (status_code == lit(3))
            ),
            free_flow_speed * lit(0.50)
        )
        .when(
            free_flow_speed.isNotNull() & (
                status_desc.contains("缓行") | (status_code == lit(2))
            ),
            free_flow_speed * lit(0.75)
        )
        .when(
            free_flow_speed.isNotNull() & (
                status_desc.contains("畅通") | status_desc.contains("通畅") | (status_code == lit(1)) | status_code.isNull()
            ),
            free_flow_speed
        )
    )


def build_speed_source_expr():
    status_desc = col("overall_status_desc")
    status_code = when(col("section_status").isNotNull(), col("section_status")).otherwise(col("overall_status"))

    return (
        when(col("raw_speed").isNotNull(), lit("api"))
        .when(
            col("free_flow_speed").isNotNull() & (
                status_desc.contains("严重") | (status_code == lit(4))
            ),
            lit("estimated_severe")
        )
        .when(
            col("free_flow_speed").isNotNull() & (
                status_desc.contains("拥堵") | (status_code == lit(3))
            ),
            lit("estimated_congested")
        )
        .when(
            col("free_flow_speed").isNotNull() & (
                status_desc.contains("缓行") | (status_code == lit(2))
            ),
            lit("estimated_slow")
        )
        .when(
            col("free_flow_speed").isNotNull() & (
                status_desc.contains("畅通") | status_desc.contains("通畅") | (status_code == lit(1)) | status_code.isNull()
            ),
            lit("estimated_free_flow")
        )
    )


def main():
    spark = create_spark_session("TrafficAvgSpeedStreamJob")

    traffic_schema = get_traffic_schema()

    kafka_df = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", "localhost:9092")
        .option("subscribe", "traffic_raw")
        .option("startingOffsets", "latest")
        .load()
    )

    value_df = kafka_df.selectExpr("CAST(value AS STRING) AS json_str")

    parsed_df = value_df.select(
        from_json(col("json_str"), traffic_schema).alias("data")
    )

    traffic_df = parsed_df.select("data.*")

    clean_df = (
        traffic_df.select(
            "collect_time",
            "dt",
            "city",
            "district",
            "road_id",
            "road_name",
            "overall_status",
            "overall_status_desc",
            "section_status",
            "speed",
            "free_flow_speed"
        )
        .filter(
            col("road_id").isNotNull() &
            col("road_name").isNotNull() &
            col("collect_time").isNotNull() &
            col("dt").isNotNull()
        )
        .withColumn("collect_ts", to_timestamp(col("collect_time"), "yyyy-MM-dd HH:mm:ss"))
        .filter(col("collect_ts").isNotNull())
        .withColumn(
            "raw_speed",
            when(
                col("speed").isNotNull() & (col("speed") >= 0) & (col("speed") <= 150),
                col("speed")
            )
        )
        .withColumn("final_speed", build_final_speed_expr())
        .withColumn("speed_source", build_speed_source_expr())
        .filter(
            col("final_speed").isNotNull() &
            (col("final_speed") >= 0) &
            (col("final_speed") <= 150)
        )
        .withColumn("minute_bucket", date_format(date_trunc("minute", col("collect_ts")), "yyyy-MM-dd HH:mm:00"))
    )

    avg_speed_df = clean_df.groupBy(
        "dt", "minute_bucket", "city", "district", "road_id", "road_name"
    ).agg(
        avg("final_speed").alias("avg_speed"),
        count(lit(1)).alias("record_count"),
        count(when(col("speed_source") == "api", 1)).alias("api_count"),
        count(when(col("speed_source") != "api", 1)).alias("estimated_count")
    )

    query = (
        avg_speed_df.writeStream
        .outputMode("update")
        .foreachBatch(write_avg_speed_batch)
        .option("checkpointLocation", "file:///mnt/d/bigdata/myproject/checkpoints/avg_speed")
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    main()
