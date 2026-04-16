from pyspark.sql.functions import col, from_json
from common import create_spark_session, get_traffic_schema


def write_detail_batch(batch_df, batch_id):
    count = batch_df.count()
    print(f"traffic_detail Batch {batch_id}, count={count}")

    if count == 0:
        return

    batch_df.show(5, truncate=False)

    (
        batch_df.write
        .mode("append")
        .partitionBy("dt")
        .parquet("hdfs://localhost:9000/traffic/history/traffic_detail")
    )


def main():
    spark = create_spark_session("TrafficDetailStreamJob")

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

    detail_df = traffic_df.select(
        "collect_time",
        "city",
        "district",
        "road_id",
        "road_name",
        "description",
        "overall_status",
        "overall_status_desc",
        "section_desc",
        "road_type",
        "congestion_distance",
        "speed",
        "section_status",
        "congestion_trend",
        "dt"
    ).filter(
        col("road_name").isNotNull() &
        col("collect_time").isNotNull() &
        col("dt").isNotNull()
    )

    query = (
        detail_df.writeStream
        .outputMode("append")
        .foreachBatch(write_detail_batch)
        .option("checkpointLocation", "file:///mnt/d/bigdata/myproject/checkpoints/traffic_detail")
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    main()