from pyspark.sql.functions import col, from_json
from common import create_spark_session, get_weather_schema


def write_weather_batch(batch_df, batch_id):
    count = batch_df.count()
    print(f"weather_detail Batch {batch_id}, count={count}")

    if count == 0:
        return

    print("weather columns:", batch_df.columns)
    batch_df.show(5, truncate=False)

    (
        batch_df.write
        .mode("append")
        .option("header", False)
        .partitionBy("dt")
        .csv("hdfs://localhost:9000/traffic/history/weather")
    )


def main():
    spark = create_spark_session("WeatherDetailStreamJob")

    weather_schema = get_weather_schema()

    kafka_df = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", "localhost:9092")
        .option("subscribe", "weather_raw")
        .option("startingOffsets", "latest")
        .load()
    )

    value_df = kafka_df.selectExpr("CAST(value AS STRING) AS json_str")

    parsed_df = value_df.select(
        from_json(col("json_str"), weather_schema).alias("data")
    )

    weather_df = parsed_df.select("data.*")

    detail_df = weather_df.select(
        "collect_time",
        "country",
        "province",
        "city",
        "district",
        "district_id",
        "weather_text",
        "temperature",
        "feels_like",
        "humidity",
        "wind_class",
        "wind_dir",
        "precipitation_1h",
        "clouds",
        "visibility",
        "aqi",
        "pm25",
        "pm10",
        "pressure",
        "dt"
    ).filter(
        col("city").isNotNull() &
        col("collect_time").isNotNull() &
        col("dt").isNotNull()
    )

    query = (
        detail_df.writeStream
        .outputMode("append")
        .foreachBatch(write_weather_batch)
        .option("checkpointLocation", "file:///mnt/d/bigdata/myproject/checkpoints/weather")
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    main()