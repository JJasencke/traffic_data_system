from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType, IntegerType, DoubleType


def create_spark_session(app_name: str):
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def get_traffic_schema():
    return StructType() \
        .add("source", StringType()) \
        .add("data_type", StringType()) \
        .add("api_status", IntegerType()) \
        .add("api_message", StringType()) \
        .add("collect_time", StringType()) \
        .add("dt", StringType()) \
        .add("road_id", StringType()) \
        .add("city", StringType()) \
        .add("district", StringType()) \
        .add("center_lng", StringType()) \
        .add("center_lat", StringType()) \
        .add("free_flow_speed", DoubleType()) \
        .add("road_name", StringType()) \
        .add("description", StringType()) \
        .add("overall_status", IntegerType()) \
        .add("overall_status_desc", StringType()) \
        .add("section_desc", StringType()) \
        .add("road_type", IntegerType()) \
        .add("congestion_distance", IntegerType()) \
        .add("speed", DoubleType()) \
        .add("section_status", IntegerType()) \
        .add("congestion_trend", StringType()) \
        .add("raw_json_full", StringType()) \
        .add("raw_json_road", StringType()) \
        .add("raw_json_section", StringType())


def get_weather_schema():
    return StructType() \
        .add("source", StringType()) \
        .add("data_type", StringType()) \
        .add("api_status", IntegerType()) \
        .add("api_message", StringType()) \
        .add("collect_time", StringType()) \
        .add("dt", StringType()) \
        .add("country", StringType()) \
        .add("province", StringType()) \
        .add("city", StringType()) \
        .add("district", StringType()) \
        .add("district_id", StringType()) \
        .add("weather_text", StringType()) \
        .add("temperature", DoubleType()) \
        .add("feels_like", DoubleType()) \
        .add("humidity", IntegerType()) \
        .add("wind_class", StringType()) \
        .add("wind_dir", StringType()) \
        .add("precipitation_1h", DoubleType()) \
        .add("clouds", IntegerType()) \
        .add("visibility", IntegerType()) \
        .add("aqi", IntegerType()) \
        .add("pm25", IntegerType()) \
        .add("pm10", IntegerType()) \
        .add("pressure", IntegerType()) \
        .add("raw_collect_time", StringType()) \
        .add("raw_json", StringType())