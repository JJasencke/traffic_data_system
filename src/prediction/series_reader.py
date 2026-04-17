from datetime import datetime

import pandas as pd
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, lit, to_timestamp
except ImportError:  # pragma: no cover - optional for pure unit tests
    SparkSession = object
    col = None
    lit = None
    to_timestamp = None

from prediction.interfaces import SeriesReader


class SparkCsvSeriesReader(SeriesReader):
    def __init__(self, spark: SparkSession, input_path: str):
        if col is None or lit is None or to_timestamp is None:
            raise ImportError("pyspark 未安装，无法使用 SparkCsvSeriesReader")
        self.spark = spark
        self.input_path = input_path

    def read(self, now: datetime, history_days: int) -> pd.DataFrame:
        try:
            raw_df = self.spark.read.option("header", True).csv(self.input_path)
        except Exception as error:
            print(f"[WARN] 读取历史均速数据失败: {error}")
            return pd.DataFrame(columns=["road_id", "road_name", "ts", "avg_speed", "free_flow_speed"])

        if raw_df.rdd.isEmpty():
            return pd.DataFrame(columns=["road_id", "road_name", "ts", "avg_speed", "free_flow_speed"])

        columns = set(raw_df.columns)
        if "minute_bucket" not in columns or "road_id" not in columns or "avg_speed" not in columns:
            print("[WARN] avg_speed 输入数据缺少必要字段(minute_bucket/road_id/avg_speed)")
            return pd.DataFrame(columns=["road_id", "road_name", "ts", "avg_speed", "free_flow_speed"])

        road_name_col = col("road_name") if "road_name" in columns else lit(None).alias("road_name")
        free_flow_col = col("free_flow_speed") if "free_flow_speed" in columns else lit(None).alias("free_flow_speed")

        selected = (
            raw_df.select(
                col("road_id").alias("road_id"),
                road_name_col.alias("road_name"),
                col("avg_speed").cast("double").alias("avg_speed"),
                free_flow_col.cast("double").alias("free_flow_speed"),
                to_timestamp(col("minute_bucket"), "yyyy-MM-dd HH:mm:00").alias("ts"),
            )
            .filter(col("road_id").isNotNull() & col("ts").isNotNull())
        )

        pdf = selected.toPandas()
        if pdf.empty:
            return pd.DataFrame(columns=["road_id", "road_name", "ts", "avg_speed", "free_flow_speed"])

        pdf["ts"] = pd.to_datetime(pdf["ts"], errors="coerce")
        cutoff = pd.Timestamp(now) - pd.Timedelta(days=history_days)
        pdf = pdf[(pdf["ts"] >= cutoff) & (pdf["ts"] <= pd.Timestamp(now))]
        return pdf


class DataFrameSeriesReader(SeriesReader):
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.copy()

    def read(self, now: datetime, history_days: int) -> pd.DataFrame:
        if self.frame.empty:
            return pd.DataFrame(columns=["road_id", "road_name", "ts", "avg_speed", "free_flow_speed"])

        frame = self.frame.copy()
        if "ts" not in frame.columns:
            if "minute_bucket" in frame.columns:
                frame["ts"] = pd.to_datetime(frame["minute_bucket"], errors="coerce")
            else:
                frame["ts"] = pd.NaT

        if "road_name" not in frame.columns:
            frame["road_name"] = None
        if "free_flow_speed" not in frame.columns:
            frame["free_flow_speed"] = None

        frame["ts"] = pd.to_datetime(frame["ts"], errors="coerce")
        frame["avg_speed"] = pd.to_numeric(frame["avg_speed"], errors="coerce")
        cutoff = pd.Timestamp(now) - pd.Timedelta(days=history_days)
        frame = frame[(frame["ts"] >= cutoff) & (frame["ts"] <= pd.Timestamp(now))]
        return frame[["road_id", "road_name", "ts", "avg_speed", "free_flow_speed"]]
