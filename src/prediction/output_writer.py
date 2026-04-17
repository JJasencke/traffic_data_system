from pathlib import Path

import pandas as pd
try:
    from pyspark.sql import SparkSession
except ImportError:  # pragma: no cover - optional for pure unit tests
    SparkSession = object

from prediction.interfaces import OutputWriter


class SparkCsvOutputWriter(OutputWriter):
    def __init__(self, spark: SparkSession):
        if SparkSession is object:
            raise ImportError("pyspark 未安装，无法使用 SparkCsvOutputWriter")
        self.spark = spark

    def write(self, output_df: pd.DataFrame, output_path: str) -> None:
        if output_df.empty:
            print("[INFO] 无预测结果可写入")
            return

        spark_df = self.spark.createDataFrame(output_df)
        (
            spark_df.write
            .mode("append")
            .option("header", True)
            .partitionBy("dt", "granularity")
            .csv(output_path)
        )
        print(f"[INFO] 预测结果已写入: {output_path}, rows={len(output_df)}")


class LocalPartitionedCsvOutputWriter(OutputWriter):
    def write(self, output_df: pd.DataFrame, output_path: str) -> None:
        if output_df.empty:
            return

        base = Path(output_path)
        for (dt, granularity), group in output_df.groupby(["dt", "granularity"]):
            target_dir = base / f"dt={dt}" / f"granularity={granularity}"
            target_dir.mkdir(parents=True, exist_ok=True)
            target_file = target_dir / "predictions.csv"
            exists = target_file.exists()
            group.to_csv(target_file, mode="a", header=not exists, index=False)
