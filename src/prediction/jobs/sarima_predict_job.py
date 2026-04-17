import argparse
from datetime import datetime
from typing import Optional

from core.config import load_config
from prediction.baseline_forecaster import WeightedBaselineForecaster
from prediction.bias_calibrator import MeanBiasCalibrator
from prediction.evaluator import SimpleEvaluator
from prediction.fallback_policy import HierarchicalFallbackPolicy
from prediction.feature_builder import DefaultFeatureBuilder
from prediction.model_registry import FileModelRegistry
from prediction.output_writer import SparkCsvOutputWriter
from prediction.pipeline import SarimaPredictionPipeline
from prediction.residual_model import SarimaxResidualModel
from prediction.series_reader import SparkCsvSeriesReader
from streaming.runtime import create_spark_session


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SARIMA prediction pipeline once.")
    parser.add_argument(
        "--now",
        type=str,
        default=None,
        help="override current time, format: YYYY-mm-dd HH:MM:SS",
    )
    return parser.parse_args()


def _parse_now(raw: Optional[str]) -> datetime:
    if not raw:
        return datetime.now()
    return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")


def main():
    args = _parse_args()
    now = _parse_now(args.now)
    config = load_config()

    spark = create_spark_session("SarimaPredictJob")
    series_reader = SparkCsvSeriesReader(spark=spark, input_path=config.paths.avg_speed_output_path)
    output_writer = SparkCsvOutputWriter(spark=spark)

    feature_builder = DefaultFeatureBuilder(peak_windows=config.prediction.peak_windows)
    baseline_forecaster = WeightedBaselineForecaster(
        yesterday_weight=config.prediction.baseline_yesterday_weight,
        lastweek_weight=config.prediction.baseline_lastweek_weight,
    )
    bias_calibrator = MeanBiasCalibrator(bias_window_minutes=config.prediction.bias_window_minutes)
    fallback_policy = HierarchicalFallbackPolicy()
    model_registry = FileModelRegistry(base_dir=config.prediction.model_dir)
    evaluator = SimpleEvaluator()

    def model_factory(granularity: str):
        if granularity == "1m":
            return SarimaxResidualModel(
                order=config.prediction.sarimax_order_1m,
                seasonal_order=config.prediction.sarimax_seasonal_order_1m,
            )
        return SarimaxResidualModel(
            order=config.prediction.sarimax_order_15m,
            seasonal_order=config.prediction.sarimax_seasonal_order_15m,
        )

    pipeline = SarimaPredictionPipeline(
        config=config,
        series_reader=series_reader,
        feature_builder=feature_builder,
        baseline_forecaster=baseline_forecaster,
        residual_model_factory=model_factory,
        bias_calibrator=bias_calibrator,
        fallback_policy=fallback_policy,
        model_registry=model_registry,
        output_writer=output_writer,
        evaluator=evaluator,
    )

    result = pipeline.run_once(now=now)
    print(
        "SARIMA预测完成: "
        f"rows_written={result.rows_written}, "
        f"roads_total={result.roads_total}, "
        f"roads_success={result.roads_success}, "
        f"roads_fallback={result.roads_fallback}"
    )
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings[:20]:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
