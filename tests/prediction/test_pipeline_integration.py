from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.config import ApiConfig, AppConfig, CollectConfig, KafkaConfig, PathConfig, PredictionConfig
from prediction.baseline_forecaster import WeightedBaselineForecaster
from prediction.bias_calibrator import MeanBiasCalibrator
from prediction.evaluator import SimpleEvaluator
from prediction.fallback_policy import HierarchicalFallbackPolicy
from prediction.feature_builder import DefaultFeatureBuilder
from prediction.model_registry import FileModelRegistry
from prediction.output_writer import LocalPartitionedCsvOutputWriter
from prediction.pipeline import SarimaPredictionPipeline
from prediction.residual_model import SarimaxResidualModel
from prediction.series_reader import DataFrameSeriesReader

try:
    import statsmodels  # noqa: F401
    STATS_AVAILABLE = True
except Exception:
    STATS_AVAILABLE = False


def _build_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        api=ApiConfig(baidu_ak="", weather_location="", timeout_seconds=15),
        kafka=KafkaConfig(
            bootstrap_servers="localhost:9092",
            weather_topic="weather_raw",
            traffic_topic="traffic_raw",
            retries=3,
            acks="all",
        ),
        collect=CollectConfig(interval_seconds=60, weather_interval_seconds=600, save_raw_json=False),
        paths=PathConfig(
            project_root=str(tmp_path),
            road_list_file=str(tmp_path / "road_list.csv"),
            raw_dir=str(tmp_path / "raw"),
            log_dir=str(tmp_path / "logs"),
            traffic_detail_output_path=str(tmp_path / "traffic_detail"),
            weather_output_path=str(tmp_path / "weather"),
            avg_speed_output_path=str(tmp_path / "avg_speed"),
            traffic_detail_checkpoint=str(tmp_path / "cp_traffic"),
            weather_checkpoint=str(tmp_path / "cp_weather"),
            avg_speed_checkpoint=str(tmp_path / "cp_avg"),
        ),
        prediction=PredictionConfig(
            output_path=str(tmp_path / "pred_output"),
            model_dir=str(tmp_path / "models"),
            history_days=14,
            forecast_steps_15m=4,
            forecast_steps_1m=5,
            bias_window_minutes=60,
            min_samples=20,
            missing_rate_threshold=0.9,
            speed_min=0.0,
            speed_max=150.0,
            peak_windows="07:00-09:00,17:00-19:00",
            baseline_yesterday_weight=0.6,
            baseline_lastweek_weight=0.4,
            sarimax_order_15m=(1, 0, 0),
            sarimax_seasonal_order_15m=(0, 0, 0, 0),
            sarimax_order_1m=(1, 0, 0),
            sarimax_seasonal_order_1m=(0, 0, 0, 0),
            enable_short_term=True,
            residual_ci_lower_q=0.1,
            residual_ci_upper_q=0.9,
            timezone="Asia/Shanghai",
        ),
    )


def _build_history_frame() -> pd.DataFrame:
    timeline = pd.date_range("2026-04-01 00:00:00", "2026-04-17 08:30:00", freq="1min")
    length = len(timeline)
    base = 45 + 8 * np.sin(np.arange(len(timeline)) / 20.0)
    frame = pd.DataFrame(
        {
            "road_id": ["r001"] * length,
            "road_name": ["TestRoad"] * length,
            "minute_bucket": timeline.strftime("%Y-%m-%d %H:%M:%S"),
            "avg_speed": base,
            "free_flow_speed": [60.0] * length,
        }
    )
    return frame


@pytest.mark.skipif(not STATS_AVAILABLE, reason="statsmodels not available")
def test_pipeline_run_once_writes_15m_and_1m_predictions(tmpdir):
    tmp_path = Path(str(tmpdir))
    config = _build_config(tmp_path)
    history = _build_history_frame()

    # keep road metadata available for congestion index
    pd.DataFrame({"road_id": ["r001"], "free_flow_speed": [60.0]}).to_csv(config.paths.road_list_file, index=False)

    reader = DataFrameSeriesReader(history)
    writer = LocalPartitionedCsvOutputWriter()
    registry = FileModelRegistry(config.prediction.model_dir)

    pipeline = SarimaPredictionPipeline(
        config=config,
        series_reader=reader,
        feature_builder=DefaultFeatureBuilder(config.prediction.peak_windows),
        baseline_forecaster=WeightedBaselineForecaster(
            config.prediction.baseline_yesterday_weight,
            config.prediction.baseline_lastweek_weight,
        ),
        residual_model_factory=lambda granularity: SarimaxResidualModel(
            order=(1, 0, 0),
            seasonal_order=(0, 0, 0, 0),
        ),
        bias_calibrator=MeanBiasCalibrator(config.prediction.bias_window_minutes),
        fallback_policy=HierarchicalFallbackPolicy(),
        model_registry=registry,
        output_writer=writer,
        evaluator=SimpleEvaluator(),
    )

    now = datetime(2026, 4, 17, 8, 30, 0)  # peak time to enable 1m short-term
    result = pipeline.run_once(now=now)

    assert result.rows_written >= (config.prediction.forecast_steps_15m + config.prediction.forecast_steps_1m)
    assert result.roads_total == 1
    assert result.roads_success == 1

    output_root = Path(config.prediction.output_path)
    assert output_root.exists()
    csv_files = list(output_root.rglob("predictions.csv"))
    assert csv_files, "should write partitioned prediction csv files"
