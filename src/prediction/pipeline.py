from __future__ import annotations

from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from core.config import AppConfig
from prediction.interfaces import (
    BaselineForecaster,
    BiasCalibrator,
    Evaluator,
    FallbackPolicy,
    FeatureBuilder,
    ModelRegistry,
    OutputWriter,
    ResidualModel,
    SeriesReader,
)
from prediction.types import PipelineResult, PredictionRecord


class SarimaPredictionPipeline:
    def __init__(
        self,
        config: AppConfig,
        series_reader: SeriesReader,
        feature_builder: FeatureBuilder,
        baseline_forecaster: BaselineForecaster,
        residual_model_factory: Callable[[str], ResidualModel],
        bias_calibrator: BiasCalibrator,
        fallback_policy: FallbackPolicy,
        model_registry: ModelRegistry,
        output_writer: OutputWriter,
        evaluator: Evaluator,
    ):
        self.config = config
        self.series_reader = series_reader
        self.feature_builder = feature_builder
        self.baseline_forecaster = baseline_forecaster
        self.residual_model_factory = residual_model_factory
        self.bias_calibrator = bias_calibrator
        self.fallback_policy = fallback_policy
        self.model_registry = model_registry
        self.output_writer = output_writer
        self.evaluator = evaluator
        self.road_free_flow_map = self._load_road_free_flow_map(config.paths.road_list_file)

    @staticmethod
    def _load_road_free_flow_map(path: str) -> Dict[str, float]:
        try:
            table = pd.read_csv(path)
        except Exception:
            return {}

        if "road_id" not in table.columns or "free_flow_speed" not in table.columns:
            return {}

        result: Dict[str, float] = {}
        for _, row in table.iterrows():
            road_id = str(row.get("road_id", "")).strip()
            value = pd.to_numeric(row.get("free_flow_speed"), errors="coerce")
            if road_id and pd.notna(value) and float(value) > 0:
                result[road_id] = float(value)
        return result

    def _estimate_free_flow(self, road_id: str, road_df: pd.DataFrame, series: pd.Series) -> float:
        if road_id in self.road_free_flow_map:
            return self.road_free_flow_map[road_id]

        if "free_flow_speed" in road_df.columns:
            value = pd.to_numeric(road_df["free_flow_speed"], errors="coerce").dropna()
            if not value.empty and float(value.iloc[-1]) > 0:
                return float(value.iloc[-1])

        if series.notna().any():
            q90 = float(series.quantile(0.90))
            if q90 > 0:
                return q90

        return max(1.0, self.config.prediction.speed_max)

    @staticmethod
    def _clip_predictions(values: np.ndarray, speed_min: float, speed_max: float) -> np.ndarray:
        if values.size == 0:
            return values
        return np.clip(values.astype(float), speed_min, speed_max)

    def _build_ci(self, pred_speed: np.ndarray, residuals: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        if residuals.empty:
            spread = np.full(pred_speed.shape, 5.0, dtype=float)
            return pred_speed - spread, pred_speed + spread

        lower_delta = float(residuals.quantile(self.config.prediction.residual_ci_lower_q))
        upper_delta = float(residuals.quantile(self.config.prediction.residual_ci_upper_q))
        if np.isnan(lower_delta) or np.isnan(upper_delta):
            std = float(residuals.std()) if pd.notna(residuals.std()) else 5.0
            lower_delta, upper_delta = -std, std
        return pred_speed + lower_delta, pred_speed + upper_delta

    def _build_records(
        self,
        road_id: str,
        road_name: str,
        future_index: pd.DatetimeIndex,
        pred_speed: np.ndarray,
        lower_ci: np.ndarray,
        upper_ci: np.ndarray,
        granularity: str,
        model_version: str,
        fallback_used: bool,
        free_flow_speed: float,
    ) -> List[PredictionRecord]:
        if len(future_index) == 0:
            return []

        step_minutes = int(pd.to_timedelta("15min" if granularity == "15m" else "1min").total_seconds() // 60)
        records: List[PredictionRecord] = []

        for idx, ts in enumerate(future_index, start=1):
            speed = float(pred_speed[idx - 1])
            ci_low = float(lower_ci[idx - 1])
            ci_high = float(upper_ci[idx - 1])
            congestion_index = 1.0 - (speed / max(1e-6, free_flow_speed))
            congestion_index = float(np.clip(congestion_index, 0.0, 1.0))

            records.append(
                PredictionRecord(
                    road_id=road_id,
                    road_name=road_name,
                    predict_time=pd.Timestamp(ts).to_pydatetime(),
                    horizon_minute=idx * step_minutes,
                    granularity=granularity,
                    pred_speed=speed,
                    pred_congestion_index=congestion_index,
                    lower_ci=ci_low,
                    upper_ci=ci_high,
                    model_version=model_version,
                    fallback_used=fallback_used,
                    dt=pd.Timestamp(ts).strftime("%Y-%m-%d"),
                )
            )
        return records

    def _forecast_granularity(
        self,
        road_id: str,
        road_name: str,
        road_df: pd.DataFrame,
        now: datetime,
        freq: str,
        steps: int,
        granularity: str,
    ) -> Tuple[List[PredictionRecord], bool, Sequence[str]]:
        if steps <= 0:
            return [], False, []

        prepared = self.feature_builder.prepare_series(
            raw_df=road_df,
            now=now,
            freq=freq,
            history_days=self.config.prediction.history_days,
            speed_min=self.config.prediction.speed_min,
            speed_max=self.config.prediction.speed_max,
        )
        series = prepared.series
        if series.empty:
            return [], True, ["empty_series"]

        future_index = self.feature_builder.build_future_index(now=now, freq=freq, steps=steps)
        free_flow_speed = self._estimate_free_flow(road_id=road_id, road_df=road_df, series=series)

        warnings: List[str] = []
        should_fallback = (
            len(series) < self.config.prediction.min_samples
            or prepared.missing_rate > self.config.prediction.missing_rate_threshold
        )
        if should_fallback:
            warnings.append(
                f"{road_id}:{granularity} fallback by threshold (samples={len(series)}, missing={prepared.missing_rate:.3f})"
            )

        if not should_fallback:
            try:
                baseline_hist = self.baseline_forecaster.forecast_on_index(series, series.index)
                baseline_hist_series = pd.Series(baseline_hist, index=series.index)
                residual_train = (series - baseline_hist_series).dropna()

                if len(residual_train) < self.config.prediction.min_samples:
                    should_fallback = True
                    warnings.append(f"{road_id}:{granularity} fallback by residual samples={len(residual_train)}")
                else:
                    model = self.residual_model_factory(granularity)
                    model.fit(residual_train)

                    baseline_future = self.baseline_forecaster.forecast(series, future_index)
                    residual_future = model.forecast(len(future_index))
                    pred_speed = baseline_future + residual_future
                    pred_speed, _ = self.bias_calibrator.apply(pred_speed, model.residuals(), freq=freq)
                    pred_speed = self._clip_predictions(
                        pred_speed,
                        self.config.prediction.speed_min,
                        self.config.prediction.speed_max,
                    )
                    lower_ci, upper_ci = self._build_ci(pred_speed, model.residuals())

                    metadata = {
                        "freq": freq,
                        "samples": int(len(series)),
                        "missing_rate": float(prepared.missing_rate),
                    }
                    model_version = self.model_registry.save(
                        road_id=road_id,
                        granularity=granularity,
                        model=model,
                        metadata=metadata,
                    )

                    records = self._build_records(
                        road_id=road_id,
                        road_name=road_name,
                        future_index=future_index,
                        pred_speed=pred_speed,
                        lower_ci=lower_ci,
                        upper_ci=upper_ci,
                        granularity=granularity,
                        model_version=model_version,
                        fallback_used=False,
                        free_flow_speed=free_flow_speed,
                    )
                    return records, False, warnings
            except Exception as error:
                should_fallback = True
                warnings.append(f"{road_id}:{granularity} fallback by model error={error}")

        fallback_values, _ = self.fallback_policy.forecast(series=series, future_index=future_index)
        fallback_values = self._clip_predictions(
            fallback_values,
            self.config.prediction.speed_min,
            self.config.prediction.speed_max,
        )
        std = float(series.std()) if pd.notna(series.std()) else 5.0
        lower_ci = fallback_values - std
        upper_ci = fallback_values + std
        records = self._build_records(
            road_id=road_id,
            road_name=road_name,
            future_index=future_index,
            pred_speed=fallback_values,
            lower_ci=lower_ci,
            upper_ci=upper_ci,
            granularity=granularity,
            model_version="fallback",
            fallback_used=True,
            free_flow_speed=free_flow_speed,
        )
        return records, True, warnings

    def run_once(self, now: Optional[datetime] = None) -> PipelineResult:
        now = now or datetime.now()
        history_df = self.series_reader.read(now=now, history_days=self.config.prediction.history_days)
        if history_df.empty:
            return PipelineResult(rows_written=0, roads_total=0, roads_success=0, roads_fallback=0, warnings=["empty_input"])

        history_df = history_df.copy()
        history_df["road_id"] = history_df["road_id"].astype(str)
        roads_total = int(history_df["road_id"].nunique())
        roads_success = 0
        roads_fallback = 0
        warnings: List[str] = []
        all_records: List[PredictionRecord] = []

        for road_id, road_df in history_df.groupby("road_id"):
            road_name = str(road_df["road_name"].dropna().iloc[0]) if road_df["road_name"].notna().any() else road_id
            road_records: List[PredictionRecord] = []
            road_fallback = False

            records_15m, fallback_15m, warns_15m = self._forecast_granularity(
                road_id=road_id,
                road_name=road_name,
                road_df=road_df,
                now=now,
                freq="15min",
                steps=self.config.prediction.forecast_steps_15m,
                granularity="15m",
            )
            road_records.extend(records_15m)
            warnings.extend(warns_15m)
            road_fallback = road_fallback or fallback_15m

            if self.config.prediction.enable_short_term and self.feature_builder.is_peak_time(now):
                records_1m, fallback_1m, warns_1m = self._forecast_granularity(
                    road_id=road_id,
                    road_name=road_name,
                    road_df=road_df,
                    now=now,
                    freq="1min",
                    steps=self.config.prediction.forecast_steps_1m,
                    granularity="1m",
                )
                road_records.extend(records_1m)
                warnings.extend(warns_1m)
                road_fallback = road_fallback or fallback_1m

            if road_records:
                all_records.extend(road_records)
                roads_success += 1
                if road_fallback:
                    roads_fallback += 1

        output_df = pd.DataFrame([item.to_dict() for item in all_records])
        if not output_df.empty:
            self.output_writer.write(output_df=output_df, output_path=self.config.prediction.output_path)

        return PipelineResult(
            rows_written=len(output_df),
            roads_total=roads_total,
            roads_success=roads_success,
            roads_fallback=roads_fallback,
            warnings=warnings,
        )
