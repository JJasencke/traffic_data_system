from datetime import datetime
from typing import Sequence, Tuple

try:
    from typing import Protocol
except ImportError:  # pragma: no cover - py<3.8
    from typing_extensions import Protocol

import numpy as np
import pandas as pd

from prediction.types import PreparedSeries


class SeriesReader(Protocol):
    def read(self, now: datetime, history_days: int) -> pd.DataFrame:
        ...


class FeatureBuilder(Protocol):
    def prepare_series(
        self,
        raw_df: pd.DataFrame,
        now: datetime,
        freq: str,
        history_days: int,
        speed_min: float,
        speed_max: float,
    ) -> PreparedSeries:
        ...

    def build_future_index(self, now: datetime, freq: str, steps: int) -> pd.DatetimeIndex:
        ...

    def is_peak_time(self, now: datetime) -> bool:
        ...


class BaselineForecaster(Protocol):
    def forecast(self, series: pd.Series, future_index: pd.DatetimeIndex) -> np.ndarray:
        ...

    def forecast_on_index(self, series: pd.Series, target_index: pd.DatetimeIndex) -> np.ndarray:
        ...


class ResidualModel(Protocol):
    def fit(self, residual_series: pd.Series) -> "ResidualModel":
        ...

    def forecast(self, steps: int) -> np.ndarray:
        ...

    def residuals(self) -> pd.Series:
        ...

    def save(self, path: str) -> None:
        ...


class BiasCalibrator(Protocol):
    def apply(self, predictions: np.ndarray, residuals: pd.Series, freq: str) -> Tuple[np.ndarray, float]:
        ...


class FallbackPolicy(Protocol):
    def forecast(self, series: pd.Series, future_index: pd.DatetimeIndex) -> Tuple[np.ndarray, Sequence[str]]:
        ...


class ModelRegistry(Protocol):
    def save(self, road_id: str, granularity: str, model: ResidualModel, metadata: dict) -> str:
        ...


class Evaluator(Protocol):
    def mae(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        ...


class OutputWriter(Protocol):
    def write(self, output_df: pd.DataFrame, output_path: str) -> None:
        ...
