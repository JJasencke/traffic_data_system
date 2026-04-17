from typing import Optional

import numpy as np
import pandas as pd

from prediction.interfaces import BaselineForecaster


class WeightedBaselineForecaster(BaselineForecaster):
    def __init__(self, yesterday_weight: float = 0.6, lastweek_weight: float = 0.4):
        total = yesterday_weight + lastweek_weight
        if total <= 0:
            self.yesterday_weight = 0.6
            self.lastweek_weight = 0.4
        else:
            self.yesterday_weight = yesterday_weight / total
            self.lastweek_weight = lastweek_weight / total

    @staticmethod
    def _safe_value(series: pd.Series, ts: pd.Timestamp) -> Optional[float]:
        value = series.get(ts)
        if value is None or pd.isna(value):
            return None
        return float(value)

    def _forecast_one(self, series: pd.Series, ts: pd.Timestamp) -> float:
        yesterday = self._safe_value(series, ts - pd.Timedelta(days=1))
        lastweek = self._safe_value(series, ts - pd.Timedelta(days=7))
        median = float(series.median()) if series.notna().any() else 0.0

        if yesterday is not None and lastweek is not None:
            return self.yesterday_weight * yesterday + self.lastweek_weight * lastweek
        if yesterday is not None:
            return yesterday
        if lastweek is not None:
            return lastweek
        return median

    def forecast(self, series: pd.Series, future_index: pd.DatetimeIndex) -> np.ndarray:
        if len(future_index) == 0:
            return np.array([], dtype=float)
        return np.array([self._forecast_one(series, pd.Timestamp(ts)) for ts in future_index], dtype=float)

    def forecast_on_index(self, series: pd.Series, target_index: pd.DatetimeIndex) -> np.ndarray:
        if len(target_index) == 0:
            return np.array([], dtype=float)
        return np.array([self._forecast_one(series, pd.Timestamp(ts)) for ts in target_index], dtype=float)
