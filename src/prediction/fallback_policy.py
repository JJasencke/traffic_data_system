from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from prediction.interfaces import FallbackPolicy


class HierarchicalFallbackPolicy(FallbackPolicy):
    @staticmethod
    def _safe_value(series: pd.Series, ts: pd.Timestamp):
        value = series.get(ts)
        if value is None or pd.isna(value):
            return None
        return float(value)

    def forecast(self, series: pd.Series, future_index: pd.DatetimeIndex) -> Tuple[np.ndarray, Sequence[str]]:
        median = float(series.median()) if series.notna().any() else 0.0
        values: List[float] = []
        sources: List[str] = []

        for ts in future_index:
            ts = pd.Timestamp(ts)
            yesterday = self._safe_value(series, ts - pd.Timedelta(days=1))
            if yesterday is not None:
                values.append(yesterday)
                sources.append("yesterday")
                continue

            lastweek = self._safe_value(series, ts - pd.Timedelta(days=7))
            if lastweek is not None:
                values.append(lastweek)
                sources.append("lastweek")
                continue

            values.append(median)
            sources.append("median")

        return np.array(values, dtype=float), sources
