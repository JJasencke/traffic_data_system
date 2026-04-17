from typing import Tuple

import numpy as np
import pandas as pd

from prediction.interfaces import BiasCalibrator


class MeanBiasCalibrator(BiasCalibrator):
    def __init__(self, bias_window_minutes: int = 60):
        self.bias_window_minutes = max(1, int(bias_window_minutes))

    def _window_points(self, freq: str) -> int:
        step_minutes = max(1, int(pd.to_timedelta(freq).total_seconds() // 60))
        return max(1, self.bias_window_minutes // step_minutes)

    def apply(self, predictions: np.ndarray, residuals: pd.Series, freq: str) -> Tuple[np.ndarray, float]:
        if predictions.size == 0:
            return predictions, 0.0
        if residuals.empty:
            return predictions, 0.0

        points = self._window_points(freq)
        bias = float(residuals.tail(points).mean())
        if pd.isna(bias):
            bias = 0.0
        return predictions + bias, bias
