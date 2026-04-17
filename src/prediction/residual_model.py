import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from prediction.interfaces import ResidualModel

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
except ImportError:  # pragma: no cover - explicit runtime guard
    SARIMAX = None
    SARIMAXResults = None


class SarimaxResidualModel(ResidualModel):
    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 96),
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self._result = None

    @staticmethod
    def _ensure_statsmodels():
        if SARIMAX is None or SARIMAXResults is None:
            raise ImportError("缺少 statsmodels 依赖，请先安装 requirements.txt")

    def fit(self, residual_series: pd.Series) -> "SarimaxResidualModel":
        self._ensure_statsmodels()

        clean = pd.to_numeric(residual_series, errors="coerce").dropna()
        if clean.empty:
            raise ValueError("残差序列为空，无法训练 SARIMAX")

        model = SARIMAX(
            clean,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._result = model.fit(disp=False)
        return self

    def forecast(self, steps: int) -> np.ndarray:
        if self._result is None:
            raise RuntimeError("模型尚未训练，无法预测")
        if steps <= 0:
            return np.array([], dtype=float)
        values = self._result.forecast(steps=steps)
        return np.asarray(values, dtype=float)

    def residuals(self) -> pd.Series:
        if self._result is None:
            return pd.Series(dtype=float)
        return pd.Series(self._result.resid).dropna()

    def save(self, path: str) -> None:
        if self._result is None:
            raise RuntimeError("模型尚未训练，无法保存")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._result.save(path)

    @classmethod
    def load(
        cls,
        path: str,
        order: Tuple[int, int, int],
        seasonal_order: Tuple[int, int, int, int],
    ) -> "SarimaxResidualModel":
        cls._ensure_statsmodels()
        instance = cls(order=order, seasonal_order=seasonal_order)
        instance._result = SARIMAXResults.load(path)
        return instance

    @property
    def fitted(self) -> bool:
        return self._result is not None

    @property
    def aic(self) -> Optional[float]:
        if self._result is None:
            return None
        return float(self._result.aic)
