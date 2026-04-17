import numpy as np
import pandas as pd
import pytest

from prediction.residual_model import SarimaxResidualModel

try:
    import statsmodels  # noqa: F401
    STATS_AVAILABLE = True
except Exception:
    STATS_AVAILABLE = False


def _build_series(length: int = 120) -> pd.Series:
    index = pd.date_range("2026-04-01 00:00:00", periods=length, freq="15min")
    values = 0.5 * np.sin(np.arange(length) / 8.0) + np.random.RandomState(7).normal(0, 0.1, size=length)
    return pd.Series(values, index=index)


@pytest.mark.skipif(not STATS_AVAILABLE, reason="statsmodels not available")
def test_sarimax_model_fit_predict_save_load(tmpdir):
    series = _build_series()
    model = SarimaxResidualModel(order=(1, 0, 1), seasonal_order=(0, 0, 0, 0))
    model.fit(series)

    pred_before = model.forecast(steps=3)
    assert len(pred_before) == 3

    model_path = tmpdir.join("model.pkl")
    model.save(str(model_path))

    loaded = SarimaxResidualModel.load(
        str(model_path),
        order=(1, 0, 1),
        seasonal_order=(0, 0, 0, 0),
    )
    pred_after = loaded.forecast(steps=3)
    assert len(pred_after) == 3
