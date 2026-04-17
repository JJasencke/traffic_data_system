import numpy as np
import pandas as pd

from prediction.bias_calibrator import MeanBiasCalibrator
from prediction.fallback_policy import HierarchicalFallbackPolicy


def test_bias_calibrator_applies_recent_mean_bias():
    calibrator = MeanBiasCalibrator(bias_window_minutes=60)
    preds = np.array([30.0, 35.0, 40.0], dtype=float)
    residuals = pd.Series([1.0, 1.0, 1.0, 1.0], index=pd.RangeIndex(4))

    calibrated, bias = calibrator.apply(predictions=preds, residuals=residuals, freq="15min")
    assert np.allclose(calibrated, np.array([31.0, 36.0, 41.0]))
    assert np.isclose(bias, 1.0)


def test_fallback_policy_honors_yesterday_then_lastweek_then_median():
    index = pd.to_datetime(
        [
            "2026-04-10 08:00:00",  # lastweek for 2026-04-17 08:00
            "2026-04-16 08:00:00",  # yesterday for 2026-04-17 08:00
            "2026-04-16 08:01:00",  # yesterday for 2026-04-17 08:01
        ]
    )
    series = pd.Series([20.0, 40.0, 41.0], index=index)
    future = pd.to_datetime(
        [
            "2026-04-17 08:00:00",
            "2026-04-17 08:01:00",
            "2026-04-17 09:00:00",
        ]
    )

    policy = HierarchicalFallbackPolicy()
    values, sources = policy.forecast(series=series, future_index=pd.DatetimeIndex(future))

    assert np.isclose(values[0], 40.0)
    assert np.isclose(values[1], 41.0)
    assert np.isclose(values[2], float(series.median()))
    assert list(sources) == ["yesterday", "yesterday", "median"]
