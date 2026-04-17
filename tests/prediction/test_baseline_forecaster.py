import numpy as np
import pandas as pd

from prediction.baseline_forecaster import WeightedBaselineForecaster


def test_weighted_baseline_prefers_yesterday_and_lastweek():
    idx = pd.to_datetime(
        [
            "2026-04-10 08:00:00",
            "2026-04-16 08:00:00",
        ]
    )
    series = pd.Series([30.0, 50.0], index=idx)
    future_index = pd.DatetimeIndex([pd.Timestamp("2026-04-17 08:00:00")])

    forecaster = WeightedBaselineForecaster(yesterday_weight=0.6, lastweek_weight=0.4)
    forecast = forecaster.forecast(series=series, future_index=future_index)

    assert forecast.shape == (1,)
    assert np.isclose(forecast[0], 42.0)


def test_weighted_baseline_falls_back_to_median_when_history_missing():
    idx = pd.to_datetime(["2026-04-16 08:00:00", "2026-04-16 08:15:00"])
    series = pd.Series([40.0, 50.0], index=idx)
    future_index = pd.DatetimeIndex([pd.Timestamp("2026-04-17 09:00:00")])

    forecaster = WeightedBaselineForecaster()
    forecast = forecaster.forecast(series=series, future_index=future_index)

    assert np.isclose(forecast[0], 45.0)
