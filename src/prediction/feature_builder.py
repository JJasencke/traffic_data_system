from datetime import datetime, time
from typing import List, Tuple

import pandas as pd

from prediction.interfaces import FeatureBuilder
from prediction.types import PreparedSeries


class DefaultFeatureBuilder(FeatureBuilder):
    def __init__(self, peak_windows: str):
        self._peak_windows = self._parse_peak_windows(peak_windows)

    @staticmethod
    def _parse_peak_windows(raw_windows: str) -> List[Tuple[time, time]]:
        windows: List[Tuple[time, time]] = []
        for item in raw_windows.split(","):
            token = item.strip()
            if not token or "-" not in token:
                continue
            start_raw, end_raw = token.split("-", 1)
            try:
                start = datetime.strptime(start_raw.strip(), "%H:%M").time()
                end = datetime.strptime(end_raw.strip(), "%H:%M").time()
            except ValueError:
                continue
            windows.append((start, end))
        return windows

    @staticmethod
    def _fill_missing(series: pd.Series, freq: str) -> pd.Series:
        if series.empty:
            return series

        freq_delta = pd.to_timedelta(freq)
        day_shift = int(pd.Timedelta(days=1) / freq_delta)
        week_shift = int(pd.Timedelta(days=7) / freq_delta)

        filled = series.copy()
        yesterday = filled.shift(day_shift)
        lastweek = filled.shift(week_shift)
        filled = filled.fillna(yesterday).fillna(lastweek)
        filled = filled.interpolate(limit_direction="both")
        median_speed = float(series.median()) if series.notna().any() else 0.0
        return filled.fillna(median_speed)

    def prepare_series(
        self,
        raw_df: pd.DataFrame,
        now: datetime,
        freq: str,
        history_days: int,
        speed_min: float,
        speed_max: float,
    ) -> PreparedSeries:
        if raw_df.empty:
            return PreparedSeries(series=pd.Series(dtype=float), missing_rate=1.0, median_speed=0.0)

        work = raw_df.copy()
        work["ts"] = pd.to_datetime(work["ts"], errors="coerce")
        work["avg_speed"] = pd.to_numeric(work["avg_speed"], errors="coerce")
        work = work.dropna(subset=["ts"])

        window_start = now - pd.Timedelta(days=history_days)
        work = work[(work["ts"] >= window_start) & (work["ts"] <= now)]
        if work.empty:
            return PreparedSeries(series=pd.Series(dtype=float), missing_rate=1.0, median_speed=0.0)

        work["avg_speed"] = work["avg_speed"].clip(lower=speed_min, upper=speed_max)
        grouped = work.groupby("ts", as_index=True)["avg_speed"].mean().sort_index()
        resampled = grouped.resample(freq).mean()
        missing_rate = float(resampled.isna().mean()) if len(resampled) else 1.0
        filled = self._fill_missing(resampled, freq)
        median_speed = float(filled.median()) if len(filled) else 0.0
        return PreparedSeries(series=filled, missing_rate=missing_rate, median_speed=median_speed)

    @staticmethod
    def build_future_index(now: datetime, freq: str, steps: int) -> pd.DatetimeIndex:
        if steps <= 0:
            return pd.DatetimeIndex([])
        floor_ts = pd.Timestamp(now).floor(freq)
        start = floor_ts + pd.to_timedelta(freq)
        return pd.date_range(start=start, periods=steps, freq=freq)

    def is_peak_time(self, now: datetime) -> bool:
        if not self._peak_windows:
            return False

        current = now.time()
        for start, end in self._peak_windows:
            if start <= current <= end:
                return True
        return False
