from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import pandas as pd


@dataclass(frozen=True)
class PreparedSeries:
    series: pd.Series
    missing_rate: float
    median_speed: float


@dataclass(frozen=True)
class PredictionRecord:
    road_id: str
    road_name: str
    predict_time: datetime
    horizon_minute: int
    granularity: str
    pred_speed: float
    pred_congestion_index: float
    lower_ci: float
    upper_ci: float
    model_version: str
    fallback_used: bool
    dt: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "road_id": self.road_id,
            "road_name": self.road_name,
            "predict_time": self.predict_time.strftime("%Y-%m-%d %H:%M:%S"),
            "horizon_minute": self.horizon_minute,
            "granularity": self.granularity,
            "pred_speed": float(self.pred_speed),
            "pred_congestion_index": float(self.pred_congestion_index),
            "lower_ci": float(self.lower_ci),
            "upper_ci": float(self.upper_ci),
            "model_version": self.model_version,
            "fallback_used": bool(self.fallback_used),
            "dt": self.dt,
        }


@dataclass(frozen=True)
class PipelineResult:
    rows_written: int
    roads_total: int
    roads_success: int
    roads_fallback: int
    warnings: List[str]
