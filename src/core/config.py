import os
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - fallback for environments without python-dotenv
    def load_dotenv():
        return False


DEFAULT_PROJECT_ROOT = "/mnt/d/bigdata/myproject"
DEFAULT_WEATHER_LOCATION = "126.642464,45.756967"


def _to_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default

    normalized = value.strip().lower()
    return normalized in {"1", "true", "yes", "y", "on"}


def _to_int(value: Optional[str], default: int) -> int:
    if value is None:
        return default

    try:
        return int(value)
    except ValueError:
        return default


def _to_float(value: Optional[str], default: float) -> float:
    if value is None:
        return default

    try:
        return float(value)
    except ValueError:
        return default


def _to_int_tuple(value: Optional[str], default: Tuple[int, ...]) -> Tuple[int, ...]:
    if value is None:
        return default

    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        return default

    try:
        parsed = tuple(int(part) for part in parts)
    except ValueError:
        return default

    if len(parsed) != len(default):
        return default

    return parsed


@dataclass(frozen=True)
class ApiConfig:
    baidu_ak: str
    weather_location: str
    timeout_seconds: int


@dataclass(frozen=True)
class KafkaConfig:
    bootstrap_servers: str
    weather_topic: str
    traffic_topic: str
    retries: int
    acks: str


@dataclass(frozen=True)
class CollectConfig:
    interval_seconds: int
    weather_interval_seconds: int
    save_raw_json: bool


@dataclass(frozen=True)
class PathConfig:
    project_root: str
    road_list_file: str
    raw_dir: str
    log_dir: str
    traffic_detail_output_path: str
    weather_output_path: str
    avg_speed_output_path: str
    traffic_detail_checkpoint: str
    weather_checkpoint: str
    avg_speed_checkpoint: str


@dataclass(frozen=True)
class PredictionConfig:
    output_path: str
    model_dir: str
    history_days: int
    forecast_steps_15m: int
    forecast_steps_1m: int
    bias_window_minutes: int
    min_samples: int
    missing_rate_threshold: float
    speed_min: float
    speed_max: float
    peak_windows: str
    baseline_yesterday_weight: float
    baseline_lastweek_weight: float
    sarimax_order_15m: Tuple[int, int, int]
    sarimax_seasonal_order_15m: Tuple[int, int, int, int]
    sarimax_order_1m: Tuple[int, int, int]
    sarimax_seasonal_order_1m: Tuple[int, int, int, int]
    enable_short_term: bool
    residual_ci_lower_q: float
    residual_ci_upper_q: float
    timezone: str


@dataclass(frozen=True)
class AppConfig:
    api: ApiConfig
    kafka: KafkaConfig
    collect: CollectConfig
    paths: PathConfig
    prediction: PredictionConfig


def load_config() -> AppConfig:
    load_dotenv()

    project_root = os.getenv("PROJECT_ROOT", DEFAULT_PROJECT_ROOT)

    paths = PathConfig(
        project_root=project_root,
        road_list_file=os.getenv("ROAD_LIST_FILE", f"{project_root}/data/config/road_list.csv"),
        raw_dir=os.getenv("RAW_DIR", f"{project_root}/data/raw"),
        log_dir=os.getenv("LOG_DIR", f"{project_root}/logs"),
        traffic_detail_output_path=os.getenv(
            "TRAFFIC_DETAIL_OUTPUT_PATH",
            "hdfs://localhost:9000/traffic/history/traffic_detail",
        ),
        weather_output_path=os.getenv(
            "WEATHER_OUTPUT_PATH",
            "hdfs://localhost:9000/traffic/history/weather",
        ),
        avg_speed_output_path=os.getenv(
            "AVG_SPEED_OUTPUT_PATH",
            "hdfs://localhost:9000/traffic/history/avg_speed",
        ),
        traffic_detail_checkpoint=os.getenv(
            "TRAFFIC_DETAIL_CHECKPOINT",
            f"file://{project_root}/checkpoints/traffic_detail",
        ),
        weather_checkpoint=os.getenv(
            "WEATHER_CHECKPOINT",
            f"file://{project_root}/checkpoints/weather",
        ),
        avg_speed_checkpoint=os.getenv(
            "AVG_SPEED_CHECKPOINT",
            f"file://{project_root}/checkpoints/avg_speed",
        ),
    )

    api = ApiConfig(
        baidu_ak=os.getenv("BAIDU_AK", "").strip(),
        weather_location=os.getenv("WEATHER_LOCATION", DEFAULT_WEATHER_LOCATION),
        timeout_seconds=_to_int(os.getenv("API_TIMEOUT_SECONDS"), 15),
    )

    kafka = KafkaConfig(
        bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        weather_topic=os.getenv("KAFKA_WEATHER_TOPIC", "weather_raw"),
        traffic_topic=os.getenv("KAFKA_TRAFFIC_TOPIC", "traffic_raw"),
        retries=_to_int(os.getenv("KAFKA_RETRIES"), 3),
        acks=os.getenv("KAFKA_ACKS", "all"),
    )

    collect = CollectConfig(
        interval_seconds=_to_int(os.getenv("COLLECT_INTERVAL_SECONDS"), 60),
        weather_interval_seconds=_to_int(os.getenv("WEATHER_INTERVAL_SECONDS"), 600),
        save_raw_json=_to_bool(os.getenv("SAVE_RAW_JSON"), True),
    )

    prediction = PredictionConfig(
        output_path=os.getenv(
            "PREDICTION_OUTPUT_PATH",
            "hdfs://localhost:9000/traffic/prediction/sarima",
        ),
        model_dir=os.getenv("PREDICTION_MODEL_DIR", f"{project_root}/models/sarima"),
        history_days=_to_int(os.getenv("PREDICTION_HISTORY_DAYS"), 21),
        forecast_steps_15m=_to_int(os.getenv("PREDICTION_FORECAST_STEPS_15M"), 8),
        forecast_steps_1m=_to_int(os.getenv("PREDICTION_FORECAST_STEPS_1M"), 30),
        bias_window_minutes=_to_int(os.getenv("PREDICTION_BIAS_WINDOW_MINUTES"), 60),
        min_samples=_to_int(os.getenv("PREDICTION_MIN_SAMPLES"), 96),
        missing_rate_threshold=_to_float(os.getenv("PREDICTION_MISSING_RATE_THRESHOLD"), 0.20),
        speed_min=_to_float(os.getenv("PREDICTION_SPEED_MIN"), 0.0),
        speed_max=_to_float(os.getenv("PREDICTION_SPEED_MAX"), 150.0),
        peak_windows=os.getenv("PREDICTION_PEAK_WINDOWS", "07:00-09:00,17:00-19:00"),
        baseline_yesterday_weight=_to_float(os.getenv("PREDICTION_BASELINE_YESTERDAY_WEIGHT"), 0.6),
        baseline_lastweek_weight=_to_float(os.getenv("PREDICTION_BASELINE_LASTWEEK_WEIGHT"), 0.4),
        sarimax_order_15m=_to_int_tuple(os.getenv("PREDICTION_SARIMAX_ORDER_15M"), (1, 1, 1)),
        sarimax_seasonal_order_15m=_to_int_tuple(
            os.getenv("PREDICTION_SARIMAX_SEASONAL_ORDER_15M"),
            (1, 1, 1, 96),
        ),
        sarimax_order_1m=_to_int_tuple(os.getenv("PREDICTION_SARIMAX_ORDER_1M"), (1, 0, 1)),
        sarimax_seasonal_order_1m=_to_int_tuple(
            os.getenv("PREDICTION_SARIMAX_SEASONAL_ORDER_1M"),
            (0, 0, 0, 0),
        ),
        enable_short_term=_to_bool(os.getenv("PREDICTION_ENABLE_SHORT_TERM"), True),
        residual_ci_lower_q=_to_float(os.getenv("PREDICTION_RESIDUAL_CI_LOWER_Q"), 0.10),
        residual_ci_upper_q=_to_float(os.getenv("PREDICTION_RESIDUAL_CI_UPPER_Q"), 0.90),
        timezone=os.getenv("PREDICTION_TIMEZONE", "Asia/Shanghai"),
    )

    return AppConfig(api=api, kafka=kafka, collect=collect, paths=paths, prediction=prediction)
