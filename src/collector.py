import csv
import json
import logging
import os
import time
from datetime import datetime

from api_client import fetch_weather, fetch_traffic
from adapters import adapt_weather, adapt_traffic
from kafka_producer import get_producer


PROJECT_ROOT = "/mnt/d/bigdata/myproject"
ROAD_LIST_FILE = f"{PROJECT_ROOT}/data/config/road_list.csv"
RAW_DIR = f"{PROJECT_ROOT}/data/raw"
LOG_DIR = f"{PROJECT_ROOT}/logs"

SAVE_RAW_JSON = True
WEATHER_INTERVAL_SECONDS = 600
_LAST_WEATHER_COLLECT_TS = None


def setup_logger():
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger("collector")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(f"{LOG_DIR}/collector.log", encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger




def _safe_float(value):
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def should_collect_weather(now_ts: float) -> bool:
    global _LAST_WEATHER_COLLECT_TS

    if _LAST_WEATHER_COLLECT_TS is None:
        return True

    return (now_ts - _LAST_WEATHER_COLLECT_TS) >= WEATHER_INTERVAL_SECONDS


def mark_weather_collected(now_ts: float):
    global _LAST_WEATHER_COLLECT_TS
    _LAST_WEATHER_COLLECT_TS = now_ts

def load_road_list():
    roads = []

    with open(ROAD_LIST_FILE, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            is_active = str(row.get("is_active", "1")).strip()
            if is_active != "1":
                continue

            roads.append({
                "road_id": row.get("road_id", "").strip(),
                "road_name": row.get("road_name", "").strip(),
                "city": row.get("city", "").strip(),
                "district": row.get("district", "").strip(),
                "center_lng": row.get("center_lng", "").strip(),
                "center_lat": row.get("center_lat", "").strip(),
                "display_order": row.get("display_order", "").strip(),
                "remark": row.get("remark", "").strip(),
                "free_flow_speed": _safe_float(row.get("free_flow_speed")),
            })

    return roads


def save_raw_json(data: dict, prefix: str):
    if not SAVE_RAW_JSON:
        return

    os.makedirs(RAW_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(RAW_DIR, f"{prefix}_{ts}.json")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def enrich_weather_record(record: dict, collect_time: str, dt: str) -> dict:
    """
    统一补充项目标准字段
    """
    record["collect_time"] = collect_time
    record["dt"] = dt
    return record


def enrich_traffic_record(record: dict, road_meta: dict, collect_time: str, dt: str) -> dict:
    """
    将道路元数据补充到 traffic_record 中
    """
    record["collect_time"] = collect_time
    record["dt"] = dt
    record["road_id"] = road_meta.get("road_id")
    record["district"] = road_meta.get("district")

    # 保留地图层需要的中心点字段
    record["center_lng"] = road_meta.get("center_lng")
    record["center_lat"] = road_meta.get("center_lat")
    record["free_flow_speed"] = road_meta.get("free_flow_speed")

    return record


def collect_once():
    logger = setup_logger()
    roads = load_road_list()
    producer = get_producer()

    now = datetime.now()
    now_ts = time.time()
    collect_time = now.strftime("%Y-%m-%d %H:%M:%S")
    dt = now.strftime("%Y-%m-%d")

    logger.info(f"本轮采集开始，重点道路数量={len(roads)}")

    # 1. 天气按 10 分钟采一次
    if should_collect_weather(now_ts):
        try:
            raw_weather = fetch_weather()
            save_raw_json(raw_weather, "weather")

            parsed_weather = adapt_weather(raw_weather)
            parsed_weather = enrich_weather_record(parsed_weather, collect_time, dt)

            producer.send("weather_raw", value=parsed_weather)
            mark_weather_collected(now_ts)
            logger.info("天气数据发送成功 -> weather_raw")
        except Exception as e:
            logger.exception(f"天气采集失败: {e}")
    else:
        remaining = max(0, WEATHER_INTERVAL_SECONDS - int(now_ts - (_LAST_WEATHER_COLLECT_TS or 0)))
        logger.info(f"本轮跳过天气采集，距下一次天气采集约 {remaining} 秒")

    # 2. 采每条路的路况
    total_sent = 0

    for road in roads:
        road_name = road["road_name"]
        city = road["city"]

        try:
            raw_traffic = fetch_traffic(road_name, city)
            save_raw_json(raw_traffic, f"traffic_{road_name}")

            parsed_traffic_list = adapt_traffic(raw_traffic, city=city)

            if not parsed_traffic_list:
                logger.warning(f"道路无可发送记录: road_name={road_name}")
                continue

            for item in parsed_traffic_list:
                item = enrich_traffic_record(item, road, collect_time, dt)
                producer.send("traffic_raw", value=item)
                total_sent += 1

            logger.info(f"路况发送成功: road_name={road_name}, message_count={len(parsed_traffic_list)}")

        except Exception as e:
            logger.exception(f"路况采集失败: road_name={road_name}, error={e}")

    producer.flush()
    producer.close()

    logger.info(f"本轮采集结束，发送完成：traffic_count={total_sent}")