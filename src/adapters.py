from typing import Any, Dict, List, Optional


def safe_get(d: Dict[str, Any], *keys, default=None):
    """
    安全读取多层字典字段
    """
    current = d
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def adapt_weather(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    百度天气原始 JSON -> 项目标准 weather_record

    说明：
    1. 这里只做字段映射，不补 collect_time / dt
    2. collect_time / dt 由 collector 层统一补齐
    """
    location = safe_get(raw, "result", "location", default={}) or {}
    now = safe_get(raw, "result", "now", default={}) or {}

    record = {
        # 基础标识字段
        "source": "baidu_weather",
        "data_type": "weather",

        # 接口状态字段
        "api_status": raw.get("status"),
        "api_message": raw.get("message"),

        # 运行时字段（由 collector 层补齐）
        "collect_time": None,
        "dt": None,

        # 位置字段
        "country": location.get("country"),
        "province": location.get("province"),
        "city": location.get("city"),
        "district": location.get("name"),
        "district_id": location.get("id"),

        # 天气核心字段
        "weather_text": now.get("text"),
        "temperature": now.get("temp"),
        "feels_like": now.get("feels_like"),
        "humidity": now.get("rh"),
        "wind_class": now.get("wind_class"),
        "wind_dir": now.get("wind_dir"),
        "precipitation_1h": now.get("prec_1h"),
        "clouds": now.get("clouds"),
        "visibility": now.get("vis"),
        "aqi": now.get("aqi"),
        "pm25": now.get("pm25"),
        "pm10": now.get("pm10"),
        "pressure": now.get("pressure"),

        # 可选调试/扩展字段
        "raw_collect_time": now.get("uptime"),
        "raw_json": raw
    }

    return record


def adapt_traffic(raw: Dict[str, Any], city: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    百度路况原始 JSON -> 项目标准 traffic_record 列表

    说明：
    1. 一条道路可能包含多个 congestion_section，所以返回 list
    2. collect_time / dt / road_id / district / center_lng / center_lat
       由 collector 层根据道路元数据统一补齐
    """
    result: List[Dict[str, Any]] = []

    api_status = raw.get("status")
    api_message = raw.get("message")
    description = raw.get("description")

    evaluation = raw.get("evaluation", {}) or {}
    overall_status = evaluation.get("status")
    overall_status_desc = evaluation.get("status_desc")

    road_traffic = raw.get("road_traffic", []) or []

    for road in road_traffic:
        road_name = road.get("road_name")
        congestion_sections = road.get("congestion_sections", []) or []

        # 如果没有拥堵片段，也保留一条道路级记录
        if not congestion_sections:
            record = {
                # 基础标识字段
                "source": "baidu_traffic",
                "data_type": "traffic",

                # 接口状态字段
                "api_status": api_status,
                "api_message": api_message,

                # 运行时字段（由 collector 层补齐）
                "collect_time": None,
                "dt": None,

                # 道路元数据字段（由 collector 层补齐）
                "road_id": None,
                "city": city,
                "district": None,
                "center_lng": None,
                "center_lat": None,

                # 交通核心字段
                "road_name": road_name,
                "description": description,
                "overall_status": overall_status,
                "overall_status_desc": overall_status_desc,
                "section_desc": None,
                "road_type": None,
                "congestion_distance": None,
                "speed": None,
                "section_status": None,
                "congestion_trend": None,

                # 调试字段
                "raw_json_full": raw,
                "raw_json_road": road,
                "raw_json_section": None
            }
            result.append(record)
            continue

        for section in congestion_sections:
            record = {
                # 基础标识字段
                "source": "baidu_traffic",
                "data_type": "traffic",

                # 接口状态字段
                "api_status": api_status,
                "api_message": api_message,

                # 运行时字段（由 collector 层补齐）
                "collect_time": None,
                "dt": None,

                # 道路元数据字段（由 collector 层补齐）
                "road_id": None,
                "city": city,
                "district": None,
                "center_lng": None,
                "center_lat": None,

                # 交通核心字段
                "road_name": road_name,
                "description": description,
                "overall_status": overall_status,
                "overall_status_desc": overall_status_desc,
                "section_desc": section.get("section_desc"),
                "road_type": section.get("road_type"),
                "congestion_distance": section.get("congestion_distance"),
                "speed": section.get("speed"),
                "section_status": section.get("status"),
                "congestion_trend": section.get("congestion_trend"),

                # 调试字段
                "raw_json_full": raw,
                "raw_json_road": road,
                "raw_json_section": section
            }
            result.append(record)

    return result