import os
import requests
from dotenv import load_dotenv

load_dotenv()

BAIDU_AK = os.getenv("BAIDU_AK")

if not BAIDU_AK:
    raise ValueError("没有读取到 BAIDU_AK，请检查 .env 文件")


def fetch_weather(location: str = "126.642464,45.756967") -> dict:
    """
    获取哈尔滨天气数据
    location: 经度在前，纬度在后
    """
    url = "https://api.map.baidu.com/weather/v1/"
    params = {
        "location": location,
        "coordtype": "wgs84",
        "data_type": "now",
        "output": "json",
        "ak": BAIDU_AK
    }

    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_traffic(road_name: str, city: str) -> dict:
    """
    获取单条道路实时路况
    """
    url = "https://api.map.baidu.com/traffic/v1/road"
    params = {
        "road_name": road_name,
        "city": city,
        "ak": BAIDU_AK
    }

    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()