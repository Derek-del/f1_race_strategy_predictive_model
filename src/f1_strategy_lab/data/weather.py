from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from f1_strategy_lab.utils.io import load_json, save_json

TRACK_COORDS: dict[str, tuple[float, float]] = {
    "bahrain": (26.0325, 50.5106),
    "saudi arabian": (21.6319, 39.1044),
    "australian": (-37.8497, 144.968),
    "japanese": (34.8431, 136.541),
    "chinese": (31.3389, 121.219),
    "miami": (25.9581, -80.2389),
    "emilia romagna": (44.3439, 11.7167),
    "monaco": (43.7347, 7.4206),
    "canadian": (45.5, -73.5228),
    "spanish": (41.57, 2.261),
    "austrian": (47.2197, 14.7647),
    "british": (52.0786, -1.0169),
    "hungarian": (47.5789, 19.2486),
    "belgian": (50.4372, 5.9714),
    "dutch": (52.3888, 4.5409),
    "italian": (45.6156, 9.2811),
    "azerbaijan": (40.3725, 49.8533),
    "singapore": (1.2914, 103.864),
    "united states": (30.1328, -97.6411),
    "mexico city": (19.4056, -99.0927),
    "sao paulo": (-23.7036, -46.6997),
    "las vegas": (36.1162, -115.174),
    "qatar": (25.49, 51.454),
    "abu dhabi": (24.4672, 54.6031),
}


def _normalize_event_name(event_name: str) -> str:
    cleaned = event_name.lower().replace("grand prix", "").replace("gp", "").strip()
    return " ".join(cleaned.split())


def _resolve_coords(event_name: str) -> tuple[float, float] | None:
    key = _normalize_event_name(event_name)
    for name, coords in TRACK_COORDS.items():
        if name in key:
            return coords
    return None


def _hourly_request(lat: float, lon: float, when: datetime) -> dict[str, Any]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,windspeed_10m,surface_pressure",
        "timezone": "UTC",
        "start_date": when.strftime("%Y-%m-%d"),
        "end_date": when.strftime("%Y-%m-%d"),
    }
    response = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def get_weather_features(
    event_name: str,
    event_datetime: datetime,
    weather_cache_dir: str,
) -> dict[str, float]:
    cache_path = Path(weather_cache_dir) / f"{event_datetime.year}_{event_name.replace(' ', '_')}.json"
    cached = load_json(cache_path)
    payload: dict[str, Any] | None = cached

    coords = _resolve_coords(event_name)
    if coords is not None and payload is None:
        lat, lon = coords
        try:
            payload = _hourly_request(lat, lon, event_datetime)
            save_json(cache_path, payload)
        except Exception:
            payload = None

    default = {
        "weather_temp_c": 28.0,
        "weather_humidity": 55.0,
        "weather_precip_mm": 0.0,
        "weather_wind_kmh": 12.0,
        "weather_pressure_hpa": 1012.0,
    }
    if payload is None:
        return default

    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return default

    target_hour = event_datetime.strftime("%Y-%m-%dT%H:00")
    idx = 0
    if target_hour in times:
        idx = times.index(target_hour)

    def pick(name: str, fallback: float) -> float:
        values = hourly.get(name, [])
        if idx < len(values) and values[idx] is not None:
            return float(values[idx])
        return fallback

    return {
        "weather_temp_c": pick("temperature_2m", default["weather_temp_c"]),
        "weather_humidity": pick("relative_humidity_2m", default["weather_humidity"]),
        "weather_precip_mm": pick("precipitation", default["weather_precip_mm"]),
        "weather_wind_kmh": pick("windspeed_10m", default["weather_wind_kmh"]),
        "weather_pressure_hpa": pick("surface_pressure", default["weather_pressure_hpa"]),
    }
