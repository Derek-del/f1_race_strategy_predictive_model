from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_training_data(n_events: int = 120, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    compounds = ["SOFT", "MEDIUM", "HARD"]
    rows: list[dict[str, float | int | str]] = []
    for i in range(n_events):
        event = f"Event_{i+1:02d}"
        temp = rng.normal(29, 7)
        humidity = np.clip(rng.normal(56, 12), 20, 95)
        wind = np.clip(rng.normal(11, 4), 1, 28)
        precip = max(0.0, rng.normal(0.5, 1.1))

        fp2_pace = rng.normal(90.5, 1.8)
        q_best = fp2_pace - abs(rng.normal(1.2, 0.6))
        deg_soft = np.clip(rng.normal(0.18, 0.07), 0.04, 0.42)
        deg_medium = np.clip(rng.normal(0.12, 0.05), 0.03, 0.3)
        deg_hard = np.clip(rng.normal(0.08, 0.04), 0.01, 0.2)
        fuel_proxy = np.clip(rng.normal(0.95, 0.35), 0.2, 2.2)

        traffic_idx = np.clip(rng.normal(0.5, 0.2), 0.05, 1.5)
        grip_idx = np.clip(rng.normal(1.0, 0.15), 0.65, 1.5)
        rain_idx = np.clip(precip / 3.0 + rng.normal(0.05, 0.08), 0.0, 1.4)

        compound = rng.choice(compounds)
        strategy_aggressiveness = {"SOFT": 1.0, "MEDIUM": 0.65, "HARD": 0.3}[compound]

        target_race_pace = (
            83.2
            + 0.62 * fp2_pace
            - 0.35 * q_best
            + 1.2 * deg_soft
            + 0.8 * deg_medium
            + 0.5 * deg_hard
            + 0.45 * fuel_proxy
            + 0.65 * traffic_idx
            - 1.7 * grip_idx
            + 1.8 * rain_idx
            + 0.02 * temp
            + 0.01 * humidity
            + 0.015 * wind
            + rng.normal(0, 0.35)
        )

        target_points = (
            25
            - 3.1 * (target_race_pace - 89.0)
            + 1.4 * strategy_aggressiveness
            - 1.8 * rain_idx
            + rng.normal(0, 2.0)
        )
        target_points = float(np.clip(target_points, 0, 26))

        rows.append(
            {
                "year": 2020 + (i % 5),
                "event_name": event,
                "team": "MCLAREN",
                "driver": "NOR",
                "fp2_avg_lap_sec": fp2_pace,
                "quali_best_lap_sec": q_best,
                "deg_soft": deg_soft,
                "deg_medium": deg_medium,
                "deg_hard": deg_hard,
                "fuel_load_proxy": fuel_proxy,
                "weather_temp_c": temp,
                "weather_humidity": humidity,
                "weather_precip_mm": precip,
                "weather_wind_kmh": wind,
                "weather_pressure_hpa": 1012.0 + rng.normal(0, 8),
                "cv_traffic_index": traffic_idx,
                "cv_grip_index": grip_idx,
                "cv_rain_index": rain_idx,
                "compound_bias": compound,
                "target_race_pace": target_race_pace,
                "target_points": target_points,
            }
        )

    return pd.DataFrame(rows)
