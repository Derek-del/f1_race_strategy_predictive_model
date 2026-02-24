from __future__ import annotations

import math

import pandas as pd


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def project_championship(
    recommendations: pd.DataFrame,
    target_driver: str,
    team: str,
) -> dict[str, float | str]:
    if recommendations.empty:
        return {
            "driver": target_driver,
            "team": team,
            "projected_driver_points": 0.0,
            "projected_teammate_points": 0.0,
            "projected_constructors_points": 0.0,
            "driver_title_probability": 0.0,
            "constructors_title_probability": 0.0,
        }

    driver_points = float(recommendations["expected_points"].sum())

    # Simple teammate proxy: team strategy quality scales second car output.
    avg_score = float(recommendations["strategy_score"].mean())
    teammate_multiplier = 0.75 + max(0.0, min(0.2, (avg_score - 8.0) / 20.0))
    teammate_points = driver_points * teammate_multiplier
    constructors_points = driver_points + teammate_points

    # Calibrated to plausible full-season ranges (without sprint modeling).
    driver_title_prob = _sigmoid((driver_points - 360.0) / 28.0)
    constructors_title_prob = _sigmoid((constructors_points - 620.0) / 38.0)

    return {
        "driver": target_driver,
        "team": team,
        "projected_driver_points": round(driver_points, 2),
        "projected_teammate_points": round(teammate_points, 2),
        "projected_constructors_points": round(constructors_points, 2),
        "driver_title_probability": round(driver_title_prob, 4),
        "constructors_title_probability": round(constructors_title_prob, 4),
    }
