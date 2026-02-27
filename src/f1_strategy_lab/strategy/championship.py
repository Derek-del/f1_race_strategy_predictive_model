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
            "driver_title_probability": 0.0,
            "constructors_title_probability": 0.0,
        }

    mean_win_prob = float(recommendations["win_probability"].mean()) if "win_probability" in recommendations else 0.0
    mean_strategy_score = float(recommendations["strategy_score"].mean()) if "strategy_score" in recommendations else 0.0
    mean_robustness = float(recommendations["robustness_window"].mean()) if "robustness_window" in recommendations else 0.0

    # Convert strategy metrics into title confidence proxies (0..1).
    score_signal = _sigmoid((mean_strategy_score - 20.5) / 0.55)
    robustness_signal = _sigmoid((16.5 - mean_robustness) / 2.2)
    driver_title_prob = max(0.0, min(1.0, 0.55 * mean_win_prob + 0.3 * score_signal + 0.15 * robustness_signal))
    constructors_title_prob = max(
        0.0, min(1.0, 0.42 * driver_title_prob + 0.38 * score_signal + 0.20 * robustness_signal)
    )

    return {
        "driver": target_driver,
        "team": team,
        "driver_title_probability": round(driver_title_prob, 4),
        "constructors_title_probability": round(constructors_title_prob, 4),
    }
