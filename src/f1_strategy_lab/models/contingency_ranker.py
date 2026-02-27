from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


@dataclass(slots=True)
class ContingencyRankResult:
    ranked: pd.DataFrame
    metrics: dict[str, float]


def _compound_features(compounds: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=compounds.index)
    out["n_soft"] = compounds.str.count("SOFT").astype(float)
    out["n_medium"] = compounds.str.count("MEDIUM").astype(float)
    out["n_hard"] = compounds.str.count("HARD").astype(float)
    return out


def _feature_frame(table: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "stops",
        "baseline_expected_race_time",
        "baseline_win_probability",
        "baseline_strategy_score",
        "baseline_robustness_window",
        "weather_change_strategy_score",
        "engine_conservation_strategy_score",
        "driver_error_recovery_strategy_score",
        "race_chaos_strategy_score",
        "weather_change_win_probability",
        "engine_conservation_win_probability",
        "driver_error_recovery_win_probability",
        "race_chaos_win_probability",
        "top3_scenario_hits",
    ]
    numeric = table[base_cols].copy()
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    return pd.concat([numeric, _compound_features(table["compounds"].astype(str))], axis=1)


def _target(table: pd.DataFrame) -> pd.Series:
    score_cols = [
        "weather_change_strategy_score",
        "engine_conservation_strategy_score",
        "driver_error_recovery_strategy_score",
        "race_chaos_strategy_score",
    ]
    win_cols = [
        "weather_change_win_probability",
        "engine_conservation_win_probability",
        "driver_error_recovery_win_probability",
        "race_chaos_win_probability",
    ]
    robustness_cols = [
        "weather_change_robustness_window",
        "engine_conservation_robustness_window",
        "driver_error_recovery_robustness_window",
        "race_chaos_robustness_window",
    ]

    scenario_score = table[score_cols].mean(axis=1)
    win_term = table[win_cols].mean(axis=1)
    robustness_term = table[robustness_cols].mean(axis=1)
    hits_term = table["top3_scenario_hits"].astype(float)

    return scenario_score + 0.2 * win_term + 0.06 * hits_term - 0.01 * robustness_term


class ContingencyRanker:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def rank(self, table: pd.DataFrame) -> ContingencyRankResult:
        work = table.copy()
        work = work.drop_duplicates(subset=["strategy"]).reset_index(drop=True)

        features = _feature_frame(work)
        target = _target(work)

        metrics = {"mae": 0.0}
        if len(work) >= 12:
            x_train, x_test, y_train, y_test = train_test_split(
                features,
                target,
                test_size=0.25,
                random_state=self.random_state,
            )
            model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=self.random_state,
            )
            model.fit(x_train, y_train)
            metrics["mae"] = float(mean_absolute_error(y_test, model.predict(x_test)))
            work["contingency_rank_score"] = model.predict(features)
        else:
            work["contingency_rank_score"] = target.to_numpy(dtype=float)

        work = work.sort_values(
            by=["contingency_rank_score", "baseline_strategy_score"],
            ascending=[False, False],
        ).reset_index(drop=True)
        return ContingencyRankResult(ranked=work, metrics=metrics)
