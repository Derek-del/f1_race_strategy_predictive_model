from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from f1_strategy_lab.config.settings import SimulationConfig
from f1_strategy_lab.strategy.simulator import StrategyCandidate, evaluate_strategies


@dataclass(slots=True)
class ContingencyScenario:
    key: str
    label: str
    base_lap_delta: float = 0.0
    deg_multiplier: float = 1.0
    fuel_delta: float = 0.0
    traffic_delta: float = 0.0
    rain_delta: float = 0.0
    pit_loss_delta: float = 0.0
    safety_car_delta: float = 0.0
    weather_uncertainty_scale: float = 1.0
    traffic_uncertainty_scale: float = 1.0


CONTINGENCY_SCENARIOS: list[ContingencyScenario] = [
    ContingencyScenario(key="baseline", label="Baseline"),
    ContingencyScenario(
        key="weather_change",
        label="Weather Change",
        deg_multiplier=1.15,
        traffic_delta=0.1,
        rain_delta=0.35,
        weather_uncertainty_scale=1.35,
    ),
    ContingencyScenario(
        key="engine_conservation",
        label="Engine Concern",
        base_lap_delta=0.45,
        fuel_delta=0.12,
        pit_loss_delta=1.1,
        safety_car_delta=0.04,
    ),
    ContingencyScenario(
        key="driver_error_recovery",
        label="Driver Error Recovery",
        base_lap_delta=0.22,
        traffic_delta=0.24,
        safety_car_delta=0.06,
        traffic_uncertainty_scale=1.3,
    ),
    ContingencyScenario(
        key="race_chaos",
        label="Race Chaos",
        traffic_delta=0.18,
        rain_delta=0.14,
        safety_car_delta=0.16,
        weather_uncertainty_scale=1.2,
        traffic_uncertainty_scale=1.25,
    ),
]


def _scenario_cfg(
    base_cfg: SimulationConfig,
    scenario: ContingencyScenario,
    n_simulations: int | None = None,
) -> SimulationConfig:
    return SimulationConfig(
        n_simulations=base_cfg.n_simulations if n_simulations is None else int(n_simulations),
        pit_loss_seconds=max(5.0, base_cfg.pit_loss_seconds + scenario.pit_loss_delta),
        safety_car_probability=min(0.95, max(0.0, base_cfg.safety_car_probability + scenario.safety_car_delta)),
        weather_uncertainty_seconds=max(
            0.01, base_cfg.weather_uncertainty_seconds * scenario.weather_uncertainty_scale
        ),
        traffic_uncertainty_seconds=max(
            0.01, base_cfg.traffic_uncertainty_seconds * scenario.traffic_uncertainty_scale
        ),
    )


def _scenario_degradation(degradation: dict[str, float], scenario: ContingencyScenario) -> dict[str, float]:
    return {k: max(0.0, float(v) * scenario.deg_multiplier) for k, v in degradation.items()}


def _trim_eval_columns(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    keep = ["strategy", "stops", "compounds", "pit_laps", "expected_race_time", "win_probability", "strategy_score", "robustness_window"]
    work = frame[keep].copy()
    rename_map = {
        "expected_race_time": f"{prefix}_expected_race_time",
        "win_probability": f"{prefix}_win_probability",
        "strategy_score": f"{prefix}_strategy_score",
        "robustness_window": f"{prefix}_robustness_window",
    }
    return work.rename(columns=rename_map)


def evaluate_strategies_with_contingencies(
    candidates: list[StrategyCandidate],
    total_laps: int,
    base_lap_time: float,
    degradation: dict[str, float],
    fuel_load_proxy: float,
    traffic_index: float,
    rain_index: float,
    cfg: SimulationConfig,
    random_state: int = 42,
) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    contingency_sim_count = max(80, int(cfg.n_simulations * 0.2))
    contingency_sim_count = min(cfg.n_simulations, contingency_sim_count)

    for i, scenario in enumerate(CONTINGENCY_SCENARIOS):
        n_sim = cfg.n_simulations if scenario.key == "baseline" else contingency_sim_count
        eval_table = evaluate_strategies(
            candidates=candidates,
            total_laps=total_laps,
            base_lap_time=base_lap_time + scenario.base_lap_delta,
            degradation=_scenario_degradation(degradation, scenario),
            fuel_load_proxy=max(0.0, fuel_load_proxy + scenario.fuel_delta),
            traffic_index=max(0.0, traffic_index + scenario.traffic_delta),
            rain_index=max(0.0, rain_index + scenario.rain_delta),
            cfg=_scenario_cfg(cfg, scenario, n_simulations=n_sim),
            random_state=random_state + (i * 97),
        )

        tables.append(_trim_eval_columns(eval_table, scenario.key))

    merged = tables[0]
    for t in tables[1:]:
        merged = merged.merge(t, on=["strategy", "stops", "compounds", "pit_laps"], how="inner")

    top3_hits = pd.Series(0, index=merged.index, dtype=int)
    for scenario in CONTINGENCY_SCENARIOS:
        col = f"{scenario.key}_strategy_score"
        if col in merged.columns:
            ranked_idx = merged[col].sort_values(ascending=False).head(3).index
            top3_hits.loc[ranked_idx] += 1
    merged["top3_scenario_hits"] = top3_hits

    merged = merged.sort_values(by=["baseline_strategy_score", "baseline_expected_race_time"], ascending=[False, True])
    return merged.reset_index(drop=True)


def contingency_reason_from_row(row: pd.Series) -> str:
    scenario_to_reason = {
        "weather_change": "Weather change or sudden rain",
        "engine_conservation": "Engine reliability concern",
        "driver_error_recovery": "Driver error recovery",
        "race_chaos": "Safety car or race chaos",
    }

    score_cols = {k: f"{k}_strategy_score" for k in scenario_to_reason}
    available = {k: float(row.get(col, float("-inf"))) for k, col in score_cols.items()}
    best = max(available, key=available.get)
    return scenario_to_reason[best]
