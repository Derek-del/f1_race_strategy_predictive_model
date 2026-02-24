from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from f1_strategy_lab.config.settings import SimulationConfig


F1_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}


@dataclass(slots=True)
class StrategyCandidate:
    name: str
    compounds: list[str]
    pit_laps: list[int]


def generate_candidate_strategies(
    total_laps: int,
    compounds: list[str] | None = None,
) -> list[StrategyCandidate]:
    compounds = compounds or ["SOFT", "MEDIUM", "HARD"]
    unique = [c.upper() for c in compounds]

    out: list[StrategyCandidate] = []

    # One-stop options
    for c1 in unique:
        for c2 in unique:
            if c1 == c2:
                continue
            for pit in [int(total_laps * 0.38), int(total_laps * 0.5), int(total_laps * 0.62)]:
                out.append(
                    StrategyCandidate(
                        name=f"ONE_STOP_{c1}_{c2}_L{pit}",
                        compounds=[c1, c2],
                        pit_laps=[pit],
                    )
                )

    # Two-stop options
    for c1 in unique:
        for c2 in unique:
            for c3 in unique:
                if len({c1, c2, c3}) < 2:
                    continue
                pit1 = int(total_laps * 0.32)
                pit2 = int(total_laps * 0.67)
                out.append(
                    StrategyCandidate(
                        name=f"TWO_STOP_{c1}_{c2}_{c3}_L{pit1}_{pit2}",
                        compounds=[c1, c2, c3],
                        pit_laps=[pit1, pit2],
                    )
                )

    dedup: dict[str, StrategyCandidate] = {c.name: c for c in out}
    return list(dedup.values())


def _compound_for_lap(strategy: StrategyCandidate, lap: int) -> str:
    if not strategy.pit_laps:
        return strategy.compounds[0]
    idx = 0
    for pit in strategy.pit_laps:
        if lap > pit:
            idx += 1
    idx = min(idx, len(strategy.compounds) - 1)
    return strategy.compounds[idx]


def _simulate_single_race(
    strategy: StrategyCandidate,
    total_laps: int,
    base_lap_time: float,
    degradation: dict[str, float],
    fuel_load_proxy: float,
    traffic_index: float,
    rain_index: float,
    cfg: SimulationConfig,
    rng: np.random.Generator,
) -> float:
    race_time = 0.0
    stint_age = 0
    pit_set = set(strategy.pit_laps)

    fuel_effect = max(0.05, min(0.3, fuel_load_proxy / max(total_laps * 0.06, 1)))

    for lap in range(1, total_laps + 1):
        if lap in pit_set:
            pit_loss = cfg.pit_loss_seconds
            if rng.random() < cfg.safety_car_probability:
                pit_loss *= rng.uniform(0.55, 0.75)
            race_time += pit_loss
            stint_age = 0

        compound = _compound_for_lap(strategy, lap)
        deg = degradation.get(compound, 0.1)

        fuel_term = -fuel_effect * (lap / total_laps) * 5.0
        tire_term = deg * stint_age
        traffic_term = traffic_index * rng.normal(0.1, 0.08)
        weather_term = rain_index * rng.normal(0.25, 0.12)
        noise = rng.normal(0.0, cfg.weather_uncertainty_seconds + cfg.traffic_uncertainty_seconds)

        lap_time = base_lap_time + fuel_term + tire_term + traffic_term + weather_term + noise
        race_time += max(lap_time, 40.0)
        stint_age += 1

    return float(race_time)


def _simulated_points(team_race_time: float, rng: np.random.Generator) -> int:
    rival_offsets = np.array([0.0, 4.0, 8.2, 12.0, 15.5, 19.8, 23.5, 27.0, 31.2], dtype=float)
    rival_noise = rng.normal(0, 2.8, size=len(rival_offsets))
    rival_times = team_race_time + rival_offsets + rival_noise

    # Lower race time is better.
    position = 1 + int(np.sum(rival_times < team_race_time))
    return int(F1_POINTS.get(position, 0))


def evaluate_strategies(
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
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, float | str]] = []

    for candidate in candidates:
        samples = np.array(
            [
                _simulate_single_race(
                    strategy=candidate,
                    total_laps=total_laps,
                    base_lap_time=base_lap_time,
                    degradation=degradation,
                    fuel_load_proxy=fuel_load_proxy,
                    traffic_index=traffic_index,
                    rain_index=rain_index,
                    cfg=cfg,
                    rng=rng,
                )
                for _ in range(cfg.n_simulations)
            ],
            dtype=float,
        )

        points = np.array([_simulated_points(t, rng=rng) for t in samples], dtype=float)
        p10 = float(np.percentile(samples, 10))
        p90 = float(np.percentile(samples, 90))
        expected = float(np.mean(samples))
        robustness = p90 - p10
        win_prob = float(np.mean(points >= 25))
        expected_points = float(np.mean(points))
        score = expected_points - 0.02 * robustness

        rows.append(
            {
                "strategy": candidate.name,
                "stops": len(candidate.pit_laps),
                "compounds": "->".join(candidate.compounds),
                "pit_laps": ",".join(str(x) for x in candidate.pit_laps),
                "expected_race_time": expected,
                "p10_time": p10,
                "p90_time": p90,
                "robustness_window": robustness,
                "win_probability": win_prob,
                "expected_points": expected_points,
                "strategy_score": score,
            }
        )

    frame = pd.DataFrame(rows)
    return frame.sort_values(by=["strategy_score", "expected_race_time"], ascending=[False, True]).reset_index(
        drop=True
    )
