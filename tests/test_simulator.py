from __future__ import annotations

from f1_strategy_lab.config.settings import SimulationConfig
from f1_strategy_lab.strategy.simulator import evaluate_strategies, generate_candidate_strategies


def test_strategy_simulation_returns_ranked_table() -> None:
    cfg = SimulationConfig(
        n_simulations=120,
        pit_loss_seconds=21.5,
        safety_car_probability=0.2,
        weather_uncertainty_seconds=0.2,
        traffic_uncertainty_seconds=0.2,
    )

    candidates = generate_candidate_strategies(total_laps=58, compounds=["SOFT", "MEDIUM", "HARD"])
    table = evaluate_strategies(
        candidates=candidates,
        total_laps=58,
        base_lap_time=90.5,
        degradation={"SOFT": 0.18, "MEDIUM": 0.12, "HARD": 0.08},
        fuel_load_proxy=0.9,
        traffic_index=0.5,
        rain_index=0.1,
        cfg=cfg,
        random_state=99,
    )

    assert not table.empty
    assert table.iloc[0]["strategy_score"] >= table.iloc[-1]["strategy_score"]
    assert {"strategy", "expected_points", "win_probability"}.issubset(table.columns)
