from __future__ import annotations

import pandas as pd

from f1_strategy_lab.models.contingency_ranker import ContingencyRanker


def test_contingency_ranker_produces_scores() -> None:
    rows = []
    for i in range(14):
        rows.append(
            {
                "strategy": f"S{i}",
                "stops": 1 + (i % 2),
                "compounds": "MEDIUM->HARD" if i % 2 == 0 else "SOFT->MEDIUM->HARD",
                "pit_laps": "30" if i % 2 == 0 else "18,40",
                "baseline_expected_race_time": 5100 + i * 1.7,
                "baseline_win_probability": 0.3 + i * 0.01,
                "baseline_strategy_score": 20.0 + i * 0.03,
                "baseline_robustness_window": 15.0 + i * 0.08,
                "weather_change_strategy_score": 19.5 + i * 0.04,
                "engine_conservation_strategy_score": 19.2 + i * 0.03,
                "driver_error_recovery_strategy_score": 19.0 + i * 0.02,
                "race_chaos_strategy_score": 19.3 + i * 0.04,
                "weather_change_win_probability": 0.28 + i * 0.008,
                "engine_conservation_win_probability": 0.27 + i * 0.007,
                "driver_error_recovery_win_probability": 0.26 + i * 0.007,
                "race_chaos_win_probability": 0.27 + i * 0.008,
                "weather_change_robustness_window": 16.0 + i * 0.1,
                "engine_conservation_robustness_window": 15.5 + i * 0.1,
                "driver_error_recovery_robustness_window": 16.3 + i * 0.1,
                "race_chaos_robustness_window": 16.8 + i * 0.1,
                "top3_scenario_hits": i % 4,
            }
        )

    frame = pd.DataFrame(rows)
    ranked = ContingencyRanker(random_state=12).rank(frame)

    assert "contingency_rank_score" in ranked.ranked.columns
    assert len(ranked.ranked) == len(frame)
    assert ranked.metrics["mae"] >= 0.0
