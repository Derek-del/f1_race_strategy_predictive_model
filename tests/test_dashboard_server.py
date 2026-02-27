from __future__ import annotations

import json
from pathlib import Path

from f1_strategy_lab.dashboard.server import build_dashboard_payload


def _write_basic_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "strategy_recommendations_2025.csv").write_text(
        "year,event_name,team,driver,predicted_base_lap_sec,best_strategy,compounds,pit_laps,stops,start_compound,first_pit_lap,strategy_plan,expected_race_time,win_probability,strategy_score,robustness_window,fallback_2_strategy,fallback_2_plan,fallback_2_trigger,fallback_3_strategy,fallback_3_plan,fallback_3_trigger\n"
        "2025,Bahrain Grand Prix,MCLAREN,NOR,90.1,ONE_STOP_MEDIUM_HARD_L31,MEDIUM->HARD,31,1,MEDIUM,31,Start on MEDIUM; Pit on lap 31 -> HARD,5120.2,0.34,18.1,14.2,TWO_STOP_SOFT_MEDIUM_HARD_L18_38,Start on SOFT; Pit on lap 18 -> MEDIUM; Pit on lap 38 -> HARD,Weather change or sudden rain,ONE_STOP_SOFT_HARD_L28,Start on SOFT; Pit on lap 28 -> HARD,Driver error recovery\n"
    )
    (run_dir / "championship_projection_2025.json").write_text(
        json.dumps(
            {
                "driver": "NOR",
                "team": "MCLAREN",
                "driver_title_probability": 0.58,
                "constructors_title_probability": 0.66,
            }
        )
    )
    (run_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "training_rows": 86,
                "inference_rows": 24,
                "metrics": {"mae": 0.42, "rmse": 0.51, "r2": 0.37},
                "outputs": {
                    "strategy_recommendations": str(run_dir / "strategy_recommendations_2025.csv"),
                    "championship_projection": str(run_dir / "championship_projection_2025.json"),
                },
            }
        )
    )


def test_build_payload_from_reports(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    _write_basic_run(reports)

    payload = build_dashboard_payload(snapshot_dir=None, lock_root=tmp_path / "locks", reports_dir=reports)

    assert payload["source"]["mode"] == "reports"
    assert len(payload["strategy_rows"]) == 1
    assert payload["kpis"]["training_rows"] == 86


def test_build_payload_from_locked_snapshot(tmp_path: Path) -> None:
    lock_dir = tmp_path / "locks" / "2025_mclaren_nor_20260224_000000Z"
    _write_basic_run(lock_dir)

    manifest = {
        "created_at_utc": "2026-02-24T00:00:00Z",
        "summary": {
            "training_rows": 90,
            "inference_rows": 24,
            "metrics": {"mae": 0.41, "rmse": 0.5, "r2": 0.38},
            "outputs": {
                "strategy_recommendations": str(lock_dir / "strategy_recommendations_2025.csv"),
                "championship_projection": str(lock_dir / "championship_projection_2025.json"),
            },
        },
        "round_validation": {
            "all_rounds_present": True,
            "expected_rounds": 24,
            "produced_rounds": 24,
            "missing_events": [],
            "extra_events": [],
        },
        "outputs": {
            "strategy_recommendations": str(lock_dir / "strategy_recommendations_2025.csv"),
            "championship_projection": str(lock_dir / "championship_projection_2025.json"),
        },
    }
    (lock_dir / "manifest.json").write_text(json.dumps(manifest))

    payload = build_dashboard_payload(snapshot_dir=None, lock_root=tmp_path / "locks", reports_dir=tmp_path / "reports")

    assert payload["source"]["mode"] == "locked"
    assert payload["round_validation"]["all_rounds_present"] is True
    assert payload["top_rounds"][0]["event_name"] == "Bahrain Grand Prix"
