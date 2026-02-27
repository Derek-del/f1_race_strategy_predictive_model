from __future__ import annotations

from pathlib import Path

import pandas as pd

from f1_strategy_lab.config.settings import load_config
from f1_strategy_lab.pipeline import run_season_pipeline


def test_pipeline_runs_with_synthetic_fallback(tmp_path: Path) -> None:
    cfg = load_config("configs/mclaren_2025.yaml")
    cfg.training_years = []
    cfg.target_year = 2025
    cfg.paths.reports_dir = str(tmp_path / "reports")
    cfg.paths.fastf1_cache = str(tmp_path / "fastf1_cache")
    cfg.paths.weather_cache = str(tmp_path / "weather_cache")
    cfg.simulation.n_simulations = 50

    summary = run_season_pipeline(cfg=cfg, videos_dir=None, synthetic_fallback=True)

    assert summary["training_rows"] > 0
    assert summary["inference_rows"] > 0
    rec_path = Path(summary["outputs"]["strategy_recommendations"])
    assert rec_path.exists()
    assert Path(summary["outputs"]["championship_projection"]).exists()

    rec_df = pd.read_csv(rec_path)
    assert {"fallback_2_strategy", "fallback_3_strategy", "strategy_plan", "start_compound"}.issubset(
        rec_df.columns
    )
    assert "expected_points" not in rec_df.columns
