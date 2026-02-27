from __future__ import annotations

import json
from pathlib import Path

from f1_strategy_lab.config.settings import ProjectConfig
from f1_strategy_lab.utils.lock import create_locked_snapshot


def test_create_locked_snapshot(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    out_a = reports / "strategy_recommendations_2025.csv"
    out_b = reports / "championship_projection_2025.json"
    out_a.write_text("event_name,strategy_score,win_probability\nBahrain Grand Prix,20.4,0.41\n")
    out_b.write_text('{"driver_title_probability": 0.74, "constructors_title_probability": 0.81}\n')

    summary = {
        "training_rows": 100,
        "inference_rows": 24,
        "metrics": {"rmse": 0.2},
        "outputs": {
            "strategy_recommendations": str(out_a),
            "championship_projection": str(out_b),
        },
    }

    cfg = ProjectConfig()
    snapshot = create_locked_snapshot(
        summary=summary,
        cfg=cfg,
        config_path="configs/mclaren_2025.yaml",
        lock_root=tmp_path / "locks",
        expected_rounds=24,
        produced_rounds=24,
        missing_events=[],
        extra_events=[],
    )

    manifest_path = snapshot / "manifest.json"
    assert manifest_path.exists()

    payload = json.loads(manifest_path.read_text())
    assert payload["round_validation"]["all_rounds_present"] is True
    assert "strategy_recommendations_2025.csv" in payload["sha256"]
    assert "championship_projection_2025.json" in payload["sha256"]
