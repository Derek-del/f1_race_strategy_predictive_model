from __future__ import annotations

import hashlib
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from f1_strategy_lab.config.settings import ProjectConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _serialize_config(cfg: ProjectConfig) -> dict[str, Any]:
    return {
        "project_name": cfg.project_name,
        "team": cfg.team,
        "target_driver": cfg.target_driver,
        "training_years": cfg.training_years,
        "target_year": cfg.target_year,
        "paths": {
            "fastf1_cache": cfg.paths.fastf1_cache,
            "weather_cache": cfg.paths.weather_cache,
            "reports_dir": cfg.paths.reports_dir,
        },
        "model": {
            "test_size": cfg.model.test_size,
            "random_state": cfg.model.random_state,
            "n_estimators": cfg.model.n_estimators,
            "learning_rate": cfg.model.learning_rate,
            "max_depth": cfg.model.max_depth,
        },
        "simulation": {
            "n_simulations": cfg.simulation.n_simulations,
            "pit_loss_seconds": cfg.simulation.pit_loss_seconds,
            "safety_car_probability": cfg.simulation.safety_car_probability,
            "weather_uncertainty_seconds": cfg.simulation.weather_uncertainty_seconds,
            "traffic_uncertainty_seconds": cfg.simulation.traffic_uncertainty_seconds,
        },
        "cv": {
            "frame_stride": cfg.cv.frame_stride,
            "min_contour_area": cfg.cv.min_contour_area,
        },
        "strategy": {
            "default_total_laps": cfg.strategy.default_total_laps,
            "compounds": cfg.strategy.compounds,
        },
    }


def create_locked_snapshot(
    summary: dict[str, Any],
    cfg: ProjectConfig,
    config_path: str,
    lock_root: str | Path,
    expected_rounds: int,
    produced_rounds: int,
    missing_events: list[str],
    extra_events: list[str],
) -> Path:
    root = Path(lock_root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    slug = f"{cfg.target_year}_{cfg.team.lower()}_{cfg.target_driver.lower()}_{stamp}"
    snapshot_dir = root / slug
    snapshot_dir.mkdir(parents=True, exist_ok=False)

    copied_outputs: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for key, path_str in summary.get("outputs", {}).items():
        src = Path(path_str)
        if not src.exists():
            continue
        dst = snapshot_dir / src.name
        shutil.copy2(src, dst)
        copied_outputs[key] = str(dst)
        hashes[src.name] = _sha256(dst)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config_path": str(Path(config_path).resolve()),
        "config": _serialize_config(cfg),
        "summary": summary,
        "round_validation": {
            "expected_rounds": expected_rounds,
            "produced_rounds": produced_rounds,
            "missing_events": missing_events,
            "extra_events": extra_events,
            "all_rounds_present": len(missing_events) == 0 and produced_rounds == expected_rounds,
        },
        "outputs": copied_outputs,
        "sha256": hashes,
    }

    (snapshot_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return snapshot_dir
