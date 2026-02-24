from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from f1_strategy_lab.config.settings import ProjectConfig
from f1_strategy_lab.cv.track_state import load_cv_features_from_directory
from f1_strategy_lab.data.fastf1_pipeline import build_prerace_dataset, build_training_dataset
from f1_strategy_lab.data.synthetic import synthetic_training_data
from f1_strategy_lab.features.feature_builder import align_inference_features, prepare_feature_set
from f1_strategy_lab.models.pace_model import PaceModel
from f1_strategy_lab.strategy.championship import project_championship
from f1_strategy_lab.strategy.simulator import evaluate_strategies, generate_candidate_strategies
from f1_strategy_lab.utils.io import ensure_dir, save_json


EVENT_LAP_HINTS = {
    "bahrain": 57,
    "saudi": 50,
    "australian": 58,
    "japanese": 53,
    "chinese": 56,
    "miami": 57,
    "imola": 63,
    "emilia romagna": 63,
    "monaco": 78,
    "canadian": 70,
    "spanish": 66,
    "austrian": 71,
    "british": 52,
    "hungarian": 70,
    "belgian": 44,
    "dutch": 72,
    "italian": 53,
    "azerbaijan": 51,
    "singapore": 62,
    "united states": 56,
    "mexico": 71,
    "sao paulo": 71,
    "las vegas": 50,
    "qatar": 57,
    "abu dhabi": 58,
}


def _event_laps(event_name: str, fallback: int) -> int:
    key = event_name.lower()
    for k, laps in EVENT_LAP_HINTS.items():
        if k in key:
            return laps
    return fallback


def _pick(row: pd.Series, options: list[str], fallback: float) -> float:
    for col in options:
        if col in row and pd.notna(row[col]):
            return float(row[col])
    return fallback


def _load_cv_features(videos_dir: str | None, cfg: ProjectConfig) -> dict[str, dict[str, float]]:
    if not videos_dir:
        return {}
    return load_cv_features_from_directory(
        videos_dir=videos_dir,
        frame_stride=cfg.cv.frame_stride,
        min_contour_area=cfg.cv.min_contour_area,
    )


def run_season_pipeline(
    cfg: ProjectConfig,
    videos_dir: str | None = None,
    synthetic_fallback: bool = True,
) -> dict[str, Any]:
    reports_dir = ensure_dir(cfg.paths.reports_dir)
    cv_features = _load_cv_features(videos_dir, cfg)

    # Training set
    train_df: pd.DataFrame
    used_synthetic_training = False
    try:
        train_df = build_training_dataset(
            years=cfg.training_years,
            team=cfg.team,
            driver=cfg.target_driver,
            fastf1_cache_dir=cfg.paths.fastf1_cache,
            weather_cache_dir=cfg.paths.weather_cache,
            cv_features_by_event=cv_features,
        )
    except Exception as exc:
        if not synthetic_fallback:
            raise
        print(f"[WARN] Falling back to synthetic training data: {exc}")
        train_df = synthetic_training_data(random_state=cfg.model.random_state)
        used_synthetic_training = True

    if train_df.empty:
        if not synthetic_fallback:
            raise RuntimeError("Training dataset is empty")
        train_df = synthetic_training_data(random_state=cfg.model.random_state)
        used_synthetic_training = True

    feature_set = prepare_feature_set(train_df, target_col="target_race_pace")
    model = PaceModel(cfg.model)
    metrics = model.train(
        frame=feature_set.frame,
        feature_cols=feature_set.feature_cols,
        target_col=feature_set.target_col,
    )
    model.save(reports_dir / "pace_model.joblib")

    train_df.to_csv(reports_dir / "training_features.csv", index=False)
    save_json(reports_dir / "model_metrics.json", metrics)

    # Inference set for 2025 race weekends
    inference_df: pd.DataFrame
    used_synthetic_inference = False
    try:
        inference_df = build_prerace_dataset(
            year=cfg.target_year,
            team=cfg.team,
            driver=cfg.target_driver,
            fastf1_cache_dir=cfg.paths.fastf1_cache,
            weather_cache_dir=cfg.paths.weather_cache,
            cv_features_by_event=cv_features,
        )
    except Exception as exc:
        if not synthetic_fallback:
            raise
        print(f"[WARN] Falling back to synthetic pre-race set: {exc}")
        inference_df = train_df.copy()
        used_synthetic_inference = True

    if inference_df.empty:
        if not synthetic_fallback:
            raise RuntimeError("Inference dataset is empty")
        inference_df = train_df.copy()
        used_synthetic_inference = True

    drop_targets = [c for c in ["target_race_pace", "target_finish_position", "target_points"] if c in inference_df]
    if drop_targets:
        inference_df = inference_df.drop(columns=drop_targets)

    # If synthetic fallback is used, generate a clean 24-race shell for 2025.
    if used_synthetic_inference:
        n = min(24, len(inference_df))
        inference_df = inference_df.head(n).copy()
        inference_df["year"] = cfg.target_year
        inference_df["event_name"] = [f"Round_{i+1:02d}" for i in range(len(inference_df))]

    x_inf = align_inference_features(feature_set.feature_cols, inference_df)
    base_pace_preds = model.predict(x_inf)

    recommendations: list[dict[str, Any]] = []
    for idx, (_, row) in enumerate(inference_df.iterrows()):
        base_lap_time = float(base_pace_preds[idx])
        total_laps = _event_laps(str(row.get("event_name", "")), cfg.strategy.default_total_laps)

        deg_soft = _pick(row, ["fp2_deg_soft", "quali_deg_soft", "deg_soft"], 0.16)
        deg_medium = _pick(row, ["fp2_deg_medium", "quali_deg_medium", "deg_medium"], 0.11)
        deg_hard = _pick(row, ["fp2_deg_hard", "quali_deg_hard", "deg_hard"], 0.08)

        fuel_proxy = _pick(row, ["fp2_fuel_load_proxy", "quali_fuel_load_proxy", "fuel_load_proxy"], 0.95)
        traffic_idx = _pick(row, ["cv_traffic_index"], 0.5)
        cv_rain = _pick(row, ["cv_rain_index"], 0.05)
        weather_rain = _pick(row, ["weather_precip_mm"], 0.0)
        rain_idx = min(2.0, cv_rain + weather_rain / 5.0)

        candidates = generate_candidate_strategies(total_laps=total_laps, compounds=cfg.strategy.compounds)
        evaluated = evaluate_strategies(
            candidates=candidates,
            total_laps=total_laps,
            base_lap_time=base_lap_time,
            degradation={"SOFT": deg_soft, "MEDIUM": deg_medium, "HARD": deg_hard},
            fuel_load_proxy=fuel_proxy,
            traffic_index=traffic_idx,
            rain_index=rain_idx,
            cfg=cfg.simulation,
            random_state=cfg.model.random_state + idx,
        )

        best = evaluated.iloc[0]
        recommendations.append(
            {
                "year": int(row.get("year", cfg.target_year)),
                "event_name": str(row.get("event_name", f"Round_{idx+1:02d}")),
                "team": cfg.team,
                "driver": cfg.target_driver,
                "predicted_base_lap_sec": round(base_lap_time, 3),
                "best_strategy": str(best["strategy"]),
                "compounds": str(best["compounds"]),
                "pit_laps": str(best["pit_laps"]),
                "stops": int(best["stops"]),
                "expected_race_time": round(float(best["expected_race_time"]), 3),
                "win_probability": round(float(best["win_probability"]), 4),
                "expected_points": round(float(best["expected_points"]), 3),
                "strategy_score": round(float(best["strategy_score"]), 3),
                "robustness_window": round(float(best["robustness_window"]), 3),
            }
        )

    rec_df = pd.DataFrame(recommendations)
    rec_df.to_csv(reports_dir / f"strategy_recommendations_{cfg.target_year}.csv", index=False)

    championship = project_championship(rec_df, target_driver=cfg.target_driver, team=cfg.team)
    championship["used_synthetic_training"] = used_synthetic_training
    championship["used_synthetic_inference"] = used_synthetic_inference
    save_json(reports_dir / f"championship_projection_{cfg.target_year}.json", championship)

    summary = {
        "training_rows": int(len(train_df)),
        "inference_rows": int(len(inference_df)),
        "metrics": {k: round(v, 6) for k, v in metrics.items()},
        "outputs": {
            "training_features": str(reports_dir / "training_features.csv"),
            "model_metrics": str(reports_dir / "model_metrics.json"),
            "model": str(reports_dir / "pace_model.joblib"),
            "strategy_recommendations": str(reports_dir / f"strategy_recommendations_{cfg.target_year}.csv"),
            "championship_projection": str(reports_dir / f"championship_projection_{cfg.target_year}.json"),
        },
    }
    save_json(reports_dir / "run_summary.json", summary)
    return summary
