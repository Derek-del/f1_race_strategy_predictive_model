# Project Playbook: F1 ML + CV Strategy Engine (2025, McLaren)

## Objective
Build a race strategy system that predicts race pace and chooses pit/tire strategy to maximize title-winning probability across the 2025 season.

## Inputs
- Practice and qualifying lap/session data (FastF1).
- Weather context by circuit and race window.
- Tire degradation and fuel-load proxies from stints.
- CV features from practice/qualifying footage.

## Core outputs
- Best strategy per race (`strategy_recommendations_2025.csv`).
- Projected driver and constructors title probabilities (`championship_projection_2025.json`).
- Model quality metrics (`model_metrics.json`).

## Pipeline flow
1. Ingest historical and pre-race data.
2. Engineer strategy-aware features.
3. Train race-pace model.
4. Add CV-derived track-state features.
5. Simulate candidate strategies under uncertainty.
6. Select primary strategy and contingency backups maximizing score and robustness.
7. Aggregate season championship projection.

## Model and optimization details
- Pace model: gradient boosting regressor with mixed feature preprocessing.
- Candidate generation: one-stop and two-stop compound sequences.
- Simulator: Monte Carlo race time and points outcomes with traffic/weather/safety-car stochasticity.
- Selection objective: strategy score and robustness across baseline + contingency scenarios.

## Validation approach
- Unit tests for model training/prediction and simulator behavior.
- Pipeline smoke test using synthetic fallback.
- With full deps and internet: compare predictions against held-out race weekends.

## Portfolio framing for F1 applications
- Highlight system design under uncertainty and race-engineering decision support.
- Show feature attribution and strategic sensitivity analysis.
- Present one or two race deep-dives (e.g., Monaco vs Monza) with scenario comparisons.

## Suggested next upgrades
1. Add per-sector pace model and overtaking probability model.
2. Introduce explicit rival-team strategy modeling.
3. Replace heuristic CV with detector/segmenter + track occupancy map.
4. Add sprint-race and safety-car event priors per circuit.
