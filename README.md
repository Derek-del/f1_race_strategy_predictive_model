# F1 Strategy Lab (2025 Season, Red Bull Focus)

An end-to-end **computer vision + machine learning** project to generate race-day strategy recommendations for a top F1 team (default: **Red Bull Racing**) using:

- Pre-season and historical performance context
- Practice + qualifying pace features
- Tire degradation and fuel-load proxies
- Weather signals by circuit and race window
- CV-derived track-state/traffic metrics from session video
- Monte Carlo strategy simulation with contingency planning

Default runtime profile in this repo now targets the **ground-effect era**:
- training years: `2022, 2023, 2024` (includes race-day targets)
- inference year: `2025` (pre-season testing baseline + practice + qualifying features)

This project is designed to be portfolio-grade for motorsport analytics roles.

## 1. What this project does

For each 2025 race weekend:
1. Ingests telemetry/lap/session data from FastF1 (FP1/FP2/FP3/Q/R where available).
2. Builds race-pace prediction features.
3. Trains a pace model on prior seasons.
4. Extracts CV features (traffic, grip proxy, rain/visibility proxy) from practice footage.
5. Simulates one-stop and two-stop strategy candidates.
6. Outputs primary + fallback strategy recommendations with human-readable pit plans.
7. Aggregates a season-level title-probability projection for drivers and constructors.

## 2. Project structure

- `configs/` runtime config (team, years, sim params)
- `scripts/` entry points to run pipeline
- `src/f1_strategy_lab/data` ingestion/weather/synthetic generators
- `src/f1_strategy_lab/cv` video feature extraction
- `src/f1_strategy_lab/features` feature engineering
- `src/f1_strategy_lab/models` pace model training/inference
- `src/f1_strategy_lab/strategy` race and championship optimization
- `tests/` unit tests for model and strategy core

## 3. Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Run the full pipeline:

```bash
python scripts/run_season_pipeline.py --config configs/redbull_2025.yaml
```

Strict real-data run for 2025 with immutable snapshot lock:

```bash
python scripts/run_realdata_locked.py --config configs/redbull_2025.yaml
```

This command:
- disables synthetic fallback
- validates all scheduled 2025 rounds are present in output
- writes a locked snapshot to `reports/locks/<year_team_driver_timestamp>/` with checksums

If FastF1 API quota is hit, run with fewer training years (still real data):

```bash
python scripts/run_realdata_locked.py --config configs/redbull_2025.yaml --max-training-years 2
```

Run with optional video directory (`<event>.mp4` naming convention):

```bash
python scripts/run_season_pipeline.py \
  --config configs/redbull_2025.yaml \
  --videos-dir ./data/videos
```

Offline demo with synthetic data:

```bash
python scripts/run_demo.py
```

Run tests:

```bash
pytest
```

Launch local dashboard for latest locked run (or `reports/` fallback):

```bash
python scripts/run_locked_dashboard.py --port 8765
```

This now serves a multi-page interactive website with:
- Overview
- Race Explorer (filter/sort + clickable row drawer)
- Championship Lab (what-if sliders)
- Data Integrity (manifest + checksum copy actions)

Export static website for Vercel:

```bash
python scripts/export_static_site.py --out-dir site
```

Then deploy:

```bash
cd site
vercel --prod
```

For GitHub-based auto deploy, import this repository directly in Vercel and set Root Directory to `site`.

## 4. Outputs

Pipeline writes to `reports/`:

- `training_features.csv`
- `strategy_recommendations_2025.csv`
- `championship_projection_2025.json`
- `model_metrics.json`

Strict locked runs also write:
- `reports/locks/<...>/manifest.json`

## 5. Resume-ready talking points

- Combined **CV-derived race context** with conventional motorsport telemetry modeling.
- Built a **race strategy optimizer** under uncertainty with Monte Carlo simulation.
- Added **season-level objective** to align race strategy with championship outcomes.
- Engineered a reproducible analytics pipeline with config-driven experiments.

## 6. Notes

- FastF1 and weather APIs require internet when fetching fresh data.
- CV module is intentionally modular: swap in YOLO/SegFormer or team-specific vision models later.
- This project is for analytics/research and does not represent official team strategy tooling.
