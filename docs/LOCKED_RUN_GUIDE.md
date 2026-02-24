# Locked Real-Data Run Guide (2025)

Use this when you want a strict, reproducible portfolio run with no synthetic fallback.

## Command

```bash
python scripts/run_realdata_locked.py --config configs/mclaren_2025.yaml
```

To reduce FastF1 API load while staying in real-data mode:

```bash
python scripts/run_realdata_locked.py --config configs/mclaren_2025.yaml --max-training-years 2
```

## What it enforces

1. Uses real data only (`synthetic_fallback=False`).
2. Builds 2025 strategy recommendations from FP/Q/weather/CV features.
3. Compares output event coverage to the official 2025 event schedule from FastF1.
4. Fails if any event is missing.
5. Writes an immutable snapshot in `reports/locks/`.

## Snapshot contents

- `training_features.csv`
- `model_metrics.json`
- `pace_model.joblib`
- `strategy_recommendations_2025.csv`
- `championship_projection_2025.json`
- `manifest.json`

`manifest.json` includes:
- run timestamp
- Python/platform metadata
- resolved config + model/sim parameters
- round completeness validation
- SHA-256 hash for each copied output

## Troubleshooting

- If the script fails on missing events, rerun after ensuring FastF1 can fetch all 2025 sessions and cache path is writable.
- If weather requests fail, check internet/DNS and rerun (weather has fallback defaults but data pipeline still requires session data).
- If you hit `RateLimitExceededError: any API: 500 calls/h`, wait for hourly reset and rerun; cache is reused automatically.
