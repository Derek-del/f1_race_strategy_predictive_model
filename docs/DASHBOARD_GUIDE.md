# Dashboard Guide

This dashboard visualizes the latest strategy run for recruiter demos.

## Start

```bash
python scripts/run_locked_dashboard.py --port 8765
```

Open: `http://127.0.0.1:8765`

It is now a multi-page website with toggle navigation:
- `Overview`
- `Race Explorer`
- `Championship Lab`
- `Data Integrity`

Direct routes also work:
- `http://127.0.0.1:8765/overview`
- `http://127.0.0.1:8765/races`
- `http://127.0.0.1:8765/simulator`
- `http://127.0.0.1:8765/integrity`

## Data source selection

Order of precedence:
1. `--snapshot-dir` if provided.
2. Latest lock under `--lock-root` (default `reports/locks`).
3. Fallback to `--reports-dir` (default `reports`).

## Useful commands

Use explicit snapshot:

```bash
python scripts/run_locked_dashboard.py --snapshot-dir reports/locks/<snapshot_id>
```

Use custom lock root:

```bash
python scripts/run_locked_dashboard.py --lock-root reports/locks --port 9000
```

## What is shown

- Run metadata (locked/reports mode, source path, timestamp)
- KPI cards (rows, error metrics, title probabilities)
- Round validation status (if lock manifest exists)
- Top rounds by strategy score
- Full strategy recommendation table with quick filter, sorting, and row drill-down drawer
- Championship what-if simulator (interactive sliders)
- Manifest checksum list with copy-to-clipboard
