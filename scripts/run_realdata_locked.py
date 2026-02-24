from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from f1_strategy_lab.config.settings import load_config
from f1_strategy_lab.data.fastf1_pipeline import FastF1RateLimitError, get_event_schedule
from f1_strategy_lab.pipeline import run_season_pipeline
from f1_strategy_lab.utils.lock import create_locked_snapshot


def _event_list(schedule: pd.DataFrame) -> list[str]:
    if "EventName" not in schedule.columns:
        return []
    events = schedule["EventName"].astype(str).tolist()
    return [e.strip() for e in events if e.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run strict real-data 2025 pipeline and freeze outputs into a locked snapshot"
    )
    parser.add_argument("--config", default="configs/mclaren_2025.yaml", help="Config YAML path")
    parser.add_argument("--videos-dir", default=None, help="Optional practice/qualifying video directory")
    parser.add_argument(
        "--lock-root",
        default="reports/locks",
        help="Directory where immutable run snapshots are stored",
    )
    parser.add_argument(
        "--max-training-years",
        type=int,
        default=None,
        help="Optional cap on training years (uses most recent N years from config)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_training_years is not None and args.max_training_years > 0:
        cfg.training_years = cfg.training_years[-args.max_training_years :]

    try:
        summary = run_season_pipeline(
            cfg=cfg,
            videos_dir=args.videos_dir,
            synthetic_fallback=False,
        )
    except FastF1RateLimitError as exc:
        raise SystemExit(
            f"{exc}\n"
            f"FastF1 cache is preserved at: {Path(cfg.paths.fastf1_cache).resolve()}\n"
            "Rerun the same command after the hourly quota resets, "
            "or reduce calls now with --max-training-years 2."
        ) from exc

    rec_path = Path(summary["outputs"]["strategy_recommendations"])
    recommendations = pd.read_csv(rec_path)

    schedule = get_event_schedule(cfg.target_year, cfg.paths.fastf1_cache)
    expected_events = _event_list(schedule)
    produced_events = recommendations["event_name"].astype(str).tolist()

    expected_set = set(expected_events)
    produced_set = set(produced_events)

    missing_events = sorted(expected_set - produced_set)
    extra_events = sorted(produced_set - expected_set)

    if missing_events or len(produced_events) != len(expected_events):
        detail = {
            "expected_rounds": len(expected_events),
            "produced_rounds": len(produced_events),
            "missing_events": missing_events,
            "extra_events": extra_events,
        }
        raise RuntimeError(f"Round completeness check failed: {detail}")

    snapshot_dir = create_locked_snapshot(
        summary=summary,
        cfg=cfg,
        config_path=args.config,
        lock_root=args.lock_root,
        expected_rounds=len(expected_events),
        produced_rounds=len(produced_events),
        missing_events=missing_events,
        extra_events=extra_events,
    )

    print("\nStrict real-data run completed")
    print(f"Expected rounds: {len(expected_events)}")
    print(f"Produced rounds: {len(produced_events)}")
    print(f"Locked snapshot: {snapshot_dir}")


if __name__ == "__main__":
    main()
