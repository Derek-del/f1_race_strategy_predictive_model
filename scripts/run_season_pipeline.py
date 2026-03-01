from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from f1_strategy_lab.config.settings import load_config
from f1_strategy_lab.pipeline import run_season_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 2025 F1 strategy pipeline")
    parser.add_argument("--config", default="configs/redbull_2025.yaml", help="Config YAML path")
    parser.add_argument("--videos-dir", default=None, help="Optional directory of practice videos")
    parser.add_argument(
        "--no-synthetic-fallback",
        action="store_true",
        help="Disable synthetic fallback when live data cannot be fetched",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    summary = run_season_pipeline(
        cfg=cfg,
        videos_dir=args.videos_dir,
        synthetic_fallback=not args.no_synthetic_fallback,
    )

    print("\nPipeline completed")
    print(f"Training rows: {summary['training_rows']}")
    print(f"Inference rows: {summary['inference_rows']}")
    print("Outputs:")
    for key, value in summary["outputs"].items():
        print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()
