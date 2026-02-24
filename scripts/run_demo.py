from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from f1_strategy_lab.config.settings import load_config
from f1_strategy_lab.pipeline import run_season_pipeline


def main() -> None:
    cfg = load_config("configs/mclaren_2025.yaml")
    cfg.training_years = []
    cfg.target_year = 2025
    cfg.paths.reports_dir = "./reports/demo"

    summary = run_season_pipeline(cfg=cfg, videos_dir=None, synthetic_fallback=True)
    print("Demo run complete")
    print(summary)


if __name__ == "__main__":
    main()
