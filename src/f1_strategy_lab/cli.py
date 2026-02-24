from __future__ import annotations

from pathlib import Path

import typer
from rich import print

from f1_strategy_lab.config.settings import load_config
from f1_strategy_lab.pipeline import run_season_pipeline

app = typer.Typer(help="F1 Strategy Lab command line")


@app.command()
def run(
    config: str = typer.Option("configs/mclaren_2025.yaml", help="Path to YAML config"),
    videos_dir: str | None = typer.Option(None, help="Directory with event video files"),
    no_synthetic_fallback: bool = typer.Option(
        False, help="Disable synthetic fallback if live data is unavailable"
    ),
) -> None:
    cfg = load_config(config)
    summary = run_season_pipeline(
        cfg=cfg,
        videos_dir=videos_dir,
        synthetic_fallback=not no_synthetic_fallback,
    )

    print("\n[bold]Pipeline completed.[/bold]")
    print(f"Training rows: {summary['training_rows']}")
    print(f"Inference rows: {summary['inference_rows']}")
    print(f"Outputs: {summary['outputs']}")


@app.command()
def demo() -> None:
    cfg = load_config("configs/mclaren_2025.yaml")
    cfg.training_years = []
    cfg.target_year = 2025
    cfg.paths.reports_dir = "./reports/demo"

    summary = run_season_pipeline(cfg=cfg, videos_dir=None, synthetic_fallback=True)
    print("\n[bold]Demo run completed (synthetic fallback).[/bold]")
    print(summary)


if __name__ == "__main__":
    app()
