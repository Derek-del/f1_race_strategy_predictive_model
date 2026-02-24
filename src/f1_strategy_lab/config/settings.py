from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class PathsConfig:
    fastf1_cache: str = "./fastf1_cache"
    weather_cache: str = "./cache/weather"
    reports_dir: str = "./reports"


@dataclass(slots=True)
class WeatherConfig:
    provider: str = "open-meteo"


@dataclass(slots=True)
class ModelConfig:
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 4


@dataclass(slots=True)
class SimulationConfig:
    n_simulations: int = 3000
    pit_loss_seconds: float = 21.5
    safety_car_probability: float = 0.18
    weather_uncertainty_seconds: float = 0.35
    traffic_uncertainty_seconds: float = 0.4


@dataclass(slots=True)
class CVConfig:
    frame_stride: int = 10
    min_contour_area: int = 220


@dataclass(slots=True)
class StrategyConfig:
    default_total_laps: int = 58
    compounds: list[str] = field(default_factory=lambda: ["SOFT", "MEDIUM", "HARD"])


@dataclass(slots=True)
class ProjectConfig:
    project_name: str = "F1 Strategy Lab"
    team: str = "MCLAREN"
    target_driver: str = "NOR"
    training_years: list[int] = field(default_factory=lambda: [2021, 2022, 2023, 2024])
    target_year: int = 2025
    paths: PathsConfig = field(default_factory=PathsConfig)
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)


def _merge_dataclass(dc_cls: type, source: dict[str, Any] | None) -> Any:
    source = source or {}
    valid = {k: v for k, v in source.items() if k in dc_cls.__dataclass_fields__}
    return dc_cls(**valid)


def load_config(path: str | Path) -> ProjectConfig:
    cfg_path = Path(path)
    data = yaml.safe_load(cfg_path.read_text()) or {}

    return ProjectConfig(
        project_name=data.get("project_name", "F1 Strategy Lab"),
        team=data.get("team", "MCLAREN"),
        target_driver=data.get("target_driver", "NOR"),
        training_years=data.get("training_years", [2021, 2022, 2023, 2024]),
        target_year=data.get("target_year", 2025),
        paths=_merge_dataclass(PathsConfig, data.get("paths")),
        weather=_merge_dataclass(WeatherConfig, data.get("weather")),
        model=_merge_dataclass(ModelConfig, data.get("model")),
        simulation=_merge_dataclass(SimulationConfig, data.get("simulation")),
        cv=_merge_dataclass(CVConfig, data.get("cv")),
        strategy=_merge_dataclass(StrategyConfig, data.get("strategy")),
    )
