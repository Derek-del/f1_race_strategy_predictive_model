from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from f1_strategy_lab.config.settings import ModelConfig


@dataclass(slots=True)
class PaceModelArtifacts:
    feature_cols: list[str]
    target_col: str
    metrics: dict[str, float]


class PaceModel:
    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.pipeline: Pipeline | None = None
        self.artifacts: PaceModelArtifacts | None = None

    def _build_pipeline(self, frame: pd.DataFrame, feature_cols: list[str]) -> Pipeline:
        numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(frame[c])]
        cat_cols = [c for c in feature_cols if c not in numeric_cols]

        numeric_pipe = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        prep = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, numeric_cols),
                ("cat", cat_pipe, cat_cols),
            ],
            remainder="drop",
        )

        model = GradientBoostingRegressor(
            n_estimators=self.cfg.n_estimators,
            learning_rate=self.cfg.learning_rate,
            max_depth=self.cfg.max_depth,
            random_state=self.cfg.random_state,
        )

        return Pipeline(steps=[("prep", prep), ("model", model)])

    def train(
        self,
        frame: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = "target_race_pace",
    ) -> dict[str, float]:
        if frame.empty:
            raise ValueError("Training frame is empty")

        x = frame[feature_cols]
        y = frame[target_col].astype(float)

        x_train, x_val, y_train, y_val = train_test_split(
            x,
            y,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
        )

        self.pipeline = self._build_pipeline(frame=frame, feature_cols=feature_cols)
        self.pipeline.fit(x_train, y_train)

        preds = self.pipeline.predict(x_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        metrics = {
            "mae": float(mean_absolute_error(y_val, preds)),
            "rmse": rmse,
            "r2": float(r2_score(y_val, preds)),
        }

        self.artifacts = PaceModelArtifacts(
            feature_cols=feature_cols,
            target_col=target_col,
            metrics=metrics,
        )
        return metrics

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Model is not trained/loaded")
        return self.pipeline.predict(frame)

    def save(self, path: str | Path) -> None:
        if self.pipeline is None or self.artifacts is None:
            raise RuntimeError("No trained model to save")

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "artifacts": self.artifacts}, target)

    @classmethod
    def load(cls, path: str | Path) -> "PaceModel":
        payload: dict[str, Any] = joblib.load(path)
        model = cls(ModelConfig())
        model.pipeline = payload["pipeline"]
        model.artifacts = payload["artifacts"]
        return model
