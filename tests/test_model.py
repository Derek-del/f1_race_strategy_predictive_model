from __future__ import annotations

import numpy as np

from f1_strategy_lab.config.settings import ModelConfig
from f1_strategy_lab.data.synthetic import synthetic_training_data
from f1_strategy_lab.features.feature_builder import prepare_feature_set
from f1_strategy_lab.models.pace_model import PaceModel


def test_pace_model_train_and_predict() -> None:
    df = synthetic_training_data(n_events=90, random_state=123)
    fset = prepare_feature_set(df, target_col="target_race_pace")

    model = PaceModel(ModelConfig(test_size=0.2, random_state=123, n_estimators=80, learning_rate=0.06, max_depth=3))
    metrics = model.train(fset.frame, feature_cols=fset.feature_cols, target_col=fset.target_col)

    assert {"mae", "rmse", "r2"}.issubset(metrics.keys())
    assert np.isfinite(metrics["rmse"])

    preds = model.predict(fset.frame[fset.feature_cols])
    assert len(preds) == len(fset.frame)
