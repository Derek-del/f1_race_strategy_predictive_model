from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


META_COLS = {"year", "event_name", "team", "driver"}


@dataclass(slots=True)
class FeatureSet:
    frame: pd.DataFrame
    feature_cols: list[str]
    target_col: str


def prepare_feature_set(df: pd.DataFrame, target_col: str = "target_race_pace") -> FeatureSet:
    if df.empty:
        return FeatureSet(frame=df.copy(), feature_cols=[], target_col=target_col)

    work = df.copy()

    numeric_cols = work.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        work[col] = work[col].fillna(work[col].median())

    cat_cols = [c for c in work.columns if c not in numeric_cols]
    for col in cat_cols:
        if work[col].isna().any():
            mode = work[col].mode(dropna=True)
            fill = mode.iloc[0] if not mode.empty else "UNKNOWN"
            work[col] = work[col].fillna(fill)

    feature_cols = [c for c in work.columns if c not in META_COLS and c != target_col]
    return FeatureSet(frame=work, feature_cols=feature_cols, target_col=target_col)


def align_inference_features(
    train_features: list[str],
    inference_df: pd.DataFrame,
) -> pd.DataFrame:
    out = inference_df.copy()
    for col in train_features:
        if col not in out.columns:
            out[col] = 0.0
    drop_cols = [c for c in out.columns if c not in train_features]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    return out[train_features]
