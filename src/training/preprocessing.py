from __future__ import annotations

import joblib
import pandas as pd

from src.config import DROP_COLUMNS, FEATURE_META_PATH, PIPELINE_PATH


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def preprocess_single(
    data: dict,
    pipeline=None,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Prepare a single application dict for inference.

    Returns a DataFrame with the correct feature columns ready for
    pipeline.predict_proba(). The pipeline handles encoding/scaling internally.
    """
    if pipeline is None:
        pipeline = joblib.load(PIPELINE_PATH)
    if feature_cols is None:
        meta = joblib.load(FEATURE_META_PATH)
        feature_cols = meta["feature_cols"]

    df = pd.DataFrame([data])
    drop_cols = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    return df[feature_cols]
