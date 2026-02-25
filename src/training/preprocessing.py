from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import (
    MODEL_PATH,
    CATEGORICAL_COLUMNS,
    DROP_COLUMNS,
    ENCODERS_PATH,
    FEATURE_META_PATH,
    NUMERIC_COLUMNS,
    SCALER_PATH,
    TARGET_COLUMN,
    TEST_RATIO,
    VAL_RATIO,
)


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _fit_encoders(df: pd.DataFrame) -> dict[str, LabelEncoder]:
    encoders: dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        le.fit(df[col])
        encoders[col] = le
    return encoders


def _apply_encoders(
    df: pd.DataFrame, encoders: dict[str, LabelEncoder]
) -> pd.DataFrame:
    df = df.copy()
    for col, le in encoders.items():
        df[col] = df[col].astype(str)
        known_classes = set(le.classes_)
        df[col] = df[col].apply(lambda x: x if x in known_classes else le.classes_[0])
        df[col] = le.transform(df[col])
    return df


def _fit_scaler(df: pd.DataFrame, feature_cols: list[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(df[feature_cols].values.astype(np.float32))
    return scaler


def _apply_scaler(
    df: pd.DataFrame, scaler: StandardScaler, feature_cols: list[str]
) -> pd.DataFrame:
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols].values.astype(np.float32))
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors="ignore")


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS if c in df.columns]


def preprocess_train(
    df: pd.DataFrame,
) -> tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    list[str],
]:
    df = prepare_features(df)
    all_feature_cols = get_feature_columns(df)

    y = df[TARGET_COLUMN].values.astype(np.float32)
    val_test_ratio = VAL_RATIO + TEST_RATIO

    df_train, df_temp, y_train, y_temp = train_test_split(
        df, y, test_size=val_test_ratio, random_state=42, stratify=y
    )

    relative_test = TEST_RATIO / val_test_ratio
    df_val, df_test, y_val, y_test = train_test_split(
        df_temp, y_temp, test_size=relative_test, random_state=42, stratify=y_temp
    )

    encoders = _fit_encoders(df_train)
    df_train = _apply_encoders(df_train, encoders)
    df_val = _apply_encoders(df_val, encoders)
    df_test = _apply_encoders(df_test, encoders)

    scaler = _fit_scaler(df_train, all_feature_cols)
    df_train = _apply_scaler(df_train, scaler, all_feature_cols)
    df_val = _apply_scaler(df_val, scaler, all_feature_cols)
    df_test = _apply_scaler(df_test, scaler, all_feature_cols)

    X_train = df_train[all_feature_cols].values.astype(np.float32)
    X_val = df_val[all_feature_cols].values.astype(np.float32)
    X_test = df_test[all_feature_cols].values.astype(np.float32)

    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    joblib.dump(
        {"feature_cols": all_feature_cols, "input_dim": len(all_feature_cols)},
        FEATURE_META_PATH,
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, all_feature_cols


def preprocess_single(
    data: dict,
    scaler: StandardScaler | None = None,
    encoders: dict[str, LabelEncoder] | None = None,
    feature_cols: list[str] | None = None,
) -> np.ndarray:
    if scaler is None:
        scaler = joblib.load(SCALER_PATH)
    if encoders is None:
        encoders = joblib.load(ENCODERS_PATH)
    if feature_cols is None:
        meta = joblib.load(FEATURE_META_PATH)
        feature_cols = meta["feature_cols"]

    df = pd.DataFrame([data])
    df = prepare_features(df)
    df = _apply_encoders(df, encoders)

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    df = _apply_scaler(df, scaler, feature_cols)
    return df[feature_cols].values.astype(np.float32)
