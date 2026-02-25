from __future__ import annotations

import argparse
import json
import logging
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import xgboost as xgb

from src.config import (
    CATEGORICAL_COLUMNS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_DEPTH,
    DEFAULT_N_ESTIMATORS,
    DROP_COLUMNS,
    EARLY_STOPPING_ROUNDS,
    FEATURE_META_PATH,
    METRICS_PATH,
    MODEL_PATH,
    NUMERIC_COLUMNS,
    PIPELINE_PATH,
    TARGET_COLUMN,
    TEST_RATIO,
    VAL_RATIO,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_and_split(
    data_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    logger.info("Loading dataset from %s", data_path)
    df = pd.read_csv(data_path)
    logger.info(
        "Dataset shape: %s | fraud rate: %.2f%%",
        df.shape,
        df[TARGET_COLUMN].mean() * 100,
    )

    drop_cols = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=drop_cols)

    feature_cols = [c for c in (NUMERIC_COLUMNS + CATEGORICAL_COLUMNS) if c in df.columns]
    logger.info("Feature columns (%d): %s", len(feature_cols), feature_cols)

    X = df[feature_cols]
    y = df[TARGET_COLUMN].values.astype(np.int32)

    val_test = VAL_RATIO + TEST_RATIO
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=val_test, random_state=42, stratify=y
    )
    rel_test = TEST_RATIO / val_test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=rel_test, random_state=42, stratify=y_temp
    )

    logger.info(
        "Train: %d | Val: %d | Test: %d",
        len(X_train), len(X_val), len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


def _build_pipeline(
    feature_cols: list[str],
    scale_pos_weight: float,
    n_estimators: int,
    max_depth: int,
    lr: float,
) -> Pipeline:
    num_cols = [c for c in NUMERIC_COLUMNS if c in feature_cols]
    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in feature_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=lr,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.05,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    return Pipeline([("preprocessor", preprocessor), ("classifier", clf)])


def _find_optimal_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (probs >= t).astype(int)
        score = f1_score(labels, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    logger.info("Optimal threshold: %.3f (F1=%.4f)", best_t, best_f1)
    return best_t


def _evaluate_and_save(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "threshold": round(threshold, 4),
        "auc_roc": round(float(roc_auc_score(labels, probs)), 4),
        "auc_pr": round(float(average_precision_score(labels, probs)), 4),
        "log_loss": round(float(log_loss(labels, probs)), 4),
        "accuracy": round(float(accuracy_score(labels, preds)), 4),
        "precision": round(float(precision_score(labels, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, preds, zero_division=0)), 4),
        "specificity": round(specificity, 4),
        "f1_score": round(float(f1_score(labels, preds, zero_division=0)), 4),
        "mcc": round(float(matthews_corrcoef(labels, preds)), 4),
        "confusion_matrix": {
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        },
        "total_samples": len(labels),
        "fraud_samples": int(labels.sum()),
        "legit_samples": int((labels == 0).sum()),
    }

    logger.info("=" * 60)
    logger.info("TEST SET RESULTS (threshold=%.3f)", threshold)
    logger.info("=" * 60)
    for k, v in metrics.items():
        if k == "confusion_matrix":
            continue
        logger.info("  %-25s %s", k + ":", v)
    logger.info("\n%s", classification_report(labels, preds, target_names=["legit", "fraud"], zero_division=0))

    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", METRICS_PATH)
    return metrics


def train(
    data_path: str,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    max_depth: int = DEFAULT_MAX_DEPTH,
    lr: float = DEFAULT_LEARNING_RATE,
) -> None:
    t0 = time.time()

    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = _load_and_split(data_path)

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = float(np.sqrt(n_neg / max(n_pos, 1)))
    logger.info(
        "Class balance — neg: %d | pos: %d | scale_pos_weight: %.3f",
        n_neg, n_pos, scale_pos_weight,
    )

    pipeline = _build_pipeline(
        feature_cols, scale_pos_weight, n_estimators, max_depth, lr
    )

    logger.info("Fitting preprocessor on training data...")
    preprocessor = pipeline.named_steps["preprocessor"]
    preprocessor.fit(X_train)
    X_train_t = preprocessor.transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    logger.info(
        "Training XGBoost (n_estimators=%d, max_depth=%d, lr=%.4f)...",
        n_estimators, max_depth, lr,
    )
    clf = pipeline.named_steps["classifier"]
    clf.fit(
        X_train_t,
        y_train,
        eval_set=[(X_train_t, y_train), (X_val_t, y_val)],
        verbose=50,
    )
    logger.info("Best iteration: %d", clf.best_iteration)

    val_probs = clf.predict_proba(X_val_t)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs)
    val_pr = average_precision_score(y_val, val_probs)
    logger.info("Validation AUC-ROC: %.4f | AUC-PR: %.4f", val_auc, val_pr)

    threshold = _find_optimal_threshold(val_probs, y_val)

    test_probs = clf.predict_proba(X_test_t)[:, 1]
    _evaluate_and_save(test_probs, y_test, threshold)

    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, PIPELINE_PATH)
    logger.info("Pipeline saved to %s (%.1f MB)", PIPELINE_PATH, PIPELINE_PATH.stat().st_size / 1024 / 1024)

    num_cols = [c for c in NUMERIC_COLUMNS if c in feature_cols]
    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in feature_cols]
    meta = {
        "feature_cols": feature_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "threshold": threshold,
        "n_features": len(feature_cols),
        "best_iteration": int(clf.best_iteration),
        "scale_pos_weight": scale_pos_weight,
        "val_auc_roc": round(val_auc, 4),
        "val_auc_pr": round(val_pr, 4),
    }
    joblib.dump(meta, FEATURE_META_PATH)
    logger.info("Feature meta saved to %s", FEATURE_META_PATH)

    elapsed = time.time() - t0
    logger.info("Training complete in %.1fs", elapsed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fraud detection model (XGBoost)")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    # Kept for backwards compat with old CLI calls, ignored
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    train(
        data_path=args.data,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
