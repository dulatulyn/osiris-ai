from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import numpy as np
import torch
import torch.nn as nn
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
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    METRICS_PATH,
    MODEL_PATH,
    MODEL_WEIGHTS_PATH,
)
from src.training.model import FraudDetector
from src.training.preprocessing import load_dataset, preprocess_train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _compute_pos_weight(y_train: np.ndarray, max_weight: float = 4.0) -> torch.Tensor:
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    weight = min(np.sqrt(n_neg / max(n_pos, 1)), max_weight)
    return torch.tensor([weight], dtype=torch.float32)


def _get_predictions(model: FraudDetector, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y_batch.numpy())

    return np.array(all_probs), np.array(all_labels)


def _evaluate(model: FraudDetector, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    all_probs, all_labels = _get_predictions(model, loader, device)
    preds = (all_probs >= 0.5).astype(int)
    auc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, preds)
    return auc, f1


def _find_optimal_threshold(model: FraudDetector, loader: DataLoader, device: torch.device) -> float:
    all_probs, all_labels = _get_predictions(model, loader, device)
    best_threshold = 0.5
    best_f1 = 0.0
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (all_probs >= t).astype(int)
        score = f1_score(all_labels, preds)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(t)
    return best_threshold


def _print_test_report(model: FraudDetector, loader: DataLoader, device: torch.device, threshold: float = 0.5) -> None:
    all_probs, all_labels = _get_predictions(model, loader, device)
    preds = (all_probs >= threshold).astype(int)
    logger.info("Using threshold: %.3f", threshold)

    tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "threshold": round(threshold, 4),
        "auc_roc": round(float(roc_auc_score(all_labels, all_probs)), 4),
        "auc_pr": round(float(average_precision_score(all_labels, all_probs)), 4),
        "log_loss": round(float(log_loss(all_labels, all_probs)), 4),
        "accuracy": round(float(accuracy_score(all_labels, preds)), 4),
        "precision": round(float(precision_score(all_labels, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(all_labels, preds, zero_division=0)), 4),
        "specificity": round(specificity, 4),
        "f1_score": round(float(f1_score(all_labels, preds, zero_division=0)), 4),
        "mcc": round(float(matthews_corrcoef(all_labels, preds)), 4),
        "confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        },
    }

    logger.info("=" * 60)
    logger.info("TEST SET RESULTS")
    logger.info("=" * 60)
    for key, value in metrics.items():
        if key == "confusion_matrix":
            continue
        logger.info("%-15s %.4f", key + ":", value)
    logger.info("")
    logger.info("Confusion Matrix:")
    logger.info("  TP=%d  FP=%d", tp, fp)
    logger.info("  FN=%d  TN=%d", fn, tn)
    logger.info("")
    logger.info("\n%s", classification_report(all_labels, preds, target_names=["legit", "fraud"], zero_division=0))

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", METRICS_PATH)


def train(
    data_path: str,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    logger.info("Loading dataset from %s", data_path)
    df = load_dataset(data_path)
    logger.info("Dataset shape: %s", df.shape)

    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = preprocess_train(df)
    logger.info("Features: %d, Train: %d, Val: %d, Test: %d", len(feature_cols), len(X_train), len(X_val), len(X_test))

    train_loader = _make_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader = _make_loader(X_val, y_val, batch_size, shuffle=False)
    test_loader = _make_loader(X_test, y_test, batch_size, shuffle=False)

    model = FraudDetector(input_dim=len(feature_cols), feature_cols=feature_cols).to(device)
    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    pos_weight = _compute_pos_weight(y_train).to(device)
    logger.info("pos_weight: %.2f", pos_weight.item())
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )

    best_auc = 0.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        val_auc, val_f1 = _evaluate(model, val_loader, device)
        scheduler.step(val_auc)

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %3d/%d | loss=%.4f | val_auc=%.4f | val_f1=%.4f | lr=%.1e | %.1fs",
            epoch, epochs, avg_loss, val_auc, val_f1, current_lr, elapsed,
        )

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            MODEL_PATH.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
            logger.info("  -> New best model saved (AUC=%.4f)", best_auc)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logger.info("Early stopping at epoch %d (best AUC=%.4f)", epoch, best_auc)
                break

    logger.info("Loading best model for final evaluation")
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device, weights_only=True))

    threshold = _find_optimal_threshold(model, val_loader, device)
    logger.info("Optimal threshold: %.3f", threshold)

    import joblib
    from src.config import FEATURE_META_PATH
    meta = joblib.load(FEATURE_META_PATH)
    meta["threshold"] = threshold
    joblib.dump(meta, FEATURE_META_PATH)
    logger.info("Threshold saved to feature_meta")

    _print_test_report(model, test_loader, device, threshold=threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the fraud detection model")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    args = parser.parse_args()

    train(data_path=args.data, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


if __name__ == "__main__":
    main()
