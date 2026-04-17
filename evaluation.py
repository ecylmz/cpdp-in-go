from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


REPORTED_METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "precision_0",
    "recall_0",
    "f1_0",
    "precision_1",
    "recall_1",
    "f1_1",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "auc",
    "pr_auc",
    "mcc",
]


def summarize_numeric_series(values: pd.Series | np.ndarray) -> dict[str, Any]:
    numeric_values = pd.to_numeric(pd.Series(values), errors="coerce")
    valid_mask = np.isfinite(numeric_values.to_numpy(dtype=float))
    valid_values = numeric_values[valid_mask]

    summary: dict[str, Any] = {"count": int(len(valid_values))}
    if valid_values.empty:
        summary.update({"mean": None, "std": None, "median": None, "min": None, "max": None})
        return summary

    summary.update(
        {
            "mean": float(valid_values.mean()),
            "std": float(valid_values.std(ddof=0)),
            "median": float(valid_values.median()),
            "min": float(valid_values.min()),
            "max": float(valid_values.max()),
        }
    )
    return summary


def compute_binary_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    unique_classes = np.unique(y_true)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": float("nan"),
        "precision_0": report.get("0", {}).get("precision", 0.0),
        "recall_0": report.get("0", {}).get("recall", 0.0),
        "f1_0": report.get("0", {}).get("f1-score", 0.0),
        "precision_1": report.get("1", {}).get("precision", 0.0),
        "recall_1": report.get("1", {}).get("recall", 0.0),
        "f1_1": report.get("1", {}).get("f1-score", 0.0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "roc_auc": float("nan"),
        "auc": float("nan"),
        "pr_auc": float("nan"),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "test_total": int(len(y_true)),
        "test_negative": int((y_true == 0).sum()),
        "test_positive": int((y_true == 1).sum()),
        "test_positive_ratio": float(np.mean(y_true)) if len(y_true) else float("nan"),
        "predicted_negative": int((y_pred == 0).sum()),
        "predicted_positive": int((y_pred == 1).sum()),
    }

    if len(unique_classes) > 1:
        metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["auc"] = metrics["roc_auc"]
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)

    return metrics


def make_predictions_frame(
    target_metadata: pd.DataFrame,
    granularity: str,
    model_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> pd.DataFrame:
    y_pred = (y_prob >= threshold).astype(int)
    prediction_frame = target_metadata.copy()
    prediction_frame["granularity"] = granularity
    prediction_frame["model_name"] = model_name
    prediction_frame["y_true"] = y_true
    prediction_frame["y_prob"] = y_prob
    prediction_frame["y_pred"] = y_pred
    ordered_columns = [
        column
        for column in ["granularity", "model_name", "project_id", "sample_id"]
        if column in prediction_frame.columns
    ]
    remaining_columns = [column for column in prediction_frame.columns if column not in ordered_columns]
    return prediction_frame[ordered_columns + remaining_columns]


def aggregate_model_results(per_project_results: pd.DataFrame, metric_columns: list[str]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    valid_results = per_project_results[per_project_results["status"] == "ok"].copy()
    if valid_results.empty:
        return summaries

    for model_name, group in valid_results.groupby("model_name"):
        summary: dict[str, Any] = {
            "model_name": model_name,
            "target_project_count": int(group["target_project"].nunique()),
        }
        for metric in metric_columns:
            metric_summary = summarize_numeric_series(group[metric])
            summary[f"valid_{metric}_count"] = metric_summary["count"]
            summary[f"mean_{metric}"] = metric_summary["mean"]
            summary[f"std_{metric}"] = metric_summary["std"]
            summary[f"median_{metric}"] = metric_summary["median"]
            summary[f"min_{metric}"] = metric_summary["min"]
            summary[f"max_{metric}"] = metric_summary["max"]
        summaries.append(summary)
    return summaries