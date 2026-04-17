from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from lopo_runner import GranularityRunResult


def _flatten_metric_summaries(summary_metrics: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for metric_name, summary in summary_metrics.items():
        row[f"valid_{metric_name}_count"] = summary.get("count")
        row[f"mean_{metric_name}"] = summary.get("mean")
        row[f"std_{metric_name}"] = summary.get("std")
        row[f"median_{metric_name}"] = summary.get("median")
        row[f"min_{metric_name}"] = summary.get("min")
        row[f"max_{metric_name}"] = summary.get("max")
    return row


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_x = len(x)
    n_y = len(y)
    if n_x == 0 or n_y == 0:
        return float("nan"), "undefined"

    greater = 0
    lower = 0
    for value in x:
        greater += np.sum(value > y)
        lower += np.sum(value < y)

    delta = (greater - lower) / (n_x * n_y)
    absolute_delta = abs(delta)
    if absolute_delta < 0.147:
        magnitude = "negligible"
    elif absolute_delta < 0.33:
        magnitude = "small"
    elif absolute_delta < 0.474:
        magnitude = "medium"
    else:
        magnitude = "large"
    return float(delta), magnitude


def build_granularity_comparison_rows(results_by_granularity: dict[str, GranularityRunResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for granularity, result in results_by_granularity.items():
        selection_summary = result.selection_summary
        rows.append(
            {
                "granularity": granularity,
                "model_name": "nested_selection",
                "selection_scope": "outer-fold model-family selection",
                "selection_rule": selection_summary.get("selection_rule"),
                "is_best_model_for_stats": True,
                "target_project_count": selection_summary.get("target_project_count"),
                **_flatten_metric_summaries(selection_summary.get("summary_metrics", {})),
            }
        )
        for model_summary in result.aggregated_results.get("models", []):
            row = {
                "granularity": granularity,
                "selection_scope": "single_model_summary",
                **model_summary,
            }
            row["is_best_model_for_stats"] = False
            rows.append(row)
    return rows


def run_pairwise_granularity_tests(
    results_by_granularity: dict[str, GranularityRunResult],
    metrics: list[str],
) -> dict[str, Any]:
    comparisons = [("commit", "file"), ("commit", "method"), ("method", "file")]
    output: dict[str, Any] = {
        "selection_rule": "for each held-out target project, choose the model with the best inner-CV primary metric"
    }

    for left, right in comparisons:
        if left not in results_by_granularity or right not in results_by_granularity:
            continue

        left_result = results_by_granularity[left]
        right_result = results_by_granularity[right]
        left_scores = left_result.selection_results.copy()
        right_scores = right_result.selection_results.copy()
        if left_scores.empty or right_scores.empty:
            continue

        merged = left_scores.merge(right_scores, on="target_project", suffixes=(f"_{left}", f"_{right}"))
        comparison_key = f"{left}_vs_{right}"
        output[comparison_key] = {
            "left_granularity": left,
            "right_granularity": right,
            "left_selection_rule": left_result.selection_summary.get("selection_rule"),
            "right_selection_rule": right_result.selection_summary.get("selection_rule"),
            "left_selected_model_counts": left_result.selection_summary.get("selected_model_counts", {}),
            "right_selected_model_counts": right_result.selection_summary.get("selected_model_counts", {}),
            "common_projects": int(len(merged)),
            "metrics": {},
        }

        for metric in metrics:
            left_values = merged[f"{metric}_{left}"].to_numpy(dtype=float)
            right_values = merged[f"{metric}_{right}"].to_numpy(dtype=float)
            valid_mask = np.isfinite(left_values) & np.isfinite(right_values)
            left_valid = left_values[valid_mask]
            right_valid = right_values[valid_mask]
            if len(left_valid) < 2:
                output[comparison_key]["metrics"][metric] = {
                    "wilcoxon_statistic": None,
                    "wilcoxon_pvalue": None,
                    "cliffs_delta": None,
                    "cliffs_delta_magnitude": "undefined",
                    "paired_sample_count": int(len(left_valid)),
                }
                continue

            try:
                wilcoxon_result = wilcoxon(left_valid, right_valid, zero_method="wilcox", alternative="two-sided")
                statistic = float(wilcoxon_result.statistic)
                pvalue = float(wilcoxon_result.pvalue)
            except ValueError:
                statistic = None
                pvalue = None

            delta, magnitude = cliffs_delta(left_valid, right_valid)
            output[comparison_key]["metrics"][metric] = {
                "wilcoxon_statistic": statistic,
                "wilcoxon_pvalue": pvalue,
                "cliffs_delta": delta,
                "cliffs_delta_magnitude": magnitude,
                "paired_sample_count": int(len(left_valid)),
            }

    return output