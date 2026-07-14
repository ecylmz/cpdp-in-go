from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, studentized_range, wilcoxon


MODEL_ORDER = ["naive_bayes", "logistic_regression", "random_forest", "xgboost"]
MODEL_LABELS = {
    "naive_bayes": "Naive Bayes",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
MODEL_COLORS = {
    "naive_bayes": "#b08968",
    "logistic_regression": "#457b9d",
    "random_forest": "#2a9d8f",
    "xgboost": "#e76f51",
}
ALL_GRANULARITIES = ["commit", "file", "method"]
GRANULARITY_LABELS = {"commit": "Commit", "file": "File", "method": "Method"}
GRANULARITY_COLORS = {"commit": "#264653", "file": "#d1495b", "method": "#6c8f3d"}
PAIRWISE_COMPARISON_ORDER = [("commit", "file"), ("commit", "method"), ("method", "file")]
SUMMARY_METRICS = ["f1_1", "balanced_accuracy", "mcc", "auc", "pr_auc", "precision_1", "recall_1"]
FOCUS_METRICS = ["f1_1", "mcc"]
FOCUS_METRIC_LABELS = {"f1_1": "$F_1$", "mcc": "MCC"}
BOOTSTRAP_RESAMPLES = 10000
CONFIDENCE_LEVEL = 0.95
LATEX_LINEBREAK = chr(92) * 2
ADEQUACY_THRESHOLDS = {
    "primary_rows": 200,
    "primary_minority": 50,
    "exploratory_rows": 100,
    "exploratory_minority": 30,
}
ADEQUACY_ORDER = ["PRIMARY", "EXPLORATORY", "INSUFFICIENT"]
ADEQUACY_SHORT_LABELS = {"PRIMARY": "P", "EXPLORATORY": "E", "INSUFFICIENT": "I"}
SUPPORT_COLORS = {"primary": "#264653", "low_support": "#d1495b"}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate tables and figures for commit/file/method LOPO baseline results."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=repo_root / "results_lopo_baseline",
        help="Directory containing the LOPO result folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "analysis_output",
        help="Directory where analysis tables and figures will be written.",
    )
    parser.add_argument(
        "--granularities",
        nargs="*",
        choices=ALL_GRANULARITIES,
        default=None,
        help="Granularities to include. Defaults to all available result folders.",
    )
    return parser.parse_args()


def latex_escape(text: Any) -> str:
    return str(text).replace("_", "\\_").replace("&", "\\&")


def json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_compatible(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_compatible(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def format_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "--"
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(value_float):
        return "--"
    if round(value_float, digits) == 0:
        value_float = 0.0
    return f"{value_float:.{digits}f}"


def format_mean_std(mean_value: Any, std_value: Any, digits: int = 3) -> str:
    if mean_value is None:
        return "--"
    try:
        mean_float = float(mean_value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(mean_float):
        return "--"

    try:
        std_float = float(std_value)
    except (TypeError, ValueError):
        std_float = 0.0
    if not np.isfinite(std_float):
        std_float = 0.0
    return f"{mean_float:.{digits}f} $\\pm$ {std_float:.{digits}f}"


def format_interval(lower_value: Any, upper_value: Any, digits: int = 3) -> str:
    return f"[{format_float(lower_value, digits)}, {format_float(upper_value, digits)}]"


def format_count_share(count: int, total: int) -> str:
    if total <= 0:
        return str(int(count))
    percentage = 100.0 * count / total
    return f"{int(count)} ({percentage:.0f}\\%)"


def format_pvalue(value: Any) -> str:
    if value is None:
        return "--"
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(value_float):
        return "--"
    if value_float < 0.001:
        return "<0.001"
    return f"{value_float:.3f}"


def format_pvalue_latex(value: Any) -> str:
    """Format a p-value safely for a LaTeX table cell."""
    formatted = format_pvalue(value)
    if formatted.startswith("<"):
        return f"$<{formatted[1:]}$"
    return formatted


def join_human_labels(labels: list[str]) -> str:
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def numeric_values(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)


def adequacy_sort_key(label: str) -> int:
    try:
        return ADEQUACY_ORDER.index(label)
    except ValueError:
        return len(ADEQUACY_ORDER)


def classify_adequacy(target_row_count: Any, bug_count: Any, non_bug_count: Any) -> str:
    row_count = int(target_row_count)
    minority_count = int(min(float(bug_count), float(non_bug_count)))
    if row_count >= ADEQUACY_THRESHOLDS["primary_rows"] and minority_count >= ADEQUACY_THRESHOLDS["primary_minority"]:
        return "PRIMARY"
    if row_count >= ADEQUACY_THRESHOLDS["exploratory_rows"] and minority_count >= ADEQUACY_THRESHOLDS["exploratory_minority"]:
        return "EXPLORATORY"
    return "INSUFFICIENT"


def paired_wilcoxon_pvalue(values: pd.Series | np.ndarray) -> float | None:
    numeric = np.asarray(values, dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    if numeric.size == 0 or np.all(np.abs(numeric) <= 1e-12):
        return None
    try:
        return float(wilcoxon(numeric, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return None


def rank_biserial_correlation(values: pd.Series | np.ndarray) -> float | None:
    numeric = np.asarray(values, dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    numeric = numeric[np.abs(numeric) > 1e-12]
    if numeric.size == 0:
        return None

    ranks = pd.Series(np.abs(numeric)).rank(method="average").to_numpy(dtype=float)
    total_rank = float(ranks.sum())
    if total_rank <= 0.0:
        return None

    positive_rank = float(ranks[numeric > 0].sum())
    negative_rank = float(ranks[numeric < 0].sum())
    return (positive_rank - negative_rank) / total_rank


def diff_win_tie_loss(values: pd.Series | np.ndarray) -> tuple[int, int, int]:
    numeric = np.asarray(values, dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    win_count = int((numeric > 1e-12).sum())
    tie_count = int((np.abs(numeric) <= 1e-12).sum())
    loss_count = int((numeric < -1e-12).sum())
    return win_count, tie_count, loss_count


def available_granularities(results_root: Path) -> list[str]:
    available: list[str] = []
    for granularity in ALL_GRANULARITIES:
        per_project_path = results_root / granularity / "per_project_results.csv"
        summary_path = results_root / granularity / "analysis_summary.json"
        if per_project_path.exists() and summary_path.exists():
            available.append(granularity)
    return available


def canonical_granularities(requested: list[str] | None, available: list[str]) -> list[str]:
    if requested:
        requested_set = set(requested)
        return [granularity for granularity in ALL_GRANULARITIES if granularity in requested_set and granularity in available]
    return [granularity for granularity in ALL_GRANULARITIES if granularity in available]


def select_nested(df: pd.DataFrame) -> pd.DataFrame:
    order_map = {model_name: index for index, model_name in enumerate(MODEL_ORDER)}
    return (
        df.assign(model_order=df["model_name"].map(order_map))
        .sort_values(
            ["target_project", "best_inner_f1", "best_inner_mcc", "model_order"],
            ascending=[True, False, False, True],
        )
        .groupby("target_project", as_index=False)
        .first()
        .drop(columns=["model_order"])
    )


def series_mean_std(series: pd.Series) -> tuple[float, float]:
    values = numeric_values(series)
    if values.size == 0:
        return np.nan, np.nan
    mean_value = float(values.mean())
    std_value = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return mean_value, std_value


def bootstrap_mean_ci(
    series: pd.Series,
    confidence_level: float = CONFIDENCE_LEVEL,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 42,
) -> tuple[float, float]:
    values = numeric_values(series)
    if values.size == 0:
        return np.nan, np.nan
    if values.size == 1:
        return float(values[0]), float(values[0])

    rng = np.random.default_rng(seed)
    bootstrap_samples = rng.choice(values, size=(n_resamples, values.size), replace=True)
    bootstrap_means = bootstrap_samples.mean(axis=1)
    alpha = 1.0 - confidence_level
    lower_value, upper_value = np.quantile(bootstrap_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lower_value), float(upper_value)


def summarize_series(series: pd.Series) -> dict[str, float]:
    mean_value, std_value = series_mean_std(series)
    ci_low, ci_high = bootstrap_mean_ci(series)
    return {
        "mean": mean_value,
        "std": std_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def summary_stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for metric in SUMMARY_METRICS:
        stats[metric] = summarize_series(df[metric])
    return stats


def nemenyi_critical_difference(granularity_count: int, project_count: int, alpha: float = 0.05) -> float | None:
    if granularity_count < 3 or project_count <= 0:
        return None
    studentized_quantile = studentized_range.ppf(1.0 - alpha, granularity_count, np.inf)
    if not np.isfinite(studentized_quantile):
        return None
    q_alpha = studentized_quantile / np.sqrt(2.0)
    return float(q_alpha * np.sqrt(granularity_count * (granularity_count + 1.0) / (6.0 * project_count)))


def load_granularity(results_root: Path, granularity: str) -> dict[str, Any]:
    per_project_path = results_root / granularity / "per_project_results.csv"
    summary_path = results_root / granularity / "analysis_summary.json"
    if not per_project_path.exists() or not summary_path.exists():
        raise FileNotFoundError(f"Missing result files for granularity '{granularity}'.")

    per_project = pd.read_csv(per_project_path)
    ok_per_project = per_project.copy()
    if "status" in ok_per_project.columns:
        ok_per_project = ok_per_project[ok_per_project["status"] == "ok"].copy()
    ok_per_project["granularity"] = granularity

    analysis_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selected = select_nested(ok_per_project).copy()
    selected["granularity"] = granularity

    project_frame = pd.DataFrame(analysis_summary["dataset_info"]["projects"]).copy()
    if "bug_ratio" not in project_frame.columns:
        project_frame["bug_ratio"] = project_frame["bug_count"] / project_frame["row_count"]

    total_bug_ratio = analysis_summary["dataset_info"]["total_bug_count"] / analysis_summary["dataset_info"]["total_rows"]
    exact_duplicate_ratio = sum(project["exact_duplicate_rows"] for project in analysis_summary["dataset_info"]["projects"]) / sum(
        project["raw_row_count"] for project in analysis_summary["dataset_info"]["projects"]
    )

    selected["always_positive_f1_1"] = 2 * selected["target_bug_ratio"] / (1 + selected["target_bug_ratio"])
    selected["majority_f1_1"] = np.where(selected["target_bug_ratio"] >= 0.5, selected["always_positive_f1_1"], 0.0)
    selected["f1_gap_to_majority"] = selected["f1_1"] - selected["majority_f1_1"]
    selected["minority_count"] = selected[["target_bug_count", "target_non_bug_count"]].min(axis=1)
    selected["adequacy"] = selected.apply(
        lambda row: classify_adequacy(row["target_row_count"], row["target_bug_count"], row["target_non_bug_count"]),
        axis=1,
    )

    best_single_model = (
        ok_per_project.groupby("model_name", as_index=False)["f1_1"].mean().sort_values("f1_1", ascending=False).iloc[0]["model_name"]
    )

    return {
        "granularity": granularity,
        "per_project": ok_per_project,
        "selected": selected,
        "analysis_summary": analysis_summary,
        "project_frame": project_frame,
        "overall_bug_ratio": float(total_bug_ratio),
        "exact_duplicate_ratio": float(exact_duplicate_ratio),
        "best_single_model": str(best_single_model),
    }


def load_statistical_tests(results_root: Path) -> dict[str, Any]:
    statistical_tests_path = results_root / "statistical_tests.json"
    if not statistical_tests_path.exists():
        return {}
    return json.loads(statistical_tests_path.read_text(encoding="utf-8"))


def build_summary_rows(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    granularity = bundle["granularity"]
    nested = summary_stats(bundle["selected"])
    best_model_name = bundle["best_single_model"]
    best_single_frame = bundle["per_project"][bundle["per_project"]["model_name"] == best_model_name]
    best_single = summary_stats(best_single_frame)

    rows: list[dict[str, Any]] = []
    for view_name, model_name, metric_payload in [
        ("Nested selected", "fold_local", nested),
        (f"Best single model ({MODEL_LABELS[best_model_name]})", best_model_name, best_single),
    ]:
        row = {
            "granularity": granularity,
            "view": view_name,
            "model_name": model_name,
        }
        for metric in SUMMARY_METRICS:
            row[f"mean_{metric}"] = metric_payload[metric]["mean"]
            row[f"std_{metric}"] = metric_payload[metric]["std"]
            row[f"ci_low_{metric}"] = metric_payload[metric]["ci_low"]
            row[f"ci_high_{metric}"] = metric_payload[metric]["ci_high"]
        rows.append(row)
    return rows


def build_model_summary(bundle: dict[str, Any]) -> pd.DataFrame:
    grouped = (
        bundle["per_project"]
        .groupby("model_name", as_index=False)
        .agg(
            mean_f1_1=("f1_1", "mean"),
            std_f1_1=("f1_1", "std"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy=("balanced_accuracy", "std"),
            mean_mcc=("mcc", "mean"),
            std_mcc=("mcc", "std"),
            mean_auc=("auc", "mean"),
            std_auc=("auc", "std"),
            mean_pr_auc=("pr_auc", "mean"),
            std_pr_auc=("pr_auc", "std"),
            mean_precision_1=("precision_1", "mean"),
            std_precision_1=("precision_1", "std"),
            mean_recall_1=("recall_1", "mean"),
            std_recall_1=("recall_1", "std"),
        )
    )
    grouped.insert(0, "granularity", bundle["granularity"])
    return grouped


def build_focus_ci_rows(bundles: dict[str, dict[str, Any]], granularities: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for granularity in granularities:
        nested_stats = summary_stats(bundles[granularity]["selected"])
        row = {"granularity": granularity}
        for metric in FOCUS_METRICS:
            row[f"mean_{metric}"] = nested_stats[metric]["mean"]
            row[f"std_{metric}"] = nested_stats[metric]["std"]
            row[f"ci_low_{metric}"] = nested_stats[metric]["ci_low"]
            row[f"ci_high_{metric}"] = nested_stats[metric]["ci_high"]
        rows.append(row)
    return rows


def build_variability_rows(bundles: dict[str, dict[str, Any]], granularities: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for granularity in granularities:
        selected = bundles[granularity]["selected"]
        f1_values = numeric_values(selected["f1_1"])
        mcc_values = numeric_values(selected["mcc"])
        rows.append(
            {
                "granularity": granularity,
                "project_count": int(len(selected)),
                "mean_f1_1": float(f1_values.mean()) if f1_values.size else np.nan,
                "std_f1_1": float(f1_values.std(ddof=1)) if f1_values.size > 1 else 0.0,
                "median_f1_1": float(np.median(f1_values)) if f1_values.size else np.nan,
                "min_f1_1": float(f1_values.min()) if f1_values.size else np.nan,
                "max_f1_1": float(f1_values.max()) if f1_values.size else np.nan,
                "mean_mcc": float(mcc_values.mean()) if mcc_values.size else np.nan,
                "std_mcc": float(mcc_values.std(ddof=1)) if mcc_values.size > 1 else 0.0,
                "median_mcc": float(np.median(mcc_values)) if mcc_values.size else np.nan,
                "min_mcc": float(mcc_values.min()) if mcc_values.size else np.nan,
                "max_mcc": float(mcc_values.max()) if mcc_values.size else np.nan,
            }
        )
    return rows


def build_selection_gain_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_frame = pd.DataFrame(summary_rows)
    if summary_frame.empty:
        return []

    rows: list[dict[str, Any]] = []
    available_granularities = set(summary_frame["granularity"])
    for granularity in [item for item in ALL_GRANULARITIES if item in available_granularities]:
        subset = summary_frame[summary_frame["granularity"] == granularity]
        nested_rows = subset[subset["view"] == "Nested selected"]
        best_single_rows = subset[subset["view"].str.startswith("Best single model", na=False)]
        if nested_rows.empty or best_single_rows.empty:
            continue

        nested_row = nested_rows.iloc[0]
        best_single_row = best_single_rows.iloc[0]
        rows.append(
            {
                "granularity": granularity,
                "best_single_model_name": best_single_row["model_name"],
                "best_single_view": best_single_row["view"],
                "delta_f1_1": float(nested_row["mean_f1_1"] - best_single_row["mean_f1_1"]),
                "delta_mcc": float(nested_row["mean_mcc"] - best_single_row["mean_mcc"]),
                "delta_auc": float(nested_row["mean_auc"] - best_single_row["mean_auc"]),
                "delta_pr_auc": float(nested_row["mean_pr_auc"] - best_single_row["mean_pr_auc"]),
            }
        )
    return rows


def build_adequacy_summary_rows(bundles: dict[str, dict[str, Any]], granularities: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for granularity in granularities:
        selected = bundles[granularity]["selected"]
        counts = selected["adequacy"].value_counts()
        rows.append(
            {
                "granularity": granularity,
                "primary_count": int(counts.get("PRIMARY", 0)),
                "exploratory_count": int(counts.get("EXPLORATORY", 0)),
                "insufficient_count": int(counts.get("INSUFFICIENT", 0)),
            }
        )
    return rows


def build_adequacy_matrix_rows(bundles: dict[str, dict[str, Any]], granularities: list[str]) -> list[dict[str, Any]]:
    project_names = sorted(
        {project_name for granularity in granularities for project_name in bundles[granularity]["selected"]["target_project"]}
    )
    rows: list[dict[str, Any]] = []
    for project_name in project_names:
        row: dict[str, Any] = {"target_project": project_name}
        for granularity in granularities:
            match = bundles[granularity]["selected"]
            match = match[match["target_project"] == project_name]
            if match.empty:
                row[f"{granularity}_adequacy"] = None
            else:
                row[f"{granularity}_adequacy"] = str(match.iloc[0]["adequacy"])
        rows.append(row)
    return rows


def build_support_robustness_rows(selected_frame: pd.DataFrame, granularities: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    available = set(granularities)

    for left_granularity, right_granularity in PAIRWISE_COMPARISON_ORDER:
        if left_granularity not in available or right_granularity not in available:
            continue

        left_frame = selected_frame[selected_frame["granularity"] == left_granularity][["target_project", "adequacy"] + FOCUS_METRICS]
        right_frame = selected_frame[selected_frame["granularity"] == right_granularity][["target_project", "adequacy"] + FOCUS_METRICS]
        merged = left_frame.merge(right_frame, on="target_project", suffixes=(f"_{left_granularity}", f"_{right_granularity}"))
        if merged.empty:
            continue

        primary_mask = (merged[f"adequacy_{left_granularity}"] == "PRIMARY") & (
            merged[f"adequacy_{right_granularity}"] == "PRIMARY"
        )
        primary_merged = merged[primary_mask].copy()

        for metric_name in FOCUS_METRICS:
            all_diff = merged[f"{metric_name}_{left_granularity}"] - merged[f"{metric_name}_{right_granularity}"]
            primary_diff = primary_merged[f"{metric_name}_{left_granularity}"] - primary_merged[f"{metric_name}_{right_granularity}"]
            all_win_count, all_tie_count, all_loss_count = diff_win_tie_loss(all_diff)
            primary_win_count, primary_tie_count, primary_loss_count = diff_win_tie_loss(primary_diff)
            rows.append(
                {
                    "comparison": f"{GRANULARITY_LABELS[left_granularity]} vs {GRANULARITY_LABELS[right_granularity]}",
                    "metric": metric_name,
                    "all_mean_diff": float(all_diff.mean()) if len(all_diff) else np.nan,
                    "all_sample_count": int(len(all_diff)),
                    "all_wilcoxon_pvalue": paired_wilcoxon_pvalue(all_diff),
                    "all_win_count": all_win_count,
                    "all_tie_count": all_tie_count,
                    "all_loss_count": all_loss_count,
                    "primary_mean_diff": float(primary_diff.mean()) if len(primary_diff) else np.nan,
                    "primary_sample_count": int(len(primary_diff)),
                    "primary_wilcoxon_pvalue": paired_wilcoxon_pvalue(primary_diff),
                    "primary_win_count": primary_win_count,
                    "primary_tie_count": primary_tie_count,
                    "primary_loss_count": primary_loss_count,
                }
            )
    return rows


def build_prevalence_diagnostic_rows(
    bundles: dict[str, dict[str, Any]], granularities: list[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for granularity in granularities:
        selected = bundles[granularity]["selected"]
        gap_values = numeric_values(selected["f1_gap_to_majority"])
        mcc_values = numeric_values(selected["mcc"])
        tie_mask = selected["f1_gap_to_majority"].abs() < 1e-12
        rows.append(
            {
                "granularity": granularity,
                "project_count": int(len(selected)),
                "beats_majority_count": int((selected["f1_gap_to_majority"] > 0).sum()),
                "ties_majority_count": int(tie_mask.sum()),
                "below_majority_count": int((selected["f1_gap_to_majority"] < 0).sum()),
                "mean_gap_f1_1": float(gap_values.mean()) if gap_values.size else np.nan,
                "median_gap_f1_1": float(np.median(gap_values)) if gap_values.size else np.nan,
                "min_gap_f1_1": float(gap_values.min()) if gap_values.size else np.nan,
                "max_gap_f1_1": float(gap_values.max()) if gap_values.size else np.nan,
                "negative_mcc_count": int((selected["mcc"] < 0).sum()),
                "mean_mcc": float(mcc_values.mean()) if mcc_values.size else np.nan,
                "median_mcc": float(np.median(mcc_values)) if mcc_values.size else np.nan,
            }
        )
    return rows


def build_data_profile_rows(bundles: dict[str, dict[str, Any]], granularities: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for granularity in granularities:
        bundle = bundles[granularity]
        project_frame = bundle["project_frame"]
        mean_target_rows, std_target_rows = series_mean_std(project_frame["row_count"])
        mean_target_bug_ratio, std_target_bug_ratio = series_mean_std(project_frame["bug_ratio"])
        rows.append(
            {
                "granularity": granularity,
                "project_count": int(project_frame["project_id"].nunique()),
                "total_rows": int(project_frame["row_count"].sum()),
                "overall_bug_ratio": bundle["overall_bug_ratio"],
                "exact_duplicate_ratio": bundle["exact_duplicate_ratio"],
                "mean_target_rows": mean_target_rows,
                "std_target_rows": std_target_rows,
                "median_target_rows": float(project_frame["row_count"].median()),
                "mean_target_bug_ratio": mean_target_bug_ratio,
                "std_target_bug_ratio": std_target_bug_ratio,
                "selected_high_f1_count": int((bundle["selected"]["f1_1"] >= 0.75).sum()),
                "selected_moderate_f1_count": int((bundle["selected"]["f1_1"] >= 0.50).sum()),
            }
        )
    return rows


def build_pairwise_rows(
    selected_frame: pd.DataFrame,
    statistical_tests: dict[str, Any],
    granularities: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    available = set(granularities)

    for left_granularity, right_granularity in PAIRWISE_COMPARISON_ORDER:
        if left_granularity not in available or right_granularity not in available:
            continue

        left_frame = selected_frame[selected_frame["granularity"] == left_granularity]
        right_frame = selected_frame[selected_frame["granularity"] == right_granularity]
        merged = left_frame.merge(right_frame, on="target_project", suffixes=(f"_{left_granularity}", f"_{right_granularity}"))
        if merged.empty:
            continue

        stats_payload = statistical_tests.get(f"{left_granularity}_vs_{right_granularity}")
        reverse_direction = False
        if stats_payload is None:
            stats_payload = statistical_tests.get(f"{right_granularity}_vs_{left_granularity}")
            reverse_direction = True

        for metric_name, stat_metric_name in [("f1_1", "f1"), ("mcc", "mcc")]:
            diff = merged[f"{metric_name}_{left_granularity}"] - merged[f"{metric_name}_{right_granularity}"]
            diff_summary = summarize_series(diff)
            win_count = int((diff > 1e-12).sum())
            tie_count = int((diff.abs() <= 1e-12).sum())
            loss_count = int((diff < -1e-12).sum())

            wilcoxon_pvalue = paired_wilcoxon_pvalue(diff)
            rank_biserial = rank_biserial_correlation(diff)
            cliffs_delta = None
            cliffs_delta_magnitude = "undefined"
            paired_sample_count = int(len(diff))
            if stats_payload is not None:
                metric_payload = stats_payload.get("metrics", {}).get(stat_metric_name, {})
                wilcoxon_pvalue = metric_payload.get("wilcoxon_pvalue")
                cliffs_delta = metric_payload.get("cliffs_delta")
                if reverse_direction and cliffs_delta is not None:
                    cliffs_delta = -float(cliffs_delta)
                cliffs_delta_magnitude = metric_payload.get("cliffs_delta_magnitude", "undefined")
                paired_sample_count = int(metric_payload.get("paired_sample_count", paired_sample_count))

            rows.append(
                {
                    "comparison": f"{GRANULARITY_LABELS[left_granularity]} vs {GRANULARITY_LABELS[right_granularity]}",
                    "left_granularity": left_granularity,
                    "right_granularity": right_granularity,
                    "metric": metric_name,
                    "mean_diff": diff_summary["mean"],
                    "std_diff": diff_summary["std"],
                    "ci_low_diff": diff_summary["ci_low"],
                    "ci_high_diff": diff_summary["ci_high"],
                    "median_diff": float(pd.Series(diff).median()),
                    "win_count": win_count,
                    "tie_count": tie_count,
                    "loss_count": loss_count,
                    "wilcoxon_pvalue": wilcoxon_pvalue,
                    "rank_biserial": rank_biserial,
                    "cliffs_delta": cliffs_delta,
                    "cliffs_delta_magnitude": cliffs_delta_magnitude,
                    "paired_sample_count": paired_sample_count,
                }
            )
    return rows


def build_rank_rows(selected_frame: pd.DataFrame, granularities: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    granularity_count = len(granularities)

    for metric in FOCUS_METRICS:
        pivot = selected_frame.pivot(index="target_project", columns="granularity", values=metric).reindex(columns=granularities)
        pivot = pivot.dropna()
        if pivot.empty:
            continue

        ranks = pivot.rank(axis=1, method="average", ascending=False)
        best_flags = ranks.eq(ranks.min(axis=1), axis=0)
        friedman_statistic = None
        friedman_pvalue = None
        if granularity_count >= 3 and len(pivot) > 1:
            friedman_statistic, friedman_pvalue = friedmanchisquare(*[pivot[granularity] for granularity in granularities])
        critical_difference = nemenyi_critical_difference(granularity_count, len(pivot))

        for granularity in granularities:
            rows.append(
                {
                    "metric": metric,
                    "granularity": granularity,
                    "average_rank": float(ranks[granularity].mean()),
                    "std_rank": float(ranks[granularity].std(ddof=1)) if len(ranks) > 1 else 0.0,
                    "best_fold_wins": int(best_flags[granularity].sum()),
                    "project_count": int(len(pivot)),
                    "friedman_statistic": float(friedman_statistic) if friedman_statistic is not None else None,
                    "friedman_pvalue": float(friedman_pvalue) if friedman_pvalue is not None else None,
                    "critical_difference": critical_difference,
                }
            )
    return rows


def write_text(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="ascii")


def write_macros(output_dir: Path, output_prefix: str, bundles: dict[str, dict[str, Any]], granularities: list[str]) -> None:
    lines = [
        f"\\newcommand{{\\LopoAdequacyPrimaryRows}}{{{ADEQUACY_THRESHOLDS['primary_rows']}}}",
        f"\\newcommand{{\\LopoAdequacyPrimaryMinority}}{{{ADEQUACY_THRESHOLDS['primary_minority']}}}",
        f"\\newcommand{{\\LopoAdequacyExploratoryRows}}{{{ADEQUACY_THRESHOLDS['exploratory_rows']}}}",
        f"\\newcommand{{\\LopoAdequacyExploratoryMinority}}{{{ADEQUACY_THRESHOLDS['exploratory_minority']}}}",
    ]
    for granularity in granularities:
        bundle = bundles[granularity]
        nested = summary_stats(bundle["selected"])
        adequacy_counts = bundle["selected"]["adequacy"].value_counts()
        label_prefix = f"Lopo{GRANULARITY_LABELS[granularity]}"
        lines.extend(
            [
                f"\\newcommand{{\\{label_prefix}NestedFOne}}{{{format_float(nested['f1_1']['mean'])}}}",
                f"\\newcommand{{\\{label_prefix}NestedFOneStd}}{{{format_float(nested['f1_1']['std'])}}}",
                f"\\newcommand{{\\{label_prefix}NestedFOneCiLow}}{{{format_float(nested['f1_1']['ci_low'])}}}",
                f"\\newcommand{{\\{label_prefix}NestedFOneCiHigh}}{{{format_float(nested['f1_1']['ci_high'])}}}",
                f"\\newcommand{{\\{label_prefix}NestedBalancedAccuracy}}{{{format_float(nested['balanced_accuracy']['mean'])}}}",
                f"\\newcommand{{\\{label_prefix}NestedBalancedAccuracyStd}}{{{format_float(nested['balanced_accuracy']['std'])}}}",
                f"\\newcommand{{\\{label_prefix}NestedMcc}}{{{format_float(nested['mcc']['mean'])}}}",
                f"\\newcommand{{\\{label_prefix}NestedMccStd}}{{{format_float(nested['mcc']['std'])}}}",
                f"\\newcommand{{\\{label_prefix}NestedMccCiLow}}{{{format_float(nested['mcc']['ci_low'])}}}",
                f"\\newcommand{{\\{label_prefix}NestedMccCiHigh}}{{{format_float(nested['mcc']['ci_high'])}}}",
                f"\\newcommand{{\\{label_prefix}NestedAuc}}{{{format_float(nested['auc']['mean'])}}}",
                f"\\newcommand{{\\{label_prefix}NestedPrAuc}}{{{format_float(nested['pr_auc']['mean'])}}}",
                f"\\newcommand{{\\{label_prefix}BugRatio}}{{{format_float(bundle['overall_bug_ratio'])}}}",
                f"\\newcommand{{\\{label_prefix}DuplicateRatio}}{{{format_float(bundle['exact_duplicate_ratio'])}}}",
                f"\\newcommand{{\\{label_prefix}HighFoneCount}}{{{int((bundle['selected']['f1_1'] >= 0.75).sum())}}}",
                f"\\newcommand{{\\{label_prefix}ModerateFoneCount}}{{{int((bundle['selected']['f1_1'] >= 0.50).sum())}}}",
                f"\\newcommand{{\\{label_prefix}PrimaryCount}}{{{int(adequacy_counts.get('PRIMARY', 0))}}}",
                f"\\newcommand{{\\{label_prefix}ExploratoryCount}}{{{int(adequacy_counts.get('EXPLORATORY', 0))}}}",
                f"\\newcommand{{\\{label_prefix}InsufficientCount}}{{{int(adequacy_counts.get('INSUFFICIENT', 0))}}}",
            ]
        )
    write_text(output_dir / f"{output_prefix}_macros.tex", "\n".join(lines))


def write_summary_table(
    output_dir: Path,
    output_prefix: str,
    summary_rows: list[dict[str, Any]],
    granularities: list[str],
) -> None:
    scope_text = join_human_labels([GRANULARITY_LABELS[granularity].lower() for granularity in granularities])
    displayed_rows = [row for row in summary_rows if row.get("view") == "Nested selected"]
    if not displayed_rows:
        displayed_rows = summary_rows
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{Summary of the fold-local selected LOPO outputs for {scope_text} granularity. Values are reported as mean $\\pm$ sample standard deviation across held-out target projects.}}",
        f"\\label{{tab:{output_prefix.replace('_', '-')}-summary}}",
        "\\small",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Granularity & $F_1$ & Bal. Acc. & MCC & AUC & PR-AUC " + LATEX_LINEBREAK,
        "\\midrule",
    ]
    for row in displayed_rows:
        lines.append(
            " & ".join(
                [
                    GRANULARITY_LABELS[row["granularity"]],
                    format_mean_std(row["mean_f1_1"], row["std_f1_1"]),
                    format_mean_std(row["mean_balanced_accuracy"], row["std_balanced_accuracy"]),
                    format_mean_std(row["mean_mcc"], row["std_mcc"]),
                    format_mean_std(row["mean_auc"], row["std_auc"]),
                    format_mean_std(row["mean_pr_auc"], row["std_pr_auc"]),
                ]
            )
            + " "
            + LATEX_LINEBREAK
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table*}"])
    write_text(output_dir / f"{output_prefix}_summary_table.tex", "\n".join(lines))


def write_focus_ci_table(output_dir: Path, output_prefix: str, focus_ci_rows: list[dict[str, Any]]) -> None:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Appendix summary of the held-out performance of the fold-local selected models. For positive-class $F_1$ and MCC, the table reports mean $\\pm$ sample standard deviation together with nonparametric bootstrap confidence intervals over held-out projects.}",
        f"\\label{{tab:{output_prefix.replace('_', '-')}-ci}}",
        "\\small",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Granularity & $F_1$ & $F_1$ 95\\% CI & MCC & MCC 95\\% CI " + LATEX_LINEBREAK,
        "\\midrule",
    ]
    for row in focus_ci_rows:
        lines.append(
            " & ".join(
                [
                    GRANULARITY_LABELS[row["granularity"]],
                    format_mean_std(row["mean_f1_1"], row["std_f1_1"]),
                    format_interval(row["ci_low_f1_1"], row["ci_high_f1_1"]),
                    format_mean_std(row["mean_mcc"], row["std_mcc"]),
                    format_interval(row["ci_low_mcc"], row["ci_high_mcc"]),
                ]
            )
            + " "
            + LATEX_LINEBREAK
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    write_text(output_dir / f"{output_prefix}_ci_table.tex", "\n".join(lines))


def write_variability_table(output_dir: Path, output_prefix: str, variability_rows: list[dict[str, Any]]) -> None:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Across-project variability of the held-out performance of the fold-local selected models. Mean $\\pm$ sample standard deviation is complemented with the median and full observed range.}",
        f"\\label{{tab:{output_prefix.replace('_', '-')}-variability}}",
        "\\small",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Granularity & $n$ & $F_1$ mean $\\pm$ sd & $F_1$ median & $F_1$ range & MCC mean $\\pm$ sd & MCC median & MCC range " + LATEX_LINEBREAK,
        "\\midrule",
    ]
    for row in variability_rows:
        lines.append(
            " & ".join(
                [
                    GRANULARITY_LABELS[row["granularity"]],
                    str(int(row["project_count"])),
                    format_mean_std(row["mean_f1_1"], row["std_f1_1"]),
                    format_float(row["median_f1_1"]),
                    format_interval(row["min_f1_1"], row["max_f1_1"]),
                    format_mean_std(row["mean_mcc"], row["std_mcc"]),
                    format_float(row["median_mcc"]),
                    format_interval(row["min_mcc"], row["max_mcc"]),
                ]
            )
            + " "
            + LATEX_LINEBREAK
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table*}"])
    write_text(output_dir / f"{output_prefix}_variability_table.tex", "\n".join(lines))


def write_selection_gain_table(output_dir: Path, output_prefix: str, selection_gain_rows: list[dict[str, Any]]) -> None:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Descriptive gain from fold-local model selection over the single best model chosen post hoc for each granularity. Positive deltas favor per-project fold-local selection. These deltas are descriptive only and are not used for significance claims.}",
        f"\\label{{tab:{output_prefix.replace('_', '-')}-selection-gain}}",
        "\\small",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Granularity & Best single model & $\\Delta F_1$ & $\\Delta$MCC & $\\Delta$AUC & $\\Delta$PR-AUC " + LATEX_LINEBREAK,
        "\\midrule",
    ]
    if not selection_gain_rows:
        lines.append("-- & -- & -- & -- & -- & -- " + LATEX_LINEBREAK)
    else:
        for row in selection_gain_rows:
            lines.append(
                " & ".join(
                    [
                        GRANULARITY_LABELS[row["granularity"]],
                        MODEL_LABELS.get(row["best_single_model_name"], latex_escape(row["best_single_view"])),
                        format_float(row["delta_f1_1"]),
                        format_float(row["delta_mcc"]),
                        format_float(row["delta_auc"]),
                        format_float(row["delta_pr_auc"]),
                    ]
                )
                + " "
                + LATEX_LINEBREAK
            )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    write_text(output_dir / f"{output_prefix}_selection_gain_table.tex", "\n".join(lines))


def write_prevalence_diagnostic_table(
    output_dir: Path,
    output_prefix: str,
    prevalence_diagnostic_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Prevalence-aware diagnostic summary for the fold-local selected models. The majority baseline uses the always-positive classifier when the held-out bug ratio exceeds 0.5 and the always-negative classifier otherwise. Negative $\\Delta F_1$ values, therefore, indicate that the selected model trails that prevalence-induced majority baseline. At file level, the majority baseline predicts the always-negative class, which yields $F_1 = 0$ for the positive class; consequently all file-level targets exceed this baseline on $F_1$ even at modest absolute values. At commit level, the majority baseline is often a stronger $F_1$ competitor because many held-out targets are bug-dense, which is why MCC remains essential for interpretation.}",
        f"\\label{{tab:{output_prefix.replace('_', '-')}-prevalence-diagnostics}}",
        "\\small",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Granularity & $\\Delta F_1 > 0$ & $\\Delta F_1 = 0$ & $\\Delta F_1 < 0$ & Mean $\\Delta F_1$ & Median $\\Delta F_1$ & $\\Delta F_1$ range & MCC $< 0$ " + LATEX_LINEBREAK,
        "\\midrule",
    ]
    for row in prevalence_diagnostic_rows:
        total = int(row["project_count"])
        lines.append(
            " & ".join(
                [
                    GRANULARITY_LABELS[row["granularity"]],
                    format_count_share(int(row["beats_majority_count"]), total),
                    format_count_share(int(row["ties_majority_count"]), total),
                    format_count_share(int(row["below_majority_count"]), total),
                    format_float(row["mean_gap_f1_1"]),
                    format_float(row["median_gap_f1_1"]),
                    format_interval(row["min_gap_f1_1"], row["max_gap_f1_1"]),
                    format_count_share(int(row["negative_mcc_count"]), total),
                ]
            )
            + " "
            + LATEX_LINEBREAK
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table*}"])
    write_text(output_dir / f"{output_prefix}_prevalence_diagnostic_table.tex", "\n".join(lines))


def write_selection_table(
    output_dir: Path,
    output_prefix: str,
    bundles: dict[str, dict[str, Any]],
    granularities: list[str],
) -> None:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Model-family selection frequencies under fold-local inner cross-validation. Counts are followed by percentages of held-out target projects in parentheses.}",
        f"\\label{{tab:{output_prefix.replace('_', '-')}-selection}}",
        "\\small",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Granularity & Naive Bayes & Logistic Regression & Random Forest & XGBoost " + LATEX_LINEBREAK,
        "\\midrule",
    ]
    for granularity in granularities:
        selected = bundles[granularity]["selected"]
        total = len(selected)
        counts = selected["model_name"].value_counts().reindex(MODEL_ORDER, fill_value=0)
        lines.append(
            " & ".join(
                [
                    GRANULARITY_LABELS[granularity],
                    format_count_share(int(counts["naive_bayes"]), total),
                    format_count_share(int(counts["logistic_regression"]), total),
                    format_count_share(int(counts["random_forest"]), total),
                    format_count_share(int(counts["xgboost"]), total),
                ]
            )
            + " "
            + LATEX_LINEBREAK
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    write_text(output_dir / f"{output_prefix}_selection_table.tex", "\n".join(lines))


def write_model_table(output_dir: Path, output_prefix: str, model_summary: pd.DataFrame) -> None:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Per-model mean and sample standard deviation across held-out target projects.}",
        f"\\label{{tab:{output_prefix.replace('_', '-')}-models}}",
        "\\small",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Granularity & Model & $F_1$ & Bal. Acc. & MCC & AUC & PR-AUC " + LATEX_LINEBREAK,
        "\\midrule",
    ]
    granularity_order = {granularity: index for index, granularity in enumerate(ALL_GRANULARITIES)}
    model_order = {model_name: index for index, model_name in enumerate(MODEL_ORDER)}
    sorted_summary = model_summary.sort_values(
        ["granularity", "model_name"],
        key=lambda series: series.map(granularity_order).fillna(-1) if series.name == "granularity" else series.map(model_order).fillna(-1),
    )
    for _, row in sorted_summary.iterrows():
        lines.append(
            " & ".join(
                [
                    GRANULARITY_LABELS[row["granularity"]],
                    MODEL_LABELS[row["model_name"]],
                    format_mean_std(row["mean_f1_1"], row["std_f1_1"]),
                    format_mean_std(row["mean_balanced_accuracy"], row["std_balanced_accuracy"]),
                    format_mean_std(row["mean_mcc"], row["std_mcc"]),
                    format_mean_std(row["mean_auc"], row["std_auc"]),
                    format_mean_std(row["mean_pr_auc"], row["std_pr_auc"]),
                ]
            )
            + " "
            + LATEX_LINEBREAK
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table*}"])
    write_text(output_dir / f"{output_prefix}_model_table.tex", "\n".join(lines))


def write_pairwise_table(output_dir: Path, output_prefix: str, pairwise_rows: list[dict[str, Any]]) -> None:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Pairwise comparisons of the held-out performance of the fold-local selected models across granularities. Positive mean differences favor the left-hand granularity. Mean differences are reported with sample standard deviations and bootstrap confidence intervals over aligned held-out projects, together with the Wilcoxon-aligned rank-biserial correlation and Cliff's $\\delta$.}",
        f"\\label{{tab:{output_prefix.replace('_', '-')}-pairwise}}",
        "\\small",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llrrrrrrr}",
        "\\toprule",
        "Comparison & Metric & Mean diff. & 95\\% CI & Win/Tie/Loss & $n$ & Wilcoxon $p$ & Rank-biserial $r_{rb}$ & Cliff's $\\delta$ " + LATEX_LINEBREAK,
        "\\midrule",
    ]
    if not pairwise_rows:
        lines.append("Pairwise comparisons & -- & -- & -- & -- & -- & -- & -- & -- " + LATEX_LINEBREAK)
    else:
        for row in pairwise_rows:
            metric_label = FOCUS_METRIC_LABELS.get(row["metric"], latex_escape(row["metric"]))
            delta_text = "--"
            if row["cliffs_delta"] is not None:
                delta_text = f"{format_float(row['cliffs_delta'])} ({latex_escape(row['cliffs_delta_magnitude'])})"
            lines.append(
                " & ".join(
                    [
                        latex_escape(row["comparison"]),
                        metric_label,
                        format_mean_std(row["mean_diff"], row["std_diff"]),
                        format_interval(row["ci_low_diff"], row["ci_high_diff"]),
                        f"{row['win_count']}/{row['tie_count']}/{row['loss_count']}",
                        str(int(row["paired_sample_count"])),
                        format_pvalue_latex(row["wilcoxon_pvalue"]),
                        format_float(row["rank_biserial"]),
                        delta_text,
                    ]
                )
                + " "
                + LATEX_LINEBREAK
            )
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table*}"])
    write_text(output_dir / f"{output_prefix}_pairwise_table.tex", "\n".join(lines))


def write_rank_table(output_dir: Path, output_prefix: str, rank_rows: list[dict[str, Any]]) -> None:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Average ranks of the held-out performance of the fold-local selected models across projects. Lower ranks are better. The Friedman $p$-value and Nemenyi critical difference are reported when at least three granularities are present.}",
        f"\\label{{tab:{output_prefix.replace('_', '-')}-ranks}}",
        "\\small",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Metric & Granularity & Avg. rank & SD rank & Best-fold wins & Friedman $p$ & CD@0.05 " + LATEX_LINEBREAK,
        "\\midrule",
    ]
    rank_frame = pd.DataFrame(rank_rows)
    metric_candidates = set(rank_frame["metric"]) if not rank_frame.empty else set()
    metric_order = [metric for metric in FOCUS_METRICS if metric in metric_candidates]
    for metric in metric_order:
        subset = rank_frame[rank_frame["metric"] == metric].copy()
        subset = subset.sort_values("average_rank")
        for _, row in subset.iterrows():
            lines.append(
                " & ".join(
                    [
                        FOCUS_METRIC_LABELS.get(metric, latex_escape(metric)),
                        GRANULARITY_LABELS[row["granularity"]],
                        format_float(row["average_rank"], 4),
                        format_float(row["std_rank"]),
                        str(int(row["best_fold_wins"])),
                        format_pvalue_latex(row["friedman_pvalue"]),
                        format_float(row["critical_difference"]),
                    ]
                )
                + " "
                + LATEX_LINEBREAK
            )
    if not metric_order:
        lines.append("-- & -- & -- & -- & -- & -- & -- " + LATEX_LINEBREAK)
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    write_text(output_dir / f"{output_prefix}_rank_table.tex", "\n".join(lines))


def write_data_profile_table(output_dir: Path, output_prefix: str, data_profile_rows: list[dict[str, Any]]) -> None:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Dataset profile after exact-duplicate removal but before train-only preprocessing. Held-out row and positive-class-proportion columns summarize per-project distributions and report mean $\\pm$ sample standard deviation. The class proportions describe the released retrospective sampling task, not repository-wide defect prevalence.}",
        f"\\label{{tab:{output_prefix.replace('_', '-')}-data-profile}}",
        "\\small",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Granularity & \\shortstack{Total\\\\rows} & \\shortstack{Pos.-class\\\\prop.} & \\shortstack{Duplicate\\\\ratio} & \\shortstack{Held-out\\\\rows} & \\shortstack{Held-out pos.-\\\\class prop.} " + LATEX_LINEBREAK,
        "\\midrule",
    ]
    for row in data_profile_rows:
        lines.append(
            " & ".join(
                [
                    GRANULARITY_LABELS[row["granularity"]],
                    f"{int(row['total_rows'])}",
                    format_float(row["overall_bug_ratio"]),
                    format_float(row["exact_duplicate_ratio"]),
                    format_mean_std(row["mean_target_rows"], row["std_target_rows"]),
                    format_mean_std(row["mean_target_bug_ratio"], row["std_target_bug_ratio"]),
                ]
            )
            + " "
            + LATEX_LINEBREAK
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    write_text(output_dir / f"{output_prefix}_data_profile_table.tex", "\n".join(lines))


def write_adequacy_summary_table(output_dir: Path, output_prefix: str, adequacy_summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        (
            "\\caption{Held-out target-project support categories used for the LOPO sensitivity screen. "
            f"PRIMARY requires at least {ADEQUACY_THRESHOLDS['primary_rows']} held-out rows and at least {ADEQUACY_THRESHOLDS['primary_minority']} minority-class instances; "
            f"EXPLORATORY requires at least {ADEQUACY_THRESHOLDS['exploratory_rows']} held-out rows and at least {ADEQUACY_THRESHOLDS['exploratory_minority']} minority-class instances; "
            "otherwise, the target is marked INSUFFICIENT. This screen is reporting-oriented: it does not alter model fitting, preprocessing, or fold-local model selection, and is used only for PRIMARY-only sensitivity summaries.}"
        ),
        f"\\label{{tab:{output_prefix.replace('_', '-')}-adequacy-summary}}",
        "\\small",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Granularity & PRIMARY & EXPLORATORY & INSUFFICIENT " + LATEX_LINEBREAK,
        "\\midrule",
    ]
    for row in adequacy_summary_rows:
        lines.append(
            " & ".join(
                [
                    GRANULARITY_LABELS[row["granularity"]],
                    str(int(row["primary_count"])),
                    str(int(row["exploratory_count"])),
                    str(int(row["insufficient_count"])),
                ]
            )
            + " "
            + LATEX_LINEBREAK
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    write_text(output_dir / f"{output_prefix}_adequacy_summary_table.tex", "\n".join(lines))


def write_adequacy_matrix_table(
    output_dir: Path,
    output_prefix: str,
    adequacy_matrix_rows: list[dict[str, Any]],
    granularities: list[str],
) -> None:
    column_spec = "l" + "c" * len(granularities)
    header_cells = ["Project"] + [GRANULARITY_LABELS[granularity] for granularity in granularities]
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Project-level support classes for the LOPO sensitivity screen. P=PRIMARY, E=EXPLORATORY, and I=INSUFFICIENT.}",
        f"\\label{{tab:{output_prefix.replace('_', '-')}-adequacy-matrix}}",
        "\\small",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\toprule",
        " & ".join(header_cells) + " " + LATEX_LINEBREAK,
        "\\midrule",
    ]
    for row in adequacy_matrix_rows:
        labels = [latex_escape(row["target_project"])]
        for granularity in granularities:
            adequacy = row.get(f"{granularity}_adequacy")
            labels.append(ADEQUACY_SHORT_LABELS.get(str(adequacy), "--") if adequacy is not None else "--")
        lines.append(" & ".join(labels) + " " + LATEX_LINEBREAK)
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    write_text(output_dir / f"{output_prefix}_adequacy_matrix_table.tex", "\n".join(lines))


def write_support_robustness_table(output_dir: Path, output_prefix: str, support_robustness_rows: list[dict[str, Any]]) -> None:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Sensitivity of the paired granularity comparisons to the target-project adequacy screen. Positive deltas favor the left-hand granularity. PRIMARY-only rows retain only projects that are PRIMARY for both granularities in the pair.}",
        f"\\label{{tab:{output_prefix.replace('_', '-')}-support-robustness}}",
        "\\small",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Comparison & Metric & All $\\Delta$ & All $n$ & All $p$ & PRIMARY $\\Delta$ & PRIMARY $n$ & PRIMARY $p$ " + LATEX_LINEBREAK,
        "\\midrule",
    ]
    if not support_robustness_rows:
        lines.append("-- & -- & -- & -- & -- & -- & -- & -- " + LATEX_LINEBREAK)
    else:
        for row in support_robustness_rows:
            lines.append(
                " & ".join(
                    [
                        latex_escape(row["comparison"]),
                        FOCUS_METRIC_LABELS.get(row["metric"], latex_escape(row["metric"])),
                        format_float(row["all_mean_diff"]),
                        str(int(row["all_sample_count"])),
                        format_pvalue_latex(row["all_wilcoxon_pvalue"]),
                        format_float(row["primary_mean_diff"]),
                        str(int(row["primary_sample_count"])),
                        format_pvalue_latex(row["primary_wilcoxon_pvalue"]),
                    ]
                )
                + " "
                + LATEX_LINEBREAK
            )
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table*}"])
    write_text(output_dir / f"{output_prefix}_support_robustness_table.tex", "\n".join(lines))


def write_top_cases_table(output_dir: Path, output_prefix: str, bundle: dict[str, Any], top_n: int = 8) -> None:
    granularity = bundle["granularity"]
    top_cases = bundle["per_project"].sort_values("f1_1", ascending=False).head(top_n)
    table_name = f"{output_prefix}_{granularity}_top_cases_table.tex"
    label_name = f"tab:{output_prefix.replace('_', '-')}-{granularity}-top-cases"
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{Best {GRANULARITY_LABELS[granularity].lower()}-level held-out cases ranked by positive-class F1.}}",
        f"\\label{{{label_name}}}",
        "\\small",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Project & Model & $F_1$ & Precision$_1$ & Recall$_1$ & MCC " + LATEX_LINEBREAK,
        "\\midrule",
    ]
    for _, row in top_cases.iterrows():
        lines.append(
            " & ".join(
                [
                    latex_escape(row["target_project"]),
                    MODEL_LABELS[row["model_name"]],
                    format_float(row["f1_1"]),
                    format_float(row["precision_1"]),
                    format_float(row["recall_1"]),
                    format_float(row["mcc"]),
                ]
            )
            + " "
            + LATEX_LINEBREAK
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    write_text(output_dir / table_name, "\n".join(lines))


def write_selected_per_project_table(output_dir: Path, output_prefix: str, selected_frame: pd.DataFrame) -> None:
    sorted_frame = selected_frame.sort_values(["granularity", "target_project"])
    lines = [
        "\\small",
        "\\setlength{\\LTleft}{\\fill}",
        "\\setlength{\\LTright}{\\fill}",
        "\\setlength{\\LTcapwidth}{\\textwidth}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{longtable}{llllrrrrr}",
        f"\\caption{{Fold-local selected models and held-out performance for every target project. All held-out targets are listed, including those marked INSUFFICIENT; these targets remain part of the all-target descriptive summaries but are excluded from PRIMARY-only sensitivity summaries.}}\\label{{tab:{output_prefix.replace('_', '-')}-selected-projects}}" + LATEX_LINEBREAK,
        "\\toprule",
        "Granularity & Project & Support & Model & Bug Ratio & $F_1$ & Bal. Acc. & MCC & AUC " + LATEX_LINEBREAK,
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        "Granularity & Project & Support & Model & Bug Ratio & $F_1$ & Bal. Acc. & MCC & AUC " + LATEX_LINEBREAK,
        "\\midrule",
        "\\endhead",
        "\\bottomrule",
        "\\endfoot",
    ]
    for _, row in sorted_frame.iterrows():
        lines.append(
            " & ".join(
                [
                    GRANULARITY_LABELS[row["granularity"]],
                    latex_escape(row["target_project"]),
                    ADEQUACY_SHORT_LABELS.get(str(row["adequacy"]), "--"),
                    MODEL_LABELS[row["model_name"]],
                    format_float(row["target_bug_ratio"]),
                    format_float(row["f1_1"]),
                    format_float(row["balanced_accuracy"]),
                    format_float(row["mcc"]),
                    format_float(row["auc"]),
                ]
            )
            + " "
            + LATEX_LINEBREAK
        )
    lines.extend(["\\end{longtable}", "\\normalsize"])
    write_text(output_dir / f"{output_prefix}_selected_per_project_table.tex", "\n".join(lines))


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.18)
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight", pad_inches=0.18, dpi=300)
    plt.close(fig)


def metric_axis_limits(values: pd.Series, metric: str) -> tuple[float, float]:
    metric_values = numeric_values(values)
    if metric_values.size == 0:
        return (-0.1, 0.1) if metric == "mcc" else (0.0, 1.0)

    min_value = float(metric_values.min())
    max_value = float(metric_values.max())
    if metric == "mcc":
        lower_bound = min(-0.1, min_value - 0.05)
        upper_bound = min(1.0, max_value + 0.08)
    else:
        lower_bound = min(0.0, min_value - 0.05)
        upper_bound = min(1.0, max_value + 0.08)

    if upper_bound <= lower_bound:
        upper_bound = lower_bound + 0.1
    return lower_bound, upper_bound


def generic_axis_limits(values: pd.Series, *, include_zero: bool = False) -> tuple[float, float]:
    numeric = numeric_values(values)
    if numeric.size == 0:
        return (0.0, 1.0)

    min_value = float(numeric.min())
    max_value = float(numeric.max())
    lower_bound = min_value - 0.05
    upper_bound = max_value + 0.08
    if include_zero:
        lower_bound = min(lower_bound, 0.0)
        upper_bound = max(upper_bound, 0.0)
    if upper_bound <= lower_bound:
        upper_bound = lower_bound + 0.1
    return lower_bound, upper_bound


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
        }
    )


def plot_metric_by_model(
    full_frame: pd.DataFrame,
    metric: str,
    y_label: str,
    stem: str,
    figures_dir: Path,
    granularities: list[str],
) -> None:
    fig, axes = plt.subplots(1, len(granularities), figsize=(5.2 * len(granularities), 4.6), sharey=True)
    if len(granularities) == 1:
        axes = [axes]
    global_values = full_frame.loc[full_frame["granularity"].isin(granularities), metric]
    lower_bound, upper_bound = metric_axis_limits(global_values, metric)
    for axis, granularity in zip(axes, granularities):
        subset = full_frame[full_frame["granularity"] == granularity]
        data = [subset.loc[subset["model_name"] == model_name, metric].dropna().to_numpy() for model_name in MODEL_ORDER]
        boxplot = axis.boxplot(
            data,
            tick_labels=[MODEL_LABELS[model_name] for model_name in MODEL_ORDER],
            patch_artist=True,
            medianprops={"color": "#111111", "linewidth": 1.2},
        )
        for patch, model_name in zip(boxplot["boxes"], MODEL_ORDER):
            patch.set_facecolor(MODEL_COLORS[model_name])
            patch.set_alpha(0.78)
        axis.set_title(f"{GRANULARITY_LABELS[granularity]} level")
        axis.set_ylabel(y_label)
        axis.set_xticklabels([MODEL_LABELS[model_name] for model_name in MODEL_ORDER], rotation=20, ha="right")
        axis.set_ylim(bottom=lower_bound, top=upper_bound)
        axis.tick_params(axis="y", labelleft=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95), pad=1.05)
    save_figure(fig, figures_dir, stem)


def plot_selected_precision_recall(
    selected_frame: pd.DataFrame,
    figures_dir: Path,
    stem: str,
    granularities: list[str],
) -> None:
    fig, axes = plt.subplots(1, len(granularities), figsize=(5.2 * len(granularities), 4.6), sharex=True, sharey=True)
    if len(granularities) == 1:
        axes = [axes]
    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=MODEL_COLORS[model_name], label=MODEL_LABELS[model_name])
        for model_name in MODEL_ORDER
    ]
    for axis, granularity in zip(axes, granularities):
        subset = selected_frame[selected_frame["granularity"] == granularity]
        colors = [MODEL_COLORS[model_name] for model_name in subset["model_name"]]
        axis.scatter(subset["recall_1"], subset["precision_1"], s=70, c=colors, edgecolors="#111111", linewidths=0.4)
        axis.set_title(f"{GRANULARITY_LABELS[granularity]} level")
        axis.set_xlabel("Recall of buggy class")
        axis.set_ylabel("Precision of buggy class")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
    axes[-1].legend(handles=legend_handles, loc="lower left", frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95), pad=1.05)
    save_figure(fig, figures_dir, stem)


def plot_f1_vs_prevalence(
    selected_frame: pd.DataFrame,
    figures_dir: Path,
    stem: str,
    granularities: list[str],
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    prevalence_grid = np.linspace(0.01, 0.99, 200)
    always_positive_curve = 2 * prevalence_grid / (1 + prevalence_grid)
    ax.plot(
        prevalence_grid,
        always_positive_curve,
        color="#444444",
        linewidth=1.2,
        linestyle="--",
        label="Always-positive baseline",
    )
    ax.axhline(0.0, color="#999999", linewidth=1.0, linestyle=":", label="Always-negative baseline")
    for granularity in granularities:
        subset = selected_frame[selected_frame["granularity"] == granularity]
        ax.scatter(
            subset["target_bug_ratio"],
            subset["f1_1"],
            s=72,
            alpha=0.88,
            label=f"{GRANULARITY_LABELS[granularity]} held-out projects",
            color=GRANULARITY_COLORS[granularity],
            edgecolors="#111111",
            linewidths=0.4,
        )
    ax.set_xlabel("Held-out project bug ratio")
    ax.set_ylabel("Held-out positive-class $F_1$ after fold-local selection")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save_figure(fig, figures_dir, stem)


def plot_selected_metric_by_granularity(
    selected_frame: pd.DataFrame,
    metric: str,
    y_label: str,
    stem: str,
    figures_dir: Path,
    granularities: list[str],
) -> None:
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    data = [selected_frame.loc[selected_frame["granularity"] == granularity, metric].dropna().to_numpy() for granularity in granularities]
    boxplot = ax.boxplot(
        data,
        tick_labels=[GRANULARITY_LABELS[granularity] for granularity in granularities],
        patch_artist=True,
        medianprops={"color": "#111111", "linewidth": 1.2},
    )
    for patch, granularity in zip(boxplot["boxes"], granularities):
        patch.set_facecolor(GRANULARITY_COLORS[granularity])
        patch.set_alpha(0.75)
    for index, granularity in enumerate(granularities, start=1):
        values = selected_frame.loc[selected_frame["granularity"] == granularity, metric].dropna().to_numpy()
        jitter = rng.normal(0.0, 0.04, size=len(values))
        ax.scatter(np.full(len(values), index) + jitter, values, s=26, color="#111111", alpha=0.55)
    ax.set_ylabel(y_label)
    lower_bound, upper_bound = metric_axis_limits(selected_frame[metric], metric)
    ax.set_ylim(lower_bound, upper_bound)
    fig.tight_layout()
    save_figure(fig, figures_dir, stem)


def plot_metric_heatmap(
    selected_frame: pd.DataFrame,
    metric: str,
    title: str,
    stem: str,
    figures_dir: Path,
    granularities: list[str],
    cmap: str,
) -> None:
    pivot = selected_frame.pivot(index="target_project", columns="granularity", values=metric).reindex(columns=granularities)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    matrix = pivot.to_numpy()

    fig_height = max(4.2, 0.34 * len(pivot) + 1.8)
    fig, ax = plt.subplots(figsize=(6.8, fig_height))
    if metric == "mcc":
        limit = max(abs(float(np.nanmin(matrix))), abs(float(np.nanmax(matrix))), 0.1)
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-limit, vmax=limit)
    else:
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(granularities)))
    ax.set_xticklabels([GRANULARITY_LABELS[granularity] for granularity in granularities])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([project for project in pivot.index])
    ax.set_title(title)

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            text_color = "white" if metric == "mcc" and abs(value) > 0.18 else ("white" if value > 0.55 else "#111111")
            ax.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=8, color=text_color)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(title)
    fig.tight_layout()
    save_figure(fig, figures_dir, stem)


def plot_f1_gap_to_baseline(
    selected_frame: pd.DataFrame,
    figures_dir: Path,
    stem: str,
    granularities: list[str],
) -> None:
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    data = [
        selected_frame.loc[selected_frame["granularity"] == granularity, "f1_gap_to_majority"].dropna().to_numpy()
        for granularity in granularities
    ]
    boxplot = ax.boxplot(
        data,
        tick_labels=[GRANULARITY_LABELS[granularity] for granularity in granularities],
        patch_artist=True,
        medianprops={"color": "#111111", "linewidth": 1.2},
    )
    for patch, granularity in zip(boxplot["boxes"], granularities):
        patch.set_facecolor(GRANULARITY_COLORS[granularity])
        patch.set_alpha(0.75)
    for index, granularity in enumerate(granularities, start=1):
        values = selected_frame.loc[selected_frame["granularity"] == granularity, "f1_gap_to_majority"].dropna().to_numpy()
        jitter = rng.normal(0.0, 0.04, size=len(values))
        ax.scatter(np.full(len(values), index) + jitter, values, s=26, color="#111111", alpha=0.55)
    ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1.1)
    ax.set_ylabel("Held-out $\\Delta F_1$ vs majority baseline")
    lower_bound, upper_bound = generic_axis_limits(selected_frame["f1_gap_to_majority"], include_zero=True)
    ax.set_ylim(lower_bound, upper_bound)
    fig.tight_layout()
    save_figure(fig, figures_dir, stem)


def plot_mcc_vs_prevalence(
    selected_frame: pd.DataFrame,
    figures_dir: Path,
    stem: str,
    granularities: list[str],
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.axhline(0.0, color="#444444", linewidth=1.1, linestyle="--", label="MCC = 0")
    ax.axvline(0.5, color="#bbbbbb", linewidth=1.0, linestyle=":")
    for granularity in granularities:
        subset = selected_frame[selected_frame["granularity"] == granularity]
        ax.scatter(
            subset["target_bug_ratio"],
            subset["mcc"],
            s=72,
            alpha=0.88,
            label=f"{GRANULARITY_LABELS[granularity]} held-out projects",
            color=GRANULARITY_COLORS[granularity],
            edgecolors="#111111",
            linewidths=0.4,
        )
    lower_bound, upper_bound = metric_axis_limits(selected_frame["mcc"], "mcc")
    ax.set_xlabel("Held-out project bug ratio")
    ax.set_ylabel("Held-out MCC after fold-local selection")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(lower_bound, upper_bound)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save_figure(fig, figures_dir, stem)


def plot_critical_difference_style(
    rank_rows: list[dict[str, Any]],
    figures_dir: Path,
    stem: str,
    granularities: list[str],
) -> None:
    rank_frame = pd.DataFrame(rank_rows)
    if rank_frame.empty:
        return

    metrics = [metric for metric in FOCUS_METRICS if metric in set(rank_frame["metric"])]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(7.2, 2.8 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for axis, metric in zip(axes, metrics):
        subset = rank_frame[rank_frame["metric"] == metric].sort_values("average_rank").reset_index(drop=True)
        y_positions = np.arange(len(subset), 0, -1, dtype=float)
        colors = [GRANULARITY_COLORS[row["granularity"]] for _, row in subset.iterrows()]

        axis.scatter(subset["average_rank"], y_positions, s=90, c=colors, edgecolors="#111111", linewidths=0.5, zorder=3)
        for y_position, (_, row) in zip(y_positions, subset.iterrows()):
            axis.hlines(y_position, 1.0, row["average_rank"], color=GRANULARITY_COLORS[row["granularity"]], alpha=0.25)
            axis.text(row["average_rank"] + 0.05, y_position, GRANULARITY_LABELS[row["granularity"]], va="center", fontsize=9)

        critical_difference = subset["critical_difference"].dropna()
        friedman_pvalue = subset["friedman_pvalue"].dropna()
        title = f"Average ranks for {FOCUS_METRIC_LABELS[metric]}"
        if not friedman_pvalue.empty:
            friedman_text = format_pvalue(friedman_pvalue.iloc[0])
            if friedman_text.startswith("<"):
                title += f" (Friedman p{friedman_text})"
            else:
                title += f" (Friedman p={friedman_text})"
        axis.set_title(title)
        axis.set_yticks([])
        axis.set_xlabel("Average rank (lower is better)")
        axis.set_xlim(0.9, max(1.0 + len(granularities), 3.4))
        axis.set_xticks(range(1, len(granularities) + 1))

        ymax = float(y_positions.max())
        if not critical_difference.empty:
            cd_value = float(critical_difference.iloc[0])
            cd_y = ymax + 0.55
            axis.plot([1.0, 1.0 + cd_value], [cd_y, cd_y], color="#111111", linewidth=2.0)
            axis.vlines([1.0, 1.0 + cd_value], cd_y - 0.08, cd_y + 0.08, color="#111111", linewidth=2.0)
            axis.text(1.0 + cd_value / 2.0, cd_y + 0.12, f"CD={cd_value:.2f}", ha="center", va="bottom")

            line_y = cd_y - 0.22
            rank_points = list(subset.itertuples(index=False))
            for left_index in range(len(rank_points)):
                for right_index in range(left_index + 1, len(rank_points)):
                    if rank_points[right_index].average_rank - rank_points[left_index].average_rank <= cd_value + 1e-12:
                        axis.plot(
                            [rank_points[left_index].average_rank, rank_points[right_index].average_rank],
                            [line_y, line_y],
                            color="#444444",
                            linewidth=3.0,
                            solid_capstyle="round",
                        )
                        line_y -= 0.12

        axis.set_ylim(0.4, ymax + 0.9)

    fig.tight_layout()
    save_figure(fig, figures_dir, stem)


def plot_pairwise_metric_scatter(
    selected_frame: pd.DataFrame,
    figures_dir: Path,
    stem: str,
    granularities: list[str],
) -> None:
    comparisons = [
        pair for pair in PAIRWISE_COMPARISON_ORDER if pair[0] in set(granularities) and pair[1] in set(granularities)
    ]
    if not comparisons:
        return

    metric_limits = {metric: metric_axis_limits(selected_frame[metric], metric) for metric in FOCUS_METRICS}
    fig, axes = plt.subplots(len(FOCUS_METRICS), len(comparisons), figsize=(4.7 * len(comparisons), 4.2 * len(FOCUS_METRICS)))
    if len(FOCUS_METRICS) == 1:
        axes = np.array([axes])
    if len(comparisons) == 1:
        axes = axes.reshape(len(FOCUS_METRICS), 1)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=SUPPORT_COLORS["primary"], label="PRIMARY pair"),
        plt.Line2D(
            [0],
            [0],
            marker="^",
            linestyle="",
            markerfacecolor="none",
            markeredgecolor=SUPPORT_COLORS["low_support"],
            color=SUPPORT_COLORS["low_support"],
            label="Any non-PRIMARY",
        ),
    ]

    for row_index, metric in enumerate(FOCUS_METRICS):
        lower_bound, upper_bound = metric_limits[metric]
        for column_index, (left_granularity, right_granularity) in enumerate(comparisons):
            axis = axes[row_index, column_index]
            merged = (
                selected_frame[selected_frame["granularity"] == left_granularity][["target_project", metric, "adequacy"]]
                .merge(
                    selected_frame[selected_frame["granularity"] == right_granularity][["target_project", metric, "adequacy"]],
                    on="target_project",
                    suffixes=(f"_{left_granularity}", f"_{right_granularity}"),
                )
                .sort_values("target_project")
            )
            x_values = merged[f"{metric}_{right_granularity}"]
            y_values = merged[f"{metric}_{left_granularity}"]
            primary_mask = (merged[f"adequacy_{left_granularity}"] == "PRIMARY") & (
                merged[f"adequacy_{right_granularity}"] == "PRIMARY"
            )

            axis.plot([lower_bound, upper_bound], [lower_bound, upper_bound], color="#666666", linestyle="--", linewidth=1.0)
            axis.scatter(
                x_values[primary_mask],
                y_values[primary_mask],
                s=52,
                color=SUPPORT_COLORS["primary"],
                alpha=0.85,
                edgecolors="#111111",
                linewidths=0.3,
            )
            axis.scatter(
                x_values[~primary_mask],
                y_values[~primary_mask],
                s=68,
                marker="^",
                facecolors="none",
                edgecolors=SUPPORT_COLORS["low_support"],
                linewidths=1.0,
            )
            axis.set_title(f"{GRANULARITY_LABELS[left_granularity]} vs {GRANULARITY_LABELS[right_granularity]}", pad=8)
            axis.set_xlabel(f"{GRANULARITY_LABELS[right_granularity]} {FOCUS_METRIC_LABELS[metric]}")
            if column_index == 0:
                axis.set_ylabel(f"{GRANULARITY_LABELS[left_granularity]} {FOCUS_METRIC_LABELS[metric]}")
            axis.set_xlim(lower_bound, upper_bound)
            axis.set_ylim(lower_bound, upper_bound)
            axis.set_aspect("equal", adjustable="box")
            axis.tick_params(labelsize=10)

    fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False, fontsize=11)
    fig.tight_layout(rect=(0.0, 0.065, 1.0, 1.0), pad=1.15)
    save_figure(fig, figures_dir, stem)


def write_machine_readable_outputs(
    output_dir: Path,
    output_prefix: str,
    summary_rows: list[dict[str, Any]],
    model_summary: pd.DataFrame,
    selected_frame: pd.DataFrame,
    pairwise_rows: list[dict[str, Any]],
    data_profile_rows: list[dict[str, Any]],
    focus_ci_rows: list[dict[str, Any]],
    rank_rows: list[dict[str, Any]],
    variability_rows: list[dict[str, Any]],
    selection_gain_rows: list[dict[str, Any]],
    prevalence_diagnostic_rows: list[dict[str, Any]],
    adequacy_summary_rows: list[dict[str, Any]],
    adequacy_matrix_rows: list[dict[str, Any]],
    support_robustness_rows: list[dict[str, Any]],
) -> None:
    pd.DataFrame(summary_rows).to_csv(output_dir / f"{output_prefix}_summary.csv", index=False)
    model_summary.to_csv(output_dir / f"{output_prefix}_model_summary.csv", index=False)
    selected_frame.to_csv(output_dir / f"{output_prefix}_nested_selected.csv", index=False)
    pd.DataFrame(pairwise_rows).to_csv(output_dir / f"{output_prefix}_pairwise_stats.csv", index=False)
    pd.DataFrame(data_profile_rows).to_csv(output_dir / f"{output_prefix}_data_profile.csv", index=False)
    pd.DataFrame(focus_ci_rows).to_csv(output_dir / f"{output_prefix}_confidence_intervals.csv", index=False)
    pd.DataFrame(rank_rows).to_csv(output_dir / f"{output_prefix}_rank_stats.csv", index=False)
    pd.DataFrame(variability_rows).to_csv(output_dir / f"{output_prefix}_variability.csv", index=False)
    pd.DataFrame(selection_gain_rows).to_csv(output_dir / f"{output_prefix}_selection_gain.csv", index=False)
    pd.DataFrame(prevalence_diagnostic_rows).to_csv(output_dir / f"{output_prefix}_prevalence_diagnostics.csv", index=False)
    pd.DataFrame(adequacy_summary_rows).to_csv(output_dir / f"{output_prefix}_adequacy_summary.csv", index=False)
    pd.DataFrame(adequacy_matrix_rows).to_csv(output_dir / f"{output_prefix}_adequacy_matrix.csv", index=False)
    pd.DataFrame(support_robustness_rows).to_csv(output_dir / f"{output_prefix}_support_robustness.csv", index=False)
    payload = {
        "summary_rows": summary_rows,
        "model_summary": model_summary.to_dict(orient="records"),
        "selected_rows": selected_frame.to_dict(orient="records"),
        "pairwise_rows": pairwise_rows,
        "data_profile_rows": data_profile_rows,
        "focus_ci_rows": focus_ci_rows,
        "rank_rows": rank_rows,
        "variability_rows": variability_rows,
        "selection_gain_rows": selection_gain_rows,
        "prevalence_diagnostic_rows": prevalence_diagnostic_rows,
        "adequacy_summary_rows": adequacy_summary_rows,
        "adequacy_matrix_rows": adequacy_matrix_rows,
        "support_robustness_rows": support_robustness_rows,
    }
    write_text(output_dir / f"{output_prefix}_summary.json", json.dumps(json_compatible(payload), indent=2))


def generate_assets(
    results_root: Path,
    output_root: Path,
    granularities: list[str] | None,
    output_prefix: str,
) -> list[str]:
    generated_dir = output_root / "generated"
    figures_dir = output_root / "figures"
    generated_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    detected_granularities = available_granularities(results_root)
    selected_granularities = canonical_granularities(granularities, detected_granularities)
    if not selected_granularities:
        raise FileNotFoundError("No compatible LOPO result folders were found for analysis generation.")

    configure_plot_style()
    bundles = {granularity: load_granularity(results_root, granularity) for granularity in selected_granularities}
    full_frame = pd.concat([bundles[granularity]["per_project"] for granularity in selected_granularities], ignore_index=True)
    model_summary = pd.concat([build_model_summary(bundles[granularity]) for granularity in selected_granularities], ignore_index=True)
    selected_frame = pd.concat([bundles[granularity]["selected"] for granularity in selected_granularities], ignore_index=True)
    summary_rows: list[dict[str, Any]] = []
    for granularity in selected_granularities:
        summary_rows.extend(build_summary_rows(bundles[granularity]))

    statistical_tests = load_statistical_tests(results_root)
    pairwise_rows = build_pairwise_rows(selected_frame, statistical_tests, selected_granularities)
    data_profile_rows = build_data_profile_rows(bundles, selected_granularities)
    focus_ci_rows = build_focus_ci_rows(bundles, selected_granularities)
    rank_rows = build_rank_rows(selected_frame, selected_granularities)
    variability_rows = build_variability_rows(bundles, selected_granularities)
    selection_gain_rows = build_selection_gain_rows(summary_rows)
    prevalence_diagnostic_rows = build_prevalence_diagnostic_rows(bundles, selected_granularities)
    adequacy_summary_rows = build_adequacy_summary_rows(bundles, selected_granularities)
    adequacy_matrix_rows = build_adequacy_matrix_rows(bundles, selected_granularities)
    support_robustness_rows = build_support_robustness_rows(selected_frame, selected_granularities)

    write_macros(generated_dir, output_prefix, bundles, selected_granularities)
    write_summary_table(generated_dir, output_prefix, summary_rows, selected_granularities)
    write_focus_ci_table(generated_dir, output_prefix, focus_ci_rows)
    write_variability_table(generated_dir, output_prefix, variability_rows)
    write_selection_gain_table(generated_dir, output_prefix, selection_gain_rows)
    write_prevalence_diagnostic_table(generated_dir, output_prefix, prevalence_diagnostic_rows)
    write_selection_table(generated_dir, output_prefix, bundles, selected_granularities)
    write_model_table(generated_dir, output_prefix, model_summary)
    write_pairwise_table(generated_dir, output_prefix, pairwise_rows)
    write_rank_table(generated_dir, output_prefix, rank_rows)
    write_data_profile_table(generated_dir, output_prefix, data_profile_rows)
    write_adequacy_summary_table(generated_dir, output_prefix, adequacy_summary_rows)
    write_support_robustness_table(generated_dir, output_prefix, support_robustness_rows)
    write_adequacy_matrix_table(generated_dir, output_prefix, adequacy_matrix_rows, selected_granularities)
    for granularity in selected_granularities:
        write_top_cases_table(generated_dir, output_prefix, bundles[granularity])
    write_selected_per_project_table(generated_dir, output_prefix, selected_frame)
    write_machine_readable_outputs(
        generated_dir,
        output_prefix,
        summary_rows,
        model_summary,
        selected_frame,
        pairwise_rows,
        data_profile_rows,
        focus_ci_rows,
        rank_rows,
        variability_rows,
        selection_gain_rows,
        prevalence_diagnostic_rows,
        adequacy_summary_rows,
        adequacy_matrix_rows,
        support_robustness_rows,
    )

    plot_metric_by_model(full_frame, "f1_1", "Positive-class F1", f"{output_prefix}_f1_by_model", figures_dir, selected_granularities)
    plot_metric_by_model(full_frame, "mcc", "Matthews correlation coefficient", f"{output_prefix}_mcc_by_model", figures_dir, selected_granularities)
    plot_selected_precision_recall(selected_frame, figures_dir, f"{output_prefix}_selected_precision_recall", selected_granularities)
    plot_f1_vs_prevalence(selected_frame, figures_dir, f"{output_prefix}_f1_vs_prevalence", selected_granularities)
    plot_selected_metric_by_granularity(
        selected_frame,
        "f1_1",
        "Held-out positive-class $F_1$ after fold-local selection",
        f"{output_prefix}_selected_f1_by_granularity",
        figures_dir,
        selected_granularities,
    )
    plot_selected_metric_by_granularity(
        selected_frame,
        "mcc",
        "Held-out MCC after fold-local selection",
        f"{output_prefix}_selected_mcc_by_granularity",
        figures_dir,
        selected_granularities,
    )
    plot_metric_heatmap(
        selected_frame,
        "f1_1",
        "Held-out positive-class $F_1$ after fold-local selection",
        f"{output_prefix}_selected_f1_heatmap",
        figures_dir,
        selected_granularities,
        cmap="YlGnBu",
    )
    plot_metric_heatmap(
        selected_frame,
        "mcc",
        "Held-out MCC after fold-local selection",
        f"{output_prefix}_selected_mcc_heatmap",
        figures_dir,
        selected_granularities,
        cmap="RdBu_r",
    )
    plot_f1_gap_to_baseline(selected_frame, figures_dir, f"{output_prefix}_f1_gap_to_baseline", selected_granularities)
    plot_mcc_vs_prevalence(selected_frame, figures_dir, f"{output_prefix}_mcc_vs_prevalence", selected_granularities)
    plot_critical_difference_style(rank_rows, figures_dir, f"{output_prefix}_critical_difference", selected_granularities)
    plot_pairwise_metric_scatter(selected_frame, figures_dir, f"{output_prefix}_pairwise_metric_scatter", selected_granularities)

    summary_lines = []
    for granularity in selected_granularities:
        nested = summary_stats(bundles[granularity]["selected"])
        summary_lines.append(
            f"{GRANULARITY_LABELS[granularity]}: held-out positive-class F1 after fold-local selection={nested['f1_1']['mean']:.3f}, MCC={nested['mcc']['mean']:.3f}, "
            f"AUC={nested['auc']['mean']:.3f}, PR-AUC={nested['pr_auc']['mean']:.3f}"
        )
    return summary_lines


def main() -> None:
    args = parse_args()
    summary_lines = generate_assets(
        results_root=args.results_root,
        output_root=args.output_root,
        granularities=args.granularities,
        output_prefix="lopo_granularity",
    )
    for line in summary_lines:
        print(line)


if __name__ == "__main__":
    main()
