from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.impute import SimpleImputer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loading import ProjectDataset, load_project_dataset, prepare_features  # noqa: E402
from evaluation import compute_binary_classification_metrics  # noqa: E402
from feature_schema import (  # noqa: E402
    GO_SPECIFIC_FILE_METRICS,
    GO_SPECIFIC_METHOD_METRICS,
    SCHEMAS,
    get_feature_columns,
    get_schema,
)
from models import get_model_specs  # noqa: E402
from preprocessing import build_modeling_pipeline  # noqa: E402


RANDOM_SEED = 42
GRANULARITIES = ("commit", "file", "method")
GRANULARITY_LABELS = {"commit": "Commit", "file": "File", "method": "Method"}
PAIR_ORDER = (("commit", "file"), ("commit", "method"), ("method", "file"))
MODEL_ORDER = ("naive_bayes", "logistic_regression", "random_forest", "xgboost")
FOCUS_METRICS = ("f1_1", "mcc")
METRIC_LABELS = {
    "f1_1": "$F_1$",
    "mcc": "MCC",
    "auc": "AUC",
    "pr_auc": "PR-AUC",
    "selected_f1": "$F_1$",
    "selected_mcc": "MCC",
}
REPLAY_METRICS = ("f1_1", "mcc", "auc", "pr_auc")
SMOTE_K_VALUES = (1, 3, 5, 7)
ADEQUACY_ROW_THRESHOLDS = (100, 200, 500)
ADEQUACY_MINORITY_THRESHOLDS = (20, 30, 50, 75, 100)
BOOTSTRAP_RESAMPLES = 10_000
VERIFICATION_ATOL = 1e-10
LATEX_LINEBREAK = chr(92) * 2

HARMONIZED_FEATURE_MAP = {
    "commit": {
        "nloc": "total_nloc",
        "token_count": "total_token_count",
        "complexity": "total_complexity",
    },
    "file": {
        "nloc": "nloc",
        "token_count": "token_count",
        "complexity": "complexity",
    },
    "method": {
        "nloc": "nloc",
        "token_count": "token_count",
        "complexity": "cyclomatic_complexity",
    },
}

RAW_FILE_NAMES = {
    "commit": ("bugs.csv", "non_bugs.csv"),
    "file": ("file_bug_metrics.csv", "file_non_bug_metrics.csv"),
    "method": ("method_bug_metrics.csv", "method_non_bug_metrics.csv"),
}

MATMUL_RUNTIME_WARNING_MESSAGES = (
    "divide by zero encountered in matmul",
    "overflow encountered in matmul",
    "invalid value encountered in matmul",
)


@dataclass
class RawLabelInfo:
    positive: pd.DataFrame
    negative: pd.DataFrame
    conflict_keys: set[str]


@dataclass
class GranularityContext:
    granularity: str
    projects: list[str]
    signature: dict[str, Any]
    per_project_results: pd.DataFrame
    selected_rows: pd.DataFrame
    datasets: dict[str, ProjectDataset]
    project_frames: dict[str, pd.DataFrame]
    all_rows: pd.DataFrame
    raw_labels: dict[str, RawLabelInfo]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic post-hoc sensitivity and diagnostic assets from the "
            "recorded strict-LOPO outputs and a matching GoBug data snapshot."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT,
        help="Root containing commit_data, file_data, and method_data.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "results_lopo_baseline",
        help="Root containing recorded per-granularity LOPO results.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "analysis_output",
        help="Output root containing generated/ and figures/.",
    )
    return parser.parse_args()


def configure_data_root(data_root: Path) -> None:
    data_root = data_root.resolve()
    for granularity, schema in list(SCHEMAS.items()):
        data_dir = data_root / f"{granularity}_data"
        if not data_dir.is_dir():
            raise FileNotFoundError(f"Missing {granularity} data directory: {data_dir}")
        SCHEMAS[granularity] = replace(schema, data_dir=data_dir)


def json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is pd.NA:
        return None
    return value


def format_float(value: Any, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(numeric):
        return "--"
    return f"{numeric:.{digits}f}"


def format_pvalue(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(numeric):
        return "--"
    if numeric < 0.001:
        return "$<0.001$"
    return f"{numeric:.3f}"


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame, sort_by: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    if sort_by:
        existing = [column for column in sort_by if column in output.columns]
        if existing:
            output = output.sort_values(existing, kind="mergesort").reset_index(drop=True)
    output.to_csv(path, index=False, float_format="%.10g")


def make_latex_table(
    caption: str,
    label: str,
    headers: list[str],
    rows: Iterable[Iterable[Any]],
    column_spec: str,
) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        " & ".join(headers) + f" {LATEX_LINEBREAK}",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(value) for value in row) + f" {LATEX_LINEBREAK}")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _identity_parts(frame: pd.DataFrame, granularity: str, project: str) -> list[pd.Series]:
    def normalized(column: str, default: str = "<missing>") -> pd.Series:
        if column not in frame.columns:
            return pd.Series(default, index=frame.index, dtype="object")
        values = frame[column].astype("object")
        return values.where(pd.notna(values), default).astype(str)

    if granularity == "commit":
        return [normalized("sha")]

    project_values = normalized("project", project)
    project_values = project_values.where(project_values.ne("<missing>"), project)
    if granularity == "file":
        return [project_values, normalized("file_path"), normalized("sha")]
    return [
        project_values,
        normalized("file_path"),
        normalized("method_name"),
        normalized("sha"),
    ]


def build_identity_keys(frame: pd.DataFrame, granularity: str, project: str) -> pd.Series:
    parts = _identity_parts(frame, granularity, project)
    keys = parts[0]
    for part in parts[1:]:
        keys = keys.str.cat(part, sep="\x1f")
    return keys


def read_raw_labels(data_root: Path, granularity: str, project: str) -> RawLabelInfo:
    positive_name, negative_name = RAW_FILE_NAMES[granularity]
    project_dir = data_root / f"{granularity}_data" / project
    positive = pd.read_csv(project_dir / positive_name)
    negative = pd.read_csv(project_dir / negative_name)
    positive["__identity_key"] = build_identity_keys(positive, granularity, project)
    negative["__identity_key"] = build_identity_keys(negative, granularity, project)
    conflict_keys = set(positive["__identity_key"]).intersection(negative["__identity_key"])
    return RawLabelInfo(positive=positive, negative=negative, conflict_keys=conflict_keys)


def select_fold_local_rows(
    per_project_results: pd.DataFrame,
    signature: dict[str, Any],
) -> pd.DataFrame:
    successful = per_project_results[per_project_results["status"].eq("ok")].copy()
    primary_metric = str(signature.get("primary_metric", "f1"))
    primary_column = f"best_inner_{primary_metric}"
    secondary_column = "best_inner_mcc" if primary_metric != "mcc" else "best_inner_f1"
    model_names = list(signature.get("model_names") or MODEL_ORDER)
    model_order = {name: index for index, name in enumerate(model_names)}
    successful["__model_order"] = successful["model_name"].map(model_order).fillna(len(model_order)).astype(int)
    selected = (
        successful.sort_values(
            ["target_project", primary_column, secondary_column, "__model_order"],
            ascending=[True, False, False, True],
            kind="mergesort",
        )
        .drop_duplicates("target_project", keep="first")
        .drop(columns="__model_order")
        .reset_index(drop=True)
    )
    expected_projects = list(signature.get("available_projects") or [])
    if expected_projects and set(selected["target_project"]) != set(expected_projects):
        raise AssertionError("Fold-local selection did not produce exactly one row for every recorded target project.")
    return selected


def load_context(
    data_root: Path,
    results_root: Path,
    granularity: str,
) -> GranularityContext:
    output_dir = results_root / granularity
    with (output_dir / "run_signature.json").open("r", encoding="utf-8") as handle:
        signature = json.load(handle)
    if int(signature.get("random_seed", RANDOM_SEED)) != RANDOM_SEED:
        raise ValueError(f"Expected random seed {RANDOM_SEED} for {granularity}.")
    if signature.get("resampling") != "smote" or int(signature.get("smote_k_neighbors", -1)) != 1:
        raise ValueError(f"Expected the recorded {granularity} baseline to use SMOTE k=1.")

    per_project_results = pd.read_csv(output_dir / "per_project_results.csv")
    selected_rows = select_fold_local_rows(per_project_results, signature)
    projects = list(signature["available_projects"])
    datasets: dict[str, ProjectDataset] = {}
    project_frames: dict[str, pd.DataFrame] = {}
    raw_labels: dict[str, RawLabelInfo] = {}

    for project in projects:
        raw_info = read_raw_labels(data_root, granularity, project)
        raw_labels[project] = raw_info
        dataset = load_project_dataset(project, granularity, exclude_go_metrics=False)
        datasets[project] = dataset
        frame = dataset.data.copy()
        frame["__identity_key"] = build_identity_keys(frame, granularity, project)
        frame["__cross_label_conflict"] = frame["__identity_key"].isin(raw_info.conflict_keys)
        project_frames[project] = frame

    all_rows = pd.concat([project_frames[project] for project in projects], ignore_index=True)
    return GranularityContext(
        granularity=granularity,
        projects=projects,
        signature=signature,
        per_project_results=per_project_results,
        selected_rows=selected_rows,
        datasets=datasets,
        project_frames=project_frames,
        all_rows=all_rows,
        raw_labels=raw_labels,
    )


def parse_best_params(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    return json.loads(value)


def fit_predict_fixed(
    model_name: str,
    best_params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    smote_k: int = 1,
) -> np.ndarray:
    class_counts = y_train.value_counts()
    if len(class_counts) < 2:
        raise ValueError("Training data contain only one class.")
    if int(class_counts.min()) <= smote_k:
        raise ValueError(
            f"SMOTE k={smote_k} requires at least {smote_k + 1} minority rows; found {int(class_counts.min())}."
        )
    model_spec = get_model_specs([model_name])[0]
    pipeline = build_modeling_pipeline(
        estimator=model_spec.estimator_factory(RANDOM_SEED),
        resampling_strategy="smote",
        random_seed=RANDOM_SEED,
        smote_k_neighbors=smote_k,
    )
    pipeline.set_params(**best_params)
    suppress_matmul = model_name in {"logistic_regression", "voting"}
    with warnings.catch_warnings():
        if suppress_matmul:
            for message in MATMUL_RUNTIME_WARNING_MESSAGES:
                warnings.filterwarnings("ignore", category=RuntimeWarning, message=message)
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
    return np.asarray(probabilities, dtype=float)


def compute_metrics(y_true: pd.Series | np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return compute_binary_classification_metrics(np.asarray(y_true, dtype=int), np.asarray(y_prob, dtype=float))


def safe_wilcoxon(left: Iterable[float], right: Iterable[float]) -> tuple[float, float, int]:
    paired = pd.DataFrame({"left": left, "right": right}).replace([np.inf, -np.inf], np.nan).dropna()
    if paired.empty:
        return float("nan"), float("nan"), 0
    differences = paired["left"].to_numpy(dtype=float) - paired["right"].to_numpy(dtype=float)
    if np.allclose(differences, 0.0):
        return 0.0, 1.0, len(paired)
    result = wilcoxon(differences, zero_method="wilcox", alternative="two-sided", method="auto")
    return float(result.statistic), float(result.pvalue), len(paired)


def harmonized_features(frame: pd.DataFrame, granularity: str) -> pd.DataFrame:
    mapping = HARMONIZED_FEATURE_MAP[granularity]
    output = pd.DataFrame(index=frame.index)
    for common_name, source_name in mapping.items():
        output[common_name] = pd.to_numeric(frame[source_name], errors="coerce")
    return output


def source_target_shift(
    source_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    granularity: str,
) -> tuple[float, int]:
    feature_columns = get_feature_columns(granularity)
    source = source_frame[feature_columns]
    target = target_frame[feature_columns]
    imputer = SimpleImputer(strategy="median")
    source_imputed = imputer.fit_transform(source)
    target_imputed = imputer.transform(target)
    source_mean = source_imputed.mean(axis=0)
    source_std = source_imputed.std(axis=0, ddof=0)
    target_mean = target_imputed.mean(axis=0)
    valid = np.isfinite(source_std) & (source_std > 0) & np.isfinite(source_mean) & np.isfinite(target_mean)
    if not valid.any():
        return float("nan"), 0
    shifts = np.abs((target_mean[valid] - source_mean[valid]) / source_std[valid])
    return float(np.mean(shifts)), int(valid.sum())


def ranking_effort_metrics(
    y_true: pd.Series,
    y_prob: np.ndarray,
    sample_ids: pd.Series,
    effort: pd.Series,
) -> dict[str, Any]:
    ranking = pd.DataFrame(
        {
            "y_true": np.asarray(y_true, dtype=int),
            "y_prob": np.asarray(y_prob, dtype=float),
            "sample_id": sample_ids.astype(str).to_numpy(),
            "effort": pd.to_numeric(effort, errors="coerce").to_numpy(dtype=float),
        }
    )
    effort_missing = int((~np.isfinite(ranking["effort"])).sum())
    ranking["effort"] = ranking["effort"].where(np.isfinite(ranking["effort"]), 0.0).clip(lower=0.0)
    ranking = ranking.sort_values(["y_prob", "sample_id"], ascending=[False, True], kind="mergesort")
    positive_total = int(ranking["y_true"].sum())
    entity_prefix_rows = max(1, int(np.ceil(0.20 * len(ranking)))) if len(ranking) else 0
    entity_recall = (
        float(ranking.iloc[:entity_prefix_rows]["y_true"].sum() / positive_total)
        if positive_total and entity_prefix_rows
        else float("nan")
    )

    total_effort = float(ranking["effort"].sum())
    effort_fallback = total_effort <= 0.0
    if effort_fallback:
        ranking["effort"] = 1.0
        total_effort = float(len(ranking))
    cumulative_effort = ranking["effort"].cumsum().to_numpy(dtype=float)
    cumulative_positive = ranking["y_true"].cumsum().to_numpy(dtype=float)
    effort_threshold = 0.20 * total_effort
    effort_prefix_rows = int(np.searchsorted(cumulative_effort, effort_threshold, side="left") + 1) if len(ranking) else 0
    effort_prefix_rows = min(effort_prefix_rows, len(ranking))
    effort_recall = (
        float(cumulative_positive[effort_prefix_rows - 1] / positive_total)
        if positive_total and effort_prefix_rows
        else float("nan")
    )
    achieved_effort_fraction = (
        float(cumulative_effort[effort_prefix_rows - 1] / total_effort) if total_effort and effort_prefix_rows else float("nan")
    )
    x = np.concatenate(([0.0], cumulative_effort / total_effort)) if total_effort else np.array([0.0])
    y = np.concatenate(([0.0], cumulative_positive / positive_total)) if positive_total else np.zeros_like(x)
    aucec = float(np.trapezoid(y, x)) if len(x) > 1 else float("nan")
    return {
        "target_rows": int(len(ranking)),
        "target_positives": positive_total,
        "entity_budget_fraction": 0.20,
        "entity_budget_rows": entity_prefix_rows,
        "recall_at_20pct_entities": entity_recall,
        "effort_budget_fraction": 0.20,
        "effort_prefix_rows": effort_prefix_rows,
        "achieved_effort_fraction": achieved_effort_fraction,
        "loc_aware_recall_at_20pct_effort": effort_recall,
        "aucec": aucec,
        "effort_total": total_effort,
        "effort_missing_rows": effort_missing,
        "effort_nonpositive_rows": int((ranking["effort"] <= 0).sum()),
        "effort_unit_fallback": effort_fallback,
    }


def label_audit_rows(context: GranularityContext) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for project in context.projects:
        info = context.raw_labels[project]
        positive = info.positive
        negative = info.negative
        positive_sha = positive.get("sha", pd.Series(dtype=object)).dropna().astype(str)
        negative_sha = negative.get("sha", pd.Series(dtype=object)).dropna().astype(str)
        positive_conflict_mask = positive["__identity_key"].isin(info.conflict_keys)
        negative_conflict_mask = negative["__identity_key"].isin(info.conflict_keys)
        modeled_frame = context.project_frames[project]
        rows.append(
            {
                "granularity": context.granularity,
                "project": project,
                "positive_rows": int(len(positive)),
                "negative_rows": int(len(negative)),
                "total_rows": int(len(positive) + len(negative)),
                "positive_unique_shas": int(positive_sha.nunique()),
                "negative_unique_shas": int(negative_sha.nunique()),
                "all_unique_shas": int(pd.concat([positive_sha, negative_sha], ignore_index=True).nunique()),
                "positive_rows_per_positive_sha": float(len(positive) / positive_sha.nunique()) if positive_sha.nunique() else np.nan,
                "cross_label_sha_conflicts": int(len(set(positive_sha).intersection(negative_sha))),
                "cross_label_identity_conflicts": int(len(info.conflict_keys)),
                "positive_conflict_rows": int(positive_conflict_mask.sum()),
                "negative_conflict_rows": int(negative_conflict_mask.sum()),
                "raw_conflict_rows": int(positive_conflict_mask.sum() + negative_conflict_mask.sum()),
                "modeled_conflict_rows": int(modeled_frame["__cross_label_conflict"].sum()),
                "missing_method_names_positive": int(positive["method_name"].isna().sum()) if "method_name" in positive else 0,
                "missing_method_names_negative": int(negative["method_name"].isna().sum()) if "method_name" in negative else 0,
                "exact_exported_row_duplicates_removed": int(context.datasets[project].report.exact_duplicate_rows),
                "modeled_rows_after_exact_duplicate_removal": int(len(modeled_frame)),
                "identity_key_definition": {
                    "commit": "sha",
                    "file": "project/file_path/sha",
                    "method": "project/file_path/method_name/sha",
                }[context.granularity],
                "duplicate_definition": "exact exported-row duplicates only; no metric-tuple deduplication",
            }
        )

    project_frame = pd.DataFrame(rows)
    aggregate: dict[str, Any] = {
        "granularity": context.granularity,
        "project": "ALL",
        "identity_key_definition": rows[0]["identity_key_definition"],
        "duplicate_definition": rows[0]["duplicate_definition"],
    }
    additive_columns = [
        "positive_rows",
        "negative_rows",
        "total_rows",
        "positive_unique_shas",
        "negative_unique_shas",
        "all_unique_shas",
        "cross_label_sha_conflicts",
        "cross_label_identity_conflicts",
        "positive_conflict_rows",
        "negative_conflict_rows",
        "raw_conflict_rows",
        "modeled_conflict_rows",
        "missing_method_names_positive",
        "missing_method_names_negative",
        "exact_exported_row_duplicates_removed",
        "modeled_rows_after_exact_duplicate_removal",
    ]
    for column in additive_columns:
        aggregate[column] = int(project_frame[column].sum())
    aggregate["positive_rows_per_positive_sha"] = (
        aggregate["positive_rows"] / aggregate["positive_unique_shas"] if aggregate["positive_unique_shas"] else np.nan
    )
    rows.append(aggregate)
    return rows


def build_commit_timestamp_map(data_root: Path, project: str) -> tuple[dict[str, float], int]:
    positive_name, negative_name = RAW_FILE_NAMES["commit"]
    project_dir = data_root / "commit_data" / project
    frames = []
    for file_name in (positive_name, negative_name):
        frame = pd.read_csv(project_dir / file_name, usecols=lambda column: column in {"sha", "commit_timestamp"})
        if "sha" in frame and "commit_timestamp" in frame:
            frames.append(frame[["sha", "commit_timestamp"]])
    if not frames:
        return {}, 0
    combined = pd.concat(frames, ignore_index=True)
    combined["sha"] = combined["sha"].astype(str)
    combined["commit_timestamp"] = pd.to_numeric(combined["commit_timestamp"], errors="coerce")
    timestamp_map: dict[str, float] = {}
    ambiguous = 0
    for sha, group in combined.dropna(subset=["commit_timestamp"]).groupby("sha", sort=False):
        timestamps = np.unique(group["commit_timestamp"].to_numpy(dtype=float))
        if len(timestamps) == 1:
            timestamp_map[str(sha)] = float(timestamps[0])
        elif len(timestamps) > 1:
            ambiguous += 1
    return timestamp_map, ambiguous


def temporal_reference_row(
    data_root: Path,
    context: GranularityContext,
    project: str,
    selected_row: pd.Series,
    target_frame: pd.DataFrame,
    full_target_probabilities: np.ndarray,
) -> dict[str, Any]:
    timestamp_map, ambiguous_timestamp_shas = build_commit_timestamp_map(data_root, project)
    sha_values = target_frame.get("sha", pd.Series(index=target_frame.index, dtype=object)).astype(str)
    timestamps = sha_values.map(timestamp_map)
    covered_mask = timestamps.notna()
    covered_rows = int(covered_mask.sum())
    unique_target_shas = int(sha_values.nunique())
    covered_unique_shas = int(sha_values[covered_mask].nunique())
    base = {
        "granularity": context.granularity,
        "target_project": project,
        "model_name": selected_row["model_name"],
        "timestamp_covered_rows": covered_rows,
        "timestamp_total_rows": int(len(target_frame)),
        "timestamp_row_coverage": float(covered_rows / len(target_frame)) if len(target_frame) else np.nan,
        "timestamp_covered_unique_shas": covered_unique_shas,
        "timestamp_total_unique_shas": unique_target_shas,
        "timestamp_sha_coverage": float(covered_unique_shas / unique_target_shas) if unique_target_shas else np.nan,
        "ambiguous_timestamp_shas": ambiguous_timestamp_shas,
        "temporal_split_unit": "unique SHA ordered by mapped commit_timestamp then SHA",
        "status": "ok",
        "error_message": "",
    }
    if covered_unique_shas < 2:
        return {**base, "status": "insufficient_timestamped_commits", "error_message": "Fewer than two timestamped commits."}

    unique_commits = (
        pd.DataFrame({"sha": sha_values[covered_mask].to_numpy(), "timestamp": timestamps[covered_mask].to_numpy(dtype=float)})
        .drop_duplicates("sha")
        .sort_values(["timestamp", "sha"], kind="mergesort")
        .reset_index(drop=True)
    )
    test_commit_count = max(1, int(np.ceil(0.20 * len(unique_commits))))
    train_commit_count = len(unique_commits) - test_commit_count
    if train_commit_count < 1:
        return {**base, "status": "insufficient_temporal_train", "error_message": "Temporal training set is empty."}
    train_shas = set(unique_commits.iloc[:train_commit_count]["sha"])
    test_shas = set(unique_commits.iloc[train_commit_count:]["sha"])
    train_mask = sha_values.isin(train_shas)
    test_mask = sha_values.isin(test_shas)
    train_frame = target_frame.loc[train_mask].copy()
    test_frame = target_frame.loc[test_mask].copy()
    X_train, y_train, _ = prepare_features(train_frame, context.granularity)
    X_test, y_test, _ = prepare_features(test_frame, context.granularity)
    best_params = parse_best_params(selected_row["best_params"])
    try:
        wpdp_probabilities = fit_predict_fixed(
            str(selected_row["model_name"]),
            best_params,
            X_train,
            y_train,
            X_test,
            smote_k=1,
        )
        wpdp_metrics = compute_metrics(y_test, wpdp_probabilities)
        lopo_late_probabilities = np.asarray(full_target_probabilities)[test_mask.to_numpy()]
        lopo_metrics = compute_metrics(y_test, lopo_late_probabilities)
    except Exception as exc:
        return {
            **base,
            "status": "fit_error",
            "error_message": str(exc),
            "temporal_train_unique_shas": train_commit_count,
            "temporal_test_unique_shas": test_commit_count,
            "temporal_train_rows": int(len(train_frame)),
            "temporal_test_rows": int(len(test_frame)),
            "temporal_train_positives": int(y_train.sum()),
            "temporal_test_positives": int(y_test.sum()),
        }

    output = {
        **base,
        "temporal_train_unique_shas": train_commit_count,
        "temporal_test_unique_shas": test_commit_count,
        "temporal_train_rows": int(len(train_frame)),
        "temporal_test_rows": int(len(test_frame)),
        "temporal_train_positives": int(y_train.sum()),
        "temporal_test_positives": int(y_test.sum()),
        "temporal_test_start_timestamp": float(unique_commits.iloc[train_commit_count]["timestamp"]),
    }
    for metric in REPLAY_METRICS:
        output[f"lopo_late_{metric}"] = lopo_metrics[metric]
        output[f"wpdp_{metric}"] = wpdp_metrics[metric]
        output[f"wpdp_minus_lopo_{metric}"] = wpdp_metrics[metric] - lopo_metrics[metric]
    return output


def bootstrap_spearman_ci(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float, int]:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 4 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan"), float("nan"), 0
    estimates: list[float] = []
    for _ in range(resamples):
        indices = rng.integers(0, len(x), size=len(x))
        sample_x = x[indices]
        sample_y = y[indices]
        if np.unique(sample_x).size < 2 or np.unique(sample_y).size < 2:
            continue
        estimate = float(spearmanr(sample_x, sample_y).statistic)
        if np.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        return float("nan"), float("nan"), 0
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high), len(estimates)


def build_transfer_boundary_rows(fold_rows: pd.DataFrame) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    predictor_columns = {
        "log_target_rows": "log target rows",
        "abs_source_target_prevalence_gap": "absolute source-target operational class-proportion gap",
        "mean_abs_standardized_feature_mean_shift": "mean absolute standardized feature-mean shift",
    }
    metric_columns = {"selected_f1": "f1_1", "selected_mcc": "mcc"}
    for granularity in GRANULARITIES:
        subset = fold_rows[fold_rows["granularity"].eq(granularity)].copy()
        for metric_name, metric_column in metric_columns.items():
            for predictor_column, predictor_label in predictor_columns.items():
                paired = subset[[metric_column, predictor_column]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(paired) >= 3 and paired[predictor_column].nunique() > 1 and paired[metric_column].nunique() > 1:
                    result = spearmanr(paired[predictor_column], paired[metric_column])
                    rho = float(result.statistic)
                    pvalue = float(result.pvalue)
                else:
                    rho = pvalue = float("nan")
                seed_offset = GRANULARITIES.index(granularity) * 100 + list(metric_columns).index(metric_name) * 10 + list(predictor_columns).index(predictor_column)
                rng = np.random.default_rng(RANDOM_SEED + seed_offset)
                ci_low, ci_high, valid_bootstraps = bootstrap_spearman_ci(
                    paired[predictor_column].to_numpy(dtype=float),
                    paired[metric_column].to_numpy(dtype=float),
                    rng,
                )
                output.append(
                    {
                        "granularity": granularity,
                        "metric": metric_name,
                        "predictor": predictor_column,
                        "predictor_label": predictor_label,
                        "n_projects": int(len(paired)),
                        "spearman_rho": rho,
                        "pvalue": pvalue,
                        "bootstrap_ci_low": ci_low,
                        "bootstrap_ci_high": ci_high,
                        "valid_bootstrap_resamples": valid_bootstraps,
                        "exploratory": True,
                        "causal_interpretation": False,
                    }
                )
    return output


def build_go_feature_correlation_rows(
    contexts: dict[str, GranularityContext],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_records: list[dict[str, Any]] = []
    go_metrics_by_granularity = {
        "file": list(GO_SPECIFIC_FILE_METRICS),
        "method": list(GO_SPECIFIC_METHOD_METRICS),
    }
    for granularity, go_metrics in go_metrics_by_granularity.items():
        context = contexts[granularity]
        all_features = get_feature_columns(granularity)
        generic_metrics = [feature for feature in all_features if feature not in set(go_metrics)]
        per_pair: dict[tuple[str, str], list[float]] = {
            (go_metric, generic_metric): [] for go_metric in go_metrics for generic_metric in generic_metrics
        }
        for project in context.projects:
            frame = context.project_frames[project][all_features]
            correlations = frame.corr(method="spearman", min_periods=3)
            for pair in per_pair:
                go_metric, generic_metric = pair
                rho = correlations.loc[go_metric, generic_metric]
                if np.isfinite(rho):
                    per_pair[pair].append(float(rho))
        for (go_metric, generic_metric), values in per_pair.items():
            array = np.asarray(values, dtype=float)
            pair_records.append(
                {
                    "granularity": granularity,
                    "go_specific_feature": go_metric,
                    "generic_feature": generic_metric,
                    "valid_project_correlations": int(len(array)),
                    "median_rho_across_projects": float(np.median(array)) if len(array) else np.nan,
                    "median_absolute_rho_across_projects": float(np.median(np.abs(array))) if len(array) else np.nan,
                    "minimum_rho_across_projects": float(np.min(array)) if len(array) else np.nan,
                    "maximum_rho_across_projects": float(np.max(array)) if len(array) else np.nan,
                    "descriptive_only": True,
                }
            )
    pair_frame = pd.DataFrame(pair_records)
    summary_rows: list[dict[str, Any]] = []
    for (granularity, go_feature), group in pair_frame.groupby(["granularity", "go_specific_feature"], sort=False):
        valid_group = group.dropna(subset=["median_absolute_rho_across_projects"])
        if valid_group.empty:
            summary_rows.append(
                {
                    "granularity": granularity,
                    "go_specific_feature": go_feature,
                    "strongest_generic_feature": "",
                    "strongest_median_absolute_rho": np.nan,
                    "median_of_pairwise_median_absolute_rho": np.nan,
                }
            )
            continue
        strongest = valid_group.sort_values(
            ["median_absolute_rho_across_projects", "generic_feature"],
            ascending=[False, True],
            kind="mergesort",
        ).iloc[0]
        summary_rows.append(
            {
                "granularity": granularity,
                "go_specific_feature": go_feature,
                "strongest_generic_feature": strongest["generic_feature"],
                "strongest_median_absolute_rho": strongest["median_absolute_rho_across_projects"],
                "median_of_pairwise_median_absolute_rho": float(valid_group["median_absolute_rho_across_projects"].median()),
                "descriptive_only": True,
            }
        )
    return pair_frame, pd.DataFrame(summary_rows)


def run_fixed_replays(
    data_root: Path,
    contexts: dict[str, GranularityContext],
) -> dict[str, pd.DataFrame]:
    verification_rows: list[dict[str, Any]] = []
    effort_rows: list[dict[str, Any]] = []
    temporal_rows: list[dict[str, Any]] = []
    fold_diagnostic_rows: list[dict[str, Any]] = []
    smote_k_rows: list[dict[str, Any]] = []
    harmonized_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    effort_columns = {"commit": "total_nloc", "file": "nloc", "method": "nloc"}

    for granularity in GRANULARITIES:
        context = contexts[granularity]
        selected_lookup = context.selected_rows.set_index("target_project", drop=False)
        for target_project in context.projects:
            logging.info("Replaying fixed configuration: granularity=%s target=%s", granularity, target_project)
            selected_row = selected_lookup.loc[target_project]
            model_name = str(selected_row["model_name"])
            best_params = parse_best_params(selected_row["best_params"])
            source_frame = context.all_rows[context.all_rows["project_id"].ne(target_project)].copy()
            target_frame = context.project_frames[target_project].copy().reset_index(drop=True)
            X_source, y_source, _ = prepare_features(source_frame, granularity)
            X_target, y_target, _ = prepare_features(target_frame, granularity)

            probabilities_k1 = fit_predict_fixed(
                model_name,
                best_params,
                X_source,
                y_source,
                X_target,
                smote_k=1,
            )
            replayed_metrics = compute_metrics(y_target, probabilities_k1)
            verification_row: dict[str, Any] = {
                "granularity": granularity,
                "target_project": target_project,
                "model_name": model_name,
                "best_params": json.dumps(best_params, sort_keys=True),
                "source_rows": int(len(source_frame)),
                "target_rows": int(len(target_frame)),
                "verification_tolerance": VERIFICATION_ATOL,
            }
            maximum_difference = 0.0
            for metric in REPLAY_METRICS:
                recorded = float(selected_row[metric])
                replayed = float(replayed_metrics[metric])
                difference = replayed - recorded
                verification_row[f"recorded_{metric}"] = recorded
                verification_row[f"replayed_{metric}"] = replayed
                verification_row[f"difference_{metric}"] = difference
                if np.isfinite(difference):
                    maximum_difference = max(maximum_difference, abs(difference))
                if not np.isclose(replayed, recorded, rtol=0.0, atol=VERIFICATION_ATOL, equal_nan=True):
                    raise AssertionError(
                        f"Replay mismatch for {granularity}/{target_project}/{metric}: "
                        f"recorded={recorded}, replayed={replayed}"
                    )
            verification_row["maximum_absolute_metric_difference"] = maximum_difference
            verification_row["verified"] = True
            verification_rows.append(verification_row)

            effort_metrics = ranking_effort_metrics(
                y_target,
                probabilities_k1,
                target_frame["sample_id"],
                target_frame[effort_columns[granularity]],
            )
            effort_rows.append(
                {
                    "granularity": granularity,
                    "target_project": target_project,
                    "model_name": model_name,
                    "effort_column": effort_columns[granularity],
                    "ranking_definition": "descending predicted defect probability; sample_id ascending breaks ties",
                    "entity_budget_definition": "ceil(20% of modeled target rows)",
                    "effort_budget_definition": "smallest ranked prefix reaching at least 20% of nonnegative LOC effort",
                    "aucec_definition": "trapezoidal area under cumulative recall versus cumulative LOC-effort fraction",
                    **effort_metrics,
                }
            )

            temporal_rows.append(
                temporal_reference_row(
                    data_root,
                    context,
                    target_project,
                    selected_row,
                    target_frame,
                    probabilities_k1,
                )
            )

            shift, shifted_feature_count = source_target_shift(source_frame, target_frame, granularity)
            fold_diagnostic_rows.append(
                {
                    "granularity": granularity,
                    "target_project": target_project,
                    "model_name": model_name,
                    "f1_1": float(selected_row["f1_1"]),
                    "mcc": float(selected_row["mcc"]),
                    "target_rows": int(len(target_frame)),
                    "log_target_rows": float(np.log1p(len(target_frame))),
                    "source_prevalence": float(y_source.mean()),
                    "target_prevalence": float(y_target.mean()),
                    "abs_source_target_prevalence_gap": float(abs(y_source.mean() - y_target.mean())),
                    "mean_abs_standardized_feature_mean_shift": shift,
                    "shifted_feature_count": shifted_feature_count,
                    "shift_definition": "mean absolute target-source feature mean difference in source-standard-deviation units after source-median imputation",
                    "exploratory": True,
                    "causal_interpretation": False,
                }
            )

            for smote_k in SMOTE_K_VALUES:
                if smote_k == 1:
                    probabilities = probabilities_k1
                    metrics = replayed_metrics
                else:
                    probabilities = fit_predict_fixed(
                        model_name,
                        best_params,
                        X_source,
                        y_source,
                        X_target,
                        smote_k=smote_k,
                    )
                    metrics = compute_metrics(y_target, probabilities)
                smote_k_rows.append(
                    {
                        "granularity": granularity,
                        "target_project": target_project,
                        "model_name": model_name,
                        "best_params": json.dumps(best_params, sort_keys=True),
                        "smote_k": smote_k,
                        "f1_1": metrics["f1_1"],
                        "mcc": metrics["mcc"],
                        "auc": metrics["auc"],
                        "pr_auc": metrics["pr_auc"],
                        "fixed_family_and_hyperparameters": True,
                    }
                )

            X_source_harmonized = harmonized_features(source_frame, granularity)
            X_target_harmonized = harmonized_features(target_frame, granularity)
            harmonized_probabilities = fit_predict_fixed(
                model_name,
                best_params,
                X_source_harmonized,
                y_source,
                X_target_harmonized,
                smote_k=1,
            )
            harmonized_metrics = compute_metrics(y_target, harmonized_probabilities)
            harmonized_row: dict[str, Any] = {
                "granularity": granularity,
                "target_project": target_project,
                "model_name": model_name,
                "best_params": json.dumps(best_params, sort_keys=True),
                "harmonized_features": "nloc;token_count;complexity",
                "source_feature_mapping": json.dumps(HARMONIZED_FEATURE_MAP[granularity], sort_keys=True),
                "fixed_family_and_hyperparameters": True,
                "partial_feature_family_harmonization_only": True,
                "remaining_confounds": "prediction unit;labels;operational class proportion;dataset composition",
            }
            for metric in REPLAY_METRICS:
                original = float(selected_row[metric])
                harmonized = float(harmonized_metrics[metric])
                harmonized_row[f"original_{metric}"] = original
                harmonized_row[f"harmonized_{metric}"] = harmonized
                harmonized_row[f"harmonized_minus_original_{metric}"] = harmonized - original
            harmonized_rows.append(harmonized_row)

            source_keep = ~source_frame["__cross_label_conflict"].astype(bool)
            target_keep = ~target_frame["__cross_label_conflict"].astype(bool)
            cleaned_source = source_frame.loc[source_keep].copy()
            cleaned_target = target_frame.loc[target_keep].copy()
            X_clean_source, y_clean_source, _ = prepare_features(cleaned_source, granularity)
            X_clean_target, y_clean_target, _ = prepare_features(cleaned_target, granularity)
            cleaned_probabilities = fit_predict_fixed(
                model_name,
                best_params,
                X_clean_source,
                y_clean_source,
                X_clean_target,
                smote_k=1,
            )
            cleaned_metrics = compute_metrics(y_clean_target, cleaned_probabilities)
            original_retained_probabilities = probabilities_k1[target_keep.to_numpy()]
            original_retained_metrics = compute_metrics(y_clean_target, original_retained_probabilities)
            conflict_row: dict[str, Any] = {
                "granularity": granularity,
                "target_project": target_project,
                "model_name": model_name,
                "best_params": json.dumps(best_params, sort_keys=True),
                "source_conflict_identity_keys": int(
                    source_frame.loc[~source_keep, ["project_id", "__identity_key"]].drop_duplicates().shape[0]
                ),
                "source_conflict_rows_removed": int((~source_keep).sum()),
                "target_conflict_identity_keys": int(target_frame.loc[~target_keep, "__identity_key"].nunique()),
                "target_conflict_rows_removed": int((~target_keep).sum()),
                "cleaned_source_rows": int(len(cleaned_source)),
                "cleaned_target_rows": int(len(cleaned_target)),
                "posthoc_benchmark_cleaning_sensitivity": True,
                "identity_definition": {
                    "commit": "sha",
                    "file": "project/file_path/sha",
                    "method": "project/file_path/method_name/sha",
                }[granularity],
            }
            for metric in REPLAY_METRICS:
                recorded = float(selected_row[metric])
                retained = float(original_retained_metrics[metric])
                cleaned = float(cleaned_metrics[metric])
                conflict_row[f"original_full_{metric}"] = recorded
                conflict_row[f"original_model_retained_test_{metric}"] = retained
                conflict_row[f"conflict_cleaned_{metric}"] = cleaned
                conflict_row[f"cleaned_minus_original_full_{metric}"] = cleaned - recorded
                conflict_row[f"cleaned_minus_original_retained_test_{metric}"] = cleaned - retained
            conflict_rows.append(conflict_row)

    return {
        "verification": pd.DataFrame(verification_rows),
        "effort": pd.DataFrame(effort_rows),
        "temporal": pd.DataFrame(temporal_rows),
        "fold_diagnostics": pd.DataFrame(fold_diagnostic_rows),
        "smote_k": pd.DataFrame(smote_k_rows),
        "harmonized": pd.DataFrame(harmonized_rows),
        "conflict_cleaned": pd.DataFrame(conflict_rows),
    }


def build_adequacy_sensitivity_rows(
    contexts: dict[str, GranularityContext],
) -> pd.DataFrame:
    selected = {
        granularity: context.selected_rows.set_index("target_project", drop=False)
        for granularity, context in contexts.items()
    }
    baseline_signs: dict[tuple[str, str, str], int] = {}
    for left, right in PAIR_ORDER:
        common = sorted(set(selected[left].index).intersection(selected[right].index))
        for metric in FOCUS_METRICS:
            differences = selected[left].loc[common, metric].to_numpy(dtype=float) - selected[right].loc[common, metric].to_numpy(dtype=float)
            baseline_signs[(left, right, metric)] = int(np.sign(np.mean(differences)))

    rows: list[dict[str, Any]] = []
    for row_threshold in ADEQUACY_ROW_THRESHOLDS:
        for minority_threshold in ADEQUACY_MINORITY_THRESHOLDS:
            adequate_projects: dict[str, set[str]] = {}
            for granularity in GRANULARITIES:
                frame = selected[granularity]
                minority = np.minimum(
                    frame["target_bug_count"].to_numpy(dtype=int),
                    frame["target_non_bug_count"].to_numpy(dtype=int),
                )
                adequate_mask = frame["target_row_count"].to_numpy(dtype=int) >= row_threshold
                adequate_mask &= minority >= minority_threshold
                adequate_projects[granularity] = set(frame.index[adequate_mask])
            for left, right in PAIR_ORDER:
                common = sorted(adequate_projects[left].intersection(adequate_projects[right]))
                for metric in FOCUS_METRICS:
                    left_values = selected[left].loc[common, metric].to_numpy(dtype=float) if common else np.array([])
                    right_values = selected[right].loc[common, metric].to_numpy(dtype=float) if common else np.array([])
                    differences = left_values - right_values
                    statistic, pvalue, paired_n = safe_wilcoxon(left_values, right_values)
                    mean_difference = float(np.mean(differences)) if len(differences) else np.nan
                    difference_sign = int(np.sign(mean_difference)) if np.isfinite(mean_difference) else 0
                    rows.append(
                        {
                            "row_threshold": row_threshold,
                            "minority_threshold": minority_threshold,
                            "left_granularity": left,
                            "right_granularity": right,
                            "comparison": f"{left}_vs_{right}",
                            "metric": metric,
                            "left_adequate_targets": len(adequate_projects[left]),
                            "right_adequate_targets": len(adequate_projects[right]),
                            "paired_target_count": paired_n,
                            "mean_paired_difference": mean_difference,
                            "median_paired_difference": float(np.median(differences)) if len(differences) else np.nan,
                            "wins": int((differences > 0).sum()),
                            "ties": int((differences == 0).sum()),
                            "losses": int((differences < 0).sum()),
                            "wilcoxon_statistic": statistic,
                            "wilcoxon_pvalue": pvalue,
                            "all_target_mean_difference_sign": baseline_signs[(left, right, metric)],
                            "thresholded_mean_difference_sign": difference_sign,
                            "sign_stable_vs_all_targets": difference_sign == baseline_signs[(left, right, metric)] and paired_n > 0,
                            "reporting_only_thresholds": True,
                        }
                    )
    return pd.DataFrame(rows)


def build_holm_rows(contexts: dict[str, GranularityContext]) -> pd.DataFrame:
    selected = {
        granularity: context.selected_rows.set_index("target_project", drop=False)
        for granularity, context in contexts.items()
    }
    rows: list[dict[str, Any]] = []
    for left, right in PAIR_ORDER:
        common = sorted(set(selected[left].index).intersection(selected[right].index))
        for metric in FOCUS_METRICS:
            statistic, pvalue, paired_n = safe_wilcoxon(
                selected[left].loc[common, metric],
                selected[right].loc[common, metric],
            )
            rows.append(
                {
                    "left_granularity": left,
                    "right_granularity": right,
                    "comparison": f"{left}_vs_{right}",
                    "metric": metric,
                    "paired_target_count": paired_n,
                    "wilcoxon_statistic": statistic,
                    "raw_pvalue": pvalue,
                }
            )
    frame = pd.DataFrame(rows)
    order = np.argsort(frame["raw_pvalue"].to_numpy(dtype=float), kind="mergesort")
    adjusted = np.empty(len(frame), dtype=float)
    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(frame) - rank) * float(frame.iloc[index]["raw_pvalue"]))
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    frame["holm_adjusted_pvalue"] = adjusted
    frame["holm_family"] = "three granularity pairs x two co-primary metrics (six tests)"
    frame["reject_at_0_05_after_holm"] = frame["holm_adjusted_pvalue"] < 0.05
    return frame


def summarize_effort(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for granularity in GRANULARITIES:
        group = frame[frame["granularity"].eq(granularity)]
        rows.append(
            {
                "granularity": granularity,
                "target_count": int(group["target_project"].nunique()),
                "mean_recall_at_20pct_entities": float(group["recall_at_20pct_entities"].mean()),
                "std_recall_at_20pct_entities": float(group["recall_at_20pct_entities"].std(ddof=1)),
                "mean_loc_aware_recall_at_20pct_effort": float(group["loc_aware_recall_at_20pct_effort"].mean()),
                "std_loc_aware_recall_at_20pct_effort": float(group["loc_aware_recall_at_20pct_effort"].std(ddof=1)),
                "mean_aucec": float(group["aucec"].mean()),
                "std_aucec": float(group["aucec"].std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def summarize_temporal(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for granularity in GRANULARITIES:
        all_group = frame[frame["granularity"].eq(granularity)]
        group = all_group[all_group["status"].eq("ok")]
        row: dict[str, Any] = {
            "granularity": granularity,
            "available_target_count": int(len(group)),
            "total_target_count": int(len(all_group)),
            "mean_timestamp_row_coverage": float(all_group["timestamp_row_coverage"].mean()),
            "mean_timestamp_sha_coverage": float(all_group["timestamp_sha_coverage"].mean()),
        }
        for metric in FOCUS_METRICS:
            lopo_column = f"lopo_late_{metric}"
            wpdp_column = f"wpdp_{metric}"
            row[f"mean_lopo_late_{metric}"] = float(group[lopo_column].mean()) if lopo_column in group else np.nan
            row[f"mean_wpdp_{metric}"] = float(group[wpdp_column].mean()) if wpdp_column in group else np.nan
            row[f"mean_wpdp_minus_lopo_{metric}"] = (
                float((group[wpdp_column] - group[lopo_column]).mean()) if wpdp_column in group and lopo_column in group else np.nan
            )
            statistic, pvalue, paired_n = safe_wilcoxon(group[wpdp_column], group[lopo_column]) if len(group) else (np.nan, np.nan, 0)
            row[f"wilcoxon_statistic_{metric}"] = statistic
            row[f"wilcoxon_pvalue_{metric}"] = pvalue
            row[f"paired_n_{metric}"] = paired_n
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_smote_k(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for granularity in GRANULARITIES:
        granularity_frame = frame[frame["granularity"].eq(granularity)]
        baseline = granularity_frame[granularity_frame["smote_k"].eq(1)].set_index("target_project")
        for smote_k in SMOTE_K_VALUES:
            group = granularity_frame[granularity_frame["smote_k"].eq(smote_k)].set_index("target_project")
            common = sorted(set(group.index).intersection(baseline.index))
            row: dict[str, Any] = {
                "granularity": granularity,
                "smote_k": smote_k,
                "target_count": len(common),
                "fixed_family_and_hyperparameters": True,
            }
            for metric in FOCUS_METRICS:
                values = group.loc[common, metric].to_numpy(dtype=float)
                base_values = baseline.loc[common, metric].to_numpy(dtype=float)
                statistic, pvalue, paired_n = safe_wilcoxon(values, base_values)
                row[f"mean_{metric}"] = float(np.mean(values))
                row[f"std_{metric}"] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
                row[f"mean_delta_vs_k1_{metric}"] = float(np.mean(values - base_values))
                row[f"wilcoxon_statistic_vs_k1_{metric}"] = statistic
                row[f"wilcoxon_pvalue_vs_k1_{metric}"] = pvalue
                row[f"paired_n_{metric}"] = paired_n
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_harmonized(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for granularity in GRANULARITIES:
        group = frame[frame["granularity"].eq(granularity)]
        for metric in FOCUS_METRICS:
            original = group[f"original_{metric}"].to_numpy(dtype=float)
            harmonized = group[f"harmonized_{metric}"].to_numpy(dtype=float)
            statistic, pvalue, paired_n = safe_wilcoxon(harmonized, original)
            rows.append(
                {
                    "granularity": granularity,
                    "metric": metric,
                    "target_count": paired_n,
                    "mean_original": float(np.mean(original)),
                    "mean_harmonized": float(np.mean(harmonized)),
                    "mean_harmonized_minus_original": float(np.mean(harmonized - original)),
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_pvalue": pvalue,
                    "partial_feature_family_harmonization_only": True,
                    "remaining_confounds": "prediction unit;labels;operational class proportion;dataset composition",
                }
            )
    return pd.DataFrame(rows)


def summarize_conflict_cleaning(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for granularity in GRANULARITIES:
        group = frame[frame["granularity"].eq(granularity)]
        for metric in FOCUS_METRICS:
            original_full = group[f"original_full_{metric}"].to_numpy(dtype=float)
            original_retained = group[f"original_model_retained_test_{metric}"].to_numpy(dtype=float)
            cleaned = group[f"conflict_cleaned_{metric}"].to_numpy(dtype=float)
            statistic_full, pvalue_full, paired_n = safe_wilcoxon(cleaned, original_full)
            statistic_retained, pvalue_retained, _ = safe_wilcoxon(cleaned, original_retained)
            rows.append(
                {
                    "granularity": granularity,
                    "metric": metric,
                    "target_count": paired_n,
                    "source_conflict_rows_removed_across_folds": int(group["source_conflict_rows_removed"].sum()),
                    "target_conflict_rows_removed_across_folds": int(group["target_conflict_rows_removed"].sum()),
                    "mean_original_full": float(np.mean(original_full)),
                    "mean_original_model_retained_test": float(np.mean(original_retained)),
                    "mean_conflict_cleaned": float(np.mean(cleaned)),
                    "mean_cleaned_minus_original_full": float(np.mean(cleaned - original_full)),
                    "wilcoxon_statistic_vs_original_full": statistic_full,
                    "wilcoxon_pvalue_vs_original_full": pvalue_full,
                    "mean_cleaned_minus_original_retained_test": float(np.mean(cleaned - original_retained)),
                    "wilcoxon_statistic_vs_original_retained_test": statistic_retained,
                    "wilcoxon_pvalue_vs_original_retained_test": pvalue_retained,
                    "posthoc_benchmark_cleaning_sensitivity": True,
                }
            )
    return pd.DataFrame(rows)


def summarize_adequacy(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (comparison, metric), group in frame.groupby(["comparison", "metric"], sort=False):
        valid = group[group["paired_target_count"].gt(0)]
        rows.append(
            {
                "comparison": comparison,
                "metric": metric,
                "grid_cells": int(len(group)),
                "valid_grid_cells": int(len(valid)),
                "sign_stable_grid_cells": int(valid["sign_stable_vs_all_targets"].sum()),
                "minimum_paired_targets": int(valid["paired_target_count"].min()) if len(valid) else 0,
                "maximum_paired_targets": int(valid["paired_target_count"].max()) if len(valid) else 0,
                "minimum_mean_difference": float(valid["mean_paired_difference"].min()) if len(valid) else np.nan,
                "maximum_mean_difference": float(valid["mean_paired_difference"].max()) if len(valid) else np.nan,
                "minimum_wilcoxon_pvalue": float(valid["wilcoxon_pvalue"].min()) if len(valid) else np.nan,
                "maximum_wilcoxon_pvalue": float(valid["wilcoxon_pvalue"].max()) if len(valid) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_latex_outputs(
    generated_dir: Path,
    frames: dict[str, pd.DataFrame],
) -> None:
    verification = frames["verification"]
    verification_summary = (
        verification.groupby("granularity", sort=False)
        .agg(
            verified_targets=("verified", "sum"),
            maximum_absolute_difference=("maximum_absolute_metric_difference", "max"),
        )
        .reset_index()
    )
    write_text(
        generated_dir / "diagnostic_replay_verification_table.tex",
        make_latex_table(
            "Deterministic replay verification of the recorded fold-local model family and hyperparameters. Each pipeline is refit on the complete non-target source pool with SMOTE ($k=1$), and the reported $F_1$, MCC, AUC, and PR-AUC are checked against the recorded held-out result with absolute tolerance $10^{-10}$.",
            "tab:diagnostic-replay-verification",
            ["Granularity", "Verified targets", "Maximum absolute difference"],
            [
                [GRANULARITY_LABELS[row.granularity], int(row.verified_targets), f"{row.maximum_absolute_difference:.2e}"]
                for row in verification_summary.itertuples(index=False)
            ],
            "lrr",
        ),
    )

    effort = frames["effort_summary"]
    write_text(
        generated_dir / "diagnostic_effort_metrics_table.tex",
        make_latex_table(
            r"LOPO review-prioritization diagnostics under the recorded fold-local configurations. Entities are ranked by descending predicted defect probability with sample identifier as a deterministic tie-breaker. Recall@20\% entities uses the top $\lceil0.2n\rceil$ rows. LOC-aware Recall@20\% effort uses the smallest ranked prefix reaching at least 20\% of total nonnegative LOC effort (\texttt{total\_nloc} for commits and \texttt{nloc} otherwise). AUCEC is the trapezoidal area under cumulative recall versus cumulative LOC-effort fraction. Values are mean $\pm$ sample standard deviation across targets.",
            "tab:diagnostic-effort",
            ["Configuration", "Targets", r"Recall@20\% entities", r"LOC Recall@20\% effort", "AUCEC"],
            [
                [
                    GRANULARITY_LABELS[row.granularity],
                    int(row.target_count),
                    f"{row.mean_recall_at_20pct_entities:.3f} $\\pm$ {row.std_recall_at_20pct_entities:.3f}",
                    f"{row.mean_loc_aware_recall_at_20pct_effort:.3f} $\\pm$ {row.std_loc_aware_recall_at_20pct_effort:.3f}",
                    f"{row.mean_aucec:.3f} $\\pm$ {row.std_aucec:.3f}",
                ]
                for row in effort.itertuples(index=False)
            ],
            "lrrrr",
        ),
    )

    temporal = frames["temporal_summary"]
    write_text(
        generated_dir / "diagnostic_temporal_reference_table.tex",
        make_latex_table(
            r"Conservative within-project temporal reference using the originally selected LOPO family and hyperparameters. The first 80\% of timestamp-mapped unique commits train the within-project model, and the final 20\% form the test set. LOPO and within-project scores use identical late test rows; the reference is diagnostic rather than an upper bound. Means are across available targets.",
            "tab:diagnostic-temporal-reference",
            ["Configuration", "Available", "Timestamp row coverage", "LOPO late $F_1$", "WPDP $F_1$", "LOPO late MCC", "WPDP MCC"],
            [
                [
                    GRANULARITY_LABELS[row.granularity],
                    f"{int(row.available_target_count)}/{int(row.total_target_count)}",
                    format_float(row.mean_timestamp_row_coverage),
                    format_float(row.mean_lopo_late_f1_1),
                    format_float(row.mean_wpdp_f1_1),
                    format_float(row.mean_lopo_late_mcc),
                    format_float(row.mean_wpdp_mcc),
                ]
                for row in temporal.itertuples(index=False)
            ],
            "lrrrrrr",
        ),
    )

    label_audit = frames["label_audit"]
    label_all = label_audit[label_audit["project"].eq("ALL")]
    write_text(
        generated_dir / "diagnostic_label_audit_table.tex",
        make_latex_table(
            "Audit of the released positive and comparison-class exports before modeling. Identity keys are SHA for commits, project/path/SHA for files, and project/path/method-name/SHA for methods. Conflicts count keys present in both classes. Duplicate removal in the baseline removes only rows identical across the exported fields; it does not deduplicate metric tuples.",
            "tab:diagnostic-label-audit",
            ["Granularity", "Positive rows", "Comp. rows", "Positive rows/SHA", "Cross-label keys", "Affected raw rows", "Missing method names", "Exact rows removed"],
            [
                [
                    GRANULARITY_LABELS[row.granularity],
                    int(row.positive_rows),
                    int(row.negative_rows),
                    format_float(row.positive_rows_per_positive_sha),
                    int(row.cross_label_identity_conflicts),
                    int(row.raw_conflict_rows),
                    int(row.missing_method_names_positive + row.missing_method_names_negative),
                    int(row.exact_exported_row_duplicates_removed),
                ]
                for row in label_all.itertuples(index=False)
            ],
            "lrrrrrrr",
        ),
    )

    adequacy = frames["adequacy_summary"]
    write_text(
        generated_dir / "diagnostic_adequacy_sensitivity_table.tex",
        make_latex_table(
            r"Sensitivity of paired configuration contrasts over the 15 reporting-only adequacy screens formed by row thresholds $\{100,200,500\}$ and minority-class thresholds $\{20,30,50,75,100\}$. Sign stability compares each thresholded mean paired difference with its all-target sign; model fitting is unchanged.",
            "tab:diagnostic-adequacy-sensitivity",
            ["Comparison", "Metric", "Stable screens", "Paired $n$ range", r"Mean $\Delta$ range", "$p$ range"],
            [
                [
                    latex_escape(row.comparison.replace("_vs_", " vs ").title()),
                    METRIC_LABELS[row.metric],
                    f"{int(row.sign_stable_grid_cells)}/{int(row.valid_grid_cells)}",
                    f"{int(row.minimum_paired_targets)}--{int(row.maximum_paired_targets)}",
                    f"[{row.minimum_mean_difference:.3f}, {row.maximum_mean_difference:.3f}]",
                    f"[{format_pvalue(row.minimum_wilcoxon_pvalue)}, {format_pvalue(row.maximum_wilcoxon_pvalue)}]",
                ]
                for row in adequacy.itertuples(index=False)
            ],
            "llrrrr",
        ),
    )

    transfer = frames["transfer_boundaries"]
    write_text(
        generated_dir / "diagnostic_transfer_boundaries_table.tex",
        make_latex_table(
            r"Exploratory transfer-boundary diagnostics across 16 target projects. Entries are Spearman correlations with percentile bootstrap 95\% confidence intervals. Feature shift is the mean absolute target-source feature-mean difference in source-standard-deviation units after source-median imputation. These associations are descriptive and do not support causal interpretation.",
            "tab:diagnostic-transfer-boundaries",
            ["Configuration", "Metric", "Target characteristic", r"$\rho$", r"95\% bootstrap CI", "$p$"],
            [
                [
                    GRANULARITY_LABELS[row.granularity],
                    METRIC_LABELS[row.metric],
                    latex_escape(row.predictor_label),
                    format_float(row.spearman_rho),
                    f"[{format_float(row.bootstrap_ci_low)}, {format_float(row.bootstrap_ci_high)}]",
                    format_pvalue(row.pvalue),
                ]
                for row in transfer.itertuples(index=False)
            ],
            "lllrrr",
        ),
    )

    go_summary = frames["go_feature_summary"]
    write_text(
        generated_dir / "diagnostic_go_feature_correlations_table.tex",
        make_latex_table(
            "Descriptive within-project correlation summary for Go-specific and generic handcrafted features. For each Go-specific feature, the table reports the generic feature with the largest median absolute Spearman correlation across projects. Correlation does not establish informational redundancy or causality.",
            "tab:diagnostic-go-feature-correlations",
            ["Configuration", "Go-specific feature", "Strongest generic feature", r"Median $|\rho|$"],
            [
                [
                    GRANULARITY_LABELS[row.granularity],
                    latex_escape(row.go_specific_feature),
                    latex_escape(row.strongest_generic_feature),
                    format_float(row.strongest_median_absolute_rho),
                ]
                for row in go_summary.itertuples(index=False)
            ],
            "lllr",
        ),
    )

    holm = frames["holm"]
    write_text(
        generated_dir / "diagnostic_holm_adjusted_table.tex",
        make_latex_table(
            "Primary paired Wilcoxon contrasts with Holm adjustment across one family of six tests (three configuration pairs by two co-primary metrics).",
            "tab:diagnostic-holm",
            ["Comparison", "Metric", "$n$", "Raw $p$", "Holm-adjusted $p$", "Reject at 0.05"],
            [
                [
                    latex_escape(row.comparison.replace("_vs_", " vs ").title()),
                    METRIC_LABELS[row.metric],
                    int(row.paired_target_count),
                    format_pvalue(row.raw_pvalue),
                    format_pvalue(row.holm_adjusted_pvalue),
                    "Yes" if row.reject_at_0_05_after_holm else "No",
                ]
                for row in holm.itertuples(index=False)
            ],
            "llrrrr",
        ),
    )

    smote = frames["smote_k_summary"]
    write_text(
        generated_dir / "diagnostic_smote_k_replay_table.tex",
        make_latex_table(
            "Fixed-family, fixed-hyperparameter replay sensitivity for SMOTE neighborhood size. Only $k$ changes; each row reports target-level means and paired mean changes relative to $k=1$. This does not reproduce fold-local family and hyperparameter selection under alternative $k$ values.",
            "tab:diagnostic-smote-k-replay",
            ["Configuration", "$k$", "$F_1$", r"$\Delta F_1$", "$p_{F_1}$", "MCC", r"$\Delta$MCC", "$p_{MCC}$"],
            [
                [
                    GRANULARITY_LABELS[row.granularity],
                    int(row.smote_k),
                    format_float(row.mean_f1_1),
                    format_float(row.mean_delta_vs_k1_f1_1),
                    format_pvalue(row.wilcoxon_pvalue_vs_k1_f1_1),
                    format_float(row.mean_mcc),
                    format_float(row.mean_delta_vs_k1_mcc),
                    format_pvalue(row.wilcoxon_pvalue_vs_k1_mcc),
                ]
                for row in smote.itertuples(index=False)
            ],
            "lrrrrrrr",
        ),
    )

    harmonized = frames["harmonized_summary"]
    write_text(
        generated_dir / "diagnostic_harmonized_replay_table.tex",
        make_latex_table(
            "Partial three-descriptor feature-family harmonization with fixed original family and hyperparameters. The aligned descriptors are NLOC, token count, and complexity. Prediction unit, label construction, operational class proportion, and dataset composition remain confounded, so the sensitivity does not isolate a pure granularity effect.",
            "tab:diagnostic-harmonized-replay",
            ["Configuration", "Metric", "Original", "Harmonized", r"Mean $\Delta$", "Paired $p$"],
            [
                [
                    GRANULARITY_LABELS[row.granularity],
                    METRIC_LABELS[row.metric],
                    format_float(row.mean_original),
                    format_float(row.mean_harmonized),
                    format_float(row.mean_harmonized_minus_original),
                    format_pvalue(row.wilcoxon_pvalue),
                ]
                for row in harmonized.itertuples(index=False)
            ],
            "llrrrr",
        ),
    )

    conflict = frames["conflict_cleaned_summary"]
    write_text(
        generated_dir / "diagnostic_conflict_cleaned_replay_table.tex",
        make_latex_table(
            "Post-hoc benchmark-cleaning sensitivity that drops every operational identity key appearing in both exported classes from source and target before fixed-family, fixed-hyperparameter replay. The same-retained-test comparison separates source-cleaning changes from the target-row removal effect; neither comparison is causal.",
            "tab:diagnostic-conflict-cleaned-replay",
            ["Configuration", "Metric", "Original full", "Original retained", "Cleaned", r"$\Delta$ vs full", "$p$", r"$\Delta$ same test", "$p$ same test"],
            [
                [
                    GRANULARITY_LABELS[row.granularity],
                    METRIC_LABELS[row.metric],
                    format_float(row.mean_original_full),
                    format_float(row.mean_original_model_retained_test),
                    format_float(row.mean_conflict_cleaned),
                    format_float(row.mean_cleaned_minus_original_full),
                    format_pvalue(row.wilcoxon_pvalue_vs_original_full),
                    format_float(row.mean_cleaned_minus_original_retained_test),
                    format_pvalue(row.wilcoxon_pvalue_vs_original_retained_test),
                ]
                for row in conflict.itertuples(index=False)
            ],
            "llrrrrrrr",
        ),
    )


def write_metadata(
    path: Path,
    data_root: Path,
    results_root: Path,
    frames: dict[str, pd.DataFrame],
) -> None:
    input_hashes: dict[str, str] = {}
    for granularity in GRANULARITIES:
        for file_name in ("run_signature.json", "per_project_results.csv"):
            input_path = results_root / granularity / file_name
            input_hashes[str(input_path)] = file_sha256(input_path)
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": str(Path(__file__).resolve()),
        "random_seed": RANDOM_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "verification_absolute_tolerance": VERIFICATION_ATOL,
        "data_root": str(data_root),
        "data_root_git_revision": git_revision(data_root),
        "repository_root": str(REPO_ROOT),
        "repository_git_revision": git_revision(REPO_ROOT),
        "results_root": str(results_root),
        "input_sha256": input_hashes,
        "definitions": {
            "fold_local_selection": "highest recorded best_inner_f1, then best_inner_mcc, then recorded model order",
            "baseline_replay": "recorded family and best_params, source-only full fit, SMOTE k=1, threshold 0.5",
            "duplicate_handling": "only exact exported-row duplicates are removed before modeling; no metric-tuple deduplication",
            "recall_at_20pct_entities": "positive recall in ceil(20% of target rows) ranked by descending probability with sample_id tie-break",
            "loc_recall_at_20pct_effort": "positive recall in the smallest ranked prefix reaching at least 20% of nonnegative total_nloc/nloc effort",
            "aucec": "trapezoidal area under cumulative positive recall versus cumulative nonnegative LOC-effort fraction",
            "temporal_reference": "first 80% versus final 20% mapped unique SHAs ordered by timestamp and SHA; LOPO and WPDP share late test rows",
            "mean_feature_shift": "mean absolute target-source feature-mean difference in source-standard-deviation units after source-median imputation",
            "harmonized_sensitivity": "fixed-family/fixed-hyperparameter replay with aligned NLOC, token-count, and complexity descriptors only; residual confounding remains",
            "conflict_cleaning": "post-hoc removal from both classes of every identity key exported in both labels",
            "causal_claims": "none; transfer-boundary, feature-correlation, harmonized, and conflict-cleaning results are exploratory/descriptive",
        },
        "row_counts": {name: int(len(frame)) for name, frame in frames.items()},
    }
    write_text(path, json.dumps(json_compatible(metadata), indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    results_root = args.results_root.resolve()
    output_root = args.output_root.resolve()
    generated_dir = output_root / "generated"
    figures_dir = output_root / "figures"
    generated_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    np.random.seed(RANDOM_SEED)
    configure_data_root(data_root)

    contexts: dict[str, GranularityContext] = {}
    for granularity in GRANULARITIES:
        logging.info("Loading %s data and recorded outputs", granularity)
        contexts[granularity] = load_context(data_root, results_root, granularity)

    label_audit = pd.DataFrame(
        [row for granularity in GRANULARITIES for row in label_audit_rows(contexts[granularity])]
    )
    go_pair_frame, go_summary_frame = build_go_feature_correlation_rows(contexts)
    replay_frames = run_fixed_replays(data_root, contexts)
    adequacy = build_adequacy_sensitivity_rows(contexts)
    holm = build_holm_rows(contexts)
    transfer_boundaries = pd.DataFrame(build_transfer_boundary_rows(replay_frames["fold_diagnostics"]))

    frames: dict[str, pd.DataFrame] = {
        **replay_frames,
        "label_audit": label_audit,
        "adequacy_sensitivity": adequacy,
        "holm": holm,
        "transfer_boundaries": transfer_boundaries,
        "go_feature_correlations": go_pair_frame,
        "go_feature_summary": go_summary_frame,
        "effort_summary": summarize_effort(replay_frames["effort"]),
        "temporal_summary": summarize_temporal(replay_frames["temporal"]),
        "smote_k_summary": summarize_smote_k(replay_frames["smote_k"]),
        "harmonized_summary": summarize_harmonized(replay_frames["harmonized"]),
        "conflict_cleaned_summary": summarize_conflict_cleaning(replay_frames["conflict_cleaned"]),
        "adequacy_summary": summarize_adequacy(adequacy),
    }

    csv_outputs = {
        "verification": "diagnostic_replay_verification.csv",
        "effort": "diagnostic_effort_metrics.csv",
        "effort_summary": "diagnostic_effort_metrics_summary.csv",
        "temporal": "diagnostic_temporal_reference.csv",
        "temporal_summary": "diagnostic_temporal_reference_summary.csv",
        "label_audit": "diagnostic_label_audit.csv",
        "adequacy_sensitivity": "diagnostic_adequacy_sensitivity.csv",
        "adequacy_summary": "diagnostic_adequacy_sensitivity_summary.csv",
        "fold_diagnostics": "diagnostic_transfer_boundary_per_target.csv",
        "transfer_boundaries": "diagnostic_transfer_boundaries.csv",
        "go_feature_correlations": "diagnostic_go_feature_correlations.csv",
        "go_feature_summary": "diagnostic_go_feature_correlations_summary.csv",
        "holm": "diagnostic_holm_adjusted.csv",
        "smote_k": "diagnostic_smote_k_replay.csv",
        "smote_k_summary": "diagnostic_smote_k_replay_summary.csv",
        "harmonized": "diagnostic_harmonized_replay.csv",
        "harmonized_summary": "diagnostic_harmonized_replay_summary.csv",
        "conflict_cleaned": "diagnostic_conflict_cleaned_replay.csv",
        "conflict_cleaned_summary": "diagnostic_conflict_cleaned_replay_summary.csv",
    }
    sort_columns = {
        "verification": ["granularity", "target_project"],
        "effort": ["granularity", "target_project"],
        "temporal": ["granularity", "target_project"],
        "label_audit": ["granularity", "project"],
        "adequacy_sensitivity": ["row_threshold", "minority_threshold", "comparison", "metric"],
        "fold_diagnostics": ["granularity", "target_project"],
        "transfer_boundaries": ["granularity", "metric", "predictor"],
        "go_feature_correlations": ["granularity", "go_specific_feature", "generic_feature"],
        "go_feature_summary": ["granularity", "go_specific_feature"],
        "holm": ["comparison", "metric"],
        "smote_k": ["granularity", "smote_k", "target_project"],
        "smote_k_summary": ["granularity", "smote_k"],
        "harmonized": ["granularity", "target_project"],
        "harmonized_summary": ["granularity", "metric"],
        "conflict_cleaned": ["granularity", "target_project"],
        "conflict_cleaned_summary": ["granularity", "metric"],
        "adequacy_summary": ["comparison", "metric"],
    }
    for key, file_name in csv_outputs.items():
        write_csv(generated_dir / file_name, frames[key], sort_columns.get(key))

    write_latex_outputs(generated_dir, frames)
    write_metadata(
        generated_dir / "diagnostic_metadata.json",
        data_root,
        results_root,
        {key: frames[key] for key in csv_outputs},
    )

    output_files = sorted(path.name for path in generated_dir.glob("diagnostic_*"))
    logging.info("Generated %d diagnostic files under %s", len(output_files), generated_dir)
    for file_name in output_files:
        logging.info("  %s", file_name)


if __name__ == "__main__":
    main()
