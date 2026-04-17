from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
import time
from typing import Any
import warnings

from joblib import parallel_backend
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, ParameterGrid, StratifiedKFold
from tqdm.auto import tqdm

from data_loading import ProjectDataset, list_available_projects, load_project_dataset, prepare_features, report_to_dict
from evaluation import REPORTED_METRIC_COLUMNS, aggregate_model_results, compute_binary_classification_metrics, make_predictions_frame, summarize_numeric_series
from models import get_model_specs
from preprocessing import build_modeling_pipeline, summarize_fitted_pipeline


METRIC_COLUMNS = REPORTED_METRIC_COLUMNS
ANALYSIS_SUMMARY_METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "precision_0",
    "recall_0",
    "f1_0",
    "precision_1",
    "recall_1",
    "f1_1",
    "auc",
    "pr_auc",
    "mcc",
]
ANALYSIS_SUMMARY_DIAGNOSTIC_COLUMNS = [
    "best_inner_f1",
    "best_inner_mcc",
    "retained_feature_count",
    "dropped_constant_feature_count",
    "source_bug_ratio",
    "target_bug_ratio",
]
MATMUL_RUNTIME_WARNING_MESSAGES = [
    "divide by zero encountered in matmul",
    "overflow encountered in matmul",
    "invalid value encountered in matmul",
]
PER_PROJECT_RESULTS_FILENAME = "per_project_results.csv"
PREDICTIONS_FILENAME = "predictions.csv"
AGGREGATED_RESULTS_FILENAME = "aggregated_results.json"
ANALYSIS_SUMMARY_FILENAME = "analysis_summary.json"
RUN_SIGNATURE_FILENAME = "run_signature.json"


def _should_suppress_matmul_runtime_warnings(model_name: str) -> bool:
    return model_name in {"logistic_regression", "voting"}


def _determine_search_n_jobs(model_name: str, requested_n_jobs: int) -> int:
    if _should_suppress_matmul_runtime_warnings(model_name):
        return 1
    return requested_n_jobs


def _determine_search_backend(model_name: str, search_n_jobs: int) -> str | None:
    if search_n_jobs == 1:
        return None
    if model_name == "random_forest":
        return None
    return "threading"


def _build_run_signature(granularity: str, config: ExperimentConfig, available_projects: list[str]) -> dict[str, Any]:
    return {
        "granularity": granularity,
        "random_seed": config.random_seed,
        "primary_metric": config.primary_metric,
        "resampling": config.resampling,
        "model_names": list(config.model_names),
        "exclude_go_metrics": config.exclude_go_metrics,
        "smote_k_neighbors": config.smote_k_neighbors,
        "save_predictions": config.save_predictions,
        "available_projects": list(available_projects),
    }


def _load_run_signature(output_dir: Path) -> dict[str, Any] | None:
    signature_path = output_dir / RUN_SIGNATURE_FILENAME
    if signature_path.exists():
        with open(signature_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    analysis_summary_path = output_dir / ANALYSIS_SUMMARY_FILENAME
    if not analysis_summary_path.exists():
        return None

    with open(analysis_summary_path, "r", encoding="utf-8") as handle:
        analysis_summary = json.load(handle)

    config = analysis_summary.get("config", {})
    projects = analysis_summary.get("dataset_info", {}).get("projects", [])
    return {
        "granularity": analysis_summary.get("granularity"),
        "random_seed": config.get("random_seed"),
        "primary_metric": config.get("primary_metric"),
        "resampling": config.get("resampling"),
        "model_names": list(config.get("model_names", [])),
        "exclude_go_metrics": config.get("exclude_go_metrics"),
        "smote_k_neighbors": config.get("smote_k_neighbors"),
        "save_predictions": config.get("save_predictions"),
        "available_projects": [project.get("project_id") for project in projects if project.get("project_id")],
    }


def _assert_compatible_run_signature(output_dir: Path, expected_signature: dict[str, Any]) -> None:
    existing_signature = _load_run_signature(output_dir)
    if existing_signature is None:
        return

    mismatches: list[str] = []
    for key, expected_value in expected_signature.items():
        if key not in existing_signature:
            continue
        if existing_signature[key] != expected_value:
            mismatches.append(f"{key}: existing={existing_signature[key]!r}, current={expected_value!r}")

    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise ValueError(
            "Existing LOPO outputs are not compatible with the current experiment settings. "
            f"Use a different output_root or clean '{output_dir}'. Mismatches: {mismatch_text}"
        )


def _write_run_signature(output_dir: Path, run_signature: dict[str, Any]) -> None:
    with open(output_dir / RUN_SIGNATURE_FILENAME, "w", encoding="utf-8") as handle:
        json.dump(_json_compatible(run_signature), handle, indent=2, ensure_ascii=False)


def _load_existing_per_project_results(output_dir: Path, granularity: str, model_names: list[str]) -> pd.DataFrame:
    results_path = output_dir / PER_PROJECT_RESULTS_FILENAME
    if not results_path.exists():
        return pd.DataFrame()

    try:
        existing_results = pd.read_csv(results_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

    if existing_results.empty:
        return existing_results
    if "granularity" in existing_results.columns:
        existing_results = existing_results[existing_results["granularity"] == granularity].copy()
    if "model_name" in existing_results.columns:
        existing_results = existing_results[existing_results["model_name"].isin(model_names)].copy()
    if {"target_project", "model_name"}.issubset(existing_results.columns):
        existing_results = existing_results.drop_duplicates(subset=["target_project", "model_name"], keep="last")
    return existing_results.reset_index(drop=True)


def _load_existing_predictions(output_dir: Path) -> pd.DataFrame:
    predictions_path = output_dir / PREDICTIONS_FILENAME
    if not predictions_path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(predictions_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _append_prediction_checkpoint(output_dir: Path, prediction_frame: pd.DataFrame) -> None:
    predictions_path = output_dir / PREDICTIONS_FILENAME
    prediction_frame.to_csv(predictions_path, mode="a", header=not predictions_path.exists(), index=False)


@dataclass(frozen=True)
class ExperimentConfig:
    output_root: Path
    random_seed: int
    n_jobs: int
    primary_metric: str
    resampling: str
    model_names: list[str]
    exclude_go_metrics: bool
    smote_k_neighbors: int
    save_predictions: bool


@dataclass
class GranularityRunResult:
    granularity: str
    output_dir: Path
    per_project_results: pd.DataFrame
    predictions: pd.DataFrame
    aggregated_results: dict[str, Any]
    selection_results: pd.DataFrame
    selection_summary: dict[str, Any]
    posthoc_best_model_name: str | None


def _json_compatible(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return _json_compatible(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return _json_compatible(value.to_dict())
    if isinstance(value, np.ndarray):
        return _json_compatible(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if pd.isna(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _build_analysis_summary(
    granularity: str,
    config: ExperimentConfig,
    available_projects: list[str],
    data_quality_rows: list[dict[str, Any]],
    per_project_results_df: pd.DataFrame,
    aggregated_model_results: list[dict[str, Any]],
    predictions_df: pd.DataFrame,
    selection_summary: dict[str, Any],
    posthoc_best_model_name: str | None,
) -> dict[str, Any]:
    project_summaries: list[dict[str, Any]] = []
    total_rows = 0
    total_bug_count = 0
    total_non_bug_count = 0
    for report in data_quality_rows:
        row_count = int(report["row_count"])
        bug_count = int(report["bug_count"])
        non_bug_count = int(report["non_bug_count"])
        total_rows += row_count
        total_bug_count += bug_count
        total_non_bug_count += non_bug_count
        enriched_report = {
            **report,
            "bug_ratio": (bug_count / row_count) if row_count else None,
        }
        project_summaries.append(_json_compatible(enriched_report))

    aggregated_lookup = {item["model_name"]: item for item in aggregated_model_results}
    model_summaries: dict[str, Any] = {}
    for model_name in config.model_names:
        all_model_rows = per_project_results_df[per_project_results_df["model_name"] == model_name].copy()
        ok_model_rows = all_model_rows[all_model_rows["status"] == "ok"].copy()

        summary_metrics: dict[str, Any] = {}
        for metric_name in ANALYSIS_SUMMARY_METRIC_COLUMNS:
            values = ok_model_rows[metric_name] if metric_name in ok_model_rows else pd.Series(dtype=float)
            summary_metrics[metric_name] = _json_compatible(summarize_numeric_series(values))

        diagnostic_summaries: dict[str, Any] = {}
        for column_name in ANALYSIS_SUMMARY_DIAGNOSTIC_COLUMNS:
            values = ok_model_rows[column_name] if column_name in ok_model_rows else pd.Series(dtype=float)
            diagnostic_summaries[column_name] = _json_compatible(summarize_numeric_series(values))

        per_target_records = all_model_rows.where(pd.notna(all_model_rows), None).to_dict(orient="records")
        error_messages = sorted(
            {
                str(message)
                for message in all_model_rows.get("error_message", pd.Series(dtype=object)).tolist()
                if message
            }
        )

        model_summaries[model_name] = {
            "is_best_model": model_name == posthoc_best_model_name,
            "requested_search_n_jobs": config.n_jobs,
            "effective_search_n_jobs": _determine_search_n_jobs(model_name, config.n_jobs),
            "target_project_count": int(all_model_rows["target_project"].nunique()) if not all_model_rows.empty else 0,
            "successful_target_project_count": int(ok_model_rows["target_project"].nunique()) if not ok_model_rows.empty else 0,
            "error_target_project_count": int((all_model_rows["status"] == "error").sum()) if not all_model_rows.empty else 0,
            "summary_metrics": summary_metrics,
            "diagnostics": diagnostic_summaries,
            "aggregated_results_row": _json_compatible(aggregated_lookup.get(model_name)),
            "error_messages": error_messages,
            "per_target_results": _json_compatible(per_target_records),
        }

    return {
        "granularity": granularity,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "random_seed": config.random_seed,
            "n_jobs": config.n_jobs,
            "primary_metric": config.primary_metric,
            "resampling": config.resampling,
            "model_names": list(config.model_names),
            "exclude_go_metrics": config.exclude_go_metrics,
            "smote_k_neighbors": config.smote_k_neighbors,
            "save_predictions": config.save_predictions,
        },
        "evaluation_protocol": {
            "outer_loop": "leave-one-project-out",
            "fold_unit": "target_project",
            "fold_count": len(available_projects),
            "outer_fold_model_selection": selection_summary.get("selection_rule"),
        },
        "dataset_info": {
            "project_count": len(available_projects),
            "total_rows": total_rows,
            "total_bug_count": total_bug_count,
            "total_non_bug_count": total_non_bug_count,
            "projects": project_summaries,
        },
        "reported_metrics": ANALYSIS_SUMMARY_METRIC_COLUMNS,
        "best_model_name": posthoc_best_model_name,
        "best_model_selection_rule": f"post-hoc highest mean held-out {config.primary_metric}; descriptive only",
        "prediction_output": {
            "saved": config.save_predictions,
            "row_count": int(len(predictions_df)),
            "file_name": "predictions.csv" if config.save_predictions else None,
        },
        "nested_selection": selection_summary,
        "models": model_summaries,
    }


def _validate_feature_alignment(X_train: pd.DataFrame, X_test: pd.DataFrame, target_project: str) -> pd.DataFrame:
    train_columns = list(X_train.columns)
    test_columns = list(X_test.columns)
    missing_columns = sorted(set(train_columns) - set(test_columns))
    extra_columns = sorted(set(test_columns) - set(train_columns))
    if missing_columns or extra_columns:
        raise ValueError(
            "Train/test feature schema mismatch for target "
            f"'{target_project}': missing_in_test={missing_columns}, extra_in_test={extra_columns}"
        )
    return X_test[train_columns]


def _selection_rule_text(primary_metric: str, secondary_metric: str) -> str:
    return (
        "For each held-out target project, choose the model with the best inner-CV "
        f"{primary_metric}; tie-break with inner-CV {secondary_metric}, then config order."
    )


def _select_nested_model_rows(per_project_results_df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    successful_rows = per_project_results_df[per_project_results_df["status"] == "ok"].copy()
    if successful_rows.empty:
        return pd.DataFrame()

    primary_column = f"best_inner_{config.primary_metric}"
    if primary_column not in successful_rows.columns:
        raise ValueError(f"Unsupported primary metric for nested selection: {config.primary_metric}")

    secondary_metric = "mcc" if config.primary_metric != "mcc" else "f1"
    secondary_column = f"best_inner_{secondary_metric}"
    model_order = {model_name: index for index, model_name in enumerate(config.model_names)}

    successful_rows["__model_order"] = successful_rows["model_name"].map(model_order).fillna(len(model_order)).astype(int)
    sort_columns = ["target_project", primary_column]
    ascending = [True, False]
    if secondary_column in successful_rows.columns:
        sort_columns.append(secondary_column)
        ascending.append(False)
    sort_columns.append("__model_order")
    ascending.append(True)

    selected_rows = (
        successful_rows.sort_values(by=sort_columns, ascending=ascending, kind="mergesort")
        .drop_duplicates(subset=["target_project"], keep="first")
        .copy()
    )
    selected_rows["selection_metric"] = config.primary_metric
    selected_rows["selection_rule"] = _selection_rule_text(config.primary_metric, secondary_metric)
    return selected_rows.drop(columns=["__model_order"])


def _summarize_nested_selection(selected_rows: pd.DataFrame, config: ExperimentConfig) -> dict[str, Any]:
    secondary_metric = "mcc" if config.primary_metric != "mcc" else "f1"
    model_counts = {
        model_name: int(selected_rows["model_name"].eq(model_name).sum()) if not selected_rows.empty else 0
        for model_name in config.model_names
    }

    summary_metrics: dict[str, Any] = {}
    for metric_name in ANALYSIS_SUMMARY_METRIC_COLUMNS:
        values = selected_rows[metric_name] if metric_name in selected_rows else pd.Series(dtype=float)
        summary_metrics[metric_name] = _json_compatible(summarize_numeric_series(values))

    diagnostic_summaries: dict[str, Any] = {}
    for column_name in ANALYSIS_SUMMARY_DIAGNOSTIC_COLUMNS:
        values = selected_rows[column_name] if column_name in selected_rows else pd.Series(dtype=float)
        diagnostic_summaries[column_name] = _json_compatible(summarize_numeric_series(values))

    per_target_rows = selected_rows.where(pd.notna(selected_rows), None).to_dict(orient="records") if not selected_rows.empty else []

    return {
        "selection_rule": _selection_rule_text(config.primary_metric, secondary_metric),
        "selection_metric": config.primary_metric,
        "secondary_tie_breaker": secondary_metric,
        "target_project_count": int(selected_rows["target_project"].nunique()) if not selected_rows.empty else 0,
        "selected_model_counts": model_counts,
        "summary_metrics": summary_metrics,
        "diagnostics": diagnostic_summaries,
        "per_target_results": _json_compatible(per_target_rows),
    }


def _build_inner_cv(groups: pd.Series, y_train: pd.Series, random_seed: int) -> tuple[Any, str]:
    unique_group_count = groups.nunique()
    if unique_group_count >= 2:
        return LeaveOneGroupOut(), "leave-one-source-project-out"

    class_counts = y_train.value_counts()
    if class_counts.empty or class_counts.min() < 2:
        raise ValueError("Inner CV fallback requires at least two samples in each class.")

    n_splits = min(3, int(class_counts.min()))
    return StratifiedKFold(n_splits=max(2, n_splits), shuffle=True, random_state=random_seed), "stratified-kfold-fallback"


def _fit_single_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_metadata: pd.DataFrame,
    config: ExperimentConfig,
    target_project: str,
    granularity: str,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    model_spec = get_model_specs([model_name])[0]
    search_n_jobs = _determine_search_n_jobs(model_name, config.n_jobs)
    search_backend = _determine_search_backend(model_name, search_n_jobs)
    backend_label = search_backend or ("joblib-default" if search_n_jobs != 1 else "serial")
    pipeline = build_modeling_pipeline(
        estimator=model_spec.estimator_factory(config.random_seed),
        resampling_strategy=config.resampling,
        random_seed=config.random_seed,
        smote_k_neighbors=config.smote_k_neighbors,
    )
    inner_cv, inner_cv_name = _build_inner_cv(groups, y_train, config.random_seed)
    scoring = {
        "f1": "f1",
        "mcc": make_scorer(matthews_corrcoef),
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=model_spec.param_grid,
        scoring=scoring,
        refit=config.primary_metric,
        cv=inner_cv,
        n_jobs=search_n_jobs,
        error_score="raise",
    )
    suppress_matmul_warnings = _should_suppress_matmul_runtime_warnings(model_name)
    fit_backend = parallel_backend(search_backend) if search_backend else nullcontext()
    param_candidate_count = len(ParameterGrid(model_spec.param_grid))

    try:
        logging.info(
            "Model phase start: target=%s, model=%s, phase=search_fit, backend=%s, search_n_jobs=%d, param_candidates=%d, inner_cv=%s, train_rows=%d, test_rows=%d",
            target_project,
            model_name,
            backend_label,
            search_n_jobs,
            param_candidate_count,
            inner_cv_name,
            len(X_train),
            len(X_test),
        )
        fit_started_at = time.perf_counter()
        with fit_backend:
            if suppress_matmul_warnings:
                with warnings.catch_warnings():
                    for warning_message in MATMUL_RUNTIME_WARNING_MESSAGES:
                        warnings.filterwarnings("ignore", category=RuntimeWarning, message=warning_message)
                    search.fit(X_train, y_train, groups=groups)
            else:
                search.fit(X_train, y_train, groups=groups)
        fit_elapsed_seconds = time.perf_counter() - fit_started_at
        logging.info(
            "Model phase done: target=%s, model=%s, phase=search_fit, elapsed=%.2fs, best_params=%s",
            target_project,
            model_name,
            fit_elapsed_seconds,
            json.dumps(search.best_params_, ensure_ascii=False, sort_keys=True),
        )
        best_pipeline = search.best_estimator_
        preprocessing_summary = summarize_fitted_pipeline(best_pipeline, list(X_train.columns))
        logging.info(
            "Model phase start: target=%s, model=%s, phase=predict_proba, test_rows=%d",
            target_project,
            model_name,
            len(X_test),
        )
        predict_started_at = time.perf_counter()
        if suppress_matmul_warnings:
            with warnings.catch_warnings():
                for warning_message in MATMUL_RUNTIME_WARNING_MESSAGES:
                    warnings.filterwarnings("ignore", category=RuntimeWarning, message=warning_message)
                y_prob = best_pipeline.predict_proba(X_test)[:, 1]
        else:
            y_prob = best_pipeline.predict_proba(X_test)[:, 1]
        predict_elapsed_seconds = time.perf_counter() - predict_started_at
        logging.info(
            "Model phase done: target=%s, model=%s, phase=predict_proba, elapsed=%.2fs",
            target_project,
            model_name,
            predict_elapsed_seconds,
        )
        metrics = compute_binary_classification_metrics(y_test.to_numpy(), y_prob)
        prediction_frame = make_predictions_frame(
            target_metadata=target_metadata,
            granularity=granularity,
            model_name=model_name,
            y_true=y_test.to_numpy(),
            y_prob=y_prob,
        )

        result_row = {
            "target_project": target_project,
            "model_name": model_name,
            "status": "ok",
            "inner_cv_strategy": inner_cv_name,
            "best_params": json.dumps(search.best_params_, ensure_ascii=False, sort_keys=True),
            "best_inner_f1": float(search.cv_results_["mean_test_f1"][search.best_index_]),
            "best_inner_mcc": float(search.cv_results_["mean_test_mcc"][search.best_index_]),
            "retained_feature_count": len(preprocessing_summary.retained_features),
            "dropped_constant_feature_count": len(preprocessing_summary.dropped_constant_features),
            "dropped_constant_features": ";".join(preprocessing_summary.dropped_constant_features),
            "error_message": "",
        }
        result_row.update(metrics)
        return result_row, prediction_frame
    except Exception as exc:
        logging.exception("LOPO fit failed for target=%s, model=%s", target_project, model_name)
        error_row = {
            "target_project": target_project,
            "model_name": model_name,
            "status": "error",
            "inner_cv_strategy": inner_cv_name,
            "best_params": "",
            "best_inner_f1": np.nan,
            "best_inner_mcc": np.nan,
            "retained_feature_count": 0,
            "dropped_constant_feature_count": 0,
            "dropped_constant_features": "",
            "error_message": str(exc),
        }
        for metric in METRIC_COLUMNS:
            error_row[metric] = np.nan
        return error_row, None


def _build_output_dir(output_root: Path, granularity: str) -> Path:
    output_dir = output_root / granularity
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _write_granularity_outputs(
    output_dir: Path,
    granularity: str,
    config: ExperimentConfig,
    available_projects: list[str],
    data_quality_rows: list[dict[str, Any]],
    per_project_results_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, dict[str, Any], str | None]:
    predictions_df = _load_existing_predictions(output_dir) if config.save_predictions else pd.DataFrame()

    valid_rows = per_project_results_df[per_project_results_df["status"] == "ok"].copy()
    aggregated_model_results = aggregate_model_results(valid_rows, METRIC_COLUMNS)
    posthoc_best_model_name = None
    if aggregated_model_results:
        posthoc_best_model_name = max(aggregated_model_results, key=lambda item: item[f"mean_{config.primary_metric}"])["model_name"]

    selection_results_df = _select_nested_model_rows(per_project_results_df, config)
    selection_summary = _summarize_nested_selection(selection_results_df, config)

    analysis_summary = _build_analysis_summary(
        granularity=granularity,
        config=config,
        available_projects=available_projects,
        data_quality_rows=data_quality_rows,
        per_project_results_df=per_project_results_df,
        aggregated_model_results=aggregated_model_results,
        predictions_df=predictions_df,
        selection_summary=selection_summary,
        posthoc_best_model_name=posthoc_best_model_name,
    )

    aggregated_results = {
        "granularity": granularity,
        "random_seed": config.random_seed,
        "primary_metric": config.primary_metric,
        "resampling": config.resampling,
        "reported_metrics": ANALYSIS_SUMMARY_METRIC_COLUMNS,
        "models": aggregated_model_results,
        "best_model_name": posthoc_best_model_name,
        "best_model_selection_rule": f"post-hoc highest mean held-out {config.primary_metric}; descriptive only",
        "nested_selection": selection_summary,
        "project_count": len(available_projects),
        "data_quality_reports": data_quality_rows,
        "analysis_summary_file": ANALYSIS_SUMMARY_FILENAME,
    }

    per_project_results_df.to_csv(output_dir / PER_PROJECT_RESULTS_FILENAME, index=False)
    if config.save_predictions:
        predictions_df.to_csv(output_dir / PREDICTIONS_FILENAME, index=False)
    with open(output_dir / AGGREGATED_RESULTS_FILENAME, "w", encoding="utf-8") as handle:
        json.dump(_json_compatible(aggregated_results), handle, indent=2, ensure_ascii=False)
    with open(output_dir / ANALYSIS_SUMMARY_FILENAME, "w", encoding="utf-8") as handle:
        json.dump(_json_compatible(analysis_summary), handle, indent=2, ensure_ascii=False)

    return predictions_df, aggregated_results, selection_results_df, selection_summary, posthoc_best_model_name


def run_lopo_experiment(granularity: str, config: ExperimentConfig) -> GranularityRunResult:
    logging.info("Running LOPO baseline for granularity '%s'", granularity)
    output_dir = _build_output_dir(config.output_root, granularity)
    available_projects = list_available_projects(granularity)
    if len(available_projects) < 2:
        raise ValueError(f"Granularity '{granularity}' needs at least two projects for LOPO.")

    run_signature = _build_run_signature(granularity, config, available_projects)
    _assert_compatible_run_signature(output_dir, run_signature)
    _write_run_signature(output_dir, run_signature)

    loaded_projects: dict[str, ProjectDataset] = {}
    data_quality_rows: list[dict[str, Any]] = []
    for project_id in available_projects:
        dataset = load_project_dataset(project_id, granularity, exclude_go_metrics=config.exclude_go_metrics)
        loaded_projects[project_id] = dataset
        data_quality_rows.append(report_to_dict(dataset.report))

    data_quality_df = pd.DataFrame(data_quality_rows)
    data_quality_df.to_csv(output_dir / "data_quality_report.csv", index=False)

    existing_results_df = _load_existing_per_project_results(output_dir, granularity, config.model_names)
    per_project_results_df = existing_results_df.copy()
    completed_pairs: set[tuple[str, str]] = set()
    if not existing_results_df.empty and {"target_project", "model_name"}.issubset(existing_results_df.columns):
        completed_pairs = {
            (str(target_project), str(model_name))
            for target_project, model_name in existing_results_df[["target_project", "model_name"]].itertuples(index=False, name=None)
        }

    total_steps = len(available_projects) * len(config.model_names)
    completed_steps = len(completed_pairs)
    if completed_steps:
        logging.info(
            "Resuming granularity '%s' from checkpoint: %d/%d target-model evaluations already recorded",
            granularity,
            completed_steps,
            total_steps,
        )

    if completed_steps == total_steps:
        logging.info("Granularity '%s' is already complete. Reusing saved per-project results.", granularity)
        predictions_df, aggregated_results, selection_results_df, selection_summary, posthoc_best_model_name = _write_granularity_outputs(
            output_dir=output_dir,
            granularity=granularity,
            config=config,
            available_projects=available_projects,
            data_quality_rows=data_quality_rows,
            per_project_results_df=per_project_results_df,
        )
        return GranularityRunResult(
            granularity=granularity,
            output_dir=output_dir,
            per_project_results=per_project_results_df,
            predictions=predictions_df,
            aggregated_results=aggregated_results,
            selection_results=selection_results_df,
            selection_summary=selection_summary,
            posthoc_best_model_name=posthoc_best_model_name,
        )

    progress_bar = tqdm(
        total=total_steps,
        initial=completed_steps,
        desc=f"{granularity} LOPO",
        unit="model",
        leave=True,
    )

    try:
        for target_project in available_projects:
            pending_models = [model_name for model_name in config.model_names if (target_project, model_name) not in completed_pairs]
            if not pending_models:
                continue

            source_projects = [project_id for project_id in available_projects if project_id != target_project]
            source_df = pd.concat([loaded_projects[project_id].data for project_id in source_projects], ignore_index=True)
            target_df = loaded_projects[target_project].data.copy().reset_index(drop=True)

            X_train, y_train, train_metadata = prepare_features(source_df, granularity, exclude_go_metrics=config.exclude_go_metrics)
            X_test, y_test, test_metadata = prepare_features(target_df, granularity, exclude_go_metrics=config.exclude_go_metrics)
            X_test = _validate_feature_alignment(X_train, X_test, target_project)
            train_groups = train_metadata["project_id"]
            source_bug_count = int(y_train.sum())
            source_non_bug_count = int((y_train == 0).sum())
            target_bug_count = int(y_test.sum())
            target_non_bug_count = int((y_test == 0).sum())

            logging.info(
                "Outer LOPO fold: target=%s, source_projects=%d, source_rows=%d, target_rows=%d",
                target_project,
                len(source_projects),
                len(source_df),
                len(target_df),
            )

            for model_name in pending_models:
                logging.info(
                    "Evaluating target=%s, model=%s, search_n_jobs=%d",
                    target_project,
                    model_name,
                    _determine_search_n_jobs(model_name, config.n_jobs),
                )
                progress_bar.set_postfix(target=target_project, model=model_name)
                model_started_at = time.perf_counter()
                result_row, prediction_frame = _fit_single_model(
                    model_name=model_name,
                    X_train=X_train,
                    y_train=y_train,
                    groups=train_groups,
                    X_test=X_test,
                    y_test=y_test,
                    target_metadata=test_metadata,
                    config=config,
                    target_project=target_project,
                    granularity=granularity,
                )
                result_row["granularity"] = granularity
                result_row["source_project_count"] = len(source_projects)
                result_row["source_row_count"] = len(source_df)
                result_row["source_bug_count"] = source_bug_count
                result_row["source_non_bug_count"] = source_non_bug_count
                result_row["source_bug_ratio"] = float(y_train.mean()) if len(y_train) else np.nan
                result_row["target_row_count"] = len(target_df)
                result_row["target_bug_count"] = target_bug_count
                result_row["target_non_bug_count"] = target_non_bug_count
                result_row["target_bug_ratio"] = float(y_test.mean()) if len(y_test) else np.nan
                result_row["resampling"] = config.resampling
                per_project_results_df = pd.concat([per_project_results_df, pd.DataFrame([result_row])], ignore_index=True)
                per_project_results_df.to_csv(output_dir / PER_PROJECT_RESULTS_FILENAME, index=False)
                if config.save_predictions and prediction_frame is not None:
                    _append_prediction_checkpoint(output_dir, prediction_frame)
                completed_pairs.add((target_project, model_name))
                completed_steps = len(completed_pairs)
                elapsed_seconds = time.perf_counter() - model_started_at
                logging.info(
                    "Completed step %d/%d for granularity '%s': target=%s, model=%s, status=%s, elapsed=%.2fs, checkpoint=%s",
                    completed_steps,
                    total_steps,
                    granularity,
                    target_project,
                    model_name,
                    result_row.get("status", "unknown"),
                    elapsed_seconds,
                    output_dir / PER_PROJECT_RESULTS_FILENAME,
                )
                progress_bar.update(1)
    except KeyboardInterrupt:
        logging.warning(
            "Interrupted granularity '%s' after saving %d/%d target-model results to %s",
            granularity,
            len(per_project_results_df),
            total_steps,
            output_dir / PER_PROJECT_RESULTS_FILENAME,
        )
        raise
    finally:
        progress_bar.close()

    predictions_df, aggregated_results, selection_results_df, selection_summary, posthoc_best_model_name = _write_granularity_outputs(
        output_dir=output_dir,
        granularity=granularity,
        config=config,
        available_projects=available_projects,
        data_quality_rows=data_quality_rows,
        per_project_results_df=per_project_results_df,
    )

    return GranularityRunResult(
        granularity=granularity,
        output_dir=output_dir,
        per_project_results=per_project_results_df,
        predictions=predictions_df,
        aggregated_results=aggregated_results,
        selection_results=selection_results_df,
        selection_summary=selection_summary,
        posthoc_best_model_name=posthoc_best_model_name,
    )