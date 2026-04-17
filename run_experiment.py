from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from lopo_runner import ExperimentConfig, GranularityRunResult, run_lopo_experiment
from stats import build_granularity_comparison_rows, run_pairwise_granularity_tests


ALL_GRANULARITIES = ["commit", "file", "method"]
SUPPORTED_PRIMARY_METRICS = {"f1", "mcc"}
PACKAGE_ROOT = Path(__file__).resolve().parent


def resolve_output_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PACKAGE_ROOT / path


def resolve_config_path(path_value: str | Path | None) -> Path:
    if path_value is None:
        return PACKAGE_ROOT / "configs" / "default.yaml"

    path = Path(path_value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return PACKAGE_ROOT / path


def setup_logging(output_root: Path) -> tuple[Path, Path]:
    output_log_dir = output_root / "logs"
    output_log_dir.mkdir(parents=True, exist_ok=True)
    output_log_path = output_log_dir / "run_experiment.log"

    repo_log_dir = PACKAGE_ROOT / "log"
    repo_log_dir.mkdir(parents=True, exist_ok=True)
    repo_log_path = repo_log_dir / "run_experiment.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(output_log_path, encoding="utf-8"),
            logging.FileHandler(repo_log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logging.captureWarnings(True)
    return output_log_path, repo_log_path


def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        loaded_config = yaml.safe_load(handle) or {}
    return loaded_config


def build_experiment_config(raw_config: dict[str, Any]) -> ExperimentConfig:
    output_root = resolve_output_path(raw_config.get("output_root", "results_lopo_baseline"))
    output_root.mkdir(parents=True, exist_ok=True)
    random_seed = int(raw_config.get("random_seed", 42))
    n_jobs = int(raw_config.get("n_jobs", 4))
    primary_metric = raw_config.get("primary_metric", "f1")
    if primary_metric not in SUPPORTED_PRIMARY_METRICS:
        raise ValueError(
            f"Unsupported primary_metric '{primary_metric}'. Supported values: {sorted(SUPPORTED_PRIMARY_METRICS)}"
        )
    resampling = raw_config.get("resampling", "smote")
    model_names = list(raw_config.get("models", ["naive_bayes", "logistic_regression", "random_forest", "xgboost", "voting"]))
    exclude_go_metrics = bool(raw_config.get("exclude_go_metrics", False))
    smote_k_neighbors = int(raw_config.get("smote_k_neighbors", 1))
    save_predictions = bool(raw_config.get("save_predictions", False))

    return ExperimentConfig(
        output_root=output_root,
        random_seed=random_seed,
        n_jobs=n_jobs,
        primary_metric=primary_metric,
        resampling=resampling,
        model_names=model_names,
        exclude_go_metrics=exclude_go_metrics,
        smote_k_neighbors=smote_k_neighbors,
        save_predictions=save_predictions,
    )


def determine_granularities(cli_granularity: str | None, raw_config: dict[str, Any]) -> list[str]:
    granularity = cli_granularity or raw_config.get("granularity", "all")
    if granularity == "all":
        return ALL_GRANULARITIES
    if granularity not in ALL_GRANULARITIES:
        raise ValueError(f"Unsupported granularity: {granularity}")
    return [granularity]


def save_global_outputs(output_root: Path, results_by_granularity: dict[str, GranularityRunResult]) -> None:
    comparison_rows = build_granularity_comparison_rows(results_by_granularity)
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(output_root / "granularity_comparison.csv", index=False)

    statistical_tests = run_pairwise_granularity_tests(results_by_granularity, metrics=["f1", "mcc"])
    with open(output_root / "statistical_tests.json", "w", encoding="utf-8") as handle:
        json.dump(statistical_tests, handle, indent=2, ensure_ascii=False)


def load_saved_granularity_result(output_root: Path, granularity: str) -> GranularityRunResult | None:
    output_dir = output_root / granularity
    analysis_summary_path = output_dir / "analysis_summary.json"
    aggregated_results_path = output_dir / "aggregated_results.json"
    per_project_results_path = output_dir / "per_project_results.csv"

    if not analysis_summary_path.exists() or not aggregated_results_path.exists() or not per_project_results_path.exists():
        return None

    with open(analysis_summary_path, "r", encoding="utf-8") as handle:
        analysis_summary = json.load(handle)
    with open(aggregated_results_path, "r", encoding="utf-8") as handle:
        aggregated_results = json.load(handle)

    per_project_results = pd.read_csv(per_project_results_path)
    nested_selection = analysis_summary.get("nested_selection", {})
    selection_results = pd.DataFrame(nested_selection.get("per_target_results", []))
    posthoc_best_model_name = analysis_summary.get("best_model_name")

    predictions_path = output_dir / "predictions.csv"
    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path)
    else:
        predictions = pd.DataFrame()

    return GranularityRunResult(
        granularity=granularity,
        output_dir=output_dir,
        per_project_results=per_project_results,
        predictions=predictions,
        aggregated_results=aggregated_results,
        selection_results=selection_results,
        selection_summary=nested_selection,
        posthoc_best_model_name=posthoc_best_model_name,
    )


def collect_results_for_global_outputs(
    output_root: Path,
    current_results: dict[str, GranularityRunResult],
    requested_granularities: list[str],
) -> dict[str, GranularityRunResult]:
    combined_results = dict(current_results)
    if set(requested_granularities) == set(ALL_GRANULARITIES):
        return combined_results

    for granularity in ALL_GRANULARITIES:
        if granularity in combined_results:
            continue
        saved_result = load_saved_granularity_result(output_root, granularity)
        if saved_result is not None:
            combined_results[granularity] = saved_result

    return combined_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LOPO baseline experiments for CPDP.")
    parser.add_argument("--granularity", choices=ALL_GRANULARITIES + ["all"], default=None, help="Granularity to run. Overrides config if provided.")
    parser.add_argument("--config", default=None, help="Path to YAML config file.")
    return parser


def main() -> None:
    parser = build_parser()
    cli_args = parser.parse_args()
    raw_config = load_config(resolve_config_path(cli_args.config))
    experiment_config = build_experiment_config(raw_config)
    output_log_path, repo_log_path = setup_logging(experiment_config.output_root)
    logging.info("Logging experiment output to %s", output_log_path)
    logging.info("Logging repository-level progress to %s", repo_log_path)
    granularities = determine_granularities(cli_args.granularity, raw_config)

    results_by_granularity: dict[str, GranularityRunResult] = {}
    try:
        for granularity in granularities:
            results_by_granularity[granularity] = run_lopo_experiment(granularity, experiment_config)

        results_for_global_outputs = collect_results_for_global_outputs(
            output_root=experiment_config.output_root,
            current_results=results_by_granularity,
            requested_granularities=granularities,
        )
        save_global_outputs(experiment_config.output_root, results_for_global_outputs)
        logging.info("Skipping report-asset generation in the GitHub release package; raw experiment outputs have been refreshed only.")
        logging.info("LOPO baseline experiment completed for granularities: %s", ", ".join(granularities))
    except KeyboardInterrupt:
        logging.warning(
            "LOPO baseline experiment interrupted. Completed target-model checkpoints remain on disk and the next run will resume automatically."
        )
        raise SystemExit(130)
    except Exception:
        logging.exception("LOPO baseline experiment failed.")
        raise


if __name__ == "__main__":
    main()