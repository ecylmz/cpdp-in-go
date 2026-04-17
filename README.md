# LOPO Experiment Package

[![DOI](https://zenodo.org/badge/1213636309.svg)](https://doi.org/10.5281/zenodo.19636460)

This package contains the code and raw outputs needed to run and inspect the strict leave-one-project-out (LOPO) experiments reported in the study.
It is intentionally limited to the experiment layer: it does not include manuscript-generation, LaTeX table generation, figure generation, or PDF build scripts.

## What This Package Contains

- `run_experiment.py`: main entry point for LOPO baseline runs.
- `lopo_runner.py`, `data_loading.py`, `evaluation.py`, `models.py`, `preprocessing.py`, `feature_schema.py`, `stats.py`: supporting experiment modules used by the entry point.
- `configs/`: experiment configurations for the main baseline and the no-Go ablation runs.
- `results_lopo_baseline/`: raw outputs for the main strict LOPO baseline across commit, file, and method granularity.
- `results_lopo_baseline_no_go_metrics_matched/`: raw outputs for the matched no-Go ablation reported in the paper.

## Required Setup

Use a Python environment that includes the packages used by the study. Install the dependencies from `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

If you use `uv`, you can install them with:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

The dependency set includes versions compatible with:

- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `imbalanced-learn`
- `PyYAML`
- `xgboost`
- `tqdm`
- `joblib`

If you use `uv`, you can still run the commands below with `uv run ...`.

## Dataset Preparation

This package does not bundle the GoBug dataset.
Download the released GoBug CSV exports from IEEE Dataport:

- `https://dx.doi.org/10.21227/bk5q-fs89`

After downloading, place the CSV files under the package root using the following directory layout.
Keep the original per-project folder structure and filenames.

### Commit-level data

- `commit_data/<project>/bugs.csv`
- `commit_data/<project>/non_bugs.csv`

### File-level data

- `file_data/<project>/file_bug_metrics.csv`
- `file_data/<project>/file_non_bug_metrics.csv`

### Method-level data

- `method_data/<project>/method_bug_metrics.csv`
- `method_data/<project>/method_non_bug_metrics.csv`

## How To Run The Experiments

Run commands from the package root.

### Main baseline across all granularities

```bash
uv run python run_experiment.py --config configs/default.yaml
```

### Single granularity only

```bash
uv run python run_experiment.py --config configs/default.yaml --granularity commit
```

Replace `commit` with `file`, `method`, or `all` as needed.

### Matched no-Go ablation

File and method ablation configs are provided under `configs/`.

```bash
uv run python run_experiment.py --config configs/no_go_metrics.yaml
uv run python run_experiment.py --config configs/no_go_metrics_method.yaml
```

## Configuration Files

- `configs/default.yaml`: main strict LOPO baseline for commit, file, and method.
- `configs/no_go_metrics.yaml`: matched no-Go ablation configuration for the file-level setting.
- `configs/no_go_metrics_method.yaml`: matched no-Go ablation configuration for the method-level setting.
- `configs/commit_example.yaml`: small example configuration for a commit-only run.

Each config controls the output directory, granularity selection, model list, resampling, and related experiment settings.

## Results Directory Guide

### `results_lopo_baseline/`

This is the main raw-results directory for the strict LOPO baseline.

- `commit/`, `file/`, `method/`: per-granularity experiment outputs.
- `granularity_comparison.csv`: combined summary rows used for cross-granularity comparison.
- `statistical_tests.json`: paired statistical comparison results for the selected granularity outputs.

Within each per-granularity directory you will find:

- `data_quality_report.csv`: project-level data loading and cleaning diagnostics.
- `per_project_results.csv`: held-out results for each target project and model.
- `aggregated_results.json`: aggregated descriptive summaries across held-out projects.
- `analysis_summary.json`: audit-style summary of the run, configuration, and selection behavior.
- `run_signature.json`: run identity metadata used to detect incompatible reruns.

### `results_lopo_baseline_no_go_metrics_matched/`

This directory stores the raw outputs for the matched ablation where Go-specific metrics are removed.

- `file/`, `method/`: per-granularity matched ablation outputs.
- `granularity_comparison.csv`: combined comparison summary for the matched ablation outputs.
- `statistical_tests.json`: paired statistical comparison results for the matched ablation outputs.

The file structure inside `file/` and `method/` matches the structure used in the main baseline output directories.
