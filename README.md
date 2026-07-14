# CPDP in Go: Experiment and Analysis Code

This repository is the code-only `v2.0` artifact for strict leave-one-project-out (LOPO) cross-project defect prediction experiments on Go repositories. It contains experiment modules, configurations, and analysis scripts.

The repository intentionally does not contain raw datasets, generated result directories, generated tables or figures, manuscript material, submission files, or PDFs.

## Repository layout

- `run_experiment.py`: main entry point for commit-, file-, and method-level LOPO runs.
- `data_loading.py`, `feature_schema.py`, `preprocessing.py`, `models.py`, `evaluation.py`, `lopo_runner.py`, and `stats.py`: experiment implementation.
- `configs/`: baseline, feature-ablation, resampling, and SMOTE-neighborhood configurations.
- `analysis/`: scripts that summarize experiment outputs and run sensitivity or diagnostic analyses.
- `tests/`: lightweight implementation checks.

## Environment

Python 3.12 and `uv` are used for all commands:

```bash
uv sync
uv run pytest -q
```

No virtual-environment activation is required.

## Dataset preparation

The raw GoBug CSV exports are not redistributed. Obtain them from IEEE Dataport using DOI [`10.21227/bk5q-fs89`](https://dx.doi.org/10.21227/bk5q-fs89), then retain the released project folders under:

```text
commit_data/<project>/bugs.csv
commit_data/<project>/non_bugs.csv
file_data/<project>/file_bug_metrics.csv
file_data/<project>/file_non_bug_metrics.csv
method_data/<project>/method_bug_metrics.csv
method_data/<project>/method_non_bug_metrics.csv
```

## Experiment commands

Run the main strict-LOPO baseline:

```bash
uv run python run_experiment.py --config configs/default.yaml
```

Run one granularity:

```bash
uv run python run_experiment.py --config configs/default.yaml --granularity commit
```

Run the matched feature-removal conditions:

```bash
uv run python run_experiment.py --config configs/no_go_metrics.yaml
uv run python run_experiment.py --config configs/no_go_metrics_method.yaml
```

Run resampling and SMOTE-neighborhood sensitivities:

```bash
uv run python run_experiment.py --config configs/resampling_random_over.yaml
uv run python run_experiment.py --config configs/resampling_none.yaml
uv run python run_experiment.py --config configs/smote_k5.yaml
```

Each experiment writes its CSV, JSON, checkpoint, and log files to the `output_root` declared in its configuration. These generated directories are ignored by Git.

## Analysis commands

Summarize the main LOPO results and generate statistical tables and figures:

```bash
uv run python analysis/summarize_lopo_results.py \
  --results-root results_lopo_baseline \
  --output-root analysis_output
```

Analyze the matched Go-feature ablation:

```bash
uv run python analysis/analyze_go_feature_ablation.py \
  --full-results-root results_lopo_baseline \
  --no-go-results-root results_lopo_baseline_no_go_metrics_matched \
  --output-root analysis_output
```

Analyze resampling and selection-objective sensitivity:

```bash
uv run python analysis/analyze_robustness.py \
  --main-results-root results_lopo_baseline \
  --random-over-results-root results_lopo_baseline_resampling_random_over \
  --no-resampling-results-root results_lopo_baseline_no_resampling \
  --output-root analysis_output
```

Compare the complete nested SMOTE `k=1` and `k=5` runs:

```bash
uv run python analysis/compare_smote_k5.py \
  --baseline-root results_lopo_baseline \
  --k5-root results_lopo_baseline_smote_k5 \
  --output-root analysis_output/generated
```

Run the deterministic label, temporal-reference, effort-aware, support-threshold, transfer-boundary, feature-correlation, harmonization, and conflict-cleaning diagnostics:

```bash
uv run python analysis/generate_diagnostics.py \
  --data-root . \
  --results-root results_lopo_baseline \
  --output-root analysis_output
```

Analysis outputs are written under `analysis_output/`, which is ignored by Git.

## Methodological constraints

- Target-project data remain excluded from fitted preprocessing, hyperparameter tuning, model-family selection, and model fitting.
- Commit, file, and method runs remain separate.
- Statistical comparisons use fold-local model selections; post-hoc best-model summaries are descriptive only.
- Resampling is applied only inside training-side pipelines.
- The method-level random-forest search retains joblib's default process backend.
