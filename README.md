# CPDP in Go: Reproducible Experiment Artifact

This repository is the `v2.0` reproducibility artifact for strict leave-one-project-out (LOPO) cross-project defect prediction experiments on Go repositories. It contains the experiment and analysis code together with the frozen machine-readable result snapshot.

The repository intentionally does not redistribute the raw GoBug dataset and does not contain manuscript or submission material, TeX sources, or PDFs.

## Repository layout

- `run_experiment.py`: main entry point for commit-, file-, and method-level LOPO runs.
- `data_loading.py`, `feature_schema.py`, `preprocessing.py`, `models.py`, `evaluation.py`, `lopo_runner.py`, and `stats.py`: experiment implementation.
- `configs/`: baseline, feature-ablation, resampling, and SMOTE-neighborhood configurations.
- `analysis/`: scripts that summarize experiment outputs and run sensitivity or diagnostic analyses.
- `results_lopo_baseline/`: primary commit-, file-, and method-level strict-LOPO outputs.
- `results_lopo_baseline_no_go_metrics_matched/`: matched fixed-family Go-feature ablation outputs.
- `results_lopo_baseline_no_resampling/`: no-resampling sensitivity outputs.
- `results_lopo_baseline_resampling_random_over/`: random-oversampling sensitivity outputs.
- `results_lopo_baseline_smote_k5/`: complete nested SMOTE `k=5` sensitivity outputs.
- `analysis_results/`: committed CSV/JSON statistical summaries and diagnostics derived from the result snapshot.
- `RESULTS_SHA256SUMS`: SHA-256 inventory for all committed experiment and analysis results.
- `tests/`: lightweight implementation checks.

## Environment

Python 3.12 and `uv` are used for all commands. Runtime dependencies are pinned to the verified analysis environment in `pyproject.toml` and `uv.lock`:

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

Each experiment writes its CSV, JSON, checkpoint, and log files to the `output_root` declared in its configuration. The five result roots committed here are the frozen `v2.0` snapshot; reruns can write to separate paths by changing `output_root` in the configuration.

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

Record the executed software environment:

```bash
uv run python analysis/capture_environment.py \
  --output analysis_output/generated/software_environment.csv
```

Analysis reruns are written under `analysis_output/`, which is ignored by Git. The release snapshot of the machine-readable CSV/JSON outputs is committed under `analysis_results/`. TeX tables and rendered figures are deliberately not tracked.

Verify the committed result snapshot with:

```bash
shasum -a 256 -c RESULTS_SHA256SUMS
```

## Methodological constraints

- Target-project data remain excluded from fitted preprocessing, hyperparameter tuning, model-family selection, and model fitting.
- Commit, file, and method runs remain separate.
- Statistical comparisons use fold-local model selections; post-hoc best-model summaries are descriptive only.
- Resampling is applied only inside training-side pipelines.
- The method-level random-forest search retains joblib's default process backend.
