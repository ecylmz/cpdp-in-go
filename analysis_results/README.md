# Machine-readable analysis results

This directory contains the frozen CSV and JSON analysis snapshot derived from the committed strict-LOPO experiment outputs. It contains data products only; formatted report tables and figures are intentionally excluded.

The files are grouped by prefix:

- `lopo_granularity_*`: primary cross-granularity summaries, bootstrap confidence intervals, paired Wilcoxon and Cliff's delta comparisons, adequacy, prevalence, variability, and rank diagnostics.
- `lopo_go_metrics_*`: matched fixed-family Go-specific feature-ablation summaries.
- `lopo_robustness_*`: resampling and selection-objective sensitivity summaries.
- `diagnostic_*`: deterministic replay verification, label audits, support-threshold sensitivity, effort-aware measures, temporal references, transfer-boundary analyses, feature summaries, harmonized-feature and conflict-cleaned replays, and multiplicity-adjusted tests.
- `smote_k5_comparison*`: complete nested SMOTE `k=1` versus `k=5` comparison.
- `software_environment.csv`: verified software versions used for the analysis snapshot.

Primary statistical comparisons use fold-local model selections. Best-single-model summaries are descriptive and are not used for inferential claims. Transfer-boundary, correlation, harmonization, and conflict-cleaning analyses are exploratory or descriptive rather than causal.

The corresponding generation commands are documented in the repository root `README.md`. Diagnostics that read raw observations require the separately obtained GoBug dataset.
