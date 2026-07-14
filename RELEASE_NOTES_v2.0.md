# v2.0

Version 2.0 is a result-complete experiment and analysis release.

It includes:

- strict LOPO experiment code for commit, file, and method configurations;
- baseline, matched feature-removal, resampling, and SMOTE `k=5` configurations;
- the frozen CSV/JSON outputs for the primary baseline and four sensitivity experiment roots;
- neutral analysis scripts for main-result summaries, paired statistical comparisons, robustness checks, label audits, support thresholds, temporal references, effort-aware measures, transfer-boundary diagnostics, and feature analyses;
- machine-readable CSV/JSON analysis summaries and diagnostics under `analysis_results/`;
- a SHA-256 inventory for the committed result snapshot;
- a locked `uv` environment and implementation tests.

It deliberately excludes raw datasets, manuscript or submission material, TeX sources, rendered tables or figures, and PDFs.
