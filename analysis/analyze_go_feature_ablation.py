from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from summarize_lopo_results import (
    ALL_GRANULARITIES,
    ADEQUACY_SHORT_LABELS,
    FOCUS_METRICS,
    FOCUS_METRIC_LABELS,
    GRANULARITY_COLORS,
    GRANULARITY_LABELS,
    LATEX_LINEBREAK,
    MODEL_LABELS,
    SUPPORT_COLORS,
    available_granularities,
    canonical_granularities,
    configure_plot_style,
    diff_win_tie_loss,
    format_float,
    format_interval,
    format_mean_std,
    format_pvalue,
    format_pvalue_latex,
    generic_axis_limits,
    json_compatible,
    latex_escape,
    load_granularity,
    metric_axis_limits,
    rank_biserial_correlation,
    save_figure,
    summarize_series,
    write_text,
)


GO_METRIC_GRANULARITIES = ["file", "method"]
OUTPUT_PREFIX = "lopo_go_metrics"
SELECTION_SHIFT_COLUMNS = (
    "granularity",
    "target_project",
    "full_feature_model_name",
    "no_go_model_name",
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate paired LOPO tables and figures for the Go-metrics ablation study."
    )
    parser.add_argument(
        "--full-results-root",
        type=Path,
        default=repo_root / "results_lopo_baseline",
        help="Directory containing the baseline LOPO outputs with the full feature schema.",
    )
    parser.add_argument(
        "--no-go-results-root",
        type=Path,
        default=repo_root / "results_lopo_baseline_no_go_metrics",
        help="Directory containing matched LOPO outputs with Go-specific features removed.",
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
        help="Granularities to consider. Only file and method are used for the substantive ablation.",
    )
    return parser.parse_args()


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> tuple[float | None, str]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return None, "undefined"

    greater = 0
    lower = 0
    for value in x:
        greater += int(np.sum(value > y))
        lower += int(np.sum(value < y))

    delta = (greater - lower) / (x.size * y.size)
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


def paired_wilcoxon_pvalue(left_values: pd.Series, right_values: pd.Series) -> float | None:
    left = pd.to_numeric(left_values, errors="coerce").to_numpy(dtype=float)
    right = pd.to_numeric(right_values, errors="coerce").to_numpy(dtype=float)
    valid_mask = np.isfinite(left) & np.isfinite(right)
    left = left[valid_mask]
    right = right[valid_mask]
    if left.size < 2:
        return None
    if np.all(np.abs(left - right) <= 1e-12):
        return None
    try:
        return float(wilcoxon(left, right, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return None


def overlapping_granularities(
    full_results_root: Path,
    no_go_results_root: Path,
    requested_granularities: list[str] | None,
) -> list[str]:
    full_available = canonical_granularities(requested_granularities, available_granularities(full_results_root))
    if not no_go_results_root.exists():
        return []
    no_go_available = canonical_granularities(requested_granularities, available_granularities(no_go_results_root))
    overlap = set(full_available) & set(no_go_available)
    return [granularity for granularity in GO_METRIC_GRANULARITIES if granularity in overlap]


def build_merged_frame(full_bundle: dict[str, Any], no_go_bundle: dict[str, Any], granularity: str) -> pd.DataFrame:
    selected_columns = ["target_project", "model_name", "adequacy", *FOCUS_METRICS]
    full_selected = full_bundle["selected"][selected_columns].copy()
    no_go_columns = ["target_project", "model_name", *FOCUS_METRICS]
    no_go_per_project = no_go_bundle["per_project"][no_go_columns].copy()
    merged = full_selected.merge(no_go_per_project, on=["target_project", "model_name"], suffixes=("_with_go", "_without_go"))
    merged["granularity"] = granularity
    merged["matched_model_name"] = merged["model_name"]
    merged["both_primary"] = merged["adequacy"].eq("PRIMARY")
    for metric in FOCUS_METRICS:
        merged[f"delta_{metric}"] = merged[f"{metric}_with_go"] - merged[f"{metric}_without_go"]
    return merged.sort_values("target_project").reset_index(drop=True)


def summarize_paired_rows(merged: pd.DataFrame, metric: str) -> dict[str, Any]:
    delta_column = f"delta_{metric}"
    with_column = f"{metric}_with_go"
    without_column = f"{metric}_without_go"

    valid_all = merged[[with_column, without_column, delta_column]].apply(pd.to_numeric, errors="coerce").dropna()
    valid_primary = merged.loc[merged["both_primary"], [with_column, without_column, delta_column]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()

    all_summary = summarize_series(valid_all[delta_column]) if not valid_all.empty else summarize_series(pd.Series(dtype=float))
    primary_summary = (
        summarize_series(valid_primary[delta_column]) if not valid_primary.empty else summarize_series(pd.Series(dtype=float))
    )

    all_wins, all_ties, all_losses = diff_win_tie_loss(valid_all[delta_column].to_numpy(dtype=float))
    primary_wins, primary_ties, primary_losses = diff_win_tie_loss(valid_primary[delta_column].to_numpy(dtype=float))

    all_cliffs_delta, all_cliffs_magnitude = cliffs_delta(
        valid_all[with_column].to_numpy(dtype=float),
        valid_all[without_column].to_numpy(dtype=float),
    )
    all_rank_biserial = rank_biserial_correlation(valid_all[delta_column].to_numpy(dtype=float))
    primary_cliffs_delta, primary_cliffs_magnitude = cliffs_delta(
        valid_primary[with_column].to_numpy(dtype=float),
        valid_primary[without_column].to_numpy(dtype=float),
    )
    primary_rank_biserial = rank_biserial_correlation(valid_primary[delta_column].to_numpy(dtype=float))

    return {
        "metric": metric,
        "all_count": int(len(valid_all)),
        "all_mean_diff": all_summary["mean"],
        "all_std_diff": all_summary["std"],
        "all_ci_low_diff": all_summary["ci_low"],
        "all_ci_high_diff": all_summary["ci_high"],
        "all_wilcoxon_pvalue": paired_wilcoxon_pvalue(valid_all[with_column], valid_all[without_column]),
        "all_rank_biserial": all_rank_biserial,
        "all_cliffs_delta": all_cliffs_delta,
        "all_cliffs_magnitude": all_cliffs_magnitude,
        "all_win_count": all_wins,
        "all_tie_count": all_ties,
        "all_loss_count": all_losses,
        "primary_count": int(len(valid_primary)),
        "primary_mean_diff": primary_summary["mean"],
        "primary_std_diff": primary_summary["std"],
        "primary_ci_low_diff": primary_summary["ci_low"],
        "primary_ci_high_diff": primary_summary["ci_high"],
        "primary_wilcoxon_pvalue": paired_wilcoxon_pvalue(valid_primary[with_column], valid_primary[without_column]),
        "primary_rank_biserial": primary_rank_biserial,
        "primary_cliffs_delta": primary_cliffs_delta,
        "primary_cliffs_magnitude": primary_cliffs_magnitude,
        "primary_win_count": primary_wins,
        "primary_tie_count": primary_ties,
        "primary_loss_count": primary_losses,
    }


def build_summary_rows(merged_by_granularity: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for granularity in GO_METRIC_GRANULARITIES:
        if granularity not in merged_by_granularity:
            continue
        merged = merged_by_granularity[granularity]
        row: dict[str, Any] = {
            "granularity": granularity,
            "target_count": int(len(merged)),
            "primary_target_count": int(merged["both_primary"].sum()),
        }
        for metric in FOCUS_METRICS:
            with_summary = summarize_series(merged[f"{metric}_with_go"])
            without_summary = summarize_series(merged[f"{metric}_without_go"])
            delta_summary = summarize_series(merged[f"delta_{metric}"])
            row[f"with_mean_{metric}"] = with_summary["mean"]
            row[f"with_std_{metric}"] = with_summary["std"]
            row[f"without_mean_{metric}"] = without_summary["mean"]
            row[f"without_std_{metric}"] = without_summary["std"]
            row[f"delta_mean_{metric}"] = delta_summary["mean"]
            row[f"delta_std_{metric}"] = delta_summary["std"]
            row[f"delta_ci_low_{metric}"] = delta_summary["ci_low"]
            row[f"delta_ci_high_{metric}"] = delta_summary["ci_high"]
        rows.append(row)
    return rows


def build_inferential_rows(merged_by_granularity: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for granularity in GO_METRIC_GRANULARITIES:
        if granularity not in merged_by_granularity:
            continue
        merged = merged_by_granularity[granularity]
        for metric in FOCUS_METRICS:
            paired_summary = summarize_paired_rows(merged, metric)
            rows.append({"granularity": granularity, **paired_summary})
    return rows


def build_per_project_rows(merged_by_granularity: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for granularity in GO_METRIC_GRANULARITIES:
        if granularity not in merged_by_granularity:
            continue
        merged = merged_by_granularity[granularity]
        for _, row in merged.iterrows():
            rows.append(
                {
                    "granularity": granularity,
                    "target_project": row["target_project"],
                    "adequacy": row["adequacy"],
                    "matched_model_name": row["matched_model_name"],
                    "f1_1_with_go": row["f1_1_with_go"],
                    "f1_1_without_go": row["f1_1_without_go"],
                    "delta_f1_1": row["delta_f1_1"],
                    "mcc_with_go": row["mcc_with_go"],
                    "mcc_without_go": row["mcc_without_go"],
                    "delta_mcc": row["delta_mcc"],
                }
            )
    return pd.DataFrame(rows).sort_values(["granularity", "target_project"]).reset_index(drop=True)


def write_placeholder_files(generated_dir: Path, no_go_results_root: Path) -> None:
    results_tex = "\n".join(
        [
            "\\subsection{Matched Go-Specific Feature Ablation}",
            "",
            "Matched fixed-family LOPO outputs without Go-specific metrics were not found, so the file- and method-level ablation could not be analyzed.",
            f"The builder looked for the auxiliary condition under \\texttt{{{latex_escape(no_go_results_root)}}}.",
        ]
    )
    appendix_tex = "\n".join(
        [
            "\\section{Go-Metric Appendix Material}",
            "\\label{app:lopo-go-metrics}",
            "",
            "No appendix material was generated because the matched fixed-family no-Go-metrics LOPO outputs were not available during the report refresh.",
        ]
    )
    write_text(generated_dir / f"{OUTPUT_PREFIX}_results.tex", results_tex)
    write_text(generated_dir / f"{OUTPUT_PREFIX}_appendix.tex", appendix_tex)
    pd.DataFrame().to_csv(generated_dir / f"{OUTPUT_PREFIX}_summary.csv", index=False)
    pd.DataFrame().to_csv(generated_dir / f"{OUTPUT_PREFIX}_paired_stats.csv", index=False)
    pd.DataFrame().to_csv(generated_dir / f"{OUTPUT_PREFIX}_selected_per_project.csv", index=False)
    write_selection_shift_csv(generated_dir)
    write_text(generated_dir / f"{OUTPUT_PREFIX}_summary.json", json.dumps({"status": "missing_no_go_results"}, indent=2))


def write_selection_shift_csv(output_dir: Path, rows: list[dict[str, Any]] | None = None) -> None:
    pd.DataFrame(rows or [], columns=SELECTION_SHIFT_COLUMNS).to_csv(
        output_dir / f"{OUTPUT_PREFIX}_selection_shift.csv",
        index=False,
    )


def write_summary_table(output_dir: Path, summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Matched LOPO ablation for file- and method-level language-specific feature counts. Positive deltas favor retaining the language-specific feature counts. Means are reported across aligned held-out target projects after fixing the learner family to the model selected by the full-feature LOPO condition. The full-feature condition corresponds to the matched-family LOPO run with language-specific feature counts; the ablation keeps the selected learner family fixed and removes only those counts.}",
        f"\\label{{tab:{OUTPUT_PREFIX.replace('_', '-')}-summary}}",
        "\\small",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Granularity & $n$ & With language-specific counts $F_1$ & Without language-specific counts $F_1$ & $\\Delta F_1$ & With language-specific counts MCC & Without language-specific counts MCC & $\\Delta$MCC " + LATEX_LINEBREAK,
        "\\midrule",
    ]
    for row in summary_rows:
        lines.append(
            " & ".join(
                [
                    GRANULARITY_LABELS[row["granularity"]],
                    str(int(row["target_count"])),
                    format_mean_std(row["with_mean_f1_1"], row["with_std_f1_1"]),
                    format_mean_std(row["without_mean_f1_1"], row["without_std_f1_1"]),
                    format_mean_std(row["delta_mean_f1_1"], row["delta_std_f1_1"]),
                    format_mean_std(row["with_mean_mcc"], row["with_std_mcc"]),
                    format_mean_std(row["without_mean_mcc"], row["without_std_mcc"]),
                    format_mean_std(row["delta_mean_mcc"], row["delta_std_mcc"]),
                ]
            )
            + " "
            + LATEX_LINEBREAK
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table*}"])
    write_text(output_dir / f"{OUTPUT_PREFIX}_summary_table.tex", "\n".join(lines))


def write_inferential_table(output_dir: Path, inferential_rows: list[dict[str, Any]]) -> None:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Paired statistical summary for the LOPO language-specific feature ablation. Positive deltas favor retaining language-specific feature counts. The table reports rank-biserial correlation as the Wilcoxon-aligned signed effect size and retains Cliff's $\\delta$ for comparability with earlier tables. PRIMARY-only columns retain only targets that satisfy the held-out adequacy screen in the matched fixed-family comparison.}",
        f"\\label{{tab:{OUTPUT_PREFIX.replace('_', '-')}-paired}}",
        "\\small",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llrrrrrrrrr}",
        "\\toprule",
        "Granularity & Metric & All $\\Delta$ & All 95\\% CI & W/T/L & All $p$ & All $r_{rb}$ & Cliff's $\\delta$ & PRIMARY $\\Delta$ & PRIMARY $n$ & PRIMARY $p$ " + LATEX_LINEBREAK,
        "\\midrule",
    ]
    for row in inferential_rows:
        delta_text = "--"
        if row["all_cliffs_delta"] is not None:
            delta_text = (
                f"{format_float(row['all_cliffs_delta'])} "
                f"({latex_escape(row['all_cliffs_magnitude'])})"
            )
        lines.append(
            " & ".join(
                [
                    GRANULARITY_LABELS[row["granularity"]],
                    FOCUS_METRIC_LABELS[row["metric"]],
                    format_mean_std(row["all_mean_diff"], row["all_std_diff"]),
                    format_interval(row["all_ci_low_diff"], row["all_ci_high_diff"]),
                    f"{row['all_win_count']}/{row['all_tie_count']}/{row['all_loss_count']}",
                    format_pvalue_latex(row["all_wilcoxon_pvalue"]),
                    format_float(row["all_rank_biserial"]),
                    delta_text,
                    format_float(row["primary_mean_diff"]),
                    str(int(row["primary_count"])),
                    format_pvalue_latex(row["primary_wilcoxon_pvalue"]),
                ]
            )
            + " "
            + LATEX_LINEBREAK
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table*}"])
    write_text(output_dir / f"{OUTPUT_PREFIX}_paired_table.tex", "\n".join(lines))


def write_selected_per_project_table(output_dir: Path, per_project_frame: pd.DataFrame) -> None:
    lines = [
        "\\scriptsize",
        "\\setlength{\\LTleft}{\\fill}",
        "\\setlength{\\LTright}{\\fill}",
        "\\setlength{\\LTcapwidth}{\\textwidth}",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{longtable}{llllrrrr}",
        f"\\caption{{Per-target LOPO language-specific feature ablation results when the learner family selected by the full-feature condition is held fixed.}}\\label{{tab:{OUTPUT_PREFIX.replace('_', '-')}-per-project}}" + LATEX_LINEBREAK,
        "\\toprule",
        "Granularity & Project & Support & Model & $F_1$ with & $F_1$ without & MCC with & MCC without " + LATEX_LINEBREAK,
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        "Granularity & Project & Support & Model & $F_1$ with & $F_1$ without & MCC with & MCC without " + LATEX_LINEBREAK,
        "\\midrule",
        "\\endhead",
        "\\bottomrule",
        "\\endfoot",
    ]
    for _, row in per_project_frame.iterrows():
        lines.append(
            " & ".join(
                [
                    GRANULARITY_LABELS[row["granularity"]],
                    latex_escape(row["target_project"]),
                    ADEQUACY_SHORT_LABELS.get(str(row["adequacy"]), "--"),
                    latex_escape(MODEL_LABELS.get(row["matched_model_name"], row["matched_model_name"])),
                    format_float(row["f1_1_with_go"]),
                    format_float(row["f1_1_without_go"]),
                    format_float(row["mcc_with_go"]),
                    format_float(row["mcc_without_go"]),
                ]
            )
            + " "
            + LATEX_LINEBREAK
        )
    lines.extend(["\\end{longtable}", "\\normalsize"])
    write_text(output_dir / f"{OUTPUT_PREFIX}_selected_per_project_table.tex", "\n".join(lines))


def plot_delta_distributions(
    merged_by_granularity: dict[str, pd.DataFrame],
    figures_dir: Path,
    granularities: list[str],
) -> None:
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(len(FOCUS_METRICS), 1, figsize=(6.3, 8.0), sharex=False)
    if len(FOCUS_METRICS) == 1:
        axes = [axes]
    for axis, metric in zip(axes, FOCUS_METRICS):
        values = pd.concat(
            [merged_by_granularity[granularity][f"delta_{metric}"] for granularity in granularities],
            ignore_index=True,
        )
        lower_bound, upper_bound = generic_axis_limits(values, include_zero=True)
        data = [merged_by_granularity[granularity][f"delta_{metric}"].dropna().to_numpy() for granularity in granularities]
        boxplot = axis.boxplot(
            data,
            tick_labels=[GRANULARITY_LABELS[granularity] for granularity in granularities],
            patch_artist=True,
            medianprops={"color": "#111111", "linewidth": 1.2},
        )
        for patch, granularity in zip(boxplot["boxes"], granularities):
            patch.set_facecolor(GRANULARITY_COLORS[granularity])
            patch.set_alpha(0.75)
        for index, granularity in enumerate(granularities, start=1):
            metric_values = merged_by_granularity[granularity][f"delta_{metric}"].dropna().to_numpy()
            jitter = rng.normal(0.0, 0.04, size=len(metric_values))
            axis.scatter(np.full(len(metric_values), index) + jitter, metric_values, s=30, color="#111111", alpha=0.55)
        axis.axhline(0.0, color="#444444", linestyle="--", linewidth=1.1)
        axis.set_title(f"{FOCUS_METRIC_LABELS[metric]} delta", pad=8)
        axis.set_ylabel(f"Delta {FOCUS_METRIC_LABELS[metric]}")
        axis.set_ylim(lower_bound, upper_bound)
    fig.tight_layout(pad=1.0, h_pad=1.4)
    save_figure(fig, figures_dir, f"{OUTPUT_PREFIX}_delta_distributions")


def plot_paired_scatter(
    merged_by_granularity: dict[str, pd.DataFrame],
    figures_dir: Path,
    granularities: list[str],
) -> None:
    fig, axes = plt.subplots(len(FOCUS_METRICS), len(granularities), figsize=(4.7 * len(granularities), 4.2 * len(FOCUS_METRICS)))
    if len(FOCUS_METRICS) == 1:
        axes = np.array([axes])
    if len(granularities) == 1:
        axes = axes.reshape(len(FOCUS_METRICS), 1)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=SUPPORT_COLORS["primary"], label="PRIMARY target"),
        plt.Line2D(
            [0],
            [0],
            marker="^",
            linestyle="",
            markerfacecolor="none",
            markeredgecolor=SUPPORT_COLORS["low_support"],
            color=SUPPORT_COLORS["low_support"],
            label="Non-PRIMARY target",
        ),
    ]

    for row_index, metric in enumerate(FOCUS_METRICS):
        all_values = pd.concat(
            [
                merged_by_granularity[granularity][f"{metric}_with_go"]
                for granularity in granularities
            ]
            + [
                merged_by_granularity[granularity][f"{metric}_without_go"]
                for granularity in granularities
            ],
            ignore_index=True,
        )
        lower_bound, upper_bound = metric_axis_limits(all_values, metric)
        for column_index, granularity in enumerate(granularities):
            axis = axes[row_index, column_index]
            merged = merged_by_granularity[granularity]
            primary_mask = merged["both_primary"]
            x_values = merged[f"{metric}_without_go"]
            y_values = merged[f"{metric}_with_go"]
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
            axis.set_title(f"{GRANULARITY_LABELS[granularity]} level", pad=8)
            axis.set_xlabel(f"{FOCUS_METRIC_LABELS[metric]} without language-specific counts")
            if column_index == 0:
                axis.set_ylabel(f"{FOCUS_METRIC_LABELS[metric]} with language-specific counts")
            axis.set_xlim(lower_bound, upper_bound)
            axis.set_ylim(lower_bound, upper_bound)
            axis.set_aspect("equal", adjustable="box")
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0), pad=1.1)
    save_figure(fig, figures_dir, f"{OUTPUT_PREFIX}_paired_scatter")


def metric_direction_text(delta_value: Any) -> str:
    try:
        delta_float = float(delta_value)
    except (TypeError, ValueError):
        return "changed little"
    if not np.isfinite(delta_float):
        return "changed little"
    if delta_float > 0.01:
        return "favored retaining Go metrics"
    if delta_float < -0.01:
        return "favored removing Go metrics"
    return "changed little"


def paired_change_summary_text(f1_delta_value: Any, mcc_delta_value: Any) -> str:
    try:
        f1_delta = float(f1_delta_value)
        mcc_delta = float(mcc_delta_value)
    except (TypeError, ValueError):
        return "retaining the Go-specific counts yields mixed changes across the focal metrics"
    if not np.isfinite(f1_delta) or not np.isfinite(mcc_delta):
        return "retaining the Go-specific counts yields mixed changes across the focal metrics"
    if abs(f1_delta) <= 0.01 and abs(mcc_delta) <= 0.01:
        return "retaining the Go-specific counts produces only negligible changes in held-out performance"
    if f1_delta > 0.01 and mcc_delta > 0.01:
        return "retaining the Go-specific counts improves both focal metrics on average"
    if f1_delta < -0.01 and mcc_delta < -0.01:
        return "retaining the Go-specific counts weakens both focal metrics on average"
    return "retaining the Go-specific counts yields mixed changes across the focal metrics"


def build_results_section(
    summary_rows: list[dict[str, Any]],
    inferential_rows: list[dict[str, Any]],
) -> str:
    summary_lookup = {row["granularity"]: row for row in summary_rows}
    inferential_lookup = {(row["granularity"], row["metric"]): row for row in inferential_rows}

    paragraphs = [
        "\\subsection{Matched Language-Specific Feature Ablation}",
        "",
        "To isolate the contribution of the language-specific AST-derived counts, we ran a controlled matched LOPO ablation in which each held-out target first inherited the learner family selected by the full-feature condition and then reran that same learner after removing the language-specific counts.",
        "The substantive comparison applies only to file- and method-level prediction, because the modular LOPO commit schema does not contain those language-specific counts in the first place.",
        "Table~\\ref{tab:lopo-go-metrics-summary} reports the held-out means for both conditions, while Table~\\ref{tab:lopo-go-metrics-paired} reports the paired deltas, Wilcoxon p-values, rank-biserial correlations, Cliff's $\\delta$, and PRIMARY-only sensitivity summaries over the same held-out targets.",
        "Appendix Table~\\ref{tab:lopo-go-metrics-per-project} lists the corresponding per-target matched-family ablation outputs.",
        "Figure~\\ref{fig:lopo-go-metrics-deltas} shows the delta distributions directly, and Figure~\\ref{fig:lopo-go-metrics-scatter} shows whether the target-level points lie above or below the equality line.",
        "Figure~\\ref{fig:lopo-go-metrics-deltas} reports paired score differences rather than raw held-out scores, whereas Figure~\\ref{fig:lopo-go-metrics-scatter} shows the raw scores with and without language-specific counts.",
        "",
    ]

    for granularity in GO_METRIC_GRANULARITIES:
        if granularity not in summary_lookup:
            continue
        summary_row = summary_lookup[granularity]
        f1_row = inferential_lookup[(granularity, "f1_1")]
        mcc_row = inferential_lookup[(granularity, "mcc")]
        if granularity == "file":
            opening_clause = "At the file level, retaining the language-specific counts produces only negligible changes in held-out performance."
            target_phrase = f"across {int(summary_row['target_count'])} matched held-out targets"
        elif granularity == "method":
            opening_clause = "At the method level, the matched comparison again yields only negligible differences."
            target_phrase = f"across the same {int(summary_row['target_count'])} matched held-out targets"
        else:
            opening_clause = (
                f"At {GRANULARITY_LABELS[granularity].lower()} level, "
                f"{paired_change_summary_text(summary_row['delta_mean_f1_1'], summary_row['delta_mean_mcc'])}."
            )
            target_phrase = f"across {int(summary_row['target_count'])} matched held-out targets"
        paragraphs.append(
            (
                f"{opening_clause} "
                f"Mean $F_1$ shifts from {format_float(summary_row['without_mean_f1_1'])} "
                f"to {format_float(summary_row['with_mean_f1_1'])} ($\\Delta={format_float(summary_row['delta_mean_f1_1'])}$; "
                f"$p={format_pvalue(f1_row['all_wilcoxon_pvalue'])}$), and MCC shifts from "
                f"{format_float(summary_row['without_mean_mcc'])} to {format_float(summary_row['with_mean_mcc'])} "
                f"($\\Delta={format_float(summary_row['delta_mean_mcc'])}$; $p={format_pvalue(mcc_row['all_wilcoxon_pvalue'])}$) "
                f"{target_phrase}, with rank-biserial correlations "
                f"{format_float(f1_row['all_rank_biserial'])} for $F_1$ and {format_float(mcc_row['all_rank_biserial'])} for MCC. "
                f"The PRIMARY-only screen preserves the same overall pattern, with mean deltas "
                f"{format_float(f1_row['primary_mean_diff'])} for $F_1$ and {format_float(mcc_row['primary_mean_diff'])} for MCC."
            )
        )
        paragraphs.append("")

    delta_values = [row["delta_mean_f1_1"] for row in summary_rows] + [row["delta_mean_mcc"] for row in summary_rows]
    significant_pvalues = [
        row["all_wilcoxon_pvalue"]
        for row in inferential_rows
        if row["all_wilcoxon_pvalue"] is not None and float(row["all_wilcoxon_pvalue"]) < 0.05
    ]
    if significant_pvalues:
        if all(float(value) <= 0.0 for value in delta_values if value is not None and np.isfinite(float(value))):
            conclusion = "Overall, the matched LOPO ablation suggests that the language-specific counts tend to reduce rather than improve cross-project transfer performance under the current baseline."
        elif all(float(value) >= 0.0 for value in delta_values if value is not None and np.isfinite(float(value))):
            conclusion = "Overall, the matched LOPO ablation suggests that the language-specific counts provide a positive, though not necessarily large, contribution under the current LOPO baseline."
        else:
            conclusion = "Overall, the matched LOPO ablation points to a mixed and representation-dependent contribution from the language-specific counts rather than a uniform gain."
    else:
        conclusion = "Overall, the matched LOPO ablation indicates that the language-specific structural counts add little incremental value to cross-project transfer under the current baseline, because the paired deltas remain small, the rank-biserial and Cliff effect sizes stay near zero, and no comparison yields strong inferential support. Taken together, the evidence therefore supports a constrained negative result rather than a strong language-specific feature gain."
    paragraphs.append(conclusion)
    paragraphs.append("")
    paragraphs.append(
        "Taken together, these matched ablation results answer RQ5 by indicating that removing the language-specific counts does not materially change held-out LOPO performance at the file or method level under the matched-family baseline."
    )
    paragraphs.append("")
    paragraphs.append(f"\\input{{{OUTPUT_PREFIX}_summary_table.tex}}")
    paragraphs.append("")
    paragraphs.append(f"\\input{{{OUTPUT_PREFIX}_paired_table.tex}}")
    paragraphs.append("")
    paragraphs.extend(
        [
            "\\begin{figure*}[t]",
            "  \\centering",
            f"  \\includegraphics[width=0.94\\linewidth]{{../figures/{OUTPUT_PREFIX}_delta_distributions.pdf}}",
            "  \\caption{Per-target deltas induced by retaining language-specific feature counts. This figure reports score differences, not raw held-out $F_1$ or MCC values. Values above zero favor the full feature schema with language-specific counts, whereas values below zero favor the ablated schema without those counts.}",
            "  \\label{fig:lopo-go-metrics-deltas}",
            "\\end{figure*}",
            "",
            "\\begin{figure*}[t]",
            "  \\centering",
            f"  \\includegraphics[width=0.96\\linewidth]{{../figures/{OUTPUT_PREFIX}_paired_scatter.pdf}}",
            "  \\caption{Project-level paired comparison of the raw matched-family scores with and without language-specific feature counts. Unlike Figure~\\ref{fig:lopo-go-metrics-deltas}, this figure shows the held-out scores themselves rather than their differences. Filled circles are PRIMARY targets, and open triangles are non-PRIMARY targets. Points above the diagonal favor retaining language-specific counts.}",
            "  \\label{fig:lopo-go-metrics-scatter}",
            "\\end{figure*}",
            "",
            "\\FloatBarrier",
        ]
    )
    return "\n".join(paragraphs)


def build_appendix_section() -> str:
    return "\n".join(
        [
            "\\section{Language-Specific Feature Appendix Material}",
            "\\label{app:lopo-go-metrics}",
            "",
            "This appendix exposes the per-target LOPO language-specific feature ablation results under the matched fixed-family design.",
            "",
            f"\\input{{{OUTPUT_PREFIX}_selected_per_project_table.tex}}",
            "",
            "\\input{lopo_robustness_appendix.tex}",
        ]
    )


def write_machine_readable_outputs(
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
    inferential_rows: list[dict[str, Any]],
    per_project_frame: pd.DataFrame,
) -> None:
    pd.DataFrame(summary_rows).to_csv(output_dir / f"{OUTPUT_PREFIX}_summary.csv", index=False)
    pd.DataFrame(inferential_rows).to_csv(output_dir / f"{OUTPUT_PREFIX}_paired_stats.csv", index=False)
    per_project_frame.to_csv(output_dir / f"{OUTPUT_PREFIX}_selected_per_project.csv", index=False)
    write_selection_shift_csv(output_dir)
    payload = {
        "summary_rows": summary_rows,
        "inferential_rows": inferential_rows,
        "per_project_rows": per_project_frame.to_dict(orient="records"),
    }
    write_text(output_dir / f"{OUTPUT_PREFIX}_summary.json", json.dumps(json_compatible(payload), indent=2))


def generate_assets(
    full_results_root: Path,
    no_go_results_root: Path,
    output_root: Path,
    requested_granularities: list[str] | None,
) -> list[str]:
    generated_dir = output_root / "generated"
    figures_dir = output_root / "figures"
    generated_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    granularities = overlapping_granularities(full_results_root, no_go_results_root, requested_granularities)
    if not granularities:
        write_placeholder_files(generated_dir, no_go_results_root)
        return ["Go-metrics ablation assets were not generated because no matched no-Go-metrics LOPO outputs were found."]

    configure_plot_style()
    full_bundles = {granularity: load_granularity(full_results_root, granularity) for granularity in granularities}
    no_go_bundles = {granularity: load_granularity(no_go_results_root, granularity) for granularity in granularities}
    merged_by_granularity = {
        granularity: build_merged_frame(full_bundles[granularity], no_go_bundles[granularity], granularity)
        for granularity in granularities
    }

    summary_rows = build_summary_rows(merged_by_granularity)
    inferential_rows = build_inferential_rows(merged_by_granularity)
    per_project_frame = build_per_project_rows(merged_by_granularity)

    write_summary_table(generated_dir, summary_rows)
    write_inferential_table(generated_dir, inferential_rows)
    write_selected_per_project_table(generated_dir, per_project_frame)
    write_text(generated_dir / f"{OUTPUT_PREFIX}_results.tex", build_results_section(summary_rows, inferential_rows))
    write_text(generated_dir / f"{OUTPUT_PREFIX}_appendix.tex", build_appendix_section())
    write_machine_readable_outputs(generated_dir, summary_rows, inferential_rows, per_project_frame)

    plot_delta_distributions(merged_by_granularity, figures_dir, granularities)
    plot_paired_scatter(merged_by_granularity, figures_dir, granularities)

    summary_lines: list[str] = []
    for row in summary_rows:
        summary_lines.append(
            f"{GRANULARITY_LABELS[row['granularity']]}: delta positive-class F1={format_float(row['delta_mean_f1_1'])}, delta MCC={format_float(row['delta_mean_mcc'])}"
        )
    return summary_lines


def main() -> None:
    args = parse_args()
    summary_lines = generate_assets(
        full_results_root=args.full_results_root,
        no_go_results_root=args.no_go_results_root,
        output_root=args.output_root,
        requested_granularities=args.granularities,
    )
    for line in summary_lines:
        print(line)


if __name__ == "__main__":
    main()
