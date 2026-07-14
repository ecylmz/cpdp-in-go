from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from summarize_lopo_results import (
    ALL_GRANULARITIES,
    GRANULARITY_LABELS,
    MODEL_ORDER,
    available_granularities,
    canonical_granularities,
    load_granularity,
)


OUTPUT_PREFIX = "lopo_robustness"
LATEX_LINEBREAK = chr(92) * 2
MODEL_FREQUENCY_LABELS = {
    "naive_bayes": "NB",
    "logistic_regression": "LR",
    "random_forest": "RF",
    "xgboost": "XGB",
}
RESAMPLING_CONDITIONS = [
    ("smote_k1", "SMOTE ($k = 1$)", "main_results_root"),
    ("random_over", "Random oversampling", "random_over_results_root"),
    ("no_resampling", "No resampling", "no_resampling_results_root"),
]
SELECTION_CONDITIONS = [
    ("f1_primary", "Inner-CV positive-class $F_1$"),
    ("mcc_primary", "Inner-CV MCC"),
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate robustness sensitivity tables and machine-readable summaries."
    )
    parser.add_argument(
        "--main-results-root",
        type=Path,
        default=repo_root / "results_lopo_baseline",
        help="Directory containing the main strict LOPO outputs with SMOTE (k=1) and F1-based selection.",
    )
    parser.add_argument(
        "--random-over-results-root",
        type=Path,
        default=repo_root / "results_lopo_baseline_resampling_random_over",
        help="Directory containing the strict LOPO outputs for the random-oversampling sensitivity run.",
    )
    parser.add_argument(
        "--no-resampling-results-root",
        type=Path,
        default=repo_root / "results_lopo_baseline_no_resampling",
        help="Directory containing the strict LOPO outputs for the no-resampling sensitivity run.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "analysis_output",
        help="Directory where generated robustness outputs will be written.",
    )
    return parser.parse_args()


def write_text(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="ascii")


def latex_escape(text: Any) -> str:
    escaped = str(text)
    for old_text, new_text in [
        ("\\", "\\textbackslash{}"),
        ("&", "\\&"),
        ("%", "\\%"),
        ("_", "\\_"),
        ("#", "\\#"),
    ]:
        escaped = escaped.replace(old_text, new_text)
    return escaped


def format_float(value: Any, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(numeric):
        return "--"
    if round(numeric, digits) == 0:
        numeric = 0.0
    return f"{numeric:.{digits}f}"


def format_mean_std(mean_value: Any, std_value: Any, digits: int = 3) -> str:
    mean_text = format_float(mean_value, digits)
    std_text = format_float(std_value, digits)
    if mean_text == "--":
        return "--"
    if std_text == "--":
        std_text = format_float(0.0, digits)
    return f"{mean_text} $\\pm$ {std_text}"


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
        return None if not np.isfinite(value) else float(value)
    return value


def selected_family_frequency_text(selected_frame: pd.DataFrame) -> str:
    counts = selected_frame["model_name"].value_counts().reindex(MODEL_ORDER, fill_value=0)
    return ", ".join(
        f"{MODEL_FREQUENCY_LABELS[model_name]}={int(counts[model_name])}" for model_name in MODEL_ORDER
    )


def metric_summary(selected_frame: pd.DataFrame, metric: str) -> tuple[float, float]:
    numeric = pd.to_numeric(selected_frame[metric], errors="coerce").dropna().to_numpy(dtype=float)
    if numeric.size == 0:
        return np.nan, np.nan
    mean_value = float(numeric.mean())
    std_value = float(numeric.std(ddof=1)) if numeric.size > 1 else 0.0
    return mean_value, std_value


def load_condition_bundles(results_root: Path) -> dict[str, dict[str, Any]]:
    if not results_root.exists():
        return {}
    granularities = canonical_granularities(None, available_granularities(results_root))
    return {granularity: load_granularity(results_root, granularity) for granularity in granularities}


def build_condition_rows(
    condition_specs: list[tuple[str, str, str]],
    roots: dict[str, Path],
    condition_column: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition_key, condition_label, root_key in condition_specs:
        bundles = load_condition_bundles(roots[root_key])
        for granularity in ALL_GRANULARITIES:
            if granularity not in bundles:
                continue
            selected = bundles[granularity]["selected"]
            mean_f1_1, std_f1_1 = metric_summary(selected, "f1_1")
            mean_mcc, std_mcc = metric_summary(selected, "mcc")
            rows.append(
                {
                    "granularity": granularity,
                    condition_column: condition_key,
                    f"{condition_column}_label": condition_label,
                    "mean_f1_1": mean_f1_1,
                    "std_f1_1": std_f1_1,
                    "mean_mcc": mean_mcc,
                    "std_mcc": std_mcc,
                    "selected_family_frequencies": selected_family_frequency_text(selected),
                }
            )
    return rows


def select_nested_rows(
    per_project_frame: pd.DataFrame,
    primary_metric: str,
    secondary_metric: str,
) -> pd.DataFrame:
    successful_rows = per_project_frame[per_project_frame["status"] == "ok"].copy()
    if successful_rows.empty:
        return successful_rows

    primary_column = f"best_inner_{primary_metric}"
    secondary_column = f"best_inner_{secondary_metric}"
    model_order = {model_name: index for index, model_name in enumerate(MODEL_ORDER)}
    successful_rows["__model_order"] = successful_rows["model_name"].map(model_order).fillna(len(model_order)).astype(int)

    return (
        successful_rows.sort_values(
            by=["target_project", primary_column, secondary_column, "__model_order"],
            ascending=[True, False, False, True],
            kind="mergesort",
        )
        .drop_duplicates(subset=["target_project"], keep="first")
        .drop(columns=["__model_order"])
        .reset_index(drop=True)
    )


def build_selection_rows(main_results_root: Path) -> list[dict[str, Any]]:
    bundles = load_condition_bundles(main_results_root)
    rows: list[dict[str, Any]] = []
    for granularity in ALL_GRANULARITIES:
        if granularity not in bundles:
            continue
        bundle = bundles[granularity]
        selection_frames = {
            "f1_primary": bundle["selected"],
            "mcc_primary": select_nested_rows(bundle["per_project"], primary_metric="mcc", secondary_metric="f1"),
        }
        for condition_key, condition_label in SELECTION_CONDITIONS:
            selected = selection_frames[condition_key]
            mean_f1_1, std_f1_1 = metric_summary(selected, "f1_1")
            mean_mcc, std_mcc = metric_summary(selected, "mcc")
            rows.append(
                {
                    "granularity": granularity,
                    "selection_objective": condition_key,
                    "selection_objective_label": condition_label,
                    "mean_f1_1": mean_f1_1,
                    "std_f1_1": std_f1_1,
                    "mean_mcc": mean_mcc,
                    "std_mcc": std_mcc,
                    "selected_family_frequencies": selected_family_frequency_text(selected),
                }
            )
    return rows


def ordering_signature(rows: list[dict[str, Any]], condition_column: str) -> dict[str, dict[str, list[str]]]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {}
    signatures: dict[str, dict[str, list[str]]] = {}
    for condition_key in frame[condition_column].drop_duplicates().tolist():
        subset = frame[frame[condition_column] == condition_key].copy()
        signatures[condition_key] = {
            "f1_order": subset.sort_values("mean_f1_1", ascending=False)["granularity"].tolist(),
            "mcc_order": subset.sort_values("mean_mcc", ascending=False)["granularity"].tolist(),
        }
    return signatures


def write_resampling_table(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    condition_order = {condition_key: index for index, (condition_key, _, _) in enumerate(RESAMPLING_CONDITIONS)}
    sorted_rows = sorted(
        rows,
        key=lambda row: (ALL_GRANULARITIES.index(row["granularity"]), condition_order[row["resampling_condition"]]),
    )
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Resampling sensitivity under the same strict LOPO protocol, fold structure, random seed, learner set, and fixed decision threshold. The main condition uses SMOTE with $k=1$, the alternative conditions use random oversampling or no resampling, and selected-family frequencies report the number of held-out targets assigned to each model family after fold-local selection.}",
        "\\label{tab:lopo-robustness-resampling}",
        "\\small",
        "\\begin{tabular}{lp{3.2cm}p{1.6cm}p{1.6cm}p{4.3cm}}",
        "\\toprule",
        "Granularity & Condition & $F_1$ & MCC & Selected-family frequencies " + LATEX_LINEBREAK,
        "\\midrule",
    ]
    if not sorted_rows:
        lines.append("-- & -- & -- & -- & -- " + LATEX_LINEBREAK)
    else:
        for row in sorted_rows:
            lines.append(
                " & ".join(
                    [
                        GRANULARITY_LABELS[row["granularity"]],
                        row["resampling_condition_label"],
                        format_mean_std(row["mean_f1_1"], row["std_f1_1"]),
                        format_mean_std(row["mean_mcc"], row["std_mcc"]),
                        latex_escape(row["selected_family_frequencies"]),
                    ]
                )
                + " "
                + LATEX_LINEBREAK
            )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    write_text(output_dir / f"{OUTPUT_PREFIX}_resampling_table.tex", "\n".join(lines))


def write_selection_objective_table(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    condition_order = {condition_key: index for index, (condition_key, _) in enumerate(SELECTION_CONDITIONS)}
    sorted_rows = sorted(
        rows,
        key=lambda row: (ALL_GRANULARITIES.index(row["granularity"]), condition_order[row["selection_objective"]]),
    )
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Selection-objective sensitivity under the same strict LOPO protocol, random seed, SMOTE ($k=1$) resampling setting, learner set, and fixed decision threshold. The default condition selects the fold-local winner by inner-CV positive-class $F_1$ with MCC as the tie-breaker; the alternative condition selects by inner-CV MCC with positive-class $F_1$ as the tie-breaker. Selected-family frequencies report the number of held-out targets assigned to each model family after fold-local selection.}",
        "\\label{tab:lopo-robustness-selection-objective}",
        "\\small",
        "\\begin{tabular}{lp{3.2cm}p{1.6cm}p{1.6cm}p{4.3cm}}",
        "\\toprule",
        "Granularity & Objective & $F_1$ & MCC & Selected-family frequencies " + LATEX_LINEBREAK,
        "\\midrule",
    ]
    if not sorted_rows:
        lines.append("-- & -- & -- & -- & -- " + LATEX_LINEBREAK)
    else:
        for row in sorted_rows:
            lines.append(
                " & ".join(
                    [
                        GRANULARITY_LABELS[row["granularity"]],
                        row["selection_objective_label"],
                        format_mean_std(row["mean_f1_1"], row["std_f1_1"]),
                        format_mean_std(row["mean_mcc"], row["std_mcc"]),
                        latex_escape(row["selected_family_frequencies"]),
                    ]
                )
                + " "
                + LATEX_LINEBREAK
            )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    write_text(output_dir / f"{OUTPUT_PREFIX}_selection_objective_table.tex", "\n".join(lines))


def build_appendix_section(resampling_rows: list[dict[str, Any]], selection_rows: list[dict[str, Any]]) -> str:
    lines = [
        "\\section{Robustness Sensitivity Material}",
        "\\label{app:lopo-robustness}",
        "",
        "This appendix reports two additional robustness checks that keep the strict LOPO boundary fixed while varying only the resampling condition or the fold-local model-family selection objective.",
        "",
    ]
    if resampling_rows:
        lines.append(f"\\input{{{OUTPUT_PREFIX}_resampling_table.tex}}")
        lines.append("")
    else:
        lines.append("Resampling-sensitivity outputs were not available for analysis.")
        lines.append("")
    if selection_rows:
        lines.append(f"\\input{{{OUTPUT_PREFIX}_selection_objective_table.tex}}")
    else:
        lines.append("Selection-objective sensitivity outputs were not available for analysis.")
    return "\n".join(lines)


def write_machine_readable_outputs(
    output_dir: Path,
    resampling_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    pd.DataFrame(resampling_rows).to_csv(output_dir / f"{OUTPUT_PREFIX}_resampling.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(output_dir / f"{OUTPUT_PREFIX}_selection_objective.csv", index=False)
    payload = {
        "resampling_orderings": ordering_signature(resampling_rows, "resampling_condition"),
        "selection_orderings": ordering_signature(selection_rows, "selection_objective"),
    }
    write_text(output_dir / f"{OUTPUT_PREFIX}_summary.json", json.dumps(json_compatible(payload), indent=2))
    return payload


def generate_assets(
    main_results_root: Path,
    random_over_results_root: Path,
    no_resampling_results_root: Path,
    output_root: Path,
) -> list[str]:
    generated_dir = output_root / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    roots = {
        "main_results_root": main_results_root,
        "random_over_results_root": random_over_results_root,
        "no_resampling_results_root": no_resampling_results_root,
    }

    resampling_rows = build_condition_rows(RESAMPLING_CONDITIONS, roots, "resampling_condition")
    selection_rows = build_selection_rows(main_results_root)

    write_resampling_table(generated_dir, resampling_rows)
    write_selection_objective_table(generated_dir, selection_rows)
    write_text(generated_dir / f"{OUTPUT_PREFIX}_appendix.tex", build_appendix_section(resampling_rows, selection_rows))
    payload = write_machine_readable_outputs(generated_dir, resampling_rows, selection_rows)

    summary_lines: list[str] = []
    if payload["resampling_orderings"]:
        main_order = payload["resampling_orderings"].get("smote_k1")
        random_over_order = payload["resampling_orderings"].get("random_over")
        no_resampling_order = payload["resampling_orderings"].get("no_resampling")
        if main_order and random_over_order:
            summary_lines.append(
                f"Resampling sensitivity random-over ordering preserved: {main_order == random_over_order}"
            )
        if main_order and no_resampling_order:
            summary_lines.append(
                f"Resampling sensitivity no-resampling ordering preserved: {main_order == no_resampling_order}"
            )
    if payload["selection_orderings"]:
        main_order = payload["selection_orderings"].get("f1_primary")
        mcc_order = payload["selection_orderings"].get("mcc_primary")
        if main_order and mcc_order:
            summary_lines.append(
                f"Selection-objective ordering preserved: {main_order == mcc_order}"
            )
    if not summary_lines:
        summary_lines.append("Robustness appendix assets were refreshed.")
    return summary_lines


def main() -> None:
    args = parse_args()
    summary_lines = generate_assets(
        main_results_root=args.main_results_root.resolve(),
        random_over_results_root=args.random_over_results_root.resolve(),
        no_resampling_results_root=args.no_resampling_results_root.resolve(),
        output_root=args.output_root.resolve(),
    )
    for line in summary_lines:
        print(line)


if __name__ == "__main__":
    main()
