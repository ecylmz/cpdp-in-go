from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[1]
GRANULARITIES = ("commit", "file", "method")
GRANULARITY_LABELS = {"commit": "Commit", "file": "File", "method": "Method"}
MODEL_ORDER = ("naive_bayes", "logistic_regression", "random_forest", "xgboost")
METRICS = ("f1_1", "mcc", "auc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the complete nested-LOPO SMOTE k=1 baseline with an independently "
            "rerun SMOTE k=5 experiment and generate sensitivity outputs."
        )
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=REPO_ROOT / "results_lopo_baseline",
    )
    parser.add_argument(
        "--k5-root",
        type=Path,
        default=REPO_ROOT / "results_lopo_baseline_smote_k5",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "analysis_output" / "generated",
    )
    return parser.parse_args()


def select_fold_local_rows(root: Path, granularity: str, expected_k: int) -> pd.DataFrame:
    output_dir = root / granularity
    signature = json.loads((output_dir / "run_signature.json").read_text(encoding="utf-8"))
    if signature.get("resampling") != "smote":
        raise ValueError(f"Expected SMOTE results at {output_dir}.")
    if int(signature.get("smote_k_neighbors", -1)) != expected_k:
        raise ValueError(f"Expected SMOTE k={expected_k} at {output_dir}.")
    if int(signature.get("random_seed", -1)) != 42:
        raise ValueError(f"Expected random seed 42 at {output_dir}.")

    frame = pd.read_csv(output_dir / "per_project_results.csv")
    successful = frame[frame["status"].eq("ok")].copy()
    primary_metric = str(signature.get("primary_metric", "f1"))
    primary_column = f"best_inner_{primary_metric}"
    secondary_column = "best_inner_mcc" if primary_metric != "mcc" else "best_inner_f1"
    model_names = list(signature.get("model_names") or MODEL_ORDER)
    order = {name: index for index, name in enumerate(model_names)}
    successful["__model_order"] = successful["model_name"].map(order).fillna(len(order)).astype(int)
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
    expected_projects = set(signature.get("available_projects") or [])
    if expected_projects and set(selected["target_project"]) != expected_projects:
        raise AssertionError(f"Incomplete fold-local selection at {output_dir}.")
    return selected


def paired_wilcoxon(left: pd.Series, right: pd.Series) -> float:
    differences = left.to_numpy(dtype=float) - right.to_numpy(dtype=float)
    if np.allclose(differences, 0.0):
        return 1.0
    return float(wilcoxon(left, right, alternative="two-sided", zero_method="wilcox").pvalue)


def format_pvalue(value: float) -> str:
    return "$<0.001$" if value < 0.001 else f"{value:.3f}"


def main() -> None:
    args = parse_args()
    baseline_root = args.baseline_root.resolve()
    k5_root = args.k5_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    per_target_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for granularity in GRANULARITIES:
        selections: dict[int, pd.DataFrame] = {}
        for smote_k, root in ((1, baseline_root), (5, k5_root)):
            selected = select_fold_local_rows(root, granularity, smote_k).set_index("target_project")
            selections[smote_k] = selected
            for project, row in selected.sort_index().iterrows():
                per_target_rows.append(
                    {
                        "granularity": granularity,
                        "smote_k": smote_k,
                        "target_project": project,
                        "selected_model": row["model_name"],
                        **{metric: float(row[metric]) for metric in METRICS},
                    }
                )

        common = sorted(set(selections[1].index).intersection(selections[5].index))
        for smote_k in (1, 5):
            selected = selections[smote_k].loc[common]
            row: dict[str, object] = {
                "granularity": granularity,
                "smote_k": smote_k,
                "target_count": len(common),
                "selected_model_counts": ";".join(
                    f"{model}={int((selected['model_name'] == model).sum())}"
                    for model in MODEL_ORDER
                    if int((selected["model_name"] == model).sum()) > 0
                ),
            }
            for metric in METRICS:
                values = selected[metric].astype(float)
                baseline = selections[1].loc[common, metric].astype(float)
                row[f"mean_{metric}"] = float(values.mean())
                row[f"std_{metric}"] = float(values.std(ddof=1))
                row[f"mean_delta_vs_k1_{metric}"] = float((values - baseline).mean())
                row[f"wilcoxon_pvalue_vs_k1_{metric}"] = paired_wilcoxon(values, baseline)
            summary_rows.append(row)

    per_target = pd.DataFrame(per_target_rows).sort_values(
        ["granularity", "smote_k", "target_project"], kind="mergesort"
    )
    summary = pd.DataFrame(summary_rows).sort_values(["granularity", "smote_k"], kind="mergesort")
    per_target.to_csv(output_root / "smote_k5_comparison.csv", index=False, float_format="%.10g")
    summary.to_csv(output_root / "smote_k5_comparison_summary.csv", index=False, float_format="%.10g")

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Complete nested-LOPO sensitivity rerun for SMOTE neighborhood size. For each value of $k$, model-family and hyperparameter selection is repeated within source-only inner folds. Values are target-level mean $\pm$ sample standard deviation; paired $p$-values compare $k=5$ with the primary $k=1$ baseline.}",
        r"\label{tab:smote-k5-comparison}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrl}",
        r"\toprule",
        r"Configuration & $k$ & $F_1$ & $p_{F_1}$ & MCC & $p_{\mathrm{MCC}}$ & AUC & $p_{\mathrm{AUC}}$ & Selected families \\",
        r"\midrule",
    ]
    for granularity in GRANULARITIES:
        for smote_k in (1, 5):
            row = summary[(summary["granularity"] == granularity) & (summary["smote_k"] == smote_k)].iloc[0]
            p_f1 = "--" if smote_k == 1 else format_pvalue(float(row["wilcoxon_pvalue_vs_k1_f1_1"]))
            p_mcc = "--" if smote_k == 1 else format_pvalue(float(row["wilcoxon_pvalue_vs_k1_mcc"]))
            p_auc = "--" if smote_k == 1 else format_pvalue(float(row["wilcoxon_pvalue_vs_k1_auc"]))
            families = str(row["selected_model_counts"]).replace("_", r"\_").replace(";", ", ")
            lines.append(
                f"{GRANULARITY_LABELS[granularity]} & {smote_k} & "
                f"{row['mean_f1_1']:.3f} $\\pm$ {row['std_f1_1']:.3f} & {p_f1} & "
                f"{row['mean_mcc']:.3f} $\\pm$ {row['std_mcc']:.3f} & {p_mcc} & "
                f"{row['mean_auc']:.3f} $\\pm$ {row['std_auc']:.3f} & {p_auc} & {families} \\\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    (output_root / "smote_k5_comparison_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
