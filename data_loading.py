from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from typing import Any

import numpy as np
import pandas as pd

from feature_schema import get_feature_columns, get_prediction_metadata_columns, get_schema


@dataclass
class DataQualityReport:
    project_id: str
    granularity: str
    raw_row_count: int
    row_count: int
    bug_count: int
    non_bug_count: int
    exact_duplicate_rows: int
    duplicate_candidate_rows: int
    missing_features: list[str]
    missing_key_columns: list[str]
    inf_counts: dict[str, int]
    coerced_nan_counts: dict[str, int]


@dataclass
class ProjectDataset:
    project_id: str
    granularity: str
    data: pd.DataFrame
    report: DataQualityReport


def list_available_projects(granularity: str) -> list[str]:
    schema = get_schema(granularity)
    if not schema.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found for granularity '{granularity}': {schema.data_dir}")
    return sorted(entry.name for entry in schema.data_dir.iterdir() if entry.is_dir() and not entry.name.startswith("_"))


def _read_labeled_file(file_path: Any, label: int) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df["is_bug"] = label
    return df


def _build_sample_ids(df: pd.DataFrame, project_id: str, granularity: str) -> tuple[pd.Series, int, list[str]]:
    schema = get_schema(granularity)
    available_id_columns = [column for column in schema.preferred_id_columns if column in df.columns]
    missing_key_columns = [column for column in schema.preferred_id_columns if column not in df.columns]

    if not available_id_columns:
        base_ids = pd.Series([f"{project_id}:{granularity}:{index}" for index in range(len(df))], index=df.index, dtype="object")
    else:
        base_ids = df[available_id_columns].fillna("<missing>").astype(str).agg("::".join, axis=1)

    duplicate_candidate_rows = int(base_ids.duplicated(keep=False).sum())
    duplicate_suffix = base_ids.groupby(base_ids).cumcount()
    sample_ids = base_ids.where(duplicate_suffix.eq(0), base_ids + "::dup" + duplicate_suffix.astype(str))
    return sample_ids, duplicate_candidate_rows, missing_key_columns


def _coerce_numeric_features(df: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, dict[str, int], dict[str, int], list[str]]:
    df = df.copy()
    missing_features = [column for column in feature_columns if column not in df.columns]
    inf_counts: dict[str, int] = {}
    coerced_nan_counts: dict[str, int] = {}

    for column in missing_features:
        logging.warning(f"Feature column '{column}' is missing and will be added as NaN for later imputation.")
        df[column] = np.nan

    for column in feature_columns:
        original_series = df[column]
        original_nan_count = int(original_series.isna().sum())
        numeric_series = pd.to_numeric(original_series, errors="coerce")
        inf_count = int(np.isinf(numeric_series.to_numpy(dtype=float, na_value=np.nan)).sum())
        numeric_series = numeric_series.replace([np.inf, -np.inf], np.nan)
        coerced_nan = max(0, int(numeric_series.isna().sum()) - original_nan_count)

        if inf_count > 0:
            logging.warning(f"Column '{column}' contains {inf_count} inf/-inf values. They will be converted to NaN.")
        if coerced_nan > 0:
            logging.warning(f"Column '{column}' produced {coerced_nan} NaN values during numeric coercion.")

        inf_counts[column] = inf_count
        coerced_nan_counts[column] = coerced_nan
        df[column] = numeric_series.astype(float)

    return df, inf_counts, coerced_nan_counts, missing_features


def load_project_dataset(project_id: str, granularity: str, exclude_go_metrics: bool = False) -> ProjectDataset:
    schema = get_schema(granularity)
    project_dir = schema.data_dir / project_id
    bug_path = project_dir / schema.bug_file
    non_bug_path = project_dir / schema.non_bug_file

    if not bug_path.exists() or not non_bug_path.exists():
        raise FileNotFoundError(
            f"Project '{project_id}' at granularity '{granularity}' is missing one of the required files: "
            f"{bug_path.name}, {non_bug_path.name}"
        )

    bugs_df = _read_labeled_file(bug_path, label=1)
    non_bugs_df = _read_labeled_file(non_bug_path, label=0)
    combined_df = pd.concat([bugs_df, non_bugs_df], ignore_index=True)

    if combined_df.empty:
        raise ValueError(f"Project '{project_id}' at granularity '{granularity}' is empty after loading.")

    combined_df["project_id"] = project_id
    feature_columns = get_feature_columns(granularity, exclude_go_metrics=exclude_go_metrics)
    combined_df, inf_counts, coerced_nan_counts, missing_features = _coerce_numeric_features(combined_df, feature_columns)

    raw_row_count = len(combined_df)
    exact_duplicate_rows = int(combined_df.duplicated().sum())
    if exact_duplicate_rows > 0:
        logging.warning(
            "Project '%s' (%s) contains %d exact duplicate rows. They will be removed before modeling.",
            project_id,
            granularity,
            exact_duplicate_rows,
        )
        combined_df = combined_df.drop_duplicates().reset_index(drop=True)

    sample_ids, duplicate_candidate_rows, missing_key_columns = _build_sample_ids(combined_df, project_id, granularity)
    combined_df["sample_id"] = sample_ids

    metadata_columns = ["project_id", "sample_id", "is_bug"] + get_prediction_metadata_columns(granularity, list(combined_df.columns))
    combined_df = combined_df[metadata_columns + feature_columns]

    report = DataQualityReport(
        project_id=project_id,
        granularity=granularity,
        raw_row_count=raw_row_count,
        row_count=len(combined_df),
        bug_count=int(combined_df["is_bug"].sum()),
        non_bug_count=int((combined_df["is_bug"] == 0).sum()),
        exact_duplicate_rows=exact_duplicate_rows,
        duplicate_candidate_rows=duplicate_candidate_rows,
        missing_features=missing_features,
        missing_key_columns=missing_key_columns,
        inf_counts=inf_counts,
        coerced_nan_counts=coerced_nan_counts,
    )

    return ProjectDataset(project_id=project_id, granularity=granularity, data=combined_df, report=report)


def prepare_features(dataset: pd.DataFrame, granularity: str, exclude_go_metrics: bool = False) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    feature_columns = get_feature_columns(granularity, exclude_go_metrics=exclude_go_metrics)
    metadata_columns = [column for column in dataset.columns if column not in feature_columns]
    X = dataset[feature_columns].copy()
    y = dataset["is_bug"].copy()
    metadata = dataset[metadata_columns].copy()
    return X, y, metadata


def report_to_dict(report: DataQualityReport) -> dict[str, Any]:
    return asdict(report)