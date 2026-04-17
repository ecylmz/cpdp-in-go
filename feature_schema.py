from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

GO_SPECIFIC_FILE_METRICS = (
    "goroutine_count",
    "channel_count",
    "defer_count",
    "context_usage_count",
    "json_tag_count",
    "pointer_receiver_count",
    "error_handling_count",
)

GO_SPECIFIC_METHOD_METRICS = (
    "defer_count",
    "channel_count",
    "goroutine_count",
    "error_handling_count",
)


@dataclass(frozen=True)
class GranularitySchema:
    name: str
    data_dir: Path
    bug_file: str
    non_bug_file: str
    feature_columns: tuple[str, ...]
    preferred_id_columns: tuple[str, ...]
    prediction_metadata_columns: tuple[str, ...]


SCHEMAS: dict[str, GranularitySchema] = {
    "commit": GranularitySchema(
        name="commit",
        data_dir=BASE_DIR / "commit_data",
        bug_file="bugs.csv",
        non_bug_file="non_bugs.csv",
        feature_columns=(
            "modified_files_count",
            "code_churn",
            "max_file_churn",
            "avg_file_churn",
            "deletions",
            "insertions",
            "net_lines",
            "dmm_unit_size",
            "dmm_unit_complexity",
            "dmm_unit_interfacing",
            "total_token_count",
            "total_nloc",
            "total_complexity",
            "total_changed_method_count",
        ),
        preferred_id_columns=("sha", "commit_timestamp"),
        prediction_metadata_columns=("sha", "commit_timestamp"),
    ),
    "file": GranularitySchema(
        name="file",
        data_dir=BASE_DIR / "file_data",
        bug_file="file_bug_metrics.csv",
        non_bug_file="file_non_bug_metrics.csv",
        feature_columns=(
            "nloc",
            "complexity",
            "token_count",
            "method_count",
            "commit_count",
            "authors_count",
            "avg_method_param_count",
            "import_count",
            "cyclo_per_loc",
            "comment_ratio",
            "struct_count",
            "interface_count",
            "loop_count",
            "error_handling_count",
            "goroutine_count",
            "channel_count",
            "defer_count",
            "context_usage_count",
            "json_tag_count",
            "variadic_function_count",
            "pointer_receiver_count",
            "avg_method_complexity",
            "avg_methods_token_count",
        ),
        preferred_id_columns=("project", "file_path", "sha", "commit_timestamp"),
        prediction_metadata_columns=("project", "file_path", "sha", "commit_timestamp"),
    ),
    "method": GranularitySchema(
        name="method",
        data_dir=BASE_DIR / "method_data",
        bug_file="method_bug_metrics.csv",
        non_bug_file="method_non_bug_metrics.csv",
        feature_columns=(
            "cyclomatic_complexity",
            "nloc",
            "token_count",
            "parameter_count",
            "defer_count",
            "channel_count",
            "goroutine_count",
            "error_handling_count",
            "loop_count",
        ),
        preferred_id_columns=("project", "file_path", "method_signature", "method_name", "sha", "commit_timestamp"),
        prediction_metadata_columns=("project", "file_path", "method_signature", "method_name", "sha", "commit_timestamp"),
    ),
}


def get_schema(granularity: str) -> GranularitySchema:
    if granularity not in SCHEMAS:
        raise ValueError(f"Unsupported granularity: {granularity}")
    return SCHEMAS[granularity]


def get_feature_columns(granularity: str, exclude_go_metrics: bool = False) -> list[str]:
    schema = get_schema(granularity)
    feature_columns = list(schema.feature_columns)

    if not exclude_go_metrics:
        return feature_columns

    if granularity == "file":
        excluded = set(GO_SPECIFIC_FILE_METRICS)
    elif granularity == "method":
        excluded = set(GO_SPECIFIC_METHOD_METRICS)
    else:
        excluded = set()

    return [column for column in feature_columns if column not in excluded]


def get_prediction_metadata_columns(granularity: str, available_columns: list[str]) -> list[str]:
    schema = get_schema(granularity)
    return [column for column in schema.prediction_metadata_columns if column in available_columns]