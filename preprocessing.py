from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


@dataclass
class PreprocessingSummary:
    retained_features: list[str]
    dropped_constant_features: list[str]


def build_modeling_pipeline(
    estimator: Any,
    resampling_strategy: str,
    random_seed: int,
    smote_k_neighbors: int,
) -> ImbPipeline:
    sampler: Any
    if resampling_strategy == "smote":
        sampler = SMOTE(random_state=random_seed, k_neighbors=smote_k_neighbors)
    elif resampling_strategy == "none":
        sampler = "passthrough"
    else:
        raise ValueError(f"Unsupported baseline resampling strategy: {resampling_strategy}")

    return ImbPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("variance_threshold", VarianceThreshold()),
            ("scaler", StandardScaler()),
            ("sampler", sampler),
            ("model", estimator),
        ]
    )


def summarize_fitted_pipeline(pipeline: ImbPipeline, feature_names: list[str]) -> PreprocessingSummary:
    variance_step = pipeline.named_steps["variance_threshold"]
    support_mask = variance_step.get_support()
    retained_features = [feature for feature, is_kept in zip(feature_names, support_mask) if is_kept]
    dropped_features = [feature for feature, is_kept in zip(feature_names, support_mask) if not is_kept]
    return PreprocessingSummary(retained_features=retained_features, dropped_constant_features=dropped_features)