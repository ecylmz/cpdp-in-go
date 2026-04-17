from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator_factory: Callable[[int], Any]
    param_grid: list[dict[str, list[Any]]] | dict[str, list[Any]]


def _build_naive_bayes(_: int) -> GaussianNB:
    return GaussianNB()


def _build_logistic_regression(random_seed: int) -> LogisticRegression:
    return LogisticRegression(class_weight="balanced", max_iter=4000, solver="lbfgs", random_state=random_seed)


def _build_random_forest(random_seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(class_weight="balanced", random_state=random_seed, n_jobs=1)


def _build_xgboost(random_seed: int) -> Any:
    if XGBClassifier is None:
        raise ImportError("xgboost is not installed. Remove 'xgboost' from the model list or install the package.")
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_seed,
        n_jobs=1,
    )


def _build_voting(random_seed: int) -> VotingClassifier:
    estimators = [
        ("lr", _build_logistic_regression(random_seed)),
        ("rf", _build_random_forest(random_seed)),
        ("nb", _build_naive_bayes(random_seed)),
    ]
    return VotingClassifier(estimators=estimators, voting="soft")


BASELINE_MODEL_SPECS: dict[str, ModelSpec] = {
    "naive_bayes": ModelSpec(
        name="naive_bayes",
        estimator_factory=_build_naive_bayes,
        param_grid={"model__var_smoothing": [1e-9, 1e-8, 1e-7]},
    ),
    "logistic_regression": ModelSpec(
        name="logistic_regression",
        estimator_factory=_build_logistic_regression,
        param_grid={"model__C": [0.1, 1.0, 10.0]},
    ),
    "random_forest": ModelSpec(
        name="random_forest",
        estimator_factory=_build_random_forest,
        param_grid={
            "model__n_estimators": [100, 300],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_leaf": [1, 2],
        },
    ),
    "xgboost": ModelSpec(
        name="xgboost",
        estimator_factory=_build_xgboost,
        param_grid={
            "model__n_estimators": [100, 300],
            "model__max_depth": [4, 6],
            "model__learning_rate": [0.03, 0.1],
            "model__subsample": [0.8, 1.0],
        },
    ),
    "voting": ModelSpec(
        name="voting",
        estimator_factory=_build_voting,
        param_grid={"model__weights": [[1, 1, 1], [2, 1, 1], [1, 2, 1]]},
    ),
}


def get_model_specs(model_names: list[str]) -> list[ModelSpec]:
    unknown_models = [model_name for model_name in model_names if model_name not in BASELINE_MODEL_SPECS]
    if unknown_models:
        raise ValueError(f"Unsupported baseline models requested: {unknown_models}")
    return [BASELINE_MODEL_SPECS[model_name] for model_name in model_names]