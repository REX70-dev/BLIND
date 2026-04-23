"""Tier 3: baseline model training."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from fairness_governance.utils.preprocessing import (
    encode_binary_sensitive,
    encode_binary_target,
    make_preprocessor,
    split_dataset,
)
from .fairness import evaluate_predictions


@dataclass
class ModelArtifacts:
    model: Pipeline
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    a_train: pd.Series
    a_test: pd.Series
    predictions: pd.Series
    probabilities: pd.Series
    metrics: dict
    diagnostics: dict


def build_logistic_pipeline(x: pd.DataFrame) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(x)),
            (
                "classifier",
                LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced"),
            ),
        ]
    )


def build_random_forest_pipeline(x: pd.DataFrame) -> Pipeline:
    """Random Forest baseline for accuracy-vs-bias comparison."""
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(x)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=180,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_baseline_model(
    df: pd.DataFrame,
    target: str,
    sensitive: str,
    metric_key: str,
    test_size: float = 0.3,
    random_state: int = 42,
) -> ModelArtifacts:
    x, y_raw, a_raw = split_dataset(df, target, sensitive)
    y = encode_binary_target(y_raw)
    a = encode_binary_sensitive(a_raw)
    assert a.nunique() == 2
    assert y.nunique() == 2
    stratify = y.astype(str) + "_" + a.astype(str)
    if stratify.value_counts().min() < 2:
        raise ValueError(
            "Each label/sensitive group combination needs at least two rows "
            "for a stratified train/test split."
        )
    x_train, x_test, y_train, y_test, a_train, a_test = train_test_split(
        x, y, a, test_size=test_size, random_state=random_state, stratify=stratify
    )
    _validate_split_balance(y_train, y_test, a_train, a_test)
    model = build_logistic_pipeline(x_train)
    model.fit(x_train, y_train)
    preds = pd.Series(model.predict(x_test), index=x_test.index, name="prediction")
    probs = pd.Series(model.predict_proba(x_test)[:, 1], index=x_test.index, name="probability")
    metrics = evaluate_predictions(metric_key, y_test, preds, a_test)
    diagnostics = split_diagnostics(df, x_train, x_test, y_train, y_test, a_train, a_test)
    return ModelArtifacts(
        model=model,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        a_train=a_train,
        a_test=a_test,
        predictions=preds,
        probabilities=probs,
        metrics=metrics,
        diagnostics=diagnostics,
    )


def _validate_split_balance(y_train, y_test, a_train, a_test, min_group_size: int = 30) -> None:
    """Audit-critical sanity checks after stratification."""
    assert pd.Series(a_train).nunique() == 2
    assert pd.Series(y_train).nunique() == 2
    train_counts = pd.Series(a_train).value_counts()
    test_counts = pd.Series(a_test).value_counts()
    if train_counts.min() < min_group_size or test_counts.min() < min_group_size:
        raise ValueError(
            "Each sensitive group must have at least "
            f"{min_group_size} samples in train and test. "
            f"Train counts: {train_counts.to_dict()}; test counts: {test_counts.to_dict()}."
        )


def split_diagnostics(df, x_train, x_test, y_train, y_test, a_train, a_test) -> dict:
    """Collect split logging values for audit reporting and dashboard display."""
    return {
        "dataset_shape": tuple(df.shape),
        "feature_shape_train": tuple(x_train.shape),
        "feature_shape_test": tuple(x_test.shape),
        "group_counts_full": pd.Series(pd.concat([a_train, a_test])).value_counts().to_dict(),
        "group_counts_train": pd.Series(a_train).value_counts().to_dict(),
        "group_counts_test": pd.Series(a_test).value_counts().to_dict(),
        "label_counts_train": pd.Series(y_train).value_counts().to_dict(),
        "label_counts_test": pd.Series(y_test).value_counts().to_dict(),
    }


def train_random_forest_from_artifacts(artifacts: ModelArtifacts, metric_key: str) -> dict:
    """Train a Random Forest on the same split used by the logistic baseline."""
    model = build_random_forest_pipeline(artifacts.x_train)
    model.fit(artifacts.x_train, artifacts.y_train)
    preds = pd.Series(model.predict(artifacts.x_test), index=artifacts.x_test.index)
    probs = pd.Series(model.predict_proba(artifacts.x_test)[:, 1], index=artifacts.x_test.index)
    return {
        "model": model,
        "predictions": preds,
        "probabilities": probs,
        "metrics": evaluate_predictions(metric_key, artifacts.y_test, preds, artifacts.a_test),
        "note": "Random Forest -> higher accuracy potential, but can amplify proxy bias.",
    }


def multi_model_comparison(logistic_artifacts: ModelArtifacts, random_forest: dict) -> pd.DataFrame:
    """Readable comparison of the two baseline model families."""
    return pd.DataFrame(
        [
            {
                "model": "Logistic Regression",
                "accuracy": logistic_artifacts.metrics["accuracy"],
                "dp_gap": logistic_artifacts.metrics["demographic_parity_gap"],
                "eo_gap": logistic_artifacts.metrics["equal_opportunity_gap"],
                "interpretation": "More stable and interpretable",
            },
            {
                "model": "Random Forest",
                "accuracy": random_forest["metrics"]["accuracy"],
                "dp_gap": random_forest["metrics"]["demographic_parity_gap"],
                "eo_gap": random_forest["metrics"]["equal_opportunity_gap"],
                "interpretation": "Often higher accuracy, potentially higher bias",
            },
        ]
    )
