"""Tier 3: baseline model training."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

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


def build_logistic_pipeline(x: pd.DataFrame) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(x)),
            (
                "classifier",
                LogisticRegression(max_iter=1000, solver="lbfgs"),
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
    stratify = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test, a_train, a_test = train_test_split(
        x, y, a, test_size=test_size, random_state=random_state, stratify=stratify
    )
    model = build_logistic_pipeline(x_train)
    model.fit(x_train, y_train)
    preds = pd.Series(model.predict(x_test), index=x_test.index, name="prediction")
    probs = pd.Series(model.predict_proba(x_test)[:, 1], index=x_test.index, name="probability")
    metrics = evaluate_predictions(metric_key, y_test, preds, a_test)
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
    )

