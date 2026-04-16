"""Tiers 4 and 5: bias mitigation and post-processing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from fairness_governance.utils.preprocessing import make_preprocessor
from .fairness import evaluate_predictions

try:
    from fairlearn.postprocessing import ThresholdOptimizer
    from fairlearn.reductions import DemographicParity, ExponentiatedGradient, TruePositiveRateParity
except Exception:  # pragma: no cover - optional dependency fallback
    ThresholdOptimizer = None
    DemographicParity = None
    ExponentiatedGradient = None
    TruePositiveRateParity = None


class PreprocessedFairlearnModel(BaseEstimator, ClassifierMixin):
    """Wrap a fitted preprocessor and a fairlearn estimator."""

    def __init__(self, preprocessor, estimator):
        self.preprocessor = preprocessor
        self.estimator = estimator

    def predict(self, x):
        return self.estimator.predict(self.preprocessor.transform(x))

    def predict_proba(self, x):
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(self.preprocessor.transform(x))
        preds = self.predict(x)
        return np.vstack([1 - preds, preds]).T


def reweighting_weights(sensitive: pd.Series) -> pd.Series:
    """Assign inverse-frequency sample weights by sensitive group."""
    counts = sensitive.astype(str).value_counts()
    weights = sensitive.astype(str).map(lambda g: 1.0 / counts[g])
    return weights / weights.mean()


def train_reweighted_model(artifacts, metric_key: str) -> dict:
    weights = reweighting_weights(artifacts.a_train)
    model = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(artifacts.x_train)),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(artifacts.x_train, artifacts.y_train, classifier__sample_weight=weights)
    preds = pd.Series(model.predict(artifacts.x_test), index=artifacts.x_test.index)
    probs = pd.Series(model.predict_proba(artifacts.x_test)[:, 1], index=artifacts.x_test.index)
    return {
        "model": model,
        "predictions": preds,
        "probabilities": probs,
        "metrics": evaluate_predictions(metric_key, artifacts.y_test, preds, artifacts.a_test),
    }


def train_fairlearn_constraint_model(artifacts, metric_key: str, epsilon: float) -> dict:
    """Train ExponentiatedGradient with the selected fairness constraint."""
    if ExponentiatedGradient is None:
        return _fallback_group_threshold_model(artifacts, metric_key)

    preprocessor = make_preprocessor(artifacts.x_train)
    x_train_pre = preprocessor.fit_transform(artifacts.x_train)
    constraint = (
        TruePositiveRateParity(difference_bound=epsilon)
        if metric_key == "equal_opportunity"
        else DemographicParity(difference_bound=epsilon)
    )
    estimator = LogisticRegression(max_iter=1000)
    mitigator = ExponentiatedGradient(estimator, constraints=constraint, eps=epsilon)
    try:
        mitigator.fit(x_train_pre, artifacts.y_train, sensitive_features=artifacts.a_train)
    except Exception:
        return _fallback_group_threshold_model(artifacts, metric_key)
    wrapped = PreprocessedFairlearnModel(preprocessor, mitigator)
    preds = pd.Series(wrapped.predict(artifacts.x_test), index=artifacts.x_test.index)
    probs = pd.Series(
        np.asarray(wrapped.predict_proba(artifacts.x_test))[:, -1],
        index=artifacts.x_test.index,
    )
    return {
        "model": wrapped,
        "predictions": preds,
        "probabilities": probs,
        "metrics": evaluate_predictions(metric_key, artifacts.y_test, preds, artifacts.a_test),
    }


def run_postprocessing(artifacts, metric_key: str) -> dict:
    """Apply fairlearn ThresholdOptimizer when available."""
    constraint = "equalized_odds" if metric_key == "equal_opportunity" else "demographic_parity"
    if ThresholdOptimizer is None:
        return _fallback_group_threshold_model(artifacts, metric_key)

    optimizer = ThresholdOptimizer(
        estimator=artifacts.model,
        constraints=constraint,
        prefit=True,
        predict_method="predict_proba",
    )
    try:
        optimizer.fit(
            artifacts.x_train,
            artifacts.y_train,
            sensitive_features=artifacts.a_train,
        )
        preds = pd.Series(
            optimizer.predict(artifacts.x_test, sensitive_features=artifacts.a_test),
            index=artifacts.x_test.index,
        )
    except Exception:
        return _fallback_group_threshold_model(artifacts, metric_key)
    return {
        "model": optimizer,
        "predictions": preds,
        "metrics": evaluate_predictions(metric_key, artifacts.y_test, preds, artifacts.a_test),
        "constraint": constraint,
    }


def _fallback_group_threshold_model(artifacts, metric_key: str) -> dict:
    """Simple functional fallback that equalizes group selection rates."""
    probs = artifacts.probabilities.copy()
    groups = artifacts.a_test.astype(str)
    target_rate = float(probs.mean())
    preds = pd.Series(0, index=probs.index)
    for group, indexes in groups.groupby(groups).groups.items():
        group_probs = probs.loc[indexes]
        cutoff = group_probs.quantile(max(0.0, min(1.0, 1 - target_rate)))
        preds.loc[indexes] = (group_probs >= cutoff).astype(int)
    return {
        "model": artifacts.model,
        "predictions": preds,
        "metrics": evaluate_predictions(metric_key, artifacts.y_test, preds, artifacts.a_test),
        "constraint": "fallback_group_threshold",
    }
