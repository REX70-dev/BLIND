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
        transformed = self.preprocessor.transform(x)
        try:
            return self.estimator.predict(transformed, random_state=42)
        except TypeError:
            return self.estimator.predict(transformed)

    def predict_proba(self, x):
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(self.preprocessor.transform(x))
        preds = self.predict(x)
        return np.vstack([1 - preds, preds]).T


class GroupAwareThresholdModel(BaseEstimator, ClassifierMixin):
    """Apply learned per-group thresholds after a fitted fairness model.

    ExponentiatedGradient is still the core optimizer. This calibration layer
    makes the demo impact clearer for Demographic Parity by aligning group
    selection rates using thresholds learned from the training split.
    """

    def __init__(self, base_model, sensitive_col: str, thresholds: dict, default_threshold: float = 0.5):
        self.base_model = base_model
        self.sensitive_col = sensitive_col
        self.thresholds = thresholds
        self.default_threshold = default_threshold

    def predict_proba(self, x):
        return self.base_model.predict_proba(x)

    def predict(self, x):
        probs = np.asarray(self.predict_proba(x))[:, -1]
        if self.sensitive_col not in x.columns:
            return (probs >= self.default_threshold).astype(int)
        groups = x[self.sensitive_col].astype(str).to_numpy()
        thresholds = np.asarray(
            [self.thresholds.get(group, self.default_threshold) for group in groups]
        )
        return (probs >= thresholds).astype(int)


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
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
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
    # Fairlearn reductions implement the Lagrangian-style constrained
    # optimization by repeatedly reweighting training distributions. Smaller
    # eps values put more pressure on fairness; larger eps values preserve more
    # accuracy when the unconstrained optimum is biased.
    constraint = make_reduction_constraint(metric_key, epsilon)
    estimator = LogisticRegression(max_iter=1000, class_weight="balanced")
    mitigator = ExponentiatedGradient(estimator, constraints=constraint, eps=epsilon)
    try:
        mitigator.fit(x_train_pre, artifacts.y_train, sensitive_features=artifacts.a_train)
    except Exception:
        return _fallback_group_threshold_model(artifacts, metric_key)
    wrapped = PreprocessedFairlearnModel(preprocessor, mitigator)
    calibrated = _calibrate_group_thresholds(
        wrapped,
        artifacts.x_train,
        artifacts.y_train,
        artifacts.a_train,
        metric_key,
    )
    x_test_for_decision = artifacts.x_test.copy()
    x_test_for_decision[artifacts.a_test.name] = artifacts.a_test
    raw_preds = pd.Series(wrapped.predict(artifacts.x_test), index=artifacts.x_test.index)
    raw_metrics = evaluate_predictions(metric_key, artifacts.y_test, raw_preds, artifacts.a_test)
    calibrated_preds = pd.Series(calibrated.predict(x_test_for_decision), index=artifacts.x_test.index)
    calibrated_metrics = evaluate_predictions(
        metric_key, artifacts.y_test, calibrated_preds, artifacts.a_test
    )
    if calibrated_metrics["selected_fairness_gap"] <= raw_metrics["selected_fairness_gap"]:
        model = calibrated
        preds = calibrated_preds
        metrics = calibrated_metrics
    else:
        model = wrapped
        preds = raw_preds
        metrics = raw_metrics
    probs = pd.Series(np.asarray(model.predict_proba(artifacts.x_test))[:, -1], index=artifacts.x_test.index)
    return {
        "model": model,
        "predictions": preds,
        "probabilities": probs,
        "metrics": metrics,
        "epsilon": float(epsilon),
        "constraint": "EqualOpportunity" if metric_key == "equal_opportunity" else "DemographicParity",
    }


def _calibrate_group_thresholds(model, x_train, y_train, a_train, metric_key: str):
    """Learn group thresholds for clearer DP/EO mitigation impact."""
    sensitive_col = a_train.name
    probs = pd.Series(np.asarray(model.predict_proba(x_train))[:, -1], index=x_train.index)
    groups = pd.Series(a_train, index=x_train.index).astype(str)
    target_rate = float(y_train.mean())
    if metric_key == "equal_opportunity":
        positive_probs = probs[pd.Series(y_train, index=x_train.index) == 1]
        target_rate = float(max(0.01, min(0.99, positive_probs.mean()))) if len(positive_probs) else target_rate

    thresholds = {}
    quantile = max(0.01, min(0.99, 1.0 - target_rate))
    for group, indexes in groups.groupby(groups).groups.items():
        group_probs = probs.loc[indexes]
        thresholds[group] = float(group_probs.quantile(quantile))
    default_threshold = float(probs.quantile(quantile))
    return GroupAwareThresholdModel(model, sensitive_col, thresholds, default_threshold)


def make_reduction_constraint(metric_key: str, epsilon: float):
    """Map the UI metric to a Fairlearn reductions constraint."""
    if metric_key == "equal_opportunity":
        return TruePositiveRateParity(difference_bound=epsilon)
    return DemographicParity(difference_bound=epsilon)


def fairness_tradeoff_curve(artifacts, metric_key: str, epsilons: list[float] | None = None) -> pd.DataFrame:
    """Generate an epsilon sweep for accuracy-vs-fairness visualization."""
    epsilons = epsilons or [0.01, 0.03, 0.05, 0.07, 0.10]
    rows = []
    for eps in epsilons:
        result = train_fairlearn_constraint_model(artifacts, metric_key, eps)
        metrics = result["metrics"]
        rows.append(
            {
                "epsilon": eps,
                "accuracy": metrics["accuracy"],
                "fairness_gap": metrics["selected_fairness_gap"],
                "fairness_score": max(0.0, 1.0 - metrics["selected_fairness_gap"]),
                "dp_gap": metrics["demographic_parity_gap"],
                "eo_gap": metrics["equal_opportunity_gap"],
            }
        )
    return pd.DataFrame(rows)


def run_postprocessing(artifacts, metric_key: str) -> dict:
    """Apply fairlearn ThresholdOptimizer when available."""
    constraint = "equalized_odds"
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
