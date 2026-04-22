"""Business-facing fairness impact and trust summaries."""

from __future__ import annotations

import numpy as np


def pct_change(before: float, after: float, lower_is_better: bool = True) -> float:
    """Percent improvement with safe handling for near-zero denominators."""
    if abs(before) < 1e-9:
        return 0.0
    delta = before - after if lower_is_better else after - before
    return float((delta / abs(before)) * 100)


def fairness_impact_summary(before: dict, after: dict) -> dict:
    """Automatically compute the impact narrative for the dashboard."""
    bias_reduction = pct_change(
        before["demographic_parity_gap"], after["demographic_parity_gap"], True
    )
    eo_improvement = pct_change(
        before["equal_opportunity_gap"], after["equal_opportunity_gap"], True
    )
    accuracy_change = pct_change(before["accuracy"], after["accuracy"], False)
    return {
        "bias_reduction_pct": bias_reduction,
        "equal_opportunity_improvement_pct": eo_improvement,
        "accuracy_change_pct": accuracy_change,
        "summary": (
            f"Bias reduced by {bias_reduction:.1f}%. "
            f"Equal Opportunity improved by {eo_improvement:.1f}%. "
            f"Accuracy changed by {accuracy_change:.1f}%."
        ),
    }


def bias_label(gap: float) -> tuple[str, str]:
    """Return label and CSS color for a fairness gap."""
    if gap >= 0.20:
        return "🚨 HIGH BIAS", "#dc2626"
    if gap >= 0.10:
        return "⚠️ MODERATE", "#d97706"
    return "✅ FAIR", "#16a34a"


def ai_trust_score(after_metrics: dict, robustness: dict, consistency: dict, uncertainty) -> dict:
    """Weighted 0-10 score using fairness, robustness, consistency, uncertainty."""
    fairness_component = max(0.0, 1.0 - after_metrics["selected_fairness_gap"])
    robustness_component = float(robustness.get("stability_score", 0.0))
    consistency_component = float(consistency.get("consistency_score", 0.0)) / 100.0
    if len(uncertainty) == 0:
        certainty_component = 0.0
    else:
        certainty_component = float(np.mean(uncertainty["uncertainty_label"] == "Auto Decision"))

    raw = (
        0.40 * fairness_component
        + 0.25 * robustness_component
        + 0.25 * consistency_component
        + 0.10 * certainty_component
    )
    score = max(0.0, min(10.0, raw * 10.0))
    return {
        "score": score,
        "fairness_component": fairness_component,
        "robustness_component": robustness_component,
        "consistency_component": consistency_component,
        "certainty_component": certainty_component,
    }
