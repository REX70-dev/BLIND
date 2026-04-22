"""Tier 9: robustness checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fairness_governance.utils.preprocessing import flip_series_values
from .fairness import fairness_gap


def run_robustness_tests(
    model,
    x,
    y_true,
    sensitive_values,
    sensitive_col: str,
    metric_key: str,
    adversarial_noise_scale: float = 0.10,
) -> dict:
    """Measure fairness degradation under noise and attribute swaps."""
    base_pred = model.predict(x)
    base_gap = fairness_gap(metric_key, y_true, base_pred, sensitive_values)

    noisy_x = x.copy()
    numeric_cols = [c for c in noisy_x.columns if pd.api.types.is_numeric_dtype(noisy_x[c])]
    rng = np.random.default_rng(42)
    for col in numeric_cols:
        std = float(noisy_x[col].std() or 1.0)
        noisy_x[col] = noisy_x[col] + rng.normal(0, adversarial_noise_scale * std, size=len(noisy_x))
    noise_pred = model.predict(noisy_x)
    noise_gap = fairness_gap(metric_key, y_true, noise_pred, sensitive_values)

    swapped_x = x.copy()
    swapped_sensitive = pd.Series(sensitive_values).astype(str).reset_index(drop=True)
    if sensitive_col in swapped_x.columns:
        swapped_x[sensitive_col] = flip_series_values(swapped_x[sensitive_col])
    swapped_pred = model.predict(swapped_x)
    swap_gap = fairness_gap(metric_key, y_true, swapped_pred, swapped_sensitive)

    degradation = max(0.0, noise_gap - base_gap, swap_gap - base_gap)
    stability_score = max(0.0, 1.0 - degradation)
    return {
        "base_gap": float(base_gap),
        "noise_gap": float(noise_gap),
        "attribute_swap_gap": float(swap_gap),
        "fairness_degradation": float(degradation),
        "stability_score": float(stability_score),
        "adversarial_noise_scale": float(adversarial_noise_scale),
    }
