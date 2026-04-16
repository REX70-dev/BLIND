"""Tier 10: uncertainty and human-review labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fairness_governance.utils.preprocessing import coerce_prediction_proba


def label_uncertainty(model, x, confidence_threshold: float = 0.65) -> pd.DataFrame:
    """Flag predictions whose max class confidence is below threshold."""
    probs = coerce_prediction_proba(model, x)
    confidence = np.maximum(probs, 1 - probs)
    return pd.DataFrame(
        {
            "probability_positive": probs,
            "confidence": confidence,
            "uncertainty_label": np.where(
                confidence < confidence_threshold, "Human Review", "Auto Decision"
            ),
        },
        index=x.index,
    )

