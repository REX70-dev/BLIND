"""Tier 1: data audit."""

from __future__ import annotations

import numpy as np

from .fairness import group_outcome_table
from fairness_governance.utils.preprocessing import encode_binary_target


def run_data_audit(df, target: str, sensitive: str, epsilon: float = 0.05) -> dict:
    """Audit labels before model training."""
    y = encode_binary_target(df[target])
    a = df[sensitive].astype(str)
    table = group_outcome_table(y, y, a)
    outcome_rates = table["outcome_rate"].dropna()
    dp_gap = float(outcome_rates.max() - outcome_rates.min()) if len(outcome_rates) else 0
    eo_gap = dp_gap
    bias_flag = bool(max(dp_gap, eo_gap) > epsilon)
    return {
        "group_table": table,
        "demographic_parity_gap": dp_gap,
        "equal_opportunity_gap": eo_gap,
        "bias_flag": bias_flag,
        "message": "Bias detected" if bias_flag else "No material bias detected",
    }

