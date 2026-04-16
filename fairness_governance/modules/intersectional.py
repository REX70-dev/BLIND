"""Tier 8: intersectional subgroup analysis."""

from __future__ import annotations

import pandas as pd

from .fairness import group_outcome_table


def run_intersectional_analysis(
    x: pd.DataFrame,
    y_true,
    y_pred,
    sensitive: str,
    second_feature: str | None = None,
    min_size: int = 30,
) -> pd.DataFrame:
    """Combine A with another feature and compute subgroup metrics."""
    if second_feature is None:
        candidates = [c for c in x.columns if c != sensitive]
        second_feature = candidates[0] if candidates else sensitive
    if second_feature not in x.columns or sensitive not in x.columns:
        return pd.DataFrame()

    combined = x[sensitive].astype(str) + " | " + x[second_feature].astype(str)
    table = group_outcome_table(y_true, y_pred, combined)
    return table[table["count"] >= min_size].reset_index(drop=True)

