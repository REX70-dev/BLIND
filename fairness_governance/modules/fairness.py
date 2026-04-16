"""Fairness metrics used across tiers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def group_outcome_table(y_true, y_pred, sensitive) -> pd.DataFrame:
    """Compute per-group outcome, selection, and true-positive rates."""
    frame = pd.DataFrame(
        {
            "y_true": np.asarray(y_true),
            "y_pred": np.asarray(y_pred),
            "group": np.asarray(pd.Series(sensitive).astype(str)),
        }
    )
    rows = []
    for group, part in frame.groupby("group", dropna=False):
        positives = part[part["y_true"] == 1]
        tpr = np.nan
        if len(positives) > 0:
            tpr = float((positives["y_pred"] == 1).mean())
        rows.append(
            {
                "group": group,
                "count": int(len(part)),
                "outcome_rate": float(part["y_true"].mean()),
                "selection_rate": float(part["y_pred"].mean()),
                "true_positive_rate": tpr,
            }
        )
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def demographic_parity_gap(y_pred, sensitive) -> float:
    frame = pd.DataFrame(
        {
            "pred": np.asarray(y_pred),
            "group": np.asarray(pd.Series(sensitive).astype(str)),
        }
    )
    rates = frame.groupby("group")["pred"].mean()
    if len(rates) <= 1:
        return 0.0
    return float(rates.max() - rates.min())


def equal_opportunity_gap(y_true, y_pred, sensitive) -> float:
    table = group_outcome_table(y_true, y_pred, sensitive)
    rates = table["true_positive_rate"].dropna()
    if len(rates) <= 1:
        return 0.0
    return float(rates.max() - rates.min())


def fairness_gap(metric_key: str, y_true, y_pred, sensitive) -> float:
    if metric_key == "equal_opportunity":
        return equal_opportunity_gap(y_true, y_pred, sensitive)
    return demographic_parity_gap(y_pred, sensitive)


def evaluate_predictions(metric_key: str, y_true, y_pred, sensitive) -> dict:
    """Return accuracy and both supported fairness gaps."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "demographic_parity_gap": demographic_parity_gap(y_pred, sensitive),
        "equal_opportunity_gap": equal_opportunity_gap(y_true, y_pred, sensitive),
        "selected_fairness_gap": fairness_gap(metric_key, y_true, y_pred, sensitive),
    }
