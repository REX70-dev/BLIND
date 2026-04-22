"""Tier 7: counterfactual sensitive-attribute testing."""

from __future__ import annotations

import pandas as pd

from fairness_governance.utils.preprocessing import flip_series_values


def run_counterfactual_test(model, x: pd.DataFrame, sensitive: str, limit: int = 10) -> dict:
    """Flip A in the input features and measure prediction changes."""
    if sensitive not in x.columns:
        return {
            "changed_percent": 0.0,
            "examples": pd.DataFrame(),
            "note": "Sensitive attribute is not present in model features.",
        }
    original = pd.Series(model.predict(x), index=x.index)
    cf_x = x.copy()
    cf_x[sensitive] = flip_series_values(cf_x[sensitive])
    changed = pd.Series(model.predict(cf_x), index=x.index)
    mask = original != changed
    examples = x.loc[mask].head(limit).copy()
    if not examples.empty:
        examples["original_prediction"] = original.loc[examples.index]
        examples["counterfactual_prediction"] = changed.loc[examples.index]
        examples[f"flipped_{sensitive}"] = cf_x.loc[examples.index, sensitive]
    return {
        "changed_percent": float(mask.mean() * 100),
        "consistency_score": float((~mask).mean() * 100),
        "examples": examples.reset_index(drop=True),
    }


def compare_consistency(before: dict, after: dict) -> pd.DataFrame:
    """Show counterfactual decision consistency before and after mitigation."""
    return pd.DataFrame(
        [
            {"stage": "Before mitigation", "consistency_score": before.get("consistency_score", 0.0)},
            {"stage": "After mitigation", "consistency_score": after.get("consistency_score", 0.0)},
        ]
    )
