"""Tier 6: model comparison and visualization helpers."""

from __future__ import annotations

import pandas as pd
import plotly.express as px


def comparison_table(before: dict, after: dict, label: str = "After") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "Baseline",
                "accuracy": before["accuracy"],
                "fairness_gap": before["selected_fairness_gap"],
            },
            {
                "model": label,
                "accuracy": after["accuracy"],
                "fairness_gap": after["selected_fairness_gap"],
            },
        ]
    )


def bar_chart(compare_df: pd.DataFrame):
    melted = compare_df.melt(id_vars="model", value_vars=["accuracy", "fairness_gap"])
    return px.bar(melted, x="model", y="value", color="variable", barmode="group")


def tradeoff_plot(compare_df: pd.DataFrame):
    return px.scatter(
        compare_df,
        x="fairness_gap",
        y="accuracy",
        color="model",
        size=[16] * len(compare_df),
        title="Accuracy vs Fairness Gap",
    )


def epsilon_tradeoff_plot(curve_df: pd.DataFrame):
    """Plot fairness strength sweep: x=accuracy, y=1-gap."""
    return px.line(
        curve_df,
        x="accuracy",
        y="fairness_score",
        text="epsilon",
        markers=True,
        title="Fairness vs Accuracy Trade-off by Epsilon",
        labels={
            "accuracy": "Accuracy",
            "fairness_score": "Fairness Score (1 - selected gap)",
            "epsilon": "Epsilon",
        },
    )
