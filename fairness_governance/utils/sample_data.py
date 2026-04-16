"""Sample dataset for end-to-end demonstration."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_sample_credit_data(n: int = 600, random_state: int = 42) -> pd.DataFrame:
    """Create a small biased credit-style dataset for demos and tests."""
    rng = np.random.default_rng(random_state)
    group = rng.choice(["Group_A", "Group_B"], size=n, p=[0.55, 0.45])
    age = rng.integers(21, 70, size=n)
    income = rng.normal(65000, 18000, size=n) + np.where(group == "Group_A", 6000, -5000)
    debt_ratio = np.clip(rng.normal(0.34, 0.15, size=n), 0.02, 0.9)
    employment = rng.choice(["salaried", "self_employed", "contract"], size=n, p=[0.6, 0.25, 0.15])
    education = rng.choice(["high_school", "college", "graduate"], size=n, p=[0.35, 0.45, 0.20])
    zipcode_band = np.where(
        group == "Group_A",
        rng.choice(["Z1", "Z2", "Z3"], size=n, p=[0.55, 0.35, 0.10]),
        rng.choice(["Z1", "Z2", "Z3"], size=n, p=[0.15, 0.35, 0.50]),
    )
    logits = (
        -1.2
        + 0.000035 * income
        - 2.0 * debt_ratio
        + 0.012 * (age - 40)
        + np.where(employment == "salaried", 0.35, -0.12)
        + np.where(education == "graduate", 0.35, 0.0)
        + np.where(group == "Group_A", 0.45, -0.30)
    )
    probability = 1 / (1 + np.exp(-logits))
    approved = rng.binomial(1, probability)
    return pd.DataFrame(
        {
            "age": age,
            "income": income.round(2),
            "debt_ratio": debt_ratio.round(3),
            "employment": employment,
            "education": education,
            "zipcode_band": zipcode_band,
            "sensitive_group": group,
            "approved": approved,
        }
    )

