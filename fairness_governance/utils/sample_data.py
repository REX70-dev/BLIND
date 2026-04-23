"""Sample dataset for end-to-end demonstration."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_sample_credit_data(n: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Create an Adult-style binary income dataset for demos and tests.

    The schema intentionally mirrors the fairness audit contract:
    target = ``income`` and sensitive attribute = ``sex``. The sensitive
    attribute is correlated with proxy features, but it is removed from model
    features before preprocessing and training.
    """
    rng = np.random.default_rng(random_state)
    sex = rng.choice(["Male", "Female"], size=n, p=[0.52, 0.48])
    age = rng.integers(19, 69, size=n)
    hours_per_week = np.clip(
        rng.normal(42, 10, size=n) + np.where(sex == "Male", 6, -4),
        15,
        75,
    )
    education = rng.choice(
        ["HS-grad", "Some-college", "Bachelors", "Masters"],
        size=n,
        p=[0.34, 0.31, 0.24, 0.11],
    )
    occupation = rng.choice(
        ["Admin", "Craft", "Exec-managerial", "Professional", "Service", "Sales"],
        size=n,
        p=[0.18, 0.16, 0.14, 0.20, 0.17, 0.15],
    )
    marital_status = rng.choice(
        ["Never-married", "Married", "Divorced"],
        size=n,
        p=[0.32, 0.50, 0.18],
    )
    workclass = rng.choice(
        ["Private", "Self-emp", "Government"],
        size=n,
        p=[0.72, 0.12, 0.16],
    )
    relationship = np.where(
        sex == "Male",
        rng.choice(["Husband", "Not-in-family", "Own-child"], size=n, p=[0.52, 0.33, 0.15]),
        rng.choice(["Wife", "Not-in-family", "Own-child"], size=n, p=[0.34, 0.46, 0.20]),
    )
    logits = (
        -2.4
        + 0.045 * (age - 35)
        + 0.055 * (hours_per_week - 38)
        + np.where(education == "Bachelors", 0.85, 0.0)
        + np.where(education == "Masters", 1.25, 0.0)
        + np.where(occupation == "Exec-managerial", 1.10, 0.0)
        + np.where(occupation == "Professional", 0.75, 0.0)
        + np.where(marital_status == "Married", 0.55, 0.0)
        + np.where(workclass == "Self-emp", 0.30, 0.0)
        + np.where(sex == "Male", 0.80, -0.20)
    )
    probability = 1 / (1 + np.exp(-logits))
    income = rng.binomial(1, probability)
    return pd.DataFrame(
        {
            "age": age,
            "workclass": workclass,
            "education": education,
            "marital_status": marital_status,
            "occupation": occupation,
            "relationship": relationship,
            "hours_per_week": hours_per_week.round(1),
            "capital_gain": np.maximum(0, rng.normal(950, 2800, size=n)).round(0),
            "sex": sex,
            "income": income,
        }
    )
