"""Shared preprocessing utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_dataset(df: pd.DataFrame, target: str, sensitive: str):
    """Split a dataframe into features, target, and sensitive attribute."""
    y = df[target]
    a = df[sensitive]
    x = df.drop(columns=[target])
    return x, y, a


def encode_binary_target(y: pd.Series) -> pd.Series:
    """Convert a target column into 0/1 labels when needed."""
    y_clean = y.dropna()
    if pd.api.types.is_numeric_dtype(y):
        unique = sorted(pd.Series(y_clean).unique())
        if set(unique).issubset({0, 1}):
            return y.astype(int)
        if len(unique) == 2:
            return pd.Series((y >= unique[-1]).astype(int), index=y.index, name=y.name)

    normalized = y.astype(str).str.strip().str.lower()
    positive = {"1", "true", "yes", "y", "approved", "approve", "accepted", "positive"}
    negative = {"0", "false", "no", "n", "denied", "reject", "rejected", "negative"}
    unique_normalized = set(normalized.dropna().unique())
    if unique_normalized and unique_normalized.issubset(positive | negative):
        return normalized.map(lambda value: 1 if value in positive else 0).astype(int)

    codes, uniques = pd.factorize(y)
    if len(uniques) != 2:
        preview = ", ".join(map(str, list(uniques[:8])))
        raise ValueError(
            "This demo expects a binary target column. "
            f"Selected target '{y.name}' has {len(uniques)} unique values: {preview}."
        )
    return pd.Series(codes, index=y.index, name=y.name).astype(int)


def encode_binary_sensitive(a: pd.Series) -> pd.Series:
    """Encode sensitive values as stable string labels for grouping."""
    return a.astype(str).fillna("missing")


def make_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    """Create a reusable tabular preprocessor."""
    categorical = [
        c for c in x.columns if not pd.api.types.is_numeric_dtype(x[c])
    ]
    numeric = [c for c in x.columns if c not in categorical]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
    )


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Best-effort feature names after preprocessing."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return [f"feature_{i}" for i in range(len(preprocessor.transformers_))]


def flip_series_values(series: pd.Series) -> pd.Series:
    """Flip binary-like sensitive values for counterfactual tests."""
    values = list(pd.Series(series).dropna().astype(str).unique())
    flipped = series.astype(str).copy()
    if len(values) >= 2:
        mapping = {values[0]: values[1], values[1]: values[0]}
        flipped = flipped.map(lambda v: mapping.get(v, v))
    return flipped


def coerce_prediction_proba(model, x: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities for estimators and wrappers."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.ndim == 2:
            return proba[:, -1]
        return proba
    preds = model.predict(x)
    return np.asarray(preds, dtype=float)
