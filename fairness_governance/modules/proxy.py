"""Tier 2: proxy leakage detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from fairness_governance.utils.preprocessing import make_preprocessor


def detect_proxy_leakage(df: pd.DataFrame, target: str, sensitive: str) -> dict:
    """Train A <- X and flag features that can predict the sensitive attribute."""
    x = df.drop(columns=[target, sensitive], errors="ignore")
    a = df[sensitive].astype(str)
    if x.empty or a.nunique() < 2:
        return {"auc": 0.0, "proxy_flag": False, "flagged_features": []}

    y_codes = pd.Series(pd.factorize(a)[0], index=a.index)
    stratify = y_codes if y_codes.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y_codes, test_size=0.3, random_state=42, stratify=stratify
    )
    clf = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(x_train)),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    clf.fit(x_train, y_train)
    try:
        scores = clf.predict_proba(x_test)[:, 1]
        auc = float(roc_auc_score(y_test, scores))
    except Exception:
        auc = float((clf.predict(x_test) == y_test).mean())

    flagged_features = []
    if auc > 0.7:
        flagged_features = _feature_proxy_scores(df, target, sensitive)
    return {"auc": auc, "proxy_flag": auc > 0.7, "flagged_features": flagged_features}


def _feature_proxy_scores(df: pd.DataFrame, target: str, sensitive: str) -> list[dict]:
    """Simple per-feature association screen for readable proxy output."""
    a_codes = pd.factorize(df[sensitive].astype(str))[0]
    scores = []
    for col in df.drop(columns=[target, sensitive], errors="ignore").columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            corr = np.corrcoef(series.fillna(series.median()), a_codes)[0, 1]
            score = 0.0 if np.isnan(corr) else abs(float(corr))
        else:
            encoded = pd.factorize(series.astype(str))[0]
            corr = np.corrcoef(encoded, a_codes)[0, 1]
            score = 0.0 if np.isnan(corr) else abs(float(corr))
        scores.append({"feature": col, "association_score": score})
    return sorted(scores, key=lambda item: item["association_score"], reverse=True)[:5]

