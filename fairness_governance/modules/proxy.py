"""Tier 2: proxy leakage detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from fairness_governance.utils.preprocessing import feature_names, make_preprocessor


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
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    min_samples_leaf=4,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    clf.fit(x_train, y_train)
    try:
        scores = clf.predict_proba(x_test)[:, 1]
        auc = float(roc_auc_score(y_test, scores))
    except Exception:
        auc = float((clf.predict(x_test) == y_test).mean())

    top_features = _rf_feature_importance(clf)
    explanation = (
        "High proxy leakage - model can infer sensitive attribute"
        if auc > 0.7
        else "Low proxy leakage"
    )
    return {
        "auc": auc,
        "proxy_flag": auc > 0.7,
        "flagged_features": top_features,
        "explanation": explanation,
        "model_type": "RandomForestClassifier",
    }


def detect_proxy_from_artifacts(artifacts) -> dict:
    """Predict A from the same train-only preprocessed features used by the model."""
    preprocessor = artifacts.model.named_steps["preprocessor"]
    x_train = preprocessor.transform(artifacts.x_train)
    x_test = preprocessor.transform(artifacts.x_test)
    a_train = pd.factorize(artifacts.a_train.astype(str))[0]
    categories = list(pd.Series(artifacts.a_train.astype(str)).drop_duplicates())
    a_test = pd.Series(artifacts.a_test.astype(str)).map(
        {category: idx for idx, category in enumerate(categories)}
    )
    if a_test.isna().any():
        return {
            "auc": 0.0,
            "proxy_flag": False,
            "flagged_features": [],
            "explanation": "Proxy audit skipped because test split contains an unseen sensitive group.",
            "model_type": "RandomForestClassifier",
        }

    clf = RandomForestClassifier(
        n_estimators=250,
        min_samples_leaf=4,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(x_train, a_train)
    scores = clf.predict_proba(x_test)[:, 1]
    auc = float(roc_auc_score(a_test.astype(int), scores))
    names = feature_names(preprocessor)
    rows = [
        {"feature": name, "importance": float(score)}
        for name, score in zip(names, clf.feature_importances_)
    ]
    top_features = sorted(rows, key=lambda item: item["importance"], reverse=True)[:3]
    if auc > 0.7:
        explanation = "High proxy risk - model can infer sensitive attribute"
    elif auc < 0.6:
        explanation = "Low proxy risk"
    else:
        explanation = "Moderate proxy risk"
    return {
        "auc": auc,
        "proxy_flag": auc > 0.7,
        "flagged_features": top_features,
        "explanation": explanation,
        "model_type": "RandomForestClassifier",
    }


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


def _rf_feature_importance(clf: Pipeline) -> list[dict]:
    """Return the top three transformed features used to infer A."""
    preprocessor = clf.named_steps["preprocessor"]
    forest = clf.named_steps["classifier"]
    names = feature_names(preprocessor)
    importances = forest.feature_importances_
    rows = [
        {"feature": name, "importance": float(score)}
        for name, score in zip(names, importances)
    ]
    return sorted(rows, key=lambda item: item["importance"], reverse=True)[:3]
