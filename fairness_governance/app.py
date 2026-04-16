"""Streamlit UI for the Fairness Governance System."""

from __future__ import annotations

import os
import sys
from datetime import datetime

import pandas as pd
import streamlit as st

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fairness_governance.config import FairnessCharter, set_global_config
from fairness_governance.modules.audit import run_data_audit
from fairness_governance.modules.counterfactual import run_counterfactual_test
from fairness_governance.modules.evaluation import bar_chart, comparison_table, tradeoff_plot
from fairness_governance.modules.intersectional import run_intersectional_analysis
from fairness_governance.modules.mitigation import (
    run_postprocessing,
    train_fairlearn_constraint_model,
    train_reweighted_model,
)
from fairness_governance.modules.model import train_baseline_model
from fairness_governance.modules.proxy import detect_proxy_leakage
from fairness_governance.modules.report import generate_pdf_report
from fairness_governance.modules.robustness import run_robustness_tests
from fairness_governance.modules.uncertainty import label_uncertainty
from fairness_governance.utils.sample_data import make_sample_credit_data


st.set_page_config(page_title="Fairness Governance System", layout="wide")


def load_dataset() -> pd.DataFrame:
    uploaded = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])
    if uploaded is not None:
        return pd.read_csv(uploaded)
    st.sidebar.info("Using generated sample credit dataset.")
    return make_sample_credit_data()


def metric_key(label: str) -> str:
    return "equal_opportunity" if label == "Equal Opportunity" else "demographic_parity"


def show_metric_cards(metrics: dict, prefix: str):
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{prefix} Accuracy", f"{metrics.get('accuracy', 0):.3f}")
    c2.metric(f"{prefix} DP Gap", f"{metrics.get('demographic_parity_gap', 0):.3f}")
    c3.metric(f"{prefix} EO Gap", f"{metrics.get('equal_opportunity_gap', 0):.3f}")


def mitigation_summary(results: dict) -> pd.DataFrame:
    rows = [
        ("Baseline", results["baseline"].metrics),
        ("Reweighted", results["reweighted"]["metrics"]),
        ("Fairlearn Constraint", results["constrained"]["metrics"]),
        ("Post-processing", results["postprocessed"]["metrics"]),
    ]
    return pd.DataFrame(
        [
            {
                "model": name,
                "accuracy": metrics["accuracy"],
                "demographic_parity_gap": metrics["demographic_parity_gap"],
                "equal_opportunity_gap": metrics["equal_opportunity_gap"],
                "selected_fairness_gap": metrics["selected_fairness_gap"],
            }
            for name, metrics in rows
        ]
    )


def run_full_analysis(df: pd.DataFrame, charter: dict) -> dict:
    audit = run_data_audit(
        df,
        charter["target"],
        charter["sensitive_attribute"],
        charter["epsilon"],
    )
    proxy = detect_proxy_leakage(
        df, charter["target"], charter["sensitive_attribute"]
    )
    baseline = train_baseline_model(
        df,
        charter["target"],
        charter["sensitive_attribute"],
        charter["metric_key"],
    )
    reweighted = train_reweighted_model(baseline, charter["metric_key"])
    constrained = train_fairlearn_constraint_model(
        baseline, charter["metric_key"], charter["epsilon"]
    )
    postprocessed = run_postprocessing(baseline, charter["metric_key"])
    candidates = {
        "Reweighted": reweighted,
        "Fairlearn Constraint": constrained,
    }
    best_name, best_result = min(
        candidates.items(),
        key=lambda item: (
            item[1]["metrics"]["selected_fairness_gap"],
            -item[1]["metrics"]["accuracy"],
        ),
    )
    cf = run_counterfactual_test(
        best_result["model"],
        baseline.x_test,
        charter["sensitive_attribute"],
    )
    second_feature = next(
        (c for c in baseline.x_test.columns if c != charter["sensitive_attribute"]),
        None,
    )
    intersectional = run_intersectional_analysis(
        baseline.x_test,
        baseline.y_test,
        best_result["predictions"],
        charter["sensitive_attribute"],
        second_feature=second_feature,
    )
    robustness = run_robustness_tests(
        best_result["model"],
        baseline.x_test,
        baseline.y_test,
        baseline.a_test,
        charter["sensitive_attribute"],
        charter["metric_key"],
    )
    uncertainty = label_uncertainty(best_result["model"], baseline.x_test)
    compare = comparison_table(baseline.metrics, best_result["metrics"], best_name)
    return {
        "audit": audit,
        "proxy": proxy,
        "baseline": baseline,
        "reweighted": reweighted,
        "constrained": constrained,
        "postprocessed": postprocessed,
        "best_name": best_name,
        "best": best_result,
        "counterfactual": cf,
        "intersectional": intersectional,
        "robustness": robustness,
        "uncertainty": uncertainty,
        "comparison": compare,
    }


def prediction_form(model, df: pd.DataFrame, target: str):
    st.subheader("Prediction Input")
    feature_df = df.drop(columns=[target])
    values = {}
    cols = st.columns(3)
    for idx, col in enumerate(feature_df.columns):
        with cols[idx % 3]:
            series = feature_df[col]
            if pd.api.types.is_numeric_dtype(series):
                values[col] = st.number_input(
                    col,
                    value=float(series.median()),
                    min_value=float(series.min()),
                    max_value=float(series.max()),
                )
            else:
                options = sorted(series.astype(str).dropna().unique().tolist())
                values[col] = st.selectbox(col, options)
    if st.button("Predict Case"):
        row = pd.DataFrame([values])
        pred = int(model.predict(row)[0])
        proba = getattr(model, "predict_proba", lambda x: [[1 - pred, pred]])(row)[0][-1]
        st.success(f"Prediction: {pred} | Positive probability: {proba:.3f}")


def main():
    st.title("Fairness Governance System")
    df = load_dataset()

    st.sidebar.header("Fairness Charter")
    target = st.sidebar.selectbox("Target (Y)", df.columns, index=len(df.columns) - 1)
    sensitive_options = [c for c in df.columns if c != target]
    default_sensitive = (
        sensitive_options.index("sensitive_group")
        if "sensitive_group" in sensitive_options
        else 0
    )
    sensitive = st.sidebar.selectbox(
        "Sensitive Attribute (A)", sensitive_options, index=default_sensitive
    )
    fairness_metric = st.sidebar.selectbox(
        "Fairness Metric", ["Demographic Parity", "Equal Opportunity"]
    )
    epsilon = st.sidebar.slider("Epsilon", min_value=0.01, max_value=0.10, value=0.05, step=0.01)
    charter = set_global_config(
        FairnessCharter(
            target=target,
            sensitive_attribute=sensitive,
            fairness_metric=fairness_metric,
            epsilon=epsilon,
        )
    )

    st.write("Active charter")
    st.json(charter)
    with st.expander("Dataset preview", expanded=True):
        st.dataframe(df.head(50), use_container_width=True)

    run_clicked = st.button("Run Bias Analysis and Fix Bias", type="primary")
    if run_clicked:
        with st.spinner("Running all governance tiers..."):
            try:
                st.session_state["results"] = run_full_analysis(df, charter)
                st.session_state["charter"] = charter
            except ValueError as exc:
                st.error(str(exc))
                st.info(
                    "Choose a target column with exactly two classes, such as "
                    "`approved` in the sample dataset."
                )
                st.stop()

    results = st.session_state.get("results")
    if not results:
        st.info("Run the analysis to detect bias, mitigate it, and generate audit outputs.")
        return

    st.header("Tier 1: Data Audit")
    c1, c2, c3 = st.columns(3)
    c1.metric("Demographic Parity Gap", f"{results['audit']['demographic_parity_gap']:.3f}")
    c2.metric("Equal Opportunity Gap", f"{results['audit']['equal_opportunity_gap']:.3f}")
    c3.metric("Bias Flag", str(results["audit"]["bias_flag"]))
    st.dataframe(results["audit"]["group_table"], use_container_width=True)

    st.header("Tier 2: Proxy Detection")
    st.metric("Sensitive Attribute AUC", f"{results['proxy']['auc']:.3f}")
    st.write("Proxy leakage flag:", results["proxy"]["proxy_flag"])
    st.dataframe(pd.DataFrame(results["proxy"]["flagged_features"]), use_container_width=True)

    st.header("Tiers 3-6: Model, Mitigation, Post-processing, Evaluation")
    show_metric_cards(results["baseline"].metrics, "Baseline")
    show_metric_cards(results["best"]["metrics"], results["best_name"])
    st.dataframe(mitigation_summary(results), use_container_width=True)
    st.dataframe(results["comparison"], use_container_width=True)
    st.plotly_chart(bar_chart(results["comparison"]), use_container_width=True)
    st.plotly_chart(tradeoff_plot(results["comparison"]), use_container_width=True)

    st.header("Tier 7: Counterfactual Engine")
    st.metric("Changed Predictions", f"{results['counterfactual']['changed_percent']:.2f}%")
    st.dataframe(results["counterfactual"]["examples"], use_container_width=True)

    st.header("Tier 8: Intersectional Analysis")
    st.dataframe(results["intersectional"], use_container_width=True)

    st.header("Tier 9: Robustness Testing")
    st.json(results["robustness"])

    st.header("Tier 10: Uncertainty Module")
    st.dataframe(results["uncertainty"].head(50), use_container_width=True)

    prediction_form(results["best"]["model"], df, target)

    st.header("Tier 12: Audit Report")
    if st.button("Generate PDF Report"):
        report_path = os.path.join(
            PACKAGE_ROOT,
            "outputs",
            f"fairness_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        )
        generated = generate_pdf_report(
            report_path,
            st.session_state["charter"],
            audit=results["audit"],
            proxy=results["proxy"],
            before_metrics=results["baseline"].metrics,
            after_metrics=results["best"]["metrics"],
            counterfactual=results["counterfactual"],
            robustness=results["robustness"],
        )
        with open(generated, "rb") as handle:
            st.download_button(
                "Download Audit Report",
                data=handle,
                file_name=os.path.basename(generated),
                mime="application/pdf" if generated.endswith(".pdf") else "text/plain",
            )


if __name__ == "__main__":
    main()
