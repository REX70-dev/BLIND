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
from fairness_governance.modules.counterfactual import compare_consistency, run_counterfactual_test
from fairness_governance.modules.evaluation import (
    bar_chart,
    comparison_table,
    epsilon_tradeoff_plot,
    tradeoff_plot,
)
from fairness_governance.modules.intersectional import run_intersectional_analysis
from fairness_governance.modules.mitigation import (
    fairness_tradeoff_curve,
    run_postprocessing,
    train_fairlearn_constraint_model,
    train_reweighted_model,
)
from fairness_governance.modules.model import (
    multi_model_comparison,
    train_baseline_model,
    train_random_forest_from_artifacts,
)
from fairness_governance.modules.proxy import detect_proxy_leakage
from fairness_governance.modules.report import generate_pdf_report
from fairness_governance.modules.robustness import run_robustness_tests
from fairness_governance.modules.summary import ai_trust_score, bias_label, fairness_impact_summary
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


def status_badge(gap: float, label: str = "Bias Status"):
    text, color = bias_label(gap)
    st.markdown(
        f"""
        <div style="border-left: 8px solid {color}; padding: 0.75rem 1rem;
                    background: {color}18; border-radius: 6px; font-weight: 700;">
            {label}: <span style="color:{color};">{text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_metric_table(df: pd.DataFrame):
    return df.style.format(
        {
            "accuracy": "{:.3f}",
            "dp_gap": "{:.3f}",
            "eo_gap": "{:.3f}",
            "demographic_parity_gap": "{:.3f}",
            "equal_opportunity_gap": "{:.3f}",
            "selected_fairness_gap": "{:.3f}",
            "fairness_gap": "{:.3f}",
            "fairness_score": "{:.3f}",
            "epsilon": "{:.2f}",
        }
    )


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
    random_forest = train_random_forest_from_artifacts(baseline, charter["metric_key"])
    model_comparison = multi_model_comparison(baseline, random_forest)
    reweighted = train_reweighted_model(baseline, charter["metric_key"])
    constrained = train_fairlearn_constraint_model(
        baseline, charter["metric_key"], charter["epsilon"]
    )
    postprocessed = run_postprocessing(baseline, charter["metric_key"])
    best_name = "Fairlearn Constraint"
    best_result = constrained
    baseline_cf = run_counterfactual_test(
        baseline.model,
        baseline.x_test,
        charter["sensitive_attribute"],
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
    tradeoff_curve = fairness_tradeoff_curve(baseline, charter["metric_key"])
    impact = fairness_impact_summary(baseline.metrics, best_result["metrics"])
    trust = ai_trust_score(best_result["metrics"], robustness, cf, uncertainty)
    return {
        "audit": audit,
        "proxy": proxy,
        "baseline": baseline,
        "random_forest": random_forest,
        "model_comparison": model_comparison,
        "reweighted": reweighted,
        "constrained": constrained,
        "postprocessed": postprocessed,
        "best_name": best_name,
        "best": best_result,
        "baseline_counterfactual": baseline_cf,
        "counterfactual": cf,
        "consistency_comparison": compare_consistency(baseline_cf, cf),
        "intersectional": intersectional,
        "robustness": robustness,
        "uncertainty": uncertainty,
        "comparison": compare,
        "tradeoff_curve": tradeoff_curve,
        "impact": impact,
        "trust": trust,
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
    epsilon = st.sidebar.slider(
        "Fairness Strength (ε)",
        min_value=0.01,
        max_value=0.10,
        value=0.05,
        step=0.01,
        help="Low ε applies stronger fairness pressure. High ε gives the model more room to optimize accuracy.",
    )
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

    analysis_key = (target, sensitive, fairness_metric, round(epsilon, 2), len(df), tuple(df.columns))
    run_clicked = st.button("Run Bias Analysis and Fix Bias", type="primary")
    should_auto_rerun = (
        "results" in st.session_state
        and st.session_state.get("analysis_key") != analysis_key
    )
    if run_clicked or should_auto_rerun:
        with st.spinner("Running all governance tiers..."):
            try:
                st.session_state["results"] = run_full_analysis(df, charter)
                st.session_state["charter"] = charter
                st.session_state["analysis_key"] = analysis_key
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

    st.header("BEFORE FIX")
    status_badge(results["baseline"].metrics["selected_fairness_gap"], "Baseline")
    show_metric_cards(results["baseline"].metrics, "Before Fix")

    st.header("Tier 1: Data Audit")
    c1, c2, c3 = st.columns(3)
    c1.metric("Demographic Parity Gap", f"{results['audit']['demographic_parity_gap']:.3f}")
    c2.metric("Equal Opportunity Gap", f"{results['audit']['equal_opportunity_gap']:.3f}")
    c3.metric("Bias Flag", str(results["audit"]["bias_flag"]))
    st.dataframe(results["audit"]["group_table"], use_container_width=True)

    st.header("Tier 2: Proxy Detection")
    st.metric("Sensitive Attribute AUC", f"{results['proxy']['auc']:.3f}")
    st.write(results["proxy"]["explanation"])
    st.dataframe(pd.DataFrame(results["proxy"]["flagged_features"]), use_container_width=True)

    st.header("Tier 3: Multi-Model Comparison")
    st.caption("Logistic -> more stable, interpretable. Random Forest -> higher accuracy potential, higher bias risk.")
    st.dataframe(style_metric_table(results["model_comparison"]), use_container_width=True)

    st.header("AFTER FIX")
    status_badge(results["best"]["metrics"]["selected_fairness_gap"], "Mitigated")
    show_metric_cards(results["best"]["metrics"], f"After Fix ({results['best_name']}, ε={epsilon:.2f})")

    st.header("Fairness Impact Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bias Reduced", f"{results['impact']['bias_reduction_pct']:.1f}%")
    c2.metric("EO Improved", f"{results['impact']['equal_opportunity_improvement_pct']:.1f}%")
    c3.metric("Accuracy Change", f"{results['impact']['accuracy_change_pct']:.1f}%")
    c4.metric("AI TRUST SCORE", f"{results['trust']['score']:.1f} / 10")
    st.success(results["impact"]["summary"])

    st.header("Tiers 4-6: Mitigation, Post-processing, Evaluation")
    st.dataframe(style_metric_table(mitigation_summary(results)), use_container_width=True)
    st.dataframe(results["comparison"], use_container_width=True)
    st.plotly_chart(bar_chart(results["comparison"]), use_container_width=True)
    st.plotly_chart(tradeoff_plot(results["comparison"]), use_container_width=True)

    st.header("Tier 9: Trade-off Visualization")
    st.caption("X-axis is accuracy. Y-axis is fairness score (1 - selected fairness gap).")
    st.dataframe(style_metric_table(results["tradeoff_curve"]), use_container_width=True)
    st.plotly_chart(epsilon_tradeoff_plot(results["tradeoff_curve"]), use_container_width=True)

    st.header("Tier 7: Counterfactual Engine")
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Before Consistency", f"{results['baseline_counterfactual']['consistency_score']:.2f}%")
    cc2.metric("After Consistency", f"{results['counterfactual']['consistency_score']:.2f}%")
    cc3.metric("After Changed Predictions", f"{results['counterfactual']['changed_percent']:.2f}%")
    st.dataframe(results["consistency_comparison"], use_container_width=True)
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
            impact=results["impact"],
            trust=results["trust"],
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
