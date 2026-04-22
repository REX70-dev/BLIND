"""MeritAI-inspired Streamlit theme helpers."""

from __future__ import annotations

import html

import streamlit as st


def inject_meritai_theme() -> None:
    """Apply the imported MeritAI visual language to Streamlit widgets."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@700&display=swap');

        :root {
            --merit-bg: #0a0c10;
            --merit-bg2: #111318;
            --merit-bg3: #181c24;
            --merit-border: rgba(255,255,255,0.09);
            --merit-border2: rgba(255,255,255,0.15);
            --merit-text: #e8eaf0;
            --merit-text2: #9ca3b0;
            --merit-text3: #5a6070;
            --merit-accent: #4f8ef7;
            --merit-accent2: #7c5cfc;
            --merit-green: #22d3a0;
            --merit-amber: #f59e0b;
            --merit-red: #f43f5e;
            --merit-radius: 10px;
            --merit-radius-sm: 8px;
        }

        html, body, [data-testid="stAppViewContainer"] {
            background: var(--merit-bg);
            color: var(--merit-text);
            font-family: 'DM Sans', system-ui, sans-serif;
        }

        [data-testid="stHeader"] {
            background: rgba(10,12,16,0.86);
            backdrop-filter: blur(16px);
            border-bottom: 1px solid var(--merit-border);
        }

        [data-testid="stSidebar"] {
            background: #0d1117;
            border-right: 1px solid var(--merit-border);
        }

        [data-testid="stSidebar"] * {
            color: var(--merit-text);
        }

        .block-container {
            max-width: 1400px;
            padding-top: 1.2rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            font-family: 'Playfair Display', Georgia, serif;
            letter-spacing: 0;
        }

        div[data-testid="stMetric"] {
            background: var(--merit-bg3);
            border: 1px solid var(--merit-border);
            border-radius: var(--merit-radius-sm);
            padding: 14px 16px;
        }

        div[data-testid="stMetricLabel"] p {
            font-size: 0.72rem;
            color: var(--merit-text3);
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        div[data-testid="stMetricValue"] {
            font-family: 'DM Mono', monospace;
            color: var(--merit-text);
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: var(--merit-radius-sm);
            border: 1px solid rgba(79,142,247,0.35);
            background: linear-gradient(135deg, var(--merit-accent), var(--merit-accent2));
            color: #fff;
            font-weight: 600;
            transition: all 0.15s ease;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            transform: translateY(-1px);
            border-color: rgba(255,255,255,0.35);
        }

        [data-testid="stDataFrame"],
        [data-testid="stTable"],
        [data-testid="stExpander"] {
            border-radius: var(--merit-radius);
            overflow: hidden;
        }

        .merit-header {
            background: linear-gradient(135deg, #0d1117 0%, #1a1f2e 52%, #0d1117 100%);
            border: 1px solid var(--merit-border);
            border-radius: var(--merit-radius);
            padding: 18px 22px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 18px;
            margin-bottom: 20px;
        }

        .merit-brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .merit-logo {
            width: 38px;
            height: 38px;
            border-radius: 9px;
            background: linear-gradient(135deg, var(--merit-accent), var(--merit-accent2));
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'Playfair Display', serif;
            font-weight: 700;
            color: white;
            font-size: 21px;
        }

        .merit-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--merit-text);
        }

        .merit-subtitle {
            font-size: 0.72rem;
            color: var(--merit-text3);
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .merit-badge {
            color: var(--merit-green);
            border: 1px solid rgba(34,211,160,0.28);
            background: rgba(34,211,160,0.10);
            border-radius: 999px;
            padding: 5px 12px;
            font-size: 0.72rem;
            font-weight: 700;
            white-space: nowrap;
        }

        .merit-tabs {
            display: flex;
            flex-wrap: wrap;
            gap: 7px;
            margin-bottom: 18px;
        }

        .merit-tab {
            border-radius: 7px;
            padding: 7px 12px;
            font-size: 0.76rem;
            color: var(--merit-text2);
            border: 1px solid var(--merit-border);
            background: rgba(255,255,255,0.035);
        }

        .merit-tab.active {
            color: var(--merit-accent);
            border-color: rgba(79,142,247,0.32);
            background: rgba(79,142,247,0.13);
        }

        .merit-section-title {
            font-family: 'Playfair Display', Georgia, serif;
            font-size: 1.7rem;
            margin: 16px 0 4px;
            color: var(--merit-text);
        }

        .merit-section-sub {
            color: var(--merit-text2);
            font-size: 0.9rem;
            margin-bottom: 18px;
        }

        .merit-card {
            background: var(--merit-bg2);
            border: 1px solid var(--merit-border);
            border-radius: var(--merit-radius);
            padding: 18px 20px;
            margin: 12px 0;
        }

        .merit-card-title {
            font-size: 0.76rem;
            font-weight: 700;
            color: var(--merit-text3);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .merit-card-title::before {
            content: '';
            display: inline-block;
            width: 3px;
            height: 13px;
            border-radius: 3px;
            background: var(--merit-accent);
        }

        .merit-notice {
            border-radius: var(--merit-radius-sm);
            padding: 12px 14px;
            border: 1px solid var(--merit-border2);
            background: rgba(255,255,255,0.04);
            color: var(--merit-text2);
            margin: 12px 0;
        }

        .merit-notice.success {
            border-color: rgba(34,211,160,0.28);
            background: rgba(34,211,160,0.08);
            color: #bff7e7;
        }

        .merit-notice.warning {
            border-color: rgba(245,158,11,0.30);
            background: rgba(245,158,11,0.08);
            color: #ffe0a3;
        }

        .merit-notice.danger {
            border-color: rgba(244,63,94,0.30);
            background: rgba(244,63,94,0.08);
            color: #ffc2cc;
        }

        .merit-hero {
            background: radial-gradient(circle at top left, rgba(79,142,247,0.20), transparent 34%),
                        radial-gradient(circle at top right, rgba(34,211,160,0.13), transparent 32%),
                        var(--merit-bg2);
            border: 1px solid var(--merit-border);
            border-radius: var(--merit-radius);
            padding: 24px;
            margin-bottom: 18px;
        }

        .merit-hero h1 {
            margin: 0 0 6px;
            font-size: 2.05rem;
        }

        .merit-hero p {
            color: var(--merit-text2);
            max-width: 760px;
            margin: 0;
        }

        .merit-pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 16px;
        }

        .merit-pill {
            font-size: 0.72rem;
            color: var(--merit-text2);
            border: 1px solid var(--merit-border);
            border-radius: 999px;
            padding: 5px 10px;
            background: rgba(255,255,255,0.04);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="merit-header">
            <div class="merit-brand">
                <div class="merit-logo">M</div>
                <div>
                    <div class="merit-title">MeritAI</div>
                    <div class="merit-subtitle">Fair Decision Intelligence</div>
                </div>
            </div>
            <div class="merit-badge">Ethical AI Platform</div>
        </div>
        <div class="merit-tabs">
            <span class="merit-tab active">1 Data Input</span>
            <span class="merit-tab active">2 Governance</span>
            <span class="merit-tab active">3 ML Model</span>
            <span class="merit-tab active">4 Fairness Audit</span>
            <span class="merit-tab active">5 Optimization</span>
            <span class="merit-tab active">6 Simulation</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="merit-hero">
            <h1>{html.escape(title)}</h1>
            <p>{html.escape(subtitle)}</p>
            <div class="merit-pill-row">
                <span class="merit-pill">Bias Detection</span>
                <span class="merit-pill">Constraint Mitigation</span>
                <span class="merit-pill">Counterfactual Testing</span>
                <span class="merit-pill">Audit-Ready Reporting</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="merit-section-title">{html.escape(title)}</div>
        <div class="merit-section-sub">{html.escape(subtitle)}</div>
        """,
        unsafe_allow_html=True,
    )


def notice(text: str, kind: str = "info") -> None:
    kind_class = kind if kind in {"success", "warning", "danger"} else ""
    st.markdown(
        f'<div class="merit-notice {kind_class}">{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )

