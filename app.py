"""Exact MeritAI frontend entrypoint.

This Streamlit wrapper renders the provided HTML file directly so the UI matches
the handcrafted frontend instead of approximating it with native Streamlit
widgets.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


FRONTEND_PATH = Path(__file__).parent / "fairness_governance" / "frontend" / "meritai_platform_fixed.html"


st.set_page_config(page_title="MeritAI", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    #MainMenu, header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {
        display: none !important;
    }
    [data-testid="stAppViewContainer"], .main, .block-container {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
    }
    iframe {
        border: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if not FRONTEND_PATH.exists():
    st.error(f"Frontend file not found: {FRONTEND_PATH}")
    st.stop()

html = FRONTEND_PATH.read_text(encoding="utf-8")
components.html(html, height=2600, scrolling=True)
