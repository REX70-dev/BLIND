"""Tier 12: PDF audit report generation."""

from __future__ import annotations

import os
from datetime import datetime

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except Exception:  # pragma: no cover - optional dependency
    canvas = None
    letter = None


def generate_pdf_report(
    output_path: str,
    charter: dict,
    audit: dict | None = None,
    proxy: dict | None = None,
    before_metrics: dict | None = None,
    after_metrics: dict | None = None,
    counterfactual: dict | None = None,
    robustness: dict | None = None,
    impact: dict | None = None,
    trust: dict | None = None,
    diagnostics: dict | None = None,
) -> str:
    """Create an audit-ready PDF with the core governance evidence."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if canvas is None:
        with open(output_path.replace(".pdf", ".txt"), "w", encoding="utf-8") as handle:
            handle.write("Install reportlab to generate PDF reports.\n")
        return output_path.replace(".pdf", ".txt")

    pdf = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    y = height - 50

    def line(text: str, gap: int = 16):
        nonlocal y
        if y < 60:
            pdf.showPage()
            y = height - 50
        pdf.drawString(50, y, str(text)[:110])
        y -= gap

    pdf.setTitle("Fairness Governance Audit Report")
    pdf.setFont("Helvetica-Bold", 16)
    line("Fairness Governance Audit Report", 24)
    pdf.setFont("Helvetica", 10)
    line(f"Generated: {datetime.now().isoformat(timespec='seconds')}", 22)

    _section(line, "Fairness Charter", charter)
    line("Disclaimer: Results are based on the UCI Adult dataset and are for demonstration purposes only.")
    if audit:
        _section(
            line,
            "Data Audit",
            {
                "demographic_parity_gap": audit.get("demographic_parity_gap"),
                "equal_opportunity_gap": audit.get("equal_opportunity_gap"),
                "bias_flag": audit.get("bias_flag"),
            },
        )
    if proxy:
        _section(line, "Proxy Detection", proxy)
    if diagnostics:
        _section(line, "Split Diagnostics", diagnostics)
    if before_metrics:
        _section(line, "Baseline Metrics", before_metrics)
    if after_metrics:
        _section(line, "Mitigated Metrics", after_metrics)
    if counterfactual:
        _section(
            line,
            "Counterfactual Summary",
            {"changed_percent": counterfactual.get("changed_percent")},
        )
    if robustness:
        _section(line, "Robustness Summary", robustness)
    if impact:
        _section(line, "Fairness Impact Summary", impact)
    if trust:
        _section(line, "AI Trust Score", trust)

    pdf.save()
    return output_path


def _section(line, title: str, values: dict):
    line("", 8)
    line(title, 18)
    for key, value in values.items():
        line(f"{key}: {value}")
