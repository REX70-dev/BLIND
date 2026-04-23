# Fairness Governance System

An end-to-end Python and Streamlit system for bias detection, mitigation, counterfactual testing, robustness checks, uncertainty labeling, and audit-ready PDF reporting.

## Run

Use Python 3.11 or 3.12. Python 3.14 is not supported for this project
because Fairlearn currently resolves to a SciPy build that may need a local C/C++
compiler on Windows.

Fast Windows path:

```powershell
cd C:\BLIND
fairness_governance\setup_windows_py312.bat
fairness_governance\run_app.bat
```

```powershell
cd C:\BLIND
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r fairness_governance\requirements.txt
streamlit run fairness_governance\app.py
```

If PowerShell blocks `Activate.ps1`, you can run everything through the venv's
Python executable without activating the environment:

```powershell
cd C:\BLIND
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r fairness_governance\requirements.txt
.\.venv\Scripts\python.exe -m streamlit run fairness_governance\app.py
```

If dependency installation fails on Python 3.14, install Python 3.12, remove the
old environment if it was created with Python 3.14, and create a fresh one:

```powershell
cd C:\BLIND
Remove-Item -Recurse -Force .venv
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r fairness_governance\requirements.txt
.\.venv\Scripts\python.exe -m streamlit run fairness_governance\app.py
```

The app works with an uploaded CSV or the built-in generated credit dataset. For the sample dataset, use:

- Target: `income`
- Sensitive attribute: `sex`
- Fairness metric: `Demographic Parity` or `Equal Opportunity`
- Epsilon: `0.03` for the strongest audit-ready default demo, or `0.01` to `0.10` for trade-off exploration

## Streamlit Community Cloud

Use these deployment settings:

- Repository: `REX70-dev/BLIND`
- Branch: `main`
- Main file path: `app.py`
- Python version: `3.12`

Set Python 3.12 from the Streamlit Cloud app's **Advanced settings**. If the
app was already created with Python 3.14, delete it and redeploy with Python
3.12 selected. Streamlit Community Cloud selects the Python version in the UI.

## Architecture

- `config.py`: Tier 0 fairness charter
- `modules/audit.py`: Tier 1 data audit
- `modules/proxy.py`: Tier 2 proxy detection
- `modules/model.py`: Tier 3 baseline model
- `modules/mitigation.py`: Tiers 4 and 5 reweighting, constraints, post-processing
- `modules/evaluation.py`: Tier 6 comparison charts
- `modules/counterfactual.py`: Tier 7 counterfactual engine
- `modules/intersectional.py`: Tier 8 intersectional analysis
- `modules/robustness.py`: Tier 9 robustness tests
- `modules/uncertainty.py`: Tier 10 uncertainty labels
- `app.py`: Tier 11 dashboard
- `modules/report.py`: Tier 12 PDF report
