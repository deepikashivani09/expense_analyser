# Expense Analyzer

Streamlit app: SQLite transactions (CRUD), Pandas analytics, Plotly dashboard,
TF-IDF + Logistic Regression expense categoriser (editable predictions),
category budgets with alerts, rule-based insights, CSV import/export.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
pytest -q
```

## Structure
- `expense_analyzer/db.py` – SQLite schema (parameterised queries everywhere)
- `expense_analyzer/services.py` – validation, CRUD, budgets, CSV import/export
- `expense_analyzer/analytics.py` – summaries, monthly trends, budgets, insights
- `expense_analyzer/ml.py` – categoriser (only load model files this app created)
- `app.py` – UI only; no business logic

## Notes
- CSV columns: `date,type,amount,category,description`. Invalid rows are reported; duplicate imports are skipped.
- Use synthetic data for public demos. On Streamlit Community Cloud local files are not persistent; use an external DB for deployment.
- Add authentication before multi-user use.
