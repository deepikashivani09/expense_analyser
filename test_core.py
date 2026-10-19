import pandas as pd
import pytest

from expense_analyzer import analytics as an, db, ml, services as sv


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(ml, "MODEL_PATH", tmp_path / "m.joblib")
    db.init_db()


def test_add_valid_and_persist():
    sv.add_transaction("2026-01-05", "expense", 20, "Food", "lunch")
    assert len(sv.list_transactions()) == 1


@pytest.mark.parametrize("amt", [0, -5, "abc"])
def test_bad_amount_rejected(amt):
    with pytest.raises(ValueError):
        sv.add_transaction("2026-01-05", "expense", amt, "Food", "x")


def test_blank_description_and_bad_date_rejected():
    with pytest.raises(ValueError):
        sv.add_transaction("2026-01-05", "expense", 5, "Food", "  ")
    with pytest.raises(ValueError):
        sv.add_transaction("not-a-date", "expense", 5, "Food", "x")


def test_summary_and_income():
    sv.add_transaction("2026-01-01", "income", 1000, "Salary", "pay")
    sv.add_transaction("2026-01-02", "expense", 250, "Food", "groceries")
    s = an.calculate_summary(sv.list_transactions())
    assert s["income"] == 1000 and s["expenses"] == 250 and s["balance"] == 750


def test_empty_state():
    s = an.calculate_summary(sv.list_transactions())
    assert s["balance"] == 0 and s["savings_rate"] is None
    assert an.category_spending(sv.list_transactions()).empty


def test_category_spending_sorted():
    sv.add_transaction("2026-01-02", "expense", 10, "Food", "a")
    sv.add_transaction("2026-01-03", "expense", 90, "Housing", "rent")
    cs = an.category_spending(sv.list_transactions())
    assert list(cs["category"]) == ["Housing", "Food"]


def test_budget_exceeded():
    sv.add_transaction("2026-01-02", "expense", 150, "Food", "feast")
    sv.set_budget("Food", 100)
    bs = an.budget_status(sv.list_transactions(), sv.list_budgets(), "2026-01")
    assert bs.iloc[0]["status"] == "Exceeded" and bs.iloc[0]["remaining"] == -50


def test_previous_month_unavailable_then_available():
    sv.add_transaction("2026-02-02", "expense", 100, "Food", "a")
    assert an.month_comparison(sv.list_transactions(), "2026-02") is None
    sv.add_transaction("2026-01-02", "expense", 50, "Food", "b")
    assert an.month_comparison(sv.list_transactions(), "2026-02")["change_pct"] == 100


def test_csv_invalid_rows_reported_and_duplicates_blocked():
    raw = pd.DataFrame({"date": ["2026-01-01", "bad"], "type": ["expense", "expense"],
                        "amount": [5, 5], "category": ["Food", "Food"], "description": ["a", "b"]})
    valid, errors = sv.validate_csv(raw)
    assert len(valid) == 1 and len(errors) == 1
    assert sv.import_rows(valid) == (1, 0)
    assert sv.import_rows(valid) == (0, 1)


def test_csv_missing_columns():
    with pytest.raises(ValueError):
        sv.validate_csv(pd.DataFrame({"date": []}))


def test_prediction():
    with pytest.raises(ValueError):
        ml.predict_category("   ")
    cat, conf = ml.predict_category("netflix subscription")
    assert cat == "Entertainment" and 0 < conf <= 1
