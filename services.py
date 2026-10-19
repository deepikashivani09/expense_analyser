"""Transaction business logic: validation, CRUD, CSV import/export."""
import hashlib
import math

import pandas as pd

from .db import get_conn

TYPES = ("income", "expense")
CATEGORIES = [
    "Food", "Transport", "Housing", "Utilities", "Entertainment",
    "Shopping", "Health", "Education", "Salary", "Other",
]
REQUIRED_COLUMNS = {"date", "type", "amount", "category", "description"}


def validate_transaction(date, type_, amount, category, description) -> dict:
    t = str(type_).strip().lower()
    if t not in TYPES:
        raise ValueError("Type must be 'income' or 'expense'.")
    d = pd.to_datetime(str(date).strip(), errors="coerce")
    if pd.isna(d):
        raise ValueError("Invalid date.")
    try:
        amt = float(amount)
    except (TypeError, ValueError):
        raise ValueError("Amount must be a number.")
    if not math.isfinite(amt) or amt <= 0:
        raise ValueError("Amount must be greater than zero.")
    match = {c.lower(): c for c in CATEGORIES}.get(str(category).strip().lower())
    if match is None:
        raise ValueError(f"Unknown category: {category}")
    desc = str(description).strip()
    if not desc or desc.lower() == "nan":
        raise ValueError("Description cannot be blank.")
    return {"date": d.strftime("%Y-%m-%d"), "type": t, "amount": round(amt, 2),
            "category": match, "description": desc}


def add_transaction(date, type_, amount, category, description, import_hash=None) -> bool:
    row = validate_transaction(date, type_, amount, category, description)
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO transactions (date, type, amount, category, description, import_hash) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (row["date"], row["type"], row["amount"], row["category"], row["description"], import_hash),
        )
        return cur.rowcount > 0


def update_transaction(tx_id, date, type_, amount, category, description):
    row = validate_transaction(date, type_, amount, category, description)
    with get_conn() as conn:
        conn.execute(
            "UPDATE transactions SET date=?, type=?, amount=?, category=?, description=? WHERE id=?",
            (row["date"], row["type"], row["amount"], row["category"], row["description"], int(tx_id)),
        )


def delete_transaction(tx_id):
    with get_conn() as conn:
        conn.execute("DELETE FROM transactions WHERE id = ?", (int(tx_id),))


def list_transactions() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query(
            "SELECT id, date, type, amount, category, description FROM transactions ORDER BY date DESC, id DESC",
            conn,
        )
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------- Budgets ----------
def set_budget(category, limit):
    if category not in CATEGORIES or float(limit) <= 0:
        raise ValueError("Choose a valid category and a positive limit.")
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO budgets (category, monthly_limit) VALUES (?, ?) "
            "ON CONFLICT(category) DO UPDATE SET monthly_limit = excluded.monthly_limit",
            (category, float(limit)),
        )


def delete_budget(category):
    with get_conn() as conn:
        conn.execute("DELETE FROM budgets WHERE category = ?", (category,))


def list_budgets() -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query("SELECT category, monthly_limit FROM budgets ORDER BY category", conn)


# ---------- CSV ----------
def _row_hash(r: dict) -> str:
    key = "|".join(str(r[k]).lower() for k in ("date", "type", "amount", "category", "description"))
    return hashlib.sha256(key.encode()).hexdigest()


def validate_csv(df: pd.DataFrame):
    """Return (valid_rows, errors). Raises ValueError if required columns are missing."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    valid, errors = [], []
    for i, rec in enumerate(df.to_dict("records"), start=2):  # row 1 is the header
        try:
            valid.append(validate_transaction(*(rec[c] for c in
                         ("date", "type", "amount", "category", "description"))))
        except ValueError as e:
            errors.append({"csv_row": i, "error": str(e), **{k: rec[k] for k in sorted(REQUIRED_COLUMNS)}})
    return pd.DataFrame(valid), pd.DataFrame(errors)


def import_rows(valid: pd.DataFrame):
    """Insert validated rows, skipping duplicates. Returns (inserted, skipped)."""
    inserted = skipped = 0
    for r in valid.to_dict("records"):
        ok = add_transaction(r["date"], r["type"], r["amount"], r["category"],
                             r["description"], import_hash=_row_hash(r))
        inserted += ok
        skipped += not ok
    return inserted, skipped


def export_csv(df: pd.DataFrame) -> bytes:
    out = df.drop(columns=["id"], errors="ignore").copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return out.to_csv(index=False).encode("utf-8")
