"""Pandas analysis and rule-based insights (no ML here)."""
import pandas as pd


def _expenses(df):
    return df[df["type"] == "expense"]


def add_month(df):
    df = df.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
    return df


def calculate_summary(df) -> dict:
    income = float(df.loc[df["type"] == "income", "amount"].sum())
    expenses = float(_expenses(df)["amount"].sum())
    return {
        "income": income,
        "expenses": expenses,
        "balance": income - expenses,
        "savings_rate": (income - expenses) / income * 100 if income > 0 else None,
    }


def category_spending(df) -> pd.DataFrame:
    exp = _expenses(df)
    if exp.empty:
        return pd.DataFrame(columns=["category", "amount"])
    return (exp.groupby("category", as_index=False)["amount"].sum()
            .sort_values("amount", ascending=False).reset_index(drop=True))


def monthly_summary(df) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["month", "income", "expense"])
    p = add_month(df).pivot_table(index="month", columns="type", values="amount",
                                  aggfunc="sum", fill_value=0)
    for col in ("income", "expense"):
        if col not in p:
            p[col] = 0.0
    return p[["income", "expense"]].reset_index()


def budget_status(df, budgets: pd.DataFrame, month: str) -> pd.DataFrame:
    cols = ["category", "limit", "spent", "remaining", "pct_used", "status"]
    if budgets.empty:
        return pd.DataFrame(columns=cols)
    exp = _expenses(df)
    if exp.empty:
        spent = pd.Series(dtype=float)
    else:
        exp = add_month(exp)
        spent = exp[exp["month"] == month].groupby("category")["amount"].sum()
    rows = []
    for b in budgets.itertuples():
        s = float(spent.get(b.category, 0.0))
        pct = s / b.monthly_limit * 100
        status = "Exceeded" if s > b.monthly_limit else "Near limit" if pct >= 80 else "OK"
        rows.append({"category": b.category, "limit": b.monthly_limit, "spent": s,
                     "remaining": b.monthly_limit - s, "pct_used": pct, "status": status})
    return pd.DataFrame(rows, columns=cols)


def month_comparison(df, month: str):
    """Compare expenses with the previous month; None if the previous month has no data."""
    exp = _expenses(df)
    if exp.empty:
        return None
    exp = add_month(exp)
    prev = str(pd.Period(month, "M") - 1)
    cur_total = float(exp.loc[exp["month"] == month, "amount"].sum())
    prev_rows = exp[exp["month"] == prev]
    if prev_rows.empty:
        return None
    prev_total = float(prev_rows["amount"].sum())
    return {"previous_month": prev, "current": cur_total, "previous": prev_total,
            "change_pct": (cur_total - prev_total) / prev_total * 100}


def generate_insights(df, budgets, month: str) -> list:
    out = []
    month_df = add_month(df)
    month_df = month_df[month_df["month"] == month]
    cs = category_spending(month_df)
    if not cs.empty:
        total = cs["amount"].sum()
        top = cs.iloc[0]
        out.append(f"{top['category']} is your biggest expense in {month}: "
                   f"{top['amount']:,.2f} ({top['amount'] / total * 100:.0f}% of spending).")
    cmp_ = month_comparison(df, month)
    if cmp_ is None:
        out.append("Month-over-month comparison is unavailable (no data for the previous month).")
    else:
        direction = "up" if cmp_["change_pct"] > 0 else "down"
        out.append(f"Spending is {direction} {abs(cmp_['change_pct']):.1f}% vs {cmp_['previous_month']}.")
    for r in budget_status(df, budgets, month).itertuples():
        if r.status == "Exceeded":
            out.append(f"Over budget on {r.category} by {-r.remaining:,.2f}.")
        elif r.status == "Near limit":
            out.append(f"{r.category} has used {r.pct_used:.0f}% of its budget.")
    s = calculate_summary(month_df)
    if s["savings_rate"] is not None:
        out.append(f"Savings rate for {month}: {s['savings_rate']:.1f}%.")
    return out
