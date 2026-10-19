import pandas as pd
import plotly.express as px
import streamlit as st

from expense_analyzer import analytics as an, ml, services as sv
from expense_analyzer.db import init_db

st.set_page_config(page_title="Expense Analyzer", page_icon="💰", layout="wide")
init_db()
st.title("💰 Expense Analyzer")

df_all = sv.list_transactions()

# ---- Sidebar date filter ----
df = df_all
if not df_all.empty:
    lo, hi = df_all["date"].min().date(), df_all["date"].max().date()
    rng = st.sidebar.date_input("Date range", (lo, hi), min_value=lo, max_value=hi)
    if isinstance(rng, tuple) and len(rng) == 2:
        df = df_all[(df_all["date"].dt.date >= rng[0]) & (df_all["date"].dt.date <= rng[1])]

tab_dash, tab_tx, tab_budget, tab_csv = st.tabs(["Dashboard", "Transactions", "Budgets", "Import / Export"])

# ================= Dashboard =================
with tab_dash:
    if df.empty:
        st.info("No transactions yet. Add one in the Transactions tab or import a CSV.")
    else:
        s = an.calculate_summary(df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Income", f"{s['income']:,.2f}")
        c2.metric("Expenses", f"{s['expenses']:,.2f}")
        c3.metric("Balance", f"{s['balance']:,.2f}")
        c4.metric("Savings rate", "n/a" if s["savings_rate"] is None else f"{s['savings_rate']:.1f}%")

        left, right = st.columns(2)
        cs = an.category_spending(df)
        if not cs.empty:
            left.plotly_chart(px.pie(cs, names="category", values="amount", title="Spending by category"),
                              use_container_width=True)
        ms = an.monthly_summary(df)
        right.plotly_chart(px.bar(ms, x="month", y=["income", "expense"], barmode="group",
                                  title="Monthly income vs expenses"), use_container_width=True)

        months = sorted(an.add_month(df)["month"].unique(), reverse=True)
        month = st.selectbox("Month for budgets & insights", months)
        budgets = sv.list_budgets()
        st.subheader("Insights")
        for line in an.generate_insights(df_all, budgets, month):
            st.write("•", line)
        bs = an.budget_status(df_all, budgets, month)
        if not bs.empty:
            st.subheader("Budget status")
            for r in bs.itertuples():
                if r.status == "Exceeded":
                    st.warning(f"⚠️ {r.category}: overspent by {-r.remaining:,.2f}")
                st.progress(min(r.pct_used / 100, 1.0),
                            text=f"{r.category}: {r.spent:,.2f} / {r.limit:,.2f} ({r.status})")

# ================= Transactions =================
with tab_tx:
    st.subheader("Add transaction")
    desc = st.text_input("Description", key="desc")
    if st.button("🤖 Suggest category"):
        try:
            cat, conf = ml.predict_category(desc)
            st.session_state["cat"] = cat
            st.caption(f"Suggested: {cat} ({conf:.0%} confidence) — change it below if it's wrong.")
        except ValueError as e:
            st.error(str(e))
    a, b, c, d = st.columns(4)
    t_date = a.date_input("Date")
    t_type = b.selectbox("Type", sv.TYPES)
    t_amount = c.number_input("Amount", min_value=0.0, step=1.0, format="%.2f")
    t_cat = d.selectbox("Category", sv.CATEGORIES, key="cat")
    if st.button("Save transaction", type="primary"):
        try:
            sv.add_transaction(t_date, t_type, t_amount, t_cat, desc)
            st.success("Saved.")
            st.rerun()
        except ValueError as e:
            st.error(str(e))

    st.subheader("All transactions")
    if df_all.empty:
        st.info("Nothing here yet.")
    else:
        st.dataframe(df_all, use_container_width=True, hide_index=True)
        pick = st.selectbox("Edit / delete transaction ID", df_all["id"])
        row = df_all[df_all["id"] == pick].iloc[0]
        e1, e2, e3, e4 = st.columns(4)
        n_date = e1.date_input("Date ", row["date"].date(), key=f"d{pick}")
        n_type = e2.selectbox("Type ", sv.TYPES, index=sv.TYPES.index(row["type"]), key=f"t{pick}")
        n_amt = e3.number_input("Amount ", value=float(row["amount"]), min_value=0.0, key=f"a{pick}")
        n_cat = e4.selectbox("Category ", sv.CATEGORIES, index=sv.CATEGORIES.index(row["category"]), key=f"c{pick}")
        n_desc = st.text_input("Description ", row["description"], key=f"x{pick}")
        u, dl = st.columns(2)
        if u.button("Update"):
            try:
                sv.update_transaction(pick, n_date, n_type, n_amt, n_cat, n_desc)
                st.rerun()
            except ValueError as e:
                st.error(str(e))
        if dl.button("Delete"):
            sv.delete_transaction(pick)
            st.rerun()

        st.divider()
        if st.button("Retrain categoriser on my corrected data"):
            _, acc = ml.train_model(df_all)
            st.success(f"Model retrained. Hold-out accuracy: {acc:.0%}")

# ================= Budgets =================
with tab_budget:
    b1, b2 = st.columns(2)
    bcat = b1.selectbox("Category", [c for c in sv.CATEGORIES if c != "Salary"], key="bcat")
    blim = b2.number_input("Monthly limit", min_value=0.0, step=10.0, key="blim")
    if st.button("Set budget"):
        try:
            sv.set_budget(bcat, blim)
            st.rerun()
        except ValueError as e:
            st.error(str(e))
    budgets = sv.list_budgets()
    if budgets.empty:
        st.info("No budgets set.")
    else:
        st.dataframe(budgets, use_container_width=True, hide_index=True)
        rm = st.selectbox("Remove budget", budgets["category"])
        if st.button("Remove"):
            sv.delete_budget(rm)
            st.rerun()

# ================= CSV =================
with tab_csv:
    st.subheader("Import")
    st.caption("Required columns: date, type, amount, category, description")
    up = st.file_uploader("Upload transaction CSV", type=["csv"])
    if up is not None:
        try:
            raw = pd.read_csv(up)
            valid, errors = sv.validate_csv(raw)
            st.write(f"**Preview** — {len(valid)} valid row(s), {len(errors)} invalid row(s)")
            st.dataframe(valid, use_container_width=True)
            if not errors.empty:
                st.error("Invalid rows (not imported):")
                st.dataframe(errors, use_container_width=True)
            if not valid.empty and st.button("Import valid rows"):
                ins, skip = sv.import_rows(valid)
                st.success(f"Imported {ins} row(s); skipped {skip} duplicate(s).")
        except ValueError as e:
            st.error(str(e))
        except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
            st.error("The CSV could not be read. Check its format.")

    st.subheader("Export")
    st.download_button("Download transactions CSV", sv.export_csv(df_all),
                       "transactions.csv", "text/csv", disabled=df_all.empty)
