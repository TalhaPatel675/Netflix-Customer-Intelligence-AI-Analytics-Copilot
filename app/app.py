"""
Netflix Customer Intelligence & AI Analytics Copilot — Streamlit app.
7 pages per PRD Section 13.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import db

st.set_page_config(page_title="Netflix Customer Intelligence", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🎬 Netflix Copilot")
st.sidebar.caption("Customer Intelligence · Churn · AI Analytics")
page = st.sidebar.radio(
    "Navigate",
    ["Executive Dashboard", "Customer Analytics", "Churn Prediction",
     "Customer Segmentation", "AI Analytics Copilot", "Support Intelligence",
     "Automated Reports"],
)

# ---------------------------------------------------------------------------
# 1. EXECUTIVE DASHBOARD
# ---------------------------------------------------------------------------
if page == "Executive Dashboard":
    st.title("📊 Executive Dashboard")
    active = db.query("SELECT COUNT(DISTINCT customer_id) AS n FROM subscriptions WHERE subscription_status='Active'").iloc[0, 0]
    total = db.query("SELECT COUNT(*) AS n FROM customers").iloc[0, 0]
    churn = db.query("SELECT ROUND(100.0*SUM(churned::int)/COUNT(*),2) AS c FROM churn_labels").iloc[0, 0]
    mrr = db.query("SELECT SUM(monthly_price) AS m FROM subscriptions WHERE subscription_status='Active'").iloc[0, 0]
    high_risk = None
    avg_csat = db.query("SELECT ROUND(AVG(customer_satisfaction_score),2) AS c FROM support_tickets WHERE customer_satisfaction_score IS NOT NULL").iloc[0, 0]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Customers", f"{total:,}")
    c2.metric("Active Subscribers", f"{active:,}")
    c3.metric("Churn Rate", f"{churn:.2f}%")
    c4.metric("Monthly Revenue", f"${mrr:,.0f}")
    c5.metric("Avg Support CSAT", f"{avg_csat:.2f}/5")

    st.subheader("Revenue by Plan")
    rev = db.query("""SELECT plan_name, SUM(subscriptions.monthly_price) AS mrr FROM subscriptions
                      JOIN subscription_plans USING (plan_id)
                      WHERE subscription_status='Active' GROUP BY plan_name ORDER BY mrr DESC""")
    st.bar_chart(rev.set_index("plan_name"))

    st.subheader("Monthly Cancellations")
    canc = db.query("""SELECT TO_CHAR(DATE_TRUNC('month',cancellation_date),'YYYY-MM') AS m, COUNT(*) AS n
                       FROM subscriptions WHERE cancellation_date IS NOT NULL
                       GROUP BY 1 ORDER BY 1""")
    st.line_chart(canc.set_index("m"))

# ---------------------------------------------------------------------------
# 2. CUSTOMER ANALYTICS
# ---------------------------------------------------------------------------
elif page == "Customer Analytics":
    st.title("👥 Customer Analytics")
    tab1, tab2, tab3 = st.tabs(["Demographics", "Engagement", "Subscription"])

    with tab1:
        st.subheader("Customers by Country (Top 15)")
        d = db.query("SELECT country, COUNT(*) AS n FROM customers GROUP BY country ORDER BY n DESC LIMIT 15")
        st.bar_chart(d.set_index("country"))

        st.subheader("Age Distribution")
        ages = db.query("SELECT age FROM customers")
        st.bar_chart(ages["age"].value_counts().sort_index().head(60))

    with tab2:
        st.subheader("Watch Minutes by Device")
        d = db.query("SELECT device_type, SUM(watch_duration_minutes) AS mins FROM viewing_activity GROUP BY device_type ORDER BY mins DESC")
        st.bar_chart(d.set_index("device_type"))

        st.subheader("Top 10 Most-Watched Titles")
        d = db.query("""SELECT title, SUM(watch_duration_minutes) AS mins FROM viewing_activity
                        JOIN content USING (content_id) GROUP BY title ORDER BY mins DESC LIMIT 10""")
        st.dataframe(d)

    with tab3:
        st.subheader("Active Subscriptions by Plan")
        d = db.query("""SELECT plan_name, COUNT(*) AS n FROM subscriptions
                        JOIN subscription_plans USING (plan_id)
                        WHERE subscription_status='Active' GROUP BY plan_name""")
        st.bar_chart(d.set_index("plan_name"))

        st.subheader("Acquisition Channel Mix")
        d = db.query("SELECT acquisition_channel, COUNT(*) AS n FROM customers GROUP BY acquisition_channel ORDER BY n DESC")
        st.dataframe(d)

# ---------------------------------------------------------------------------
# 3. CHURN PREDICTION
# ---------------------------------------------------------------------------
elif page == "Churn Prediction":
    st.title("🎯 Churn Prediction")
    feats = db.load_features()
    cust_id = st.selectbox("Select a customer", feats["customer_id"].tolist())

    if st.button("Predict Churn Risk", type="primary"):
        art = db.load_churn_model()
        model, scaler = art["model"], art["scaler"]
        row = feats[feats["customer_id"] == cust_id].iloc[0]
        X = row[[f for f in art["features"] if f in row.index]].to_frame().T
        for c in art["categorical"]:
            if c in row.index and c in art["encoders"]:
                X[c] = art["encoders"][c].transform([str(row[c])])
        X = X[art["features"] + art["categorical"]]
        proba = float(model.predict_proba(scaler.transform(X))[0][1])

        st.metric("Churn Probability", f"{proba:.1%}")
        st.progress(min(proba, 1.0))
        risk = "HIGH" if proba >= 0.5 else "MODERATE" if proba >= 0.3 else "LOW"
        st.info(f"Risk level: **{risk}**")

        st.subheader("Key Signals")
        sig = {
            "Days since last activity": row["days_since_last_activity"],
            "Failed payments": row["failed_payment_count"],
            "Support tickets": row["support_ticket_count"],
            "Watch-time trend": round(row["monthly_watch_time_trend"], 1),
            "Engagement score": round(row["engagement_score"], 2),
            "Lifetime value": round(row["customer_lifetime_value"], 2),
        }
        st.json(sig)

# ---------------------------------------------------------------------------
# 4. CUSTOMER SEGMENTATION
# ---------------------------------------------------------------------------
elif page == "Customer Segmentation":
    st.title("🧩 Customer Segmentation")
    try:
        art = db.load_segment_model()
        feats = db.load_features()
        seg_feats = art["features"]
        X = feats[seg_feats].fillna(0)
        labels = art["model"].predict(art["scaler"].transform(X))
        with open(ROOT / "models" / "segment_labels.json") as f:
            seg_labels = json.load(f)
        names = seg_labels["names"]
        feats["segment"] = [names.get(str(l), f"Cluster {l}") for l in labels]

        st.subheader("Segment Sizes")
        counts = feats["segment"].value_counts()
        st.bar_chart(counts)

        st.subheader("Segment Profiles")
        profile = feats.groupby("segment")[["engagement_score", "customer_lifetime_value",
                                            "customer_tenure_days", "failed_payment_rate"]].mean().round(2)
        st.dataframe(profile)

        st.caption(f"K-Means with K={seg_labels['best_k']} (silhouette {seg_labels['silhouette']:.3f})")
    except Exception as e:
        st.error(f"Segmentation model not available: {e}")

# ---------------------------------------------------------------------------
# 5. AI ANALYTICS COPILOT
# ---------------------------------------------------------------------------
elif page == "AI Analytics Copilot":
    st.title("🤖 AI Analytics Copilot")
    st.caption("Ask business questions in plain English — backed by the real database, trained models, and unstructured text.")
    sys.path.insert(0, str(ROOT))
    from src.llm import copilot

    q = st.text_input("Your question", placeholder="e.g. What is the churn rate? What are customers saying about payment failures?")
    col1, col2, col3 = st.columns(3)
    cust_id = col1.text_input("Customer ID (for predict/explain/retention)", "")
    region = col2.text_input("Region (for root-cause)", "")
    go = col3.button("Ask", type="primary")

    if go and q:
        with st.spinner("Thinking..."):
            res = copilot.handle(q, customer_id=cust_id or None, region=region or None)
        st.markdown("**Answer:**")
        st.write(res["answer"])
        if "rows" in res and res["rows"]:
            st.subheader("Result Table")
            st.dataframe(pd.DataFrame(res["rows"]))
        if "log" in res:
            with st.expander("Query log (auditability)"):
                st.json(res["log"])
        if "records" in res:
            st.caption(f"Grounded in {res['records']} retrieved records.")

# ---------------------------------------------------------------------------
# 6. SUPPORT INTELLIGENCE
# ---------------------------------------------------------------------------
elif page == "Support Intelligence":
    st.title("🎧 Support Intelligence")
    tab1, tab2 = st.tabs(["Complaint Analysis", "Sentiment"])

    with tab1:
        st.subheader("Ticket Volume by Issue Category")
        d = db.query("SELECT issue_category, COUNT(*) AS n FROM support_tickets GROUP BY issue_category ORDER BY n DESC")
        st.bar_chart(d.set_index("issue_category"))

        st.subheader("Top Complaint Subcategories")
        d = db.query("SELECT issue_subcategory, COUNT(*) AS n FROM support_tickets GROUP BY issue_subcategory ORDER BY n DESC LIMIT 15")
        st.dataframe(d)

    with tab2:
        st.subheader("Feedback Sentiment Distribution")
        d = db.query("SELECT sentiment_label, COUNT(*) AS n FROM customer_feedback GROUP BY sentiment_label")
        st.bar_chart(d.set_index("sentiment_label"))

        st.subheader("Avg Rating by Sentiment")
        d = db.query("SELECT sentiment_label, ROUND(AVG(rating),2) AS avg_rating FROM customer_feedback GROUP BY sentiment_label")
        st.dataframe(d)

        st.subheader("AI Summary of Customer Sentiment")
        sys.path.insert(0, str(ROOT))
        from src.llm import copilot
        res = copilot.capability2_rag("what are the main complaints and what is customer sentiment?")
        st.write(res["answer"])

# ---------------------------------------------------------------------------
# 7. AUTOMATED REPORTS
# ---------------------------------------------------------------------------
elif page == "Automated Reports":
    st.title("📄 Automated Reports")
    period = st.selectbox("Report period", ["weekly", "monthly", "quarterly"])
    if st.button("Generate Report", type="primary"):
        sys.path.insert(0, str(ROOT))
        from src.llm import copilot
        with st.spinner("Generating report..."):
            res = copilot.capability4_report(period)
        st.markdown(res["answer"])
        st.download_button("Download report (.txt)", res["answer"],
                           file_name=f"netflix_report_{period}.txt")
