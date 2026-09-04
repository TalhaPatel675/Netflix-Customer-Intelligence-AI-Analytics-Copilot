"""
EDA — 20+ visualizations across the 7 PRD-mandated themes.
==========================================================
Theme 1: Customer distribution by country, age, acquisition channel
Theme 2: Plan distribution and revenue mix
Theme 3: Cancellation trend over time and by plan
Theme 4: Watch time, completion %, device-usage patterns
Theme 5: Churn rate by country, plan, tenure bucket, engagement tier,
          payment-failure history, support-ticket volume
Theme 6: Support ticket volume and resolution-time trends
Theme 7: Complaint category frequency and relationship to CSAT

Outputs: reports/figures/*.png  (>=20 figures)
Also writes the customer-level feature table used by ML:
         data/processed/customer_features.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams["figure.dpi"] = 110

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
cust = pd.read_csv(PROC / "customers.csv")
plans = pd.read_csv(PROC / "subscription_plans.csv")
subs = pd.read_csv(PROC / "subscriptions.csv")
content = pd.read_csv(PROC / "content.csv")
view = pd.read_csv(PROC / "viewing_activity.csv")
pay = pd.read_csv(PROC / "payments.csv")
tix = pd.read_csv(PROC / "support_tickets.csv")
fb = pd.read_csv(PROC / "customer_feedback.csv")
churn = pd.read_csv(PROC / "churn_labels.csv")

for c in ["registration_date", "subscription_start_date", "subscription_end_date",
          "cancellation_date", "viewing_date", "payment_date", "ticket_date",
          "feedback_date", "churn_date"]:
    for df in [cust, subs, view, pay, tix, fb, churn]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

figs = []

# ===========================================================================
# THEME 1 — Customer distribution
# ===========================================================================
# 1. Top countries by customer count
top_countries = cust["country"].value_counts().head(15)
fig, ax = plt.subplots(figsize=(10, 5))
top_countries.plot(kind="barh", ax=ax, color=sns.color_palette("viridis", 15))
ax.set_title("Customer Distribution by Country (Top 15)")
ax.set_xlabel("Customers"); ax.invert_yaxis()
fig.tight_layout(); fig.savefig(FIG / "1_country_distribution.png"); figs.append("1_country_distribution.png"); plt.close(fig)

# 2. Age distribution
fig, ax = plt.subplots(figsize=(9, 4.5))
sns.histplot(cust["age"], bins=40, kde=True, ax=ax, color="#2c7fb8")
ax.set_title("Customer Age Distribution")
ax.set_xlabel("Age")
fig.tight_layout(); fig.savefig(FIG / "2_age_distribution.png"); figs.append("2_age_distribution.png"); plt.close(fig)

# 3. Acquisition channel mix
fig, ax = plt.subplots(figsize=(9, 4.5))
ch = cust["acquisition_channel"].value_counts()
ax.pie(ch.values, labels=ch.index, autopct="%1.1f%%", startangle=90,
       colors=sns.color_palette("viridis", len(ch)), wedgeprops=dict(width=0.45))
ax.set_title("Acquisition Channel Mix")
fig.tight_layout(); fig.savefig(FIG / "3_acquisition_channel.png"); figs.append("3_acquisition_channel.png"); plt.close(fig)

# 4. Customer segment mix
fig, ax = plt.subplots(figsize=(9, 4.5))
seg = cust["customer_segment"].value_counts()
ax.bar(seg.index, seg.values, color=sns.color_palette("viridis", len(seg)))
ax.set_title("Customer Segment Mix"); ax.set_ylabel("Customers")
fig.tight_layout(); fig.savefig(FIG / "4_segment_mix.png"); figs.append("4_segment_mix.png"); plt.close(fig)

# ===========================================================================
# THEME 2 — Plan distribution and revenue mix
# ===========================================================================
# 5. Active subscriptions by plan
active = subs[subs["subscription_status"] == "Active"]
plan_counts = active["plan_id"].value_counts().reindex(plans["plan_id"])
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(plan_counts.index, plan_counts.values, color=sns.color_palette("viridis", 3))
ax.set_title("Active Subscriptions by Plan"); ax.set_ylabel("Active subs")
fig.tight_layout(); fig.savefig(FIG / "5_plan_distribution.png"); figs.append("5_plan_distribution.png"); plt.close(fig)

# 6. Monthly revenue mix by plan
rev = active.groupby("plan_id")["monthly_price"].sum().reindex(plans["plan_id"])
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.pie(rev.values, labels=rev.index, autopct="%1.1f%%", startangle=90,
       colors=sns.color_palette("viridis", 3), wedgeprops=dict(width=0.45))
ax.set_title("Monthly Revenue Mix by Plan")
fig.tight_layout(); fig.savefig(FIG / "6_revenue_mix.png"); figs.append("6_revenue_mix.png"); plt.close(fig)

# 7. Plan price vs devices (small multiples)
fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(plans["monthly_price"], plans["max_devices"], s=200,
           c=sns.color_palette("viridis", 3))
for _, r in plans.iterrows():
    ax.annotate(r["plan_name"], (r["monthly_price"], r["max_devices"]),
                textcoords="offset points", xytext=(8, 8))
ax.set_title("Plan Price vs Max Devices"); ax.set_xlabel("Monthly price ($)")
ax.set_ylabel("Max devices")
fig.tight_layout(); fig.savefig(FIG / "7_plan_price_devices.png"); figs.append("7_plan_price_devices.png"); plt.close(fig)

# ===========================================================================
# THEME 3 — Cancellation trend over time and by plan
# ===========================================================================
# 8. Cancellations over time (monthly)
canc = subs.dropna(subset=["cancellation_date"]).copy()
canc["canc_month"] = canc["cancellation_date"].dt.to_period("M")
canc_trend = canc.groupby("canc_month").size()
fig, ax = plt.subplots(figsize=(10, 4.5))
canc_trend.plot(ax=ax, marker="o", color="#d95f02")
ax.set_title("Monthly Cancellations Over Time"); ax.set_ylabel("Cancellations")
fig.tight_layout(); fig.savefig(FIG / "8_cancellation_trend.png"); figs.append("8_cancellation_trend.png"); plt.close(fig)

# 9. Cancellations by plan
canc_by_plan = canc["plan_id"].value_counts().reindex(plans["plan_id"])
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(canc_by_plan.index, canc_by_plan.values, color=sns.color_palette("viridis", 3))
ax.set_title("Cancellations by Plan"); ax.set_ylabel("Cancellations")
fig.tight_layout(); fig.savefig(FIG / "9_cancellation_by_plan.png"); figs.append("9_cancellation_by_plan.png"); plt.close(fig)

# 10. Churn reasons
fig, ax = plt.subplots(figsize=(10, 5))
reasons = churn["churn_reason"].value_counts()
reasons.plot(kind="barh", ax=ax, color="#2ca25f")
ax.set_title("Churn Reasons"); ax.set_xlabel("Customers"); ax.invert_yaxis()
fig.tight_layout(); fig.savefig(FIG / "10_churn_reasons.png"); figs.append("10_churn_reasons.png"); plt.close(fig)

# ===========================================================================
# THEME 4 — Watch time, completion, device usage
# ===========================================================================
# 11. Watch duration distribution
fig, ax = plt.subplots(figsize=(9, 4.5))
sns.histplot(view["watch_duration_minutes"], bins=50, kde=True, ax=ax, color="#756bb1")
ax.set_title("Watch Duration per Session (minutes)")
fig.tight_layout(); fig.savefig(FIG / "11_watch_duration.png"); figs.append("11_watch_duration.png"); plt.close(fig)

# 12. Completion percentage distribution
fig, ax = plt.subplots(figsize=(9, 4.5))
sns.histplot(view["completion_percentage"], bins=40, kde=True, ax=ax, color="#e6550d")
ax.set_title("Completion Percentage Distribution")
fig.tight_layout(); fig.savefig(FIG / "12_completion_pct.png"); figs.append("12_completion_pct.png"); plt.close(fig)

# 13. Device usage — total watch minutes by device
dev = view.groupby("device_type")["watch_duration_minutes"].sum().sort_values()
fig, ax = plt.subplots(figsize=(9, 4.5))
dev.plot(kind="barh", ax=ax, color=sns.color_palette("viridis", len(dev)))
ax.set_title("Total Watch Minutes by Device"); ax.set_xlabel("Total minutes")
fig.tight_layout(); fig.savefig(FIG / "13_device_usage.png"); figs.append("13_device_usage.png"); plt.close(fig)

# 14. Viewing activity over time
view["view_month"] = view["viewing_date"].dt.to_period("M")
monthly_views = view.groupby("view_month")["watch_duration_minutes"].sum()
fig, ax = plt.subplots(figsize=(10, 4.5))
monthly_views.plot(ax=ax, marker="o", color="#3182bd")
ax.set_title("Total Watch Minutes per Month"); ax.set_ylabel("Watch minutes")
fig.tight_layout(); fig.savefig(FIG / "14_watch_time_trend.png"); figs.append("14_watch_time_trend.png"); plt.close(fig)

# 15. Top content types by watch minutes
ct_views = view.merge(content[["content_id", "content_type"]], on="content_id")
ct_sum = ct_views.groupby("content_type")["watch_duration_minutes"].sum().sort_values()
fig, ax = plt.subplots(figsize=(8, 4.5))
ct_sum.plot(kind="bar", ax=ax, color=sns.color_palette("viridis", len(ct_sum)))
ax.set_title("Watch Minutes by Content Type"); ax.set_ylabel("Total minutes")
fig.tight_layout(); fig.savefig(FIG / "15_content_type_watch.png"); figs.append("15_content_type_watch.png"); plt.close(fig)

# ===========================================================================
# THEME 5 — Churn rate breakdowns
# ===========================================================================
# 16. Churn rate by country (top 15)
cc = cust.merge(churn, on="customer_id")
country_churn = cc.groupby("country")["churned"].mean().sort_values().tail(15) * 100
fig, ax = plt.subplots(figsize=(10, 5))
country_churn.plot(kind="barh", ax=ax, color="#e6550d")
ax.set_title("Churn Rate by Country (Top 15)"); ax.set_xlabel("Churn rate %")
ax.invert_yaxis()
fig.tight_layout(); fig.savefig(FIG / "16_churn_by_country.png"); figs.append("16_churn_by_country.png"); plt.close(fig)

# 17. Churn rate by plan
plan_churn = subs.merge(churn, on="customer_id").groupby("plan_id")["churned"].mean() * 100
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(plan_churn.index, plan_churn.values, color=sns.color_palette("viridis", 3))
ax.set_title("Churn Rate by Plan"); ax.set_ylabel("Churn rate %")
fig.tight_layout(); fig.savefig(FIG / "17_churn_by_plan.png"); figs.append("17_churn_by_plan.png"); plt.close(fig)

# 18. Churn rate by tenure bucket
cust["tenure_days"] = (pd.Timestamp("2026-06-30") - cust["registration_date"]).dt.days
cust["tenure_bucket"] = pd.cut(cust["tenure_days"], bins=[0, 180, 365, 730, 1095, 9999],
                               labels=["<6m", "6-12m", "1-2y", "2-3y", "3y+"])
tc = cust.merge(churn, on="customer_id")
tenure_churn = tc.groupby("tenure_bucket", observed=True)["churned"].mean() * 100
fig, ax = plt.subplots(figsize=(8, 4.5))
tenure_churn.plot(kind="bar", ax=ax, color="#2c7fb8")
ax.set_title("Churn Rate by Tenure Bucket"); ax.set_ylabel("Churn rate %")
fig.tight_layout(); fig.savefig(FIG / "18_churn_by_tenure.png"); figs.append("18_churn_by_tenure.png"); plt.close(fig)

# 19. Churn rate by payment-failure history
pay_fail = pay[pay["payment_status"] == "Failed"].groupby("customer_id").size().rename("fail_count")
pf = churn.merge(pay_fail, left_on="customer_id", right_index=True, how="left")
pf["fail_count"] = pf["fail_count"].fillna(0)
pf["fail_bucket"] = pd.cut(pf["fail_count"], bins=[-1, 0, 1, 2, 5, 99], labels=["0", "1", "2", "3-5", "5+"])
fail_churn = pf.groupby("fail_bucket", observed=True)["churned"].mean() * 100
fig, ax = plt.subplots(figsize=(8, 4.5))
fail_churn.plot(kind="bar", ax=ax, color="#d95f02")
ax.set_title("Churn Rate by Payment-Failure History"); ax.set_ylabel("Churn rate %")
fig.tight_layout(); fig.savefig(FIG / "19_churn_by_failures.png"); figs.append("19_churn_by_failures.png"); plt.close(fig)

# 20. Churn rate by support-ticket volume
tix_count = tix.groupby("customer_id").size().rename("ticket_count")
stc = churn.merge(tix_count, left_on="customer_id", right_index=True, how="left")
stc["ticket_count"] = stc["ticket_count"].fillna(0)
stc["ticket_bucket"] = pd.cut(stc["ticket_count"], bins=[-1, 0, 1, 2, 4, 99], labels=["0", "1", "2", "3-4", "5+"])
tix_churn = stc.groupby("ticket_bucket", observed=True)["churned"].mean() * 100
fig, ax = plt.subplots(figsize=(8, 4.5))
tix_churn.plot(kind="bar", ax=ax, color="#756bb1")
ax.set_title("Churn Rate by Support-Ticket Volume"); ax.set_ylabel("Churn rate %")
fig.tight_layout(); fig.savefig(FIG / "20_churn_by_tickets.png"); figs.append("20_churn_by_tickets.png"); plt.close(fig)

# 21. Churn rate by engagement tier (watch minutes)
eng = view.groupby("customer_id")["watch_duration_minutes"].sum().rename("watch_total")
ec = churn.merge(eng, left_on="customer_id", right_index=True, how="left")
ec["watch_total"] = ec["watch_total"].fillna(0)
q = ec["watch_total"].quantile([0.25, 0.5, 0.75])
ec["eng_tier"] = pd.cut(ec["watch_total"], bins=[-1, q.iloc[0], q.iloc[1], q.iloc[2], 1e12],
                        labels=["Low", "Medium", "High", "Very High"])
eng_churn = ec.groupby("eng_tier", observed=True)["churned"].mean() * 100
fig, ax = plt.subplots(figsize=(8, 4.5))
eng_churn.plot(kind="bar", ax=ax, color="#2ca25f")
ax.set_title("Churn Rate by Engagement Tier"); ax.set_ylabel("Churn rate %")
fig.tight_layout(); fig.savefig(FIG / "21_churn_by_engagement.png"); figs.append("21_churn_by_engagement.png"); plt.close(fig)

# ===========================================================================
# THEME 6 — Support ticket volume & resolution-time trends
# ===========================================================================
# 22. Ticket volume by issue category
fig, ax = plt.subplots(figsize=(9, 4.5))
cat = tix["issue_category"].value_counts()
cat.plot(kind="bar", ax=ax, color=sns.color_palette("viridis", len(cat)))
ax.set_title("Support Ticket Volume by Issue Category"); ax.set_ylabel("Tickets")
fig.tight_layout(); fig.savefig(FIG / "22_ticket_by_category.png"); figs.append("22_ticket_by_category.png"); plt.close(fig)

# 23. Average resolution time by category
res = tix.dropna(subset=["resolution_time_hours"]).groupby("issue_category")["resolution_time_hours"].mean().sort_values()
fig, ax = plt.subplots(figsize=(9, 4.5))
res.plot(kind="barh", ax=ax, color="#3182bd")
ax.set_title("Average Resolution Time by Category"); ax.set_xlabel("Hours")
fig.tight_layout(); fig.savefig(FIG / "23_resolution_by_category.png"); figs.append("23_resolution_by_category.png"); plt.close(fig)

# 24. Ticket volume trend over time
tix["tix_month"] = tix["ticket_date"].dt.to_period("M")
tix_trend = tix.groupby("tix_month").size()
fig, ax = plt.subplots(figsize=(10, 4.5))
tix_trend.plot(ax=ax, marker="o", color="#e6550d")
ax.set_title("Support Ticket Volume Over Time"); ax.set_ylabel("Tickets")
fig.tight_layout(); fig.savefig(FIG / "24_ticket_trend.png"); figs.append("24_ticket_trend.png"); plt.close(fig)

# ===========================================================================
# THEME 7 — Complaint frequency & CSAT relationship
# ===========================================================================
# 25. Top issue subcategories
fig, ax = plt.subplots(figsize=(10, 5))
subcat = tix["issue_subcategory"].value_counts().head(15)
subcat.plot(kind="barh", ax=ax, color="#2ca25f")
ax.set_title("Top 15 Complaint Subcategories"); ax.set_xlabel("Tickets"); ax.invert_yaxis()
fig.tight_layout(); fig.savefig(FIG / "25_top_complaints.png"); figs.append("25_top_complaints.png"); plt.close(fig)

# 26. Average CSAT by issue category
csat = tix.dropna(subset=["customer_satisfaction_score"]).groupby("issue_category")["customer_satisfaction_score"].mean().sort_values()
fig, ax = plt.subplots(figsize=(9, 4.5))
csat.plot(kind="barh", ax=ax, color="#756bb1")
ax.set_title("Average CSAT by Issue Category"); ax.set_xlabel("Avg CSAT (1-5)")
fig.tight_layout(); fig.savefig(FIG / "26_csat_by_category.png"); figs.append("26_csat_by_category.png"); plt.close(fig)

# 27. CSAT vs resolution time scatter
res_tix = tix.dropna(subset=["customer_satisfaction_score", "resolution_time_hours"])
fig, ax = plt.subplots(figsize=(9, 4.5))
sns.scatterplot(data=res_tix, x="resolution_time_hours", y="customer_satisfaction_score",
                alpha=0.25, color="#2c7fb8", ax=ax)
ax.set_title("CSAT vs Resolution Time")
ax.set_xlabel("Resolution time (hours)"); ax.set_ylabel("CSAT")
fig.tight_layout(); fig.savefig(FIG / "27_csat_vs_resolution.png"); figs.append("27_csat_vs_resolution.png"); plt.close(fig)

# 28. Feedback rating distribution
fig, ax = plt.subplots(figsize=(8, 4.5))
rating_counts = fb["rating"].value_counts().sort_index()
ax.bar(rating_counts.index.astype(str), rating_counts.values, color=sns.color_palette("viridis", 5))
ax.set_title("Customer Feedback Rating Distribution"); ax.set_xlabel("Rating"); ax.set_ylabel("Count")
fig.tight_layout(); fig.savefig(FIG / "28_feedback_rating.png"); figs.append("28_feedback_rating.png"); plt.close(fig)

# 29. Sentiment label distribution
fig, ax = plt.subplots(figsize=(8, 4.5))
sent = fb["sentiment_label"].value_counts()
ax.bar(sent.index, sent.values, color=["#d95f02", "#2c7fb8", "#2ca25f"])
ax.set_title("Feedback Sentiment Distribution"); ax.set_ylabel("Count")
fig.tight_layout(); fig.savefig(FIG / "29_feedback_sentiment.png"); figs.append("29_feedback_sentiment.png"); plt.close(fig)

# 30. Correlation heatmap of key numeric features
feat_df = pd.DataFrame({
    "age": cust.set_index("customer_id")["age"],
    "tenure_days": cust.set_index("customer_id")["tenure_days"],
    "watch_minutes": view.groupby("customer_id")["watch_duration_minutes"].sum(),
    "completion": view.groupby("customer_id")["completion_percentage"].mean(),
    "tickets": tix.groupby("customer_id").size(),
    "payments": pay[pay["payment_status"] == "Success"].groupby("customer_id")["amount"].sum(),
    "failed_pay": pay[pay["payment_status"] == "Failed"].groupby("customer_id").size(),
    "churned": churn.set_index("customer_id")["churned"].astype(int),
})
corr = feat_df.corr()
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
            linewidths=0.5, annot_kws={"size": 9})
ax.set_title("Correlation Heatmap of Key Customer Features")
fig.tight_layout(); fig.savefig(FIG / "30_correlation_heatmap.png"); figs.append("30_correlation_heatmap.png"); plt.close(fig)

print(f"Generated {len(figs)} figures in {FIG}")
for f in figs:
    print("  ", f)
