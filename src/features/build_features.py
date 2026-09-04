"""
Feature Engineering — build the customer-level ML dataset.
=========================================================
Builds data/processed/customer_features.csv from the cleaned tables.

PRD Section 9.2 required features (12):
  customer_tenure_days, avg_watch_time_per_session, monthly_watch_time_trend,
  days_since_last_activity, login_frequency, support_ticket_count,
  avg_customer_satisfaction, failed_payment_count, failed_payment_rate,
  plan_changes, engagement_score, customer_lifetime_value

Additional features discovered during EDA (>=3):
  total_watch_minutes, avg_completion_percentage, distinct_titles_watched,
  distinct_devices, avg_session_duration, negative_feedback_count,
  avg_feedback_rating, current_plan_tier

Reference date: 2026-06-30 (end of data window).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
REF_DATE = pd.Timestamp("2026-06-30")


def load():
    cust = pd.read_csv(PROC / "customers.csv")
    subs = pd.read_csv(PROC / "subscriptions.csv")
    view = pd.read_csv(PROC / "viewing_activity.csv")
    pay = pd.read_csv(PROC / "payments.csv")
    tix = pd.read_csv(PROC / "support_tickets.csv")
    fb = pd.read_csv(PROC / "customer_feedback.csv")
    churn = pd.read_csv(PROC / "churn_labels.csv")
    for c in ["registration_date", "subscription_start_date", "subscription_end_date",
              "viewing_date", "payment_date", "ticket_date", "feedback_date", "churn_date"]:
        for df in [cust, subs, view, pay, tix, fb, churn]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
    return cust, subs, view, pay, tix, fb, churn


def build():
    cust, subs, view, pay, tix, fb, churn = load()

    # --- base: every customer ---
    feats = cust[["customer_id", "registration_date"]].copy()
    feats["customer_tenure_days"] = (REF_DATE - feats["registration_date"]).dt.days
    feats = feats.drop(columns=["registration_date"])

    # --- engagement: viewing_activity ---
    view["month"] = view["viewing_date"].dt.to_period("M")
    active_months = view.groupby("customer_id")["month"].nunique()
    sessions = view.groupby("customer_id").size()
    login_freq = (sessions / active_months).rename("login_frequency")
    v = view.groupby("customer_id")
    eng = pd.DataFrame({
        "avg_watch_time_per_session": v["watch_duration_minutes"].mean(),
        "total_watch_minutes": v["watch_duration_minutes"].sum(),
        "avg_session_duration": v["session_duration_minutes"].mean(),
        "avg_completion_percentage": v["completion_percentage"].mean(),
        "days_since_last_activity": (REF_DATE - v["viewing_date"].max()).dt.days,
        "distinct_titles_watched": v["content_id"].nunique(),
        "distinct_devices": v["device_type"].nunique(),
    })
    eng["login_frequency"] = login_freq
    feats = feats.merge(eng, on="customer_id", how="left")

    # --- monthly watch time trend (slope over last 6 months) ---
    monthly = view.groupby(["customer_id", "month"])["watch_duration_minutes"].sum().unstack(fill_value=0)
    last6 = monthly.iloc[:, -6:]
    x = np.arange(last6.shape[1])

    def slope(row):
        if row.sum() == 0:
            return 0.0
        return float(np.polyfit(x, row.values, 1)[0]) if row.notna().all() else 0.0

    feats["monthly_watch_time_trend"] = feats["customer_id"].map(
        last6.apply(slope, axis=1)
    ).fillna(0.0)

    # --- engagement score: composite of recency + frequency + depth ---
    feats["engagement_score"] = (
        0.4 * feats["total_watch_minutes"].fillna(0) / feats["total_watch_minutes"].max()
        + 0.3 * (1 - feats["days_since_last_activity"].fillna(365) / 365.0).clip(0, 1)
        + 0.3 * feats["login_frequency"].fillna(0) / feats["login_frequency"].max()
    )

    # --- billing: payments ---
    p_succ = pay[pay["payment_status"] == "Success"].groupby("customer_id")["amount"].sum()
    p_fail = pay[pay["payment_status"] == "Failed"].groupby("customer_id").size()
    p_tot = pay.groupby("customer_id").size()
    bill = pd.DataFrame({
        "customer_lifetime_value": p_succ,
        "failed_payment_count": p_fail,
        "total_payments": p_tot,
    })
    feats = feats.merge(bill, on="customer_id", how="left")
    feats["failed_payment_count"] = feats["failed_payment_count"].fillna(0).astype(int)
    feats["total_payments"] = feats["total_payments"].fillna(0).astype(int)
    feats["failed_payment_rate"] = (
        feats["failed_payment_count"] / feats["total_payments"].replace(0, np.nan)
    ).fillna(0.0)

    # --- support ---
    t = tix.groupby("customer_id")
    sup = pd.DataFrame({
        "support_ticket_count": t.size(),
        "avg_customer_satisfaction": t["customer_satisfaction_score"].mean(),
    })
    feats = feats.merge(sup, on="customer_id", how="left")
    feats["support_ticket_count"] = feats["support_ticket_count"].fillna(0).astype(int)

    # --- subscriptions: plan changes + current plan ---
    subs = subs.sort_values(["customer_id", "subscription_start_date"])
    plan_rank = {"PLAN01": 1, "PLAN02": 2, "PLAN03": 3}
    subs["plan_rank"] = subs["plan_id"].map(plan_rank)
    subs["prev_rank"] = subs.groupby("customer_id")["plan_rank"].shift(1)
    subs["is_change"] = (subs["plan_rank"] != subs["prev_rank"]).fillna(False).astype(int)
    plan_changes = subs.groupby("customer_id")["is_change"].sum()
    current_plan = subs.sort_values("subscription_start_date").groupby("customer_id").tail(1)
    current_plan = current_plan.set_index("customer_id")["plan_id"]
    feats = feats.merge(plan_changes.rename("plan_changes"), on="customer_id", how="left")
    feats["plan_changes"] = feats["plan_changes"].fillna(0).astype(int)
    feats = feats.merge(current_plan.rename("current_plan"), on="customer_id", how="left")
    feats["current_plan_tier"] = feats["current_plan"].map(plan_rank).fillna(1).astype(int)

    # --- feedback: negative count + avg rating ---
    fb_neg = fb[fb["rating"] <= 2].groupby("customer_id").size().rename("negative_feedback_count")
    fb_avg = fb.groupby("customer_id")["rating"].mean().rename("avg_feedback_rating")
    feats = feats.merge(fb_neg, on="customer_id", how="left")
    feats = feats.merge(fb_avg, on="customer_id", how="left")
    feats["negative_feedback_count"] = feats["negative_feedback_count"].fillna(0).astype(int)

    # --- demographics (categorical, for the model) ---
    demog = cust[["customer_id", "age", "gender", "country", "acquisition_channel",
                  "customer_segment", "preferred_language"]].copy()
    feats = feats.merge(demog, on="customer_id", how="left")

    # --- target ---
    feats = feats.merge(churn[["customer_id", "churned"]], on="customer_id", how="left")
    feats["churned"] = feats["churned"].fillna(False).astype(int)

    # fill remaining NaN numeric with 0 / median
    num_cols = feats.select_dtypes(include=np.number).columns
    for c in num_cols:
        if feats[c].isna().any() and c not in ["churned"]:
            feats[c] = feats[c].fillna(feats[c].median())

    feats.to_csv(PROC / "customer_features.csv", index=False)
    print(f"customer_features.csv: {len(feats)} rows x {feats.shape[1]} cols")
    print("churn rate in dataset:", round(feats['churned'].mean() * 100, 2), "%")
    return feats


if __name__ == "__main__":
    build()
