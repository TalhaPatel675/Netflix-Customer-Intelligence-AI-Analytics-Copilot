"""
Data Cleaning Pipeline — Netflix Customer Intelligence Platform
===============================================================
Reads the 9 raw CSVs from data/raw/, applies the PRD Section 9.1 cleaning
checklist, and writes clean tables to data/processed/.

Cleaning checklist covered:
  1. Standardize inconsistent category values (country, device_type)
  2. Handle missing values: age, country, city, preferred_language,
     completion_percentage, device_type, customer_satisfaction_score
  3. Detect and resolve duplicate customer records
  4. Detect and handle invalid/malformed dates in support_tickets.ticket_date
  5. Detect and handle outliers in age, watch_duration_minutes,
     session_duration_minutes, payments.amount
  6. Reconcile payments.subscription_id (shipped blank) by matching each
     payment to the customer's active subscription window on payment_date
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
OUT = Path(__file__).resolve().parents[2] / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Canonical mappings (observed raw variants -> canonical value)
# ---------------------------------------------------------------------------
COUNTRY_MAP = {
    # Argentina
    "AR": "Argentina", "ARGENTINA": "Argentina", "Argentina": "Argentina", "argentina": "Argentina",
    # Australia
    "AU": "Australia", "AUSTRALIA": "Australia", "Australia": "Australia", "australia": "Australia",
    # Brazil
    "BR": "Brazil", "BRAZIL": "Brazil", "Brazil": "Brazil", "brazil": "Brazil",
    # Canada
    "CA": "Canada", "CANADA": "Canada", "Canada": "Canada", "canada": "Canada",
    # France
    "FR": "France", "FRANCE": "France", "France": "France", "france": "France",
    # Germany
    "DE": "Germany", "GERMANY": "Germany", "Germany": "Germany", "germany": "Germany",
    # India
    "IN": "India", "INDIA": "India", "India": "India", "india": "India", "Bharat": "India",
    # Indonesia
    "ID": "Indonesia", "INDONESIA": "Indonesia", "Indonesia": "Indonesia", "indonesia": "Indonesia",
    # Italy
    "IT": "Italy", "ITALY": "Italy", "Italy": "Italy", "italy": "Italy",
    # Japan
    "JP": "Japan", "JAPAN": "Japan", "Japan": "Japan", "japan": "Japan",
    # Mexico
    "MX": "Mexico", "MEXICO": "Mexico", "Mexico": "Mexico", "mexico": "Mexico",
    # Nigeria
    "NG": "Nigeria", "NIGERIA": "Nigeria", "Nigeria": "Nigeria", "nigeria": "Nigeria",
    # Philippines
    "PH": "Philippines", "PHILIPPINES": "Philippines", "Philippines": "Philippines", "philippines": "Philippines",
    # Poland
    "PL": "Poland", "POLAND": "Poland", "Poland": "Poland", "poland": "Poland",
    # South Africa
    "ZA": "South Africa", "SOUTH AFRICA": "South Africa", "South Africa": "South Africa", "south africa": "South Africa",
    # South Korea
    "KR": "South Korea", "S. Korea": "South Korea", "South Korea": "South Korea", "south korea": "South Korea",
    # Spain
    "ES": "Spain", "SPAIN": "Spain", "Spain": "Spain", "spain": "Spain",
    # United Arab Emirates
    "U.A.E": "United Arab Emirates", "UAE": "United Arab Emirates", "uae": "United Arab Emirates",
    "United Arab Emirates": "United Arab Emirates",
    # United Kingdom
    "U.K.": "United Kingdom", "UK": "United Kingdom", "Britain": "United Kingdom",
    "United Kingdom": "United Kingdom", "united kingdom": "United Kingdom",
    # United States
    "U.S.A": "United States", "US": "United States", "USA": "United States",
    "United States": "United States", "united states": "United States",
}

DEVICE_MAP = {
    "MOBILE": "Mobile", "Mobile": "Mobile", "Phone": "Mobile", "Smartphone": "Mobile", "mobile": "Mobile",
    "TABLET": "Tablet", "Tablet": "Tablet", "iPad": "Tablet", "tablet": "Tablet",
    "Desktop": "Desktop", "PC": "Desktop", "LAPTOP": "Desktop", "Laptop": "Desktop", "laptop": "Desktop",
    "SMART TV": "Smart TV", "Smart TV": "Smart TV", "SmartTV": "Smart TV", "TV": "Smart TV", "smart tv": "Smart TV",
    "Console": "Console", "Gaming Console": "Console", "PlayStation/Xbox": "Console", "gaming console": "Console",
}


def _fill_mode(s: pd.Series) -> pd.Series:
    mode = s.mode().iloc[0] if not s.mode().empty else "Unknown"
    return s.fillna(mode)


def clean_customers() -> pd.DataFrame:
    df = pd.read_csv(RAW / "customers.csv")
    n_raw = len(df)
    # 3) dedupe duplicate customer records (keep first occurrence)
    df = df.drop_duplicates(subset="customer_id", keep="first")
    # 1) standardize country
    df["country"] = df["country"].map(COUNTRY_MAP).fillna(df["country"])
    # 5) age outliers -> clip to plausible subscriber range, then impute median
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df.loc[df["age"] < 18, "age"] = np.nan
    df.loc[df["age"] > 100, "age"] = np.nan
    df["age"] = df["age"].fillna(df["age"].median()).round().astype("int64")
    # 2) missing values
    df["country"] = _fill_mode(df["country"])
    df["city"] = df["city"].fillna("Unknown")
    df["preferred_language"] = _fill_mode(df["preferred_language"])
    df["state_or_region"] = df["state_or_region"].fillna("Unknown")
    df["gender"] = _fill_mode(df["gender"])
    df["registration_date"] = pd.to_datetime(df["registration_date"], errors="coerce")
    df.to_csv(OUT / "customers.csv", index=False)
    print(f"customers: {n_raw} -> {len(df)} (dropped {n_raw-len(df)} dupes)")
    return df


def clean_subscription_plans() -> pd.DataFrame:
    df = pd.read_csv(RAW / "subscription_plans.csv")
    df.to_csv(OUT / "subscription_plans.csv", index=False)
    print(f"subscription_plans: {len(df)} rows")
    return df


def clean_subscriptions() -> pd.DataFrame:
    df = pd.read_csv(RAW / "subscriptions.csv")
    for c in ["subscription_start_date", "subscription_end_date", "cancellation_date"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["auto_renew"] = df["auto_renew"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    df.to_csv(OUT / "subscriptions.csv", index=False)
    print(f"subscriptions: {len(df)} rows")
    return df


def clean_content() -> pd.DataFrame:
    df = pd.read_csv(RAW / "content.csv")
    df.to_csv(OUT / "content.csv", index=False)
    print(f"content: {len(df)} rows")
    return df


def clean_viewing_activity() -> pd.DataFrame:
    df = pd.read_csv(RAW / "viewing_activity.csv")
    n_raw = len(df)
    # 1) standardize device_type
    df["device_type"] = df["device_type"].map(DEVICE_MAP).fillna(df["device_type"])
    # 2) missing completion_percentage -> median; device_type -> mode
    df["completion_percentage"] = pd.to_numeric(df["completion_percentage"], errors="coerce")
    df["completion_percentage"] = df["completion_percentage"].fillna(
        df["completion_percentage"].median()
    )
    df["device_type"] = _fill_mode(df["device_type"])
    # 5) outliers in watch/session duration -> cap at 480 min (8h, a sane single-session max)
    for c in ["watch_duration_minutes", "session_duration_minutes"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].clip(upper=480)
    df["viewing_date"] = pd.to_datetime(df["viewing_date"], errors="coerce")
    df.to_csv(OUT / "viewing_activity.csv", index=False)
    print(f"viewing_activity: {n_raw} rows (capped {c} outliers)")
    return df


def clean_payments(subs: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(RAW / "payments.csv")
    n_raw = len(df)
    df["payment_date"] = pd.to_datetime(df["payment_date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    # 5) amount outliers -> winsorize at 99th percentile, floor at 0.01
    cap = df["amount"].quantile(0.99)
    df["amount"] = df["amount"].clip(lower=0.01, upper=cap)
    # 6) reconcile subscription_id: match payment to the customer's active
    #    subscription window (start <= payment_date <= end, end NULL = ongoing)
    #    NOTE: drop the blank subscription_id column first to avoid a merge
    #    column collision (pandas would suffix it to _x/_y).
    df = df.drop(columns=["subscription_id"])
    subs = subs.copy()
    subs["_end"] = subs["subscription_end_date"].fillna(pd.Timestamp.max)
    merged = df.merge(
        subs[["customer_id", "subscription_id", "subscription_start_date", "_end"]],
        on="customer_id", how="left",
    )
    mask = (merged["payment_date"] >= merged["subscription_start_date"]) & (
        merged["payment_date"] <= merged["_end"]
    )
    df["subscription_id"] = merged.loc[mask, "subscription_id"]
    df.to_csv(OUT / "payments.csv", index=False)
    matched = df["subscription_id"].notna().sum()
    print(f"payments: {n_raw} rows, subscription_id matched for {matched:,} ({matched/n_raw*100:.1f}%)")
    return df


def clean_support_tickets() -> pd.DataFrame:
    df = pd.read_csv(RAW / "support_tickets.csv")
    n_raw = len(df)
    # 4) invalid/malformed dates (e.g. '2025-13-45') -> coerce to NaT and drop
    df["ticket_date"] = pd.to_datetime(df["ticket_date"], errors="coerce")
    n_bad = df["ticket_date"].isna().sum()
    df = df.dropna(subset=["ticket_date"])
    # 2) CSAT / resolution_time missing only on unresolved tickets -> keep NULL
    df["resolution_time_hours"] = pd.to_numeric(df["resolution_time_hours"], errors="coerce")
    df["customer_satisfaction_score"] = pd.to_numeric(
        df["customer_satisfaction_score"], errors="coerce"
    )
    df.to_csv(OUT / "support_tickets.csv", index=False)
    print(f"support_tickets: {n_raw} -> {len(df)} (dropped {n_bad} malformed dates)")
    return df


def clean_customer_feedback() -> pd.DataFrame:
    df = pd.read_csv(RAW / "customer_feedback.csv")
    df["feedback_date"] = pd.to_datetime(df["feedback_date"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df.to_csv(OUT / "customer_feedback.csv", index=False)
    print(f"customer_feedback: {len(df)} rows")
    return df


def clean_churn_labels() -> pd.DataFrame:
    df = pd.read_csv(RAW / "churn_labels.csv")
    df["churned"] = df["churned"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    df["churn_date"] = pd.to_datetime(df["churn_date"], errors="coerce")
    df.to_csv(OUT / "churn_labels.csv", index=False)
    print(f"churn_labels: {len(df)} rows")
    return df


def run_all() -> None:
    print("=" * 60)
    print("CLEANING PIPELINE START")
    print("=" * 60)
    clean_customers()
    clean_subscription_plans()
    subs = clean_subscriptions()
    clean_content()
    clean_viewing_activity()
    clean_payments(subs)
    clean_support_tickets()
    clean_customer_feedback()
    clean_churn_labels()
    print("=" * 60)
    print("CLEANING PIPELINE COMPLETE ->", OUT)
    print("=" * 60)


if __name__ == "__main__":
    run_all()
