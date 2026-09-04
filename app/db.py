"""
Shared database + model helpers for the Streamlit app.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

ROOT = Path(__file__).resolve().parents[1]
DB = dict(host="localhost", dbname="netflix", user="postgres", password="postgres")


def query(sql: str, params=None) -> pd.DataFrame:
    conn = psycopg2.connect(**DB)
    try:
        return pd.read_sql(sql, conn, params=params)
    finally:
        conn.close()


def load_features() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "processed" / "customer_features.csv")


def load_churn_model():
    with open(ROOT / "models" / "churn_model.pkl", "rb") as f:
        return pickle.load(f)


def load_segment_model():
    with open(ROOT / "models" / "segment_model.pkl", "rb") as f:
        return pickle.load(f)
    # labels
