"""
Load cleaned CSVs into PostgreSQL (netflix DB).
================================================
Run order follows the schema's FK dependencies:
customers -> subscription_plans -> subscriptions -> content -> viewing_activity
-> payments -> support_tickets -> customer_feedback -> churn_labels
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SCHEMA_SQL = Path(__file__).resolve().parent / "schema.sql"

DB_URL = "postgresql+psycopg2://postgres:postgres@localhost:5432/netflix"

# table -> csv file (in load order)
LOAD_ORDER = [
    ("customers", "customers.csv"),
    ("subscription_plans", "subscription_plans.csv"),
    ("subscriptions", "subscriptions.csv"),
    ("content", "content.csv"),
    ("viewing_activity", "viewing_activity.csv"),
    ("payments", "payments.csv"),
    ("support_tickets", "support_tickets.csv"),
    ("customer_feedback", "customer_feedback.csv"),
    ("churn_labels", "churn_labels.csv"),
]

DTYPE_HINTS = {
    "customers": {"age": "int16"},
    "subscriptions": {"auto_renew": "bool"},
    "churn_labels": {"churned": "bool"},
    "payments": {"amount": "float64"},
}


def main() -> None:
    engine = create_engine(DB_URL)

    # 0) drop all tables with CASCADE (reverse dependency order) for a clean reload
    with engine.begin() as conn:
        conn.execute(text(
            "DROP TABLE IF EXISTS churn_labels, customer_feedback, support_tickets, "
            "payments, viewing_activity, content, subscriptions, subscription_plans, "
            "customers CASCADE;"
        ))

    # 1) create schema (tables + indexes + FKs)
    with engine.begin() as conn:
        conn.execute(text(SCHEMA_SQL.read_text()))

    # 2) load each cleaned table (append into freshly-created empty tables)
    for table, csv_name in LOAD_ORDER:
        df = pd.read_csv(PROCESSED / csv_name)
        # normalize dtypes for boolean / numeric columns
        if table in DTYPE_HINTS:
            for col, dtype in DTYPE_HINTS[table].items():
                if col in df.columns:
                    df[col] = df[col].astype(dtype)
        df.to_sql(table, engine, if_exists="append", index=False, method="multi", chunksize=2000)
        with engine.connect() as conn:
            n = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
        print(f"Loaded {table:20s} -> {n:>8,} rows")

    # 3) verify FK integrity
    with engine.connect() as conn:
        orphan_checks = {
            "subscriptions": "customer_id",
            "viewing_activity": "customer_id",
            "payments": "customer_id",
            "support_tickets": "customer_id",
            "customer_feedback": "customer_id",
            "churn_labels": "customer_id",
        }
        for table, col in orphan_checks.items():
            q = text(
                f"SELECT COUNT(*) FROM {table} t LEFT JOIN customers c ON c.customer_id=t.{col} "
                f"WHERE c.customer_id IS NULL"
            )
            orphans = conn.execute(q).scalar()
            print(f"FK check {table}.{col}: {orphans} orphans")

    print("\nLoad complete.")


if __name__ == "__main__":
    main()
