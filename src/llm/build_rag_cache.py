"""
Precompute real semantic embeddings for support tickets + customer feedback
and cache them to disk, so the running app never has to recompute them.

Run this once, and again any time the underlying data changes:
    python src/llm/build_rag_cache.py
"""
from __future__ import annotations
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()
from src.llm.llm_client import embed  # noqa: E402

CACHE_PATH = ROOT / "data" / "processed" / "rag_cache.npz"
BATCH_SIZE = 96


def _db_url() -> str:
    secrets_path = ROOT / ".streamlit" / "secrets.toml"
    with open(secrets_path, "rb") as f:
        s = tomllib.load(f)
    return (f"postgresql+psycopg2://{s['DB_USER']}:{s['DB_PASSWORD']}"
            f"@{s['DB_HOST']}:{s['DB_PORT']}/{s['DB_NAME']}")


def main():
    engine = create_engine(_db_url())
    tix = pd.read_sql(
        "SELECT ticket_id AS doc_id, customer_id, issue_category, "
        "ticket_description AS text, 'ticket' AS kind FROM support_tickets "
        "WHERE ticket_description IS NOT NULL", engine)
    fb = pd.read_sql(
        "SELECT feedback_id AS doc_id, customer_id, NULL AS issue_category, "
        "feedback_text AS text, 'feedback' AS kind FROM customer_feedback "
        "WHERE feedback_text IS NOT NULL", engine)
    docs = pd.concat([tix, fb], ignore_index=True)
    texts = docs["text"].tolist()
    print(f"Embedding {len(texts)} documents in batches of {BATCH_SIZE}...")

    all_vecs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        all_vecs.extend(embed(batch))
        print(f"  {min(i + BATCH_SIZE, len(texts))}/{len(texts)} done")
        time.sleep(0.5)

    matrix = np.array(all_vecs, dtype=np.float32)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE_PATH,
        matrix=matrix,
        doc_id=docs["doc_id"].to_numpy(),
        customer_id=docs["customer_id"].to_numpy(),
        kind=docs["kind"].to_numpy(),
        text=docs["text"].to_numpy(),
        issue_category=docs["issue_category"].fillna("").to_numpy(),
    )
    print(f"Saved cache to {CACHE_PATH} — matrix shape {matrix.shape}")


if __name__ == "__main__":
    main()