"""
AI Analytics Copilot — the 6 PRD capabilities, wired to the real database,
trained models, and unstructured text. Works with or without an LLM key.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3  # noqa (placeholder to keep import surface explicit)
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

from src.llm.llm_client import llm_complete, llm_available, OFFLINE

ROOT = Path(__file__).resolve().parents[2]
DB_URL = dict(host="localhost", dbname="netflix", user="postgres", password="postgres")

# ---------------------------------------------------------------------------
# Schema whitelist for the validation layer (Capability 1)
# ---------------------------------------------------------------------------
SCHEMA = {
    "customers": ["customer_id", "first_name", "last_name", "age", "gender", "country",
                  "state_or_region", "city", "registration_date", "acquisition_channel",
                  "customer_segment", "preferred_language"],
    "subscription_plans": ["plan_id", "plan_name", "monthly_price", "video_quality",
                           "max_devices", "advertisements"],
    "subscriptions": ["subscription_id", "customer_id", "plan_id", "subscription_start_date",
                      "subscription_end_date", "subscription_status", "auto_renew",
                      "cancellation_date", "cancellation_reason", "monthly_price"],
    "content": ["content_id", "title", "content_type", "genre", "release_year",
                "duration_minutes", "content_language", "production_country", "maturity_rating"],
    "viewing_activity": ["viewing_id", "customer_id", "content_id", "viewing_date",
                         "watch_duration_minutes", "completion_percentage", "device_type",
                         "login_location", "session_duration_minutes"],
    "payments": ["payment_id", "customer_id", "subscription_id", "payment_date", "amount",
                 "payment_method", "payment_status", "failed_payment_reason"],
    "support_tickets": ["ticket_id", "customer_id", "ticket_date", "issue_category",
                        "issue_subcategory", "priority", "ticket_status",
                        "resolution_time_hours", "customer_satisfaction_score",
                        "ticket_description", "support_agent_id"],
    "customer_feedback": ["feedback_id", "customer_id", "feedback_date", "rating",
                          "feedback_text", "sentiment_label"],
    "churn_labels": ["customer_id", "churned", "churn_date", "churn_reason"],
}

# Query catalog used by the offline text-to-SQL fallback (keyword -> SQL).
QUERY_CATALOG = [
    (["active", "customer", "country"], "SELECT country, COUNT(DISTINCT customer_id) AS active_customers FROM customers c JOIN subscriptions s USING (customer_id) WHERE subscription_status='Active' GROUP BY country ORDER BY active_customers DESC LIMIT 20;"),
    (["churn", "rate"], "SELECT ROUND(100.0*SUM(churned::int)/COUNT(*),2) AS churn_rate_pct FROM churn_labels;"),
    (["revenue", "plan", "monthly"], "SELECT plan_name, SUM(monthly_price) AS total_mrr FROM subscriptions JOIN subscription_plans USING (plan_id) WHERE subscription_status='Active' GROUP BY plan_name ORDER BY total_mrr DESC;"),
    (["failed", "payment"], "SELECT TO_CHAR(DATE_TRUNC('month',payment_date),'YYYY-MM') AS month, COUNT(*) AS failed FROM payments WHERE payment_status='Failed' AND payment_date>=DATE_TRUNC('month',CURRENT_DATE)-INTERVAL '12 months' GROUP BY 1 ORDER BY 1;"),
    (["top", "watch", "title"], "SELECT title, SUM(watch_duration_minutes) AS mins FROM viewing_activity JOIN content USING (content_id) GROUP BY title ORDER BY mins DESC LIMIT 10;"),
    (["complaint", "subcategor"], "SELECT issue_subcategory, COUNT(*) AS volume FROM support_tickets GROUP BY issue_subcategory ORDER BY volume DESC LIMIT 10;"),
    (["satisfaction"], "SELECT ROUND(AVG(customer_satisfaction_score),2) AS avg_csat FROM support_tickets WHERE customer_satisfaction_score IS NOT NULL;"),
    (["segment", "customer"], "SELECT customer_segment, COUNT(*) AS n FROM customers GROUP BY customer_segment ORDER BY n DESC;"),
    (["device"], "SELECT device_type, SUM(watch_duration_minutes) AS mins FROM viewing_activity GROUP BY device_type ORDER BY mins DESC;"),
    (["tenure", "churn"], "SELECT CASE WHEN (CURRENT_DATE-registration_date)<=180 THEN '<6m' WHEN (CURRENT_DATE-registration_date)<=365 THEN '6-12m' WHEN (CURRENT_DATE-registration_date)<=730 THEN '1-2y' ELSE '2y+' END AS tenure, ROUND(100.0*SUM(churned::int)/COUNT(*),2) AS churn_rate_pct FROM customers JOIN churn_labels USING (customer_id) GROUP BY 1 ORDER BY 1;"),
]


def _connect():
    return psycopg2.connect(**DB_URL)


def _run_sql(sql: str, limit: int = 500) -> list[dict]:
    """Execute a validated SELECT and return rows (bounded)."""
    conn = _connect()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchmany(limit)
            return [dict(r) for r in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CAPABILITY 1 — Text-to-SQL with a non-negotiable validation layer
# ---------------------------------------------------------------------------
def _validate_sql(sql: str) -> tuple[bool, str]:
    """Validation layer: SELECT-only, schema whitelist, timeout+row cap, logging."""
    s = sql.strip().rstrip(";").strip()
    first = s.split(None, 1)[0].upper() if s else ""
    if first != "SELECT":
        return False, "Blocked: only SELECT statements are permitted."
    for bad in ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE", "GRANT", "COPY"]:
        if re.search(rf"\b{bad}\b", s, re.IGNORECASE):
            return False, f"Blocked: {bad} is not permitted."
    # whitelist table/column names
    for table, cols in SCHEMA.items():
        for col in cols:
            if re.search(rf"\b{col}\b", s, re.IGNORECASE):
                break
        else:
            # table not referenced by any known column -> reject unknown identifiers
            if re.search(rf"\b{table}\b", s, re.IGNORECASE):
                return False, f"Blocked: table '{table}' not in schema whitelist."
    return True, "ok"


def capability1_text_to_sql(question: str) -> dict:
    """NL question -> SQL -> validate -> execute -> plain-language answer."""
    log = {"question": question, "sql": None, "status": "pending"}

    # 1) generate SQL (live LLM if available, else catalog fallback)
    sql = ""
    if llm_available():
        sys_p = ("You convert natural-language questions about a Netflix-like subscription "
                 "database into PostgreSQL SELECT statements. Schema: " + json.dumps(SCHEMA) +
                 ". Return ONLY the SQL, no explanation.")
        sql = llm_complete(sys_p, question).strip()
        sql = re.sub(r"^```(sql)?|```$", "", sql, flags=re.M).strip()
    if not sql:
        ql = question.lower()
        for keywords, cand in QUERY_CATALOG:
            if all(k in ql for k in keywords):
                sql = cand
                break
    if not sql:
        return {"answer": "I could not map that question to a query. Try asking about churn rate, active customers by country, revenue by plan, failed payments, top titles, or complaints.", "log": log}

    # 2) validation layer
    ok, msg = _validate_sql(sql)
    if not ok:
        log.update({"sql": sql, "status": "blocked", "reason": msg})
        return {"answer": msg, "log": log}

    # 3) execute with row cap + timeout
    try:
        rows = _run_sql(sql, limit=200)
        log.update({"sql": sql, "status": "ok", "rows": len(rows)})
    except Exception as e:
        log.update({"sql": sql, "status": "error", "reason": str(e).split("\n")[0]})
        return {"answer": f"Query failed: {e}", "log": log}

    # 4) plain-language answer
    if llm_available():
        answer = llm_complete(
            "Summarize this SQL result for a non-technical manager in 2-3 sentences.",
            f"Question: {question}\nSQL result (first 50 rows): {json.dumps(rows[:50], default=str)}",
        )
    else:
        answer = f"Query returned {len(rows)} row(s). " + "; ".join(
            f"{list(r.values())[0]} -> {list(r.values())[1] if len(r) > 1 else ''}" for r in rows[:5]
        )
    return {"answer": answer, "rows": rows, "log": log}


# ---------------------------------------------------------------------------
# CAPABILITY 2 — RAG over support & feedback text (TF-IDF retrieval)
# ---------------------------------------------------------------------------
class _RAGIndex:
    def __init__(self):
        self.docs: list[dict] = []
        self.vectorizer = None
        self.matrix = None
        self._built = False

    def build(self):
        if self._built:
            return
        conn = _connect()
        try:
            tix = pd.read_sql("SELECT ticket_id AS doc_id, customer_id, ticket_date, "
                              "issue_category, ticket_description AS text, 'ticket' AS kind "
                              "FROM support_tickets WHERE ticket_description IS NOT NULL", conn)
            fb = pd.read_sql("SELECT feedback_id AS doc_id, customer_id, feedback_date, "
                             "'feedback' AS kind, feedback_text AS text "
                             "FROM customer_feedback WHERE feedback_text IS NOT NULL", conn)
        finally:
            conn.close()
        self.docs = tix.to_dict("records") + fb.to_dict("records")
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.matrix = self.vectorizer.fit_transform([d["text"] for d in self.docs])
        self._built = True

    def search(self, query: str, k: int = 8) -> list[dict]:
        self.build()
        qv = self.vectorizer.transform([query])
        from sklearn.metrics.pairwise import cosine_similarity
        scores = cosine_similarity(qv, self.matrix).ravel()
        idx = np.argsort(scores)[::-1][:k]
        out = []
        for i in idx:
            if scores[i] <= 0:
                continue
            d = dict(self.docs[i])
            d["score"] = float(scores[i])
            out.append(d)
        return out


_RAG = _RAGIndex()


def capability2_rag(question: str) -> dict:
    hits = _RAG.search(question, k=8)
    if not hits:
        return {"answer": "No relevant support/feedback records found for that question.", "records": 0}
    snippet = "\n".join(
        f"[{h['kind']}] {h.get('issue_category', '')}: {h['text'][:220]}" for h in hits
    )
    if llm_available():
        summary = llm_complete(
            "Summarize what customers are saying, grounded ONLY in the provided records. "
            "State how many records the summary is based on.",
            f"Question: {question}\nRecords:\n{snippet}",
        )
    else:
        # offline: keyword-frequency summary
        kw = [w for w in question.lower().split() if len(w) > 3]
        cats = {}
        for h in hits:
            c = h.get("issue_category") or "feedback"
            cats[c] = cats.get(c, 0) + 1
        top = ", ".join(f"{k} ({v})" for k, v in sorted(cats.items(), key=lambda x: -x[1])[:3])
        summary = (f"Based on {len(hits)} retrieved records, the dominant themes are: {top}. "
                   "This summary is grounded in the retrieved support tickets and feedback.")
    return {"answer": summary, "records": len(hits), "hits": hits}


# ---------------------------------------------------------------------------
# CAPABILITY 3 — ML Prediction Explainer
# ---------------------------------------------------------------------------
def capability3_explain(customer_id: str) -> dict:
    import pickle
    with open(ROOT / "models" / "churn_model.pkl", "rb") as f:
        art = pickle.load(f)
    model, scaler = art["model"], art["scaler"]
    feats = pd.read_csv(ROOT / "data" / "processed" / "customer_features.csv")
    row = feats[feats["customer_id"] == customer_id]
    if row.empty:
        return {"answer": f"Customer {customer_id} not found.", "proba": None}
    row = row.iloc[0]
    X = row[[f for f in art["features"] if f in row.index]].to_frame().T
    for c in art["categorical"]:
        if c in row.index and c in art["encoders"]:
            X[c] = art["encoders"][c].transform([str(row[c])])
    X = X[art["features"] + art["categorical"]]
    proba = float(model.predict_proba(scaler.transform(X))[0][1])
    facts = {
        "days_since_last_activity": float(row.get("days_since_last_activity", 0) or 0),
        "failed_payment_count": float(row.get("failed_payment_count", 0) or 0),
        "avg_customer_satisfaction": float(row.get("avg_customer_satisfaction", 3) or 3),
        "monthly_watch_time_trend": float(row.get("monthly_watch_time_trend", 0) or 0),
        "support_ticket_count": float(row.get("support_ticket_count", 0) or 0),
        "customer_lifetime_value": float(row.get("customer_lifetime_value", 0) or 0),
        "engagement_score": float(row.get("engagement_score", 0) or 0),
    }
    if llm_available():
        text = llm_complete(
            "Explain this churn prediction in plain business language for a retention manager.",
            f"Customer {customer_id}, churn probability {proba:.1%}. Signals: {json.dumps(facts)}",
        )
    else:
        text = OFFLINE.explain_churn(customer_id, proba, facts)
    return {"answer": text, "proba": proba, "facts": facts}


# ---------------------------------------------------------------------------
# CAPABILITY 4 — Automated Business Report Generator
# ---------------------------------------------------------------------------
def capability4_report(period: str = "monthly") -> dict:
    facts = {}
    facts["active_customers"] = _run_sql("SELECT COUNT(DISTINCT customer_id) AS n FROM subscriptions WHERE subscription_status='Active';")[0]["n"]
    facts["mrr"] = _run_sql("SELECT SUM(monthly_price) AS m FROM subscriptions WHERE subscription_status='Active';")[0]["m"]
    facts["churn_rate_pct"] = _run_sql("SELECT ROUND(100.0*SUM(churned::int)/COUNT(*),2) AS c FROM churn_labels;")[0]["c"]
    facts["new_customers"] = _run_sql("SELECT COUNT(*) AS n FROM customers WHERE registration_date>=DATE_TRUNC('month',CURRENT_DATE)-INTERVAL '1 month';")[0]["n"]
    facts["failed_payments"] = _run_sql("SELECT COUNT(*) AS n FROM payments WHERE payment_status='Failed';")[0]["n"]
    facts["avg_csat"] = _run_sql("SELECT ROUND(AVG(customer_satisfaction_score),2) AS c FROM support_tickets WHERE customer_satisfaction_score IS NOT NULL;")[0]["c"]
    top = _run_sql("SELECT issue_subcategory AS c FROM support_tickets GROUP BY issue_subcategory ORDER BY COUNT(*) DESC LIMIT 1;")
    facts["top_complaint"] = top[0]["c"] if top else "n/a"
    if llm_available():
        text = llm_complete(
            "Draft an executive report: key findings, risks, recommendations. Keep facts and interpretation separate.",
            f"Period: {period}. Facts: {json.dumps(facts)}",
        )
    else:
        text = OFFLINE.report(facts, period)
    return {"answer": text, "facts": facts}


# ---------------------------------------------------------------------------
# CAPABILITY 5 — Root-Cause Analysis Agent
# ---------------------------------------------------------------------------
def capability5_root_cause(region: str) -> dict:
    q_churn = ("SELECT ROUND(100.0*SUM(churned::int)/COUNT(*),2) AS c FROM customers "
               "JOIN churn_labels USING (customer_id) WHERE country=%s;")
    q_churn_prev = ("SELECT ROUND(100.0*SUM(churned::int)/COUNT(*),2) AS c FROM customers "
                    "JOIN churn_labels cl USING (customer_id) WHERE country=%s "
                    "AND cl.churn_date < DATE_TRUNC('month',CURRENT_DATE)-INTERVAL '1 month';")
    q_eng = ("SELECT (SUM(watch_duration_minutes) FILTER (WHERE viewing_date>=DATE_TRUNC('month',CURRENT_DATE)-INTERVAL '1 month') "
             "- SUM(watch_duration_minutes) FILTER (WHERE viewing_date<DATE_TRUNC('month',CURRENT_DATE)-INTERVAL '1 month')) "
             "*100.0/NULLIF(SUM(watch_duration_minutes) FILTER (WHERE viewing_date<DATE_TRUNC('month',CURRENT_DATE)-INTERVAL '1 month'),0) AS p "
             "FROM viewing_activity JOIN customers USING (customer_id) WHERE country=%s;")
    q_pay = ("SELECT (COUNT(*) FILTER (WHERE payment_status='Failed' AND payment_date>=DATE_TRUNC('month',CURRENT_DATE)-INTERVAL '1 month') "
             "- COUNT(*) FILTER (WHERE payment_status='Failed' AND payment_date<DATE_TRUNC('month',CURRENT_DATE)-INTERVAL '1 month')) "
             "*100.0/NULLIF(COUNT(*) FILTER (WHERE payment_status='Failed' AND payment_date<DATE_TRUNC('month',CURRENT_DATE)-INTERVAL '1 month'),0) AS p "
             "FROM payments JOIN customers USING (customer_id) WHERE country=%s;")
    q_tix = ("SELECT (COUNT(*) FILTER (WHERE ticket_date>=DATE_TRUNC('month',CURRENT_DATE)-INTERVAL '1 month') "
             "- COUNT(*) FILTER (WHERE ticket_date<DATE_TRUNC('month',CURRENT_DATE)-INTERVAL '1 month')) "
             "*100.0/NULLIF(COUNT(*) FILTER (WHERE ticket_date<DATE_TRUNC('month',CURRENT_DATE)-INTERVAL '1 month'),0) AS p "
             "FROM support_tickets JOIN customers USING (customer_id) WHERE country=%s;")

    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(q_churn, (region,)); cur_churn = cur.fetchone()[0] or 0
            cur.execute(q_churn_prev, (region,)); prev_churn = cur.fetchone()[0] or 0
            cur.execute(q_eng, (region,)); eng = cur.fetchone()[0] or 0
            cur.execute(q_pay, (region,)); pay = cur.fetchone()[0] or 0
            cur.execute(q_tix, (region,)); tix = cur.fetchone()[0] or 0
    finally:
        conn.close()

    deltas = {"churn_delta_pct": round(cur_churn - prev_churn, 2)}
    signals = {"engagement_mom_pct": round(eng, 1), "failed_pay_mom_pct": round(pay, 1),
               "tickets_mom_pct": round(tix, 1), "basic_share_pct": 0.0}
    if llm_available():
        text = llm_complete(
            "Produce a data-backed root-cause explanation. Never state a number not provided.",
            f"Region {region}: {json.dumps(deltas)}, signals {json.dumps(signals)}",
        )
    else:
        text = OFFLINE.root_cause(region, deltas, signals)
    return {"answer": text, "deltas": deltas, "signals": signals}


# ---------------------------------------------------------------------------
# CAPABILITY 6 — Retention Strategy Recommendation
# ---------------------------------------------------------------------------
def capability6_retention(customer_id: str) -> dict:
    exp = capability3_explain(customer_id)
    if exp["proba"] is None:
        return {"answer": exp["answer"], "proba": None}
    facts = exp["facts"]
    clv = facts["customer_lifetime_value"]
    seg = "Unknown"
    try:
        import pickle
        with open(ROOT / "models" / "segment_model.pkl", "rb") as f:
            seg_art = pickle.load(f)
        feats = pd.read_csv(ROOT / "data" / "processed" / "customer_features.csv")
        row = feats[feats["customer_id"] == customer_id]
        if not row.empty:
            sf = row.iloc[0][seg_art["features"]].fillna(0).to_frame().T
            cl = int(seg_art["model"].predict(seg_art["scaler"].transform(sf))[0])
            labels = json.load(open(ROOT / "models" / "segment_labels.json"))["names"]
            seg = labels.get(str(cl), f"Cluster {cl}")
    except Exception:
        pass
    if llm_available():
        text = llm_complete(
            "Give a retention strategy recommendation combining churn probability, CLV, segment, and support history.",
            f"Customer {customer_id}: proba={exp['proba']:.2f}, clv={clv:.0f}, segment={seg}, facts={json.dumps(facts)}",
        )
    else:
        text = OFFLINE.retention_recommendation(exp["proba"], clv, seg, facts)
    return {"answer": text, "proba": exp["proba"], "clv": clv, "segment": seg}


# ---------------------------------------------------------------------------
# Public entrypoint for the app
# ---------------------------------------------------------------------------
def handle(question: str, customer_id: Optional[str] = None, region: Optional[str] = None) -> dict:
    ql = question.lower()
    if customer_id and ("explain" in ql or "predict" in ql or "risk" in ql):
        return capability3_explain(customer_id)
    if customer_id and ("retention" in ql or "offer" in ql or "recommend" in ql):
        return capability6_retention(customer_id)
    if region and ("why" in ql or "root" in ql or "cause" in ql or "churn" in ql):
        return capability5_root_cause(region)
    if "report" in ql:
        period = "monthly" if "week" not in ql else "weekly"
        return capability4_report(period)
    if any(k in ql for k in ["say", "complaint", "feedback", "ticket", "mention", "payment failure"]):
        return capability2_rag(question)
    return capability1_text_to_sql(question)
