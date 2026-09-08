"""
Pluggable LLM client with an offline fallback engine.
======================================================
If OPENAI_API_KEY is set, uses an OpenAI-compatible chat endpoint.
Otherwise falls back to the built-in OfflineEngine (deterministic,
rule/template based) so the copilot works with zero API keys.

The copilot never depends on which backend is active — the rest of the
code calls `llm_complete()` / `llm_json()` and gets a string back.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Backend 1: OpenAI-compatible (used only when a key is present)
# ---------------------------------------------------------------------------
def _openai_complete(system: str, prompt: str, max_tokens: int = 800) -> str:
    try:
        from openai import OpenAI
    except Exception:
        return ""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return ""
    try:
        client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
        resp = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "openai/gpt-oss-20b"),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Backend 2: Offline deterministic engine (works with no key)
# ---------------------------------------------------------------------------
class OfflineEngine:
    """Rule/template based fallback so every capability works without an LLM."""

    def complete(self, system: str, prompt: str, max_tokens: int = 800) -> str:
        # A tiny keyword-aware response generator. For text-to-SQL the caller
        # already produced SQL via the query catalog; here we only produce a
        # plain-language gloss when no live LLM is available.
        return ""

    def explain_churn(self, cust_id: str, proba: float, facts: dict) -> str:
        lines = [
            f"Customer {cust_id} has a predicted churn probability of {proba:.1%}.",
        ]
        if facts.get("days_since_last_activity", 0) > 60:
            lines.append(f"- No viewing activity for {facts['days_since_last_activity']:.0f} days (recency risk).")
        if facts.get("failed_payment_count", 0) >= 2:
            lines.append(f"- {facts['failed_payment_count']:.0f} failed payment(s) on record (billing risk).")
        if facts.get("avg_customer_satisfaction", 5) is not None and facts["avg_customer_satisfaction"] <= 2.5:
            lines.append(f"- Low average support satisfaction ({facts['avg_customer_satisfaction']:.1f}/5).")
        if facts.get("monthly_watch_time_trend", 0) < 0:
            lines.append("- Watch time has been trending downward.")
        if facts.get("support_ticket_count", 0) >= 3:
            lines.append(f"- {facts['support_ticket_count']:.0f} support tickets raised (friction signal).")
        if len(lines) == 1:
            lines.append("- No dominant risk factor detected; risk is driven by a combination of engagement and billing signals.")
        return "\n".join(lines)

    def retention_recommendation(self, proba: float, clv: float, segment: str, facts: dict) -> str:
        high_value = clv >= 200
        high_risk = proba >= 0.5
        if high_value and high_risk:
            return ("IMMEDIATE PERSONALIZED OUTREACH: high lifetime value + high churn risk. "
                    "Assign a retention specialist, offer a plan discount or exclusive content access, "
                    "and address the specific friction signals (payment, support, engagement) within 48 hours.")
        if high_value and not high_risk:
            return ("NURTURE: high value, low risk. Maintain engagement with loyalty perks and "
                    "premium content recommendations; monitor monthly for any risk shift.")
        if not high_value and high_risk:
            return ("AUTOMATED PROMOTIONAL CAMPAIGN: low value + high risk. Send an automated win-back "
                    "offer (e.g. 1-month discount) — the cost of the offer is below the customer's lifetime value, "
                    "so it is worth one automated attempt but not a manual specialist.")
        return ("STANDARD MONITORING: low value, low risk. No proactive outreach needed; "
                "track in the monthly health-score review.")

    def root_cause(self, region: str, deltas: dict, signals: dict) -> str:
        parts = [f"Root-cause analysis for churn in {region}:"]
        parts.append(f"- Churn delta vs prior period: {deltas.get('churn_delta_pct', 0):+.2f} pp")
        parts.append(f"- Engagement change (MoM watch minutes): {signals.get('engagement_mom_pct', 0):+.1f}%")
        parts.append(f"- Failed-payment rate change: {signals.get('failed_pay_mom_pct', 0):+.1f}%")
        parts.append(f"- Support-ticket volume change: {signals.get('tickets_mom_pct', 0):+.1f}%")
        parts.append(f"- Plan-mix shift (share of Basic plan): {signals.get('basic_share_pct', 0):+.1f} pp")
        drivers = [k for k, v in signals.items() if v and abs(v) > 5]
        if drivers:
            parts.append(f"Primary drivers detected: {', '.join(drivers)}.")
        else:
            parts.append("No single signal moved more than 5%; the change is spread across multiple factors.")
        return "\n".join(parts)

    def report(self, facts: dict, period: str) -> str:
        lines = [
            f"=== {period.upper()} BUSINESS REPORT (facts computed from the database) ===",
            f"Active customers: {facts.get('active_customers', 0):,}",
            f"Monthly revenue (MRR): ${facts.get('mrr', 0):,.2f}",
            f"Overall churn rate: {facts.get('churn_rate_pct', 0):.2f}%",
            f"New customers this period: {facts.get('new_customers', 0):,}",
            f"Failed payments: {facts.get('failed_payments', 0):,}",
            f"Avg support satisfaction: {facts.get('avg_csat', 0):.2f}/5",
            f"Top complaint: {facts.get('top_complaint', 'n/a')}",
            "",
            "AI INTERPRETATION (separate from facts):",
        ]
        if facts.get("churn_rate_pct", 0) > 3:
            lines.append("- Churn is above the 3% monthly threshold; retention outreach should be prioritized.")
        else:
            lines.append("- Churn is within normal range; maintain current retention programs.")
        if facts.get("failed_payments", 0) > 500:
            lines.append("- Failed-payment volume is elevated; review billing/payment-method health.")
        if facts.get("avg_csat", 5) < 3:
            lines.append("- Support satisfaction is low; investigate ticket resolution quality.")
        lines.append("- All figures above are computed directly from the database; the interpretation is AI-generated.")
        return "\n".join(lines)


OFFLINE = OfflineEngine()


def llm_complete(system: str, prompt: str, max_tokens: int = 800) -> str:
    """Try live LLM; fall back to offline engine (returns '' for pure generation)."""
    if os.environ.get("OPENAI_API_KEY"):
        out = _openai_complete(system, prompt, max_tokens)
        if out:
            return out
    return OFFLINE.complete(system, prompt, max_tokens)


def llm_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))

_ST_MODEL = None

def embed(texts: list[str]) -> list[list[float]]:
    """Get real semantic embeddings using a local, free sentence-transformers model."""
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    vecs = _ST_MODEL.encode(texts, show_progress_bar=False)
    return vecs.tolist()
