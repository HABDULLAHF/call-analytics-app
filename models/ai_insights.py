# models/ai_insights.py
from __future__ import annotations
from typing import Optional, Dict, Any, List
import json
import math

from openai import OpenAI  # pip install openai>=1.40.0
from config.settings import settings  # centralized config (OPENAI_API_KEY, openai_model)

DEFAULT_MODEL = settings.openai_model  # e.g., "gpt-4o-mini" or "gpt-4.1-mini"


# -------------------- small helpers --------------------
def _take(lst, n=50):
    if isinstance(lst, list):
        return lst[: max(0, int(n))]
    return []

def _safe_float(x, default=None):
    try:
        f = float(x)
        if math.isfinite(f):
            return f
        return default
    except Exception:
        return default


# -------------------- prompts --------------------
INSIGHTS_SYSTEM_PROMPT = """\
You are a careful analyst. You will receive a compact JSON summary of call analytics
for ONE main contact (and maybe a selected associated contact). The JSON is already
computed using deterministic rules—DO NOT invent numbers. Instead, propose clear,
concise insights in bullet points, covering patterns such as:
- call volume (count, share of main)
- timing (hours of day, frequent days)
- duration patterns (avg/median, extremes)
- direction and call type balance
- notable date-wise clusters or gaps
- any anomalies (e.g., bursts, very long/short calls)

Write a short executive summary first (1–2 lines), then bullets. Keep it practical.
If something is missing, just skip it—no guesses. Timezone is Asia/Karachi.
"""

INSIGHTS_USER_TEMPLATE = """\
MAIN: {main}
ASSOCIATED (optional): {associated}

SUMMARY JSON (precomputed by app; do not alter numbers):
{summary_json}
"""


# -------------------- core transformation --------------------
def _compact_summary_from_simple(simple_payload: Dict[str, Any], max_calls: int = 120) -> Dict[str, Any]:
    """
    Convert the simple model dict into a compact JSON blob for the LLM.
    We avoid large tables; keep a small sample of calls and key aggregates only.
    """
    if not isinstance(simple_payload, dict):
        return {"error": "simple_payload_not_dict"}

    total_calls = int(simple_payload.get("total_calls") or 0)
    percent_of_total = simple_payload.get("percent_of_total", None)

    per_date_summary = simple_payload.get("per_date_summary") or []
    direction_counts = simple_payload.get("direction_counts") or []
    call_type_counts = simple_payload.get("call_type_counts") or []
    hourly_counts = simple_payload.get("hourly_counts") or {}
    duration_summary = simple_payload.get("duration_summary") or {}

    # Calls — include a small sample with just a few columns
    calls_raw = simple_payload.get("calls") or []
    calls_sample: List[Dict[str, Any]] = []
    if isinstance(calls_raw, list) and len(calls_raw) > 0:
        for r in _take(calls_raw, n=max_calls):
            if not isinstance(r, dict):
                continue
            calls_sample.append({
                "date": r.get("date"),
                "day": r.get("day"),
                "time": r.get("time"),
                "duration_seconds": _safe_float(r.get("duration_seconds"), default=None),
                "direction": r.get("direction"),
                "call_type": r.get("call_type"),
                "associated_key": r.get("associated_key"),
            })

    # Normalize hourly keys to strings "0".."23"
    hourly_norm = {}
    for k, v in hourly_counts.items() if isinstance(hourly_counts, dict) else []:
        ks = str(k)
        if ks.isdigit():
            try:
                hourly_norm[ks] = int(v)
            except Exception:
                hourly_norm[ks] = 0

    summary = {
        "total_calls": total_calls,
        "percent_of_total": percent_of_total,  # may be null
        "duration_summary": {
            "total_sec": _safe_float(duration_summary.get("total_sec"), 0) or 0,
            "avg_sec": _safe_float(duration_summary.get("avg_sec"), 0) or 0,
            "median_sec": _safe_float(duration_summary.get("median_sec"), 0) or 0,
            "min_sec": _safe_float(duration_summary.get("min_sec"), 0) or 0,
            "max_sec": _safe_float(duration_summary.get("max_sec"), 0) or 0,
            "total_hms": duration_summary.get("total_hms") or "0:00:00",
            "avg_hms": duration_summary.get("avg_hms") or "0:00:00",
            "median_hms": duration_summary.get("median_hms") or "0:00:00",
            "min_hms": duration_summary.get("min_hms") or "0:00:00",
            "max_hms": duration_summary.get("max_hms") or "0:00:00",
        },
        "direction_counts": direction_counts,
        "call_type_counts": call_type_counts,
        "hourly_counts": hourly_norm,
        "per_date_summary": _take(per_date_summary, 120),
        "calls_sample": calls_sample,
    }
    return summary


def _call_openai_markdown(client: OpenAI, model: str, system_prompt: str, user_prompt: str) -> str:
    """
    Plain chat completion that returns a markdown string.
    (Avoid function-calling/JSON here because we want natural-language insights.)
    """
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content or ""


# -------------------- public API --------------------
def ai_insights_from_simple(
    simple_payload: Dict[str, Any],
    main_number: str,
    associated: Optional[str] = None,
    *,
    model: str = DEFAULT_MODEL,
    max_calls_for_prompt: int = 120
) -> str:
    """
    Turn the already-computed 'simple model' dict into a short, useful markdown insight section.
    No DataFrame operations here—only dict/list processing.
    """
    if not settings.openai_api_key:
        return "⚠️ OpenAI not configured (set OPENAI_API_KEY in your environment)."

    summary = _compact_summary_from_simple(simple_payload, max_calls=max_calls_for_prompt)
    summary_json = json.dumps(summary, ensure_ascii=False)

    user_prompt = INSIGHTS_USER_TEMPLATE.format(
        main=main_number,
        associated=associated or "(not provided)",
        summary_json=summary_json,
    )

    client = OpenAI(api_key=settings.openai_api_key)

    try:
        md = _call_openai_markdown(
            client=client,
            model=model,
            system_prompt=INSIGHTS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        return md.strip()
    except Exception as e:
        return f"⚠️ openai_error in insights: {e}"
