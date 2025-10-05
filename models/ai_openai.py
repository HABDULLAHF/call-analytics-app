# models/ai_openai.py
from __future__ import annotations
from typing import Optional, Dict, Any, List
import io
import json
import pandas as pd
from openai import OpenAI

from config.settings import settings  # centralized config (.env)
from app.dataloader import normalize_number

DEFAULT_MODEL = settings.openai_model  # e.g., "gpt-4.1-mini"

# ---------------------------------------------------------------------
# Output schema (Structured Outputs) — strict enough for older validators
# ---------------------------------------------------------------------
AI_JSON_SCHEMA = {
    "name": "call_analytics_payload",
    "schema": {
        "type": "object",
        "properties": {
            "total_calls": {"type": "integer"},
            "percent_of_total": {"type": ["number", "null"]},
            "calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "date": {"type": ["string", "null"]},
                        "day": {"type": ["string", "null"]},
                        "time": {"type": ["string", "null"]},
                        "start_datetime": {"type": ["string", "null"]},
                        "end_datetime": {"type": ["string", "null"]},
                        "duration_seconds": {"type": ["number", "null"]},
                        "associated_number": {"type": ["string", "null"]},
                        "associated_label": {"type": ["string", "null"]},
                        "associated_key": {"type": ["string", "null"]},
                        "direction": {"type": ["string", "null"]},
                        "call_type": {"type": ["string", "null"]},
                        "source_file": {"type": ["string", "null"]}
                    },
                    "required": [
                        "date","day","time","start_datetime","end_datetime",
                        "duration_seconds","associated_number","associated_label",
                        "associated_key","direction","call_type","source_file"
                    ],
                    "additionalProperties": False
                }
            },
            "contacts_overview": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "associated_key": {"type": "string"},
                        "call_count": {"type": "integer"},
                        "percent": {"type": "number"}
                    },
                    "required": ["associated_key","call_count","percent"],
                    "additionalProperties": False
                }
            },
            "per_date_summary": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "Date": {"type": "string"},
                        "Day": {"type": "string"},
                        "Associated (counts)": {"type": "string"},
                        "Call Count": {"type": "integer"},
                        "Call Timing": {"type": "string"},
                        "Call Durations (each)": {"type": "string"},
                        "Total Duration": {"type": "string"},
                        "Avg Duration": {"type": "string"}
                    },
                    "required": [
                        "Date","Day","Associated (counts)","Call Count","Call Timing",
                        "Call Durations (each)","Total Duration","Avg Duration"
                    ],
                    "additionalProperties": False
                }
            },
            "direction_counts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "direction": {"type": ["string","null"]},
                        "count": {"type": "integer"}
                    },
                    "required": ["direction","count"],
                    "additionalProperties": False
                }
            },
            "call_type_counts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "call_type": {"type": ["string","null"]},
                        "count": {"type": "integer"}
                    },
                    "required": ["call_type","count"],
                    "additionalProperties": False
                }
            },
            "hourly_counts": {
                "type": "object",
                "patternProperties": {
                    "^(?:[0-9]|1[0-9]|2[0-3])$": {"type": "integer"}
                },
                "additionalProperties": False
            },
            "duration_summary": {
                "type": "object",
                "properties": {
                    "total_sec": {"type": "integer"},
                    "avg_sec": {"type": "integer"},
                    "median_sec": {"type": "integer"},
                    "max_sec": {"type": "integer"},
                    "min_sec": {"type": "integer"},
                    "total_hms": {"type": "string"},
                    "avg_hms": {"type": "string"},
                    "median_hms": {"type": "string"},
                    "max_hms": {"type": "string"},
                    "min_hms": {"type": "string"}
                },
                "required": [
                    "total_sec","avg_sec","median_sec","max_sec","min_sec",
                    "total_hms","avg_hms","median_hms","max_hms","min_hms"
                ],
                "additionalProperties": False
            },
            "error": {"type": ["string","null"]}
        },
        "required": [
            "total_calls","percent_of_total","calls","contacts_overview","per_date_summary",
            "direction_counts","call_type_counts","hourly_counts","duration_summary","error"
        ],
        "additionalProperties": False
    }
}

# ---------------------------------------------------------------------
# Prompt — mirrors the simple/pandas logic explicitly
# ---------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a precise telecom call-log analyst.

You will receive:
- A MAIN contact number (Pakistan MSISDN like 92300...),
- Optionally an ASSOCIATED (phone or label),
- A CSV containing ONLY rows for the MAIN number (never other mains).

STRICT rules:
1) The CSV timestamps are already in Asia/Karachi; DO NOT rebase timezone.
2) Use ONLY the rows in the CSV. No inventing or deduplicating.
3) Compute the following based solely on this main-only CSV:
   - total_calls: integer = number of rows in the CSV.
   - calls: For EACH row (chronological), include
        date, day, time, start_datetime, end_datetime, duration_seconds,
        associated_number/label/key, direction, call_type, source_file.
   - contacts_overview: counts per associated_key across this main-only CSV,
        with percent = call_count / total_calls * 100 (2 decimals).
   - percent_of_total:
        * If ASSOCIATED is provided, set to the percent for that associated_key from contacts_overview.
        * Otherwise null.
   - per_date_summary: per date,
        "Associated (counts)" = "key (count), key (count), ...",
        "Call Count", "Call Timing" (comma-separated, time-ascending),
        "Call Durations (each)" as H:MM:SS in call order,
        "Total Duration" and "Avg Duration" as H:MM:SS.
   - direction_counts, call_type_counts: frequency over this main-only CSV.
   - hourly_counts: keys "0".."23" (strings) with integer counts; fill missing with 0.
   - duration_summary: totals/avg/median/min/max in seconds and H:MM:SS strings.

Output MUST be valid JSON per the schema exactly. Use null/empty arrays where needed. No explanations outside JSON.
"""

USER_TEMPLATE = """\
MAIN: {main_msisdn}
ASSOCIATED (optional): {assoc_display}

NOTE: The CSV below contains ONLY rows for the MAIN number.
If ASSOCIATED is provided, compute its stats WITHIN this CSV and
percent_of_total = calls_to_associated / total_calls_in_this_CSV * 100.

COLUMNS (CSV): {columns_note}
ROWS: {rows_count}

CSV (main-only; max {max_rows} rows):
{csv_blob}
"""


# ---------------------------------------------------------------------
# Pre-slice rows to match the simple approach exactly
# ---------------------------------------------------------------------
def _slice_for_ai(df: pd.DataFrame, main: str, associated: Optional[str], max_rows: int = 2000) -> pd.DataFrame:
    """
    IMPORTANT: Provide ONLY the MAIN number's rows to the model.
    If 'associated' is provided, the model will filter within this main-only slice.
    """
    main_norm = normalize_number(main)
    if not main_norm:
        return pd.DataFrame()

    # Strictly main-only data
    dfm = df[df["main_number"] == main_norm].copy()

    cols = [
        "date", "day_name", "time",
        "start_datetime", "end_datetime", "duration_seconds",
        "associated_number", "associated_label", "associated_key",
        "direction", "call_type", "source_file",
    ]
    cols = [c for c in cols if c in dfm.columns]
    dfm = dfm[cols].copy()

    # Sort chronologically like simple path
    if "start_datetime" in dfm.columns:
        dfm = dfm.sort_values("start_datetime", kind="stable")
    elif all(c in dfm.columns for c in ("date", "time")):
        dfm = dfm.sort_values(["date", "time"], kind="stable")

    # Keep Karachi tz as-is; render datetimes as strings
    for c in ("start_datetime", "end_datetime"):
        if c in dfm.columns:
            s = pd.to_datetime(dfm[c], errors="coerce")
            dfm[c] = s.astype(str)

    # Rename to match simple model naming ("day" instead of "day_name")
    if "day_name" in dfm.columns:
        dfm = dfm.rename(columns={"day_name": "day"})

    # Limit token cost
    if len(dfm) > max_rows:
        dfm = dfm.head(max_rows)

    return dfm

# ---------------------------------------------------------------------
# OpenAI call with Responses API -> fallback to Chat+Tools
# ---------------------------------------------------------------------
def _call_openai_structured(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    json_schema: Dict[str, Any],
) -> Dict[str, Any]:
    # 1) Preferred: Responses API with structured outputs
    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            response_format={"type": "json_schema", "json_schema": json_schema},
        )
        if hasattr(resp, "output_json") and resp.output_json is not None:
            return resp.output_json
        # try best-effort text parse
        try:
            txt = resp.output_text
            if txt:
                return json.loads(txt)
        except Exception:
            pass
        raise TypeError("responses.create produced no JSON; falling back to tools")
    except TypeError:
        # older SDK without response_format
        pass
    except Exception:
        # any runtime issue -> fallback
        pass

    # 2) Fallback: Chat Completions + Tools (function calling)
    tool_name = json_schema.get("name", "call_analytics_payload")
    parameters_schema = json_schema.get("schema", {"type": "object"})

    chat = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tools=[{
            "type": "function",
            "function": {
                "name": tool_name,
                "description": "Return the call analytics payload in strict JSON.",
                "parameters": parameters_schema,
            },
        }],
        tool_choice={"type": "function", "function": {"name": tool_name}},
    )

    try:
        tc = chat.choices[0].message.tool_calls
        if not tc:
            raise ValueError("No tool_calls in chat completion")
        args_str = tc[0].function.arguments
        return json.loads(args_str)
    except Exception as e:
        try:
            content = chat.choices[0].message.content
            return json.loads(content)
        except Exception:
            raise RuntimeError(f"OpenAI structured output fallback failed: {e}")

# ---------------------------------------------------------------------
# Post-processing: make the AI payload match simple model exactly
# ---------------------------------------------------------------------
def _secs_to_hms(v: Optional[float | int]) -> str:
    try:
        if v is None or pd.isna(v):
            return "0:00:00"
        x = int(float(v))
    except Exception:
        return "0:00:00"
    h = x // 3600
    m = (x % 3600) // 60
    s = x % 60
    return f"{h:d}:{m:02d}:{s:02d}"

def _finalize_payload(
    payload: Dict[str, Any],
    df_slice: pd.DataFrame,
    main: str,
    associated: Optional[str],
) -> Dict[str, Any]:
    # Ensure keys exist
    payload.setdefault("total_calls", int(len(df_slice)))
    payload.setdefault("calls", [])
    payload.setdefault("contacts_overview", [])
    payload.setdefault("percent_of_total", None)
    payload.setdefault("per_date_summary", [])
    payload.setdefault("direction_counts", [])
    payload.setdefault("call_type_counts", [])
    payload.setdefault("hourly_counts", {str(i): 0 for i in range(24)})
    payload.setdefault("duration_summary", {
        "total_sec": 0, "avg_sec": 0, "median_sec": 0, "max_sec": 0, "min_sec": 0,
        "total_hms": "0:00:00", "avg_hms": "0:00:00", "median_hms": "0:00:00",
        "max_hms": "0:00:00", "min_hms": "0:00:00",
    })

    # Coerce hourly 0..23 as strings
    hour_map = {str(i): 0 for i in range(24)}
    for k, v in (payload.get("hourly_counts") or {}).items():
        try:
            hour = int(k)
            if 0 <= hour <= 23:
                hour_map[str(hour)] = int(v)
        except Exception:
            pass
    payload["hourly_counts"] = hour_map

    # Fix calls list: ensure required fields exist, correct types
    fixed_calls: List[Dict[str, Any]] = []
    for r in payload.get("calls", []):
        fixed_calls.append({
            "date": (r.get("date") if r.get("date") is not None else None),
            "day": (r.get("day") if r.get("day") is not None else None),
            "time": (r.get("time") if r.get("time") is not None else None),
            "start_datetime": r.get("start_datetime"),
            "end_datetime": r.get("end_datetime"),
            "duration_seconds": (float(r.get("duration_seconds")) if r.get("duration_seconds") not in (None, "", "null") else None),
            "associated_number": r.get("associated_number"),
            "associated_label": r.get("associated_label"),
            "associated_key": r.get("associated_key"),
            "direction": r.get("direction"),
            "call_type": r.get("call_type"),
            "source_file": r.get("source_file"),
        })
    payload["calls"] = fixed_calls

    # If contacts_overview missing/empty, compute it locally on df_slice
    if not payload.get("contacts_overview"):
        if "associated_key" in df_slice.columns and len(df_slice) > 0:
            vc = (
                df_slice["associated_key"]
                .fillna("Unknown").astype(str)
                .value_counts(dropna=False)
                .reset_index()
            )
            vc.columns = ["associated_key", "call_count"]
            denom = float(len(df_slice)) if len(df_slice) else 1.0
            vc["percent"] = (vc["call_count"] / denom * 100.0).round(2)
            payload["contacts_overview"] = vc.to_dict(orient="records")

    # If percent_of_total is missing and associated is provided → compute from contacts_overview
    if payload.get("percent_of_total") is None and associated:
        assoc_norm = normalize_number(associated)
        key = assoc_norm if assoc_norm else associated.strip()
        co = pd.DataFrame(payload.get("contacts_overview") or [])
        if not co.empty and "associated_key" in co.columns:
            row = co[co["associated_key"] == key]
            if not row.empty:
                payload["percent_of_total"] = float(row["percent"].iloc[0])
            else:
                payload["percent_of_total"] = 0.0

    # Normalize duration_summary strings (HMS) in case model returned weird values
    ds = payload.get("duration_summary") or {}
    for k in ("total_hms","avg_hms","median_hms","max_hms","min_hms"):
        sec_key = k.replace("_hms", "_sec")
        if ds.get(k) in (None, "", "NaN"):
            ds[k] = _secs_to_hms(ds.get(sec_key))
    payload["duration_summary"] = ds

    return payload

# ---------------------------------------------------------------------
# Public entry — AI model path
# ---------------------------------------------------------------------
def ai_model_openai(
    df: pd.DataFrame,
    main_number: str,
    associated: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    max_rows: int = 2000
) -> Dict[str, Any]:
    if not settings.openai_api_key:
        return {
            "total_calls": 0,
            "percent_of_total": None,
            "calls": [],
            "per_date_summary": [],
            "direction_counts": [],
            "call_type_counts": [],
            "hourly_counts": {str(h): 0 for h in range(24)},
            "duration_summary": {
                "total_sec": 0, "avg_sec": 0, "median_sec": 0, "min_sec": 0, "max_sec": 0,
                "total_hms": "0:00:00", "avg_hms": "0:00:00", "median_hms": "0:00:00",
                "min_hms": "0:00:00", "max_hms": "0:00:00",
            },
            "error": "OPENAI_API_KEY missing (set it in .env)",
        }

    df_slice = _slice_for_ai(df, main_number, associated, max_rows=max_rows)

    # Build CSV blob for the model (exactly what it will operate on)
    if df_slice.empty:
        csv_blob = "(no rows for this main/associated)"
        columns_note = "(no columns)"
        rows_count = 0
    else:
        buf = io.StringIO()
        df_slice.to_csv(buf, index=False)
        csv_blob = buf.getvalue()
        columns_note = ", ".join(df_slice.columns)
        rows_count = len(df_slice)

    assoc_display = associated if associated else "(not provided)"
    user_prompt = USER_TEMPLATE.format(
        main_msisdn=main_number,
        assoc_display=assoc_display,
        columns_note=columns_note,
        rows_count=rows_count,
        max_rows=max_rows,
        csv_blob=csv_blob,
    )

    client = OpenAI(api_key=settings.openai_api_key)

    try:
        raw = _call_openai_structured(
            client=client,
            model=model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            json_schema=AI_JSON_SCHEMA,
        )
    except Exception as e:
        return {
            "total_calls": int(rows_count),
            "percent_of_total": None,
            "calls": [],
            "per_date_summary": [],
            "direction_counts": [],
            "call_type_counts": [],
            "hourly_counts": {str(h): 0 for h in range(24)},
            "duration_summary": {
                "total_sec": 0, "avg_sec": 0, "median_sec": 0, "min_sec": 0, "max_sec": 0,
                "total_hms": "0:00:00", "avg_hms": "0:00:00", "median_hms": "0:00:00",
                "min_hms": "0:00:00", "max_hms": "0:00:00",
            },
            "error": f"openai_error: {e}",
        }

    # Finalize to match simple approach exactly (and fill anything missing)
    payload = _finalize_payload(raw or {}, df_slice, main_number, associated)

    return payload
