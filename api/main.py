# api/main.py
from __future__ import annotations
from typing import Optional, List, Dict, Any
import os
from collections import Counter, defaultdict

import pandas as pd
from fastapi import FastAPI, Query, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from config.settings import settings
from app.dataloader import load_calls
from models.analytics import simple_model  # server-side pandas analytics
from models.ai_openai import ai_model_openai  # OpenAI-powered analytics
# near other imports
from models.ai_insights import ai_insights_from_simple

# =============================================================================
# App & data
# =============================================================================
DATA_DIR = settings.data_dir

app = FastAPI(title="Call Analytics API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load once at process start
CALLS_DF = load_calls(DATA_DIR)

# =============================================================================
# Pydantic models (for stable responses)
# =============================================================================
class ContactsOverview(BaseModel):
    associated_key: str
    call_count: int
    percent: float

class ContactsResponse(BaseModel):
    total_calls: int
    contacts: List[ContactsOverview]

class CallRecord(BaseModel):
    date: Optional[str] = None
    day: Optional[str] = None
    time: Optional[str] = None
    start_datetime: Optional[str] = None
    end_datetime: Optional[str] = None
    duration_seconds: Optional[float] = None
    associated_number: Optional[str] = None
    associated_label: Optional[str] = None
    associated_key: Optional[str] = None
    direction: Optional[str] = None
    call_type: Optional[str] = None
    source_file: Optional[str] = None

class DirectionCount(BaseModel):
    direction: Optional[str] = None
    count: int

class CallTypeCount(BaseModel):
    call_type: Optional[str] = None
    count: int

class DurationSummary(BaseModel):
    total_sec: int
    avg_sec: int
    median_sec: int
    max_sec: int
    min_sec: int
    total_hms: str
    avg_hms: str
    median_hms: str
    max_hms: str
    min_hms: str

class StatsResponse(BaseModel):
    total_calls: int
    percent_of_total: Optional[float] = None
    contacts_overview: List[Any] | None = None  # [{associated_key, call_count, percent}]
    calls: List[CallRecord]
    direction_counts: List[DirectionCount] | None = None
    call_type_counts: List[CallTypeCount] | None = None
    duration_summary: DurationSummary | None = None
    hourly_counts: Dict[str, int] | None = None  # keys "0".."23" for UI compatibility
    per_date_summary: List[Dict[str, Any]] | None = None
    error: Optional[str] = None

# =============================================================================
# Helpers (NaN/Inf-safe + aggregates)
# =============================================================================


# --- JSON sanitizers ---
import math
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

def _is_na_like(x) -> bool:
    try:
        # catches pandas NA/NaT/NaN etc.
        import pandas as pd  # already installed
        if pd.isna(x):
            return True
    except Exception:
        pass
    return False

def _finite_float_or_none(x):
    try:
        f = float(x)
        return f if math.isfinite(f) else None
    except Exception:
        return None

def _sanitize_scalar(x):
    # order matters: NA-like first
    if _is_na_like(x):
        return None
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    return x

def _normalize_hourly(obj):
    """
    Ensure hourly_counts is a dict with string keys "0".."23" and integer values (no NaN/Inf).
    """
    base = {str(h): 0 for h in range(24)}
    if isinstance(obj, dict):
        for k, v in obj.items():
            try:
                h = int(k)
                if 0 <= h <= 23:
                    base[str(h)] = int(_finite_float_or_none(v) or 0)
            except Exception:
                continue
    return base

def sanitize(obj):
    """
    Recursively sanitize dict/list/scalars: replace NaN/NaT/Inf with None,
    normalize hourly_counts to "0".."23" keys, and coerce nested structures.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "hourly_counts":
                out[k] = _normalize_hourly(v)
            else:
                out[k] = sanitize(v)
        return out
    elif isinstance(obj, list):
        return [sanitize(v) for v in obj]
    else:
        return _sanitize_scalar(obj)
    
def _str_or_none(v) -> Optional[str]:
    try:
        if v is None or pd.isna(v):
            return None
    except Exception:
        pass
    s = str(v).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "inf", "-inf"}:
        return None
    return s

def _float_or_none(v) -> Optional[float]:
    try:
        if v is None or pd.isna(v):
            return None
    except Exception:
        pass
    try:
        x = float(v)
        if not pd.isna(x) and pd.notna(x) and x not in (float("inf"), float("-inf")):
            return x
    except Exception:
        return None
    return None

def _fmt_hms(seconds: float | int | None) -> str:
    if seconds is None or not pd.notna(seconds):
        return "0:00:00"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:d}:{m:02d}:{s:02d}"

def _aggregate_from_calls_list(calls: List[dict]) -> dict:
    """
    Build direction/call_type/hourly/duration aggregates from a list of call dicts.
    Always returns JSON-safe values (no NaN/Inf), hourly keys as strings "0".."23".
    """
    if not calls:
        return {
            "direction_counts": [],
            "call_type_counts": [],
            "duration_summary": {
                "total_sec": 0, "avg_sec": 0, "median_sec": 0, "max_sec": 0, "min_sec": 0,
                "total_hms": "0:00:00", "avg_hms": "0:00:00", "median_hms": "0:00:00",
                "max_hms": "0:00:00", "min_hms": "0:00:00",
            },
            "hourly_counts": {str(i): 0 for i in range(24)},
        }

    dir_counter = Counter()
    type_counter = Counter()
    durations: List[float] = []
    hours: List[int] = []

    for r in calls:
        d = _str_or_none(r.get("direction")) or "Unknown"
        t = _str_or_none(r.get("call_type")) or "Unknown"
        dir_counter[d] += 1
        type_counter[t] += 1

        sec = _float_or_none(r.get("duration_seconds"))
        if sec is not None:
            durations.append(sec)

        tm = _str_or_none(r.get("time"))
        if tm:
            parts = tm.split(":")
            try:
                hr = int(parts[0])
                if 0 <= hr <= 23:
                    hours.append(hr)
            except Exception:
                pass

    direction_counts = [{"direction": k, "count": int(v)} for k, v in dir_counter.most_common()]
    call_type_counts = [{"call_type": k, "count": int(v)} for k, v in type_counter.most_common()]

    if durations:
        s = pd.Series(durations, dtype="float64")
        total = int(float(s.sum()))
        avg = int(float(s.mean())) if len(s) else 0
        med = int(float(s.median())) if len(s) else 0
        mx = int(float(s.max()))
        mn = int(float(s.min()))
    else:
        total = avg = med = mx = mn = 0

    duration_summary = {
        "total_sec": total, "avg_sec": avg, "median_sec": med, "max_sec": mx, "min_sec": mn,
        "total_hms": _fmt_hms(total), "avg_hms": _fmt_hms(avg), "median_hms": _fmt_hms(med),
        "max_hms": _fmt_hms(mx), "min_hms": _fmt_hms(mn),
    }

    hourly = defaultdict(int)
    for h in hours:
        hourly[h] += 1
    hourly_counts = {str(i): int(hourly.get(i, 0)) for i in range(24)}

    return {
        "direction_counts": direction_counts,
        "call_type_counts": call_type_counts,
        "duration_summary": duration_summary,
        "hourly_counts": hourly_counts,
    }

def _json_sanitize(obj: Any) -> Any:
    """
    Recursively remove NaN/Inf and ensure JSON-safe dict/list/scalars.
    """
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float):
        if not pd.notna(obj) or obj in (float("inf"), float("-inf")):
            return 0
        return obj
    if isinstance(obj, (int, str)) or obj is None:
        return obj
    # Fallback for pandas/NumPy scalars
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj

# =============================================================================
# Endpoints
# =============================================================================
@app.get("/health")
def health():
    return {"status": "ok", "rows": int(len(CALLS_DF))}

@app.get("/contacts", response_model=ContactsResponse)
def contacts(main: str = Query(..., description="Main contact number (e.g., 923009605339)")):
    if not main:
        raise HTTPException(400, "Query param 'main' is required")

    payload = simple_model(CALLS_DF, main, associated=None)
    # Newer model returns 'contacts_overview'; older code used 'contacts'
    contacts_list = payload.get("contacts_overview") or payload.get("contacts") or []

    contacts_payload = [
        ContactsOverview(
            associated_key=_str_or_none(c.get("associated_key")) or "",
            call_count=int(c.get("call_count", 0)),
            percent=float(c.get("percent", 0.0)),
        )
        for c in contacts_list
    ]

    return ContactsResponse(
        total_calls=int(payload.get("total_calls", 0)),
        contacts=contacts_payload
    )

@app.get("/stats", response_model=StatsResponse)
def stats(
    main: str = Query(..., description="Main contact number (e.g., 923009605339)"),
    associated: Optional[str] = Query(
        None,
        description="Either a phone (e.g., 923001234567) or a label (e.g., 'Jazz')."
    )
):
    if not main:
        raise HTTPException(400, "Query param 'main' is required")

    payload = simple_model(CALLS_DF, main, associated)

    # Calls list -> Pydantic
    calls_in = payload.get("calls", [])
    calls: List[CallRecord] = []
    for r in calls_in:
        calls.append(
            CallRecord(
                date=_str_or_none(r.get("date")),
                day=_str_or_none(r.get("day")),
                time=_str_or_none(r.get("time")),
                start_datetime=_str_or_none(r.get("start_datetime")),
                end_datetime=_str_or_none(r.get("end_datetime")),
                duration_seconds=_float_or_none(r.get("duration_seconds")),
                associated_number=_str_or_none(r.get("associated_number")),
                associated_label=_str_or_none(r.get("associated_label")),
                associated_key=_str_or_none(r.get("associated_key")),
                direction=_str_or_none(r.get("direction")),
                call_type=_str_or_none(r.get("call_type")),
                source_file=_str_or_none(r.get("source_file")),
            )
        )

    # Prefer model-provided aggregates; else compute
    direction_counts = payload.get("direction_counts")
    call_type_counts = payload.get("call_type_counts")
    hourly_counts = payload.get("hourly_counts")
    duration_summary = payload.get("duration_summary")
    per_date_summary = payload.get("per_date_summary")

    if not direction_counts or not call_type_counts or not hourly_counts or not duration_summary:
        agg = _aggregate_from_calls_list([c.model_dump() for c in calls])
        direction_counts = direction_counts or agg["direction_counts"]
        call_type_counts = call_type_counts or agg["call_type_counts"]
        hourly_counts = hourly_counts or agg["hourly_counts"]
        duration_summary = duration_summary or agg["duration_summary"]

    resp = StatsResponse(
        total_calls=int(payload.get("total_calls", 0)),
        percent_of_total=(None if payload.get("percent_of_total") is None else float(payload.get("percent_of_total"))),
        contacts_overview=payload.get("contacts_overview") or payload.get("contacts") or [],
        calls=calls,
        direction_counts=[DirectionCount(**d) for d in direction_counts],
        call_type_counts=[CallTypeCount(**d) for d in call_type_counts],
        duration_summary=DurationSummary(**duration_summary),
        hourly_counts={str(k): int(v) for k, v in (hourly_counts or {}).items()},
        per_date_summary=per_date_summary or [],
    )

    # Ensure fully JSON-safe
    safe = _json_sanitize(resp.model_dump())
    return JSONResponse(content=jsonable_encoder(safe), status_code=200)

@app.get("/stats_ai", response_model=StatsResponse)
def stats_ai(
    main: str = Query(..., description="Main contact number (e.g., 923009605339)"),
    associated: Optional[str] = Query(None)
):
    if not main:
        raise HTTPException(400, "Query param 'main' is required")

    payload = ai_model_openai(CALLS_DF, main, associated)
    # Fill missing optional keys so the UI can render consistently
    payload.setdefault("contacts_overview", payload.get("contacts") or [])
    payload.setdefault("calls", [])
    payload.setdefault("direction_counts", [])
    payload.setdefault("call_type_counts", [])
    payload.setdefault("hourly_counts", {str(i): 0 for i in range(24)})
    payload.setdefault("duration_summary", {
        "total_sec": 0, "avg_sec": 0, "median_sec": 0, "max_sec": 0, "min_sec": 0,
        "total_hms": "0:00:00", "avg_hms": "0:00:00", "median_hms": "0:00:00",
        "max_hms": "0:00:00", "min_hms": "0:00:00",
    })
    payload.setdefault("per_date_summary", [])

    # Convert calls to CallRecord for schema
    calls: List[CallRecord] = []
    for r in payload.get("calls", []):
        calls.append(
            CallRecord(
                date=_str_or_none(r.get("date")),
                day=_str_or_none(r.get("day")),
                time=_str_or_none(r.get("time")),
                start_datetime=_str_or_none(r.get("start_datetime")),
                end_datetime=_str_or_none(r.get("end_datetime")),
                duration_seconds=_float_or_none(r.get("duration_seconds")),
                associated_number=_str_or_none(r.get("associated_number")),
                associated_label=_str_or_none(r.get("associated_label")),
                associated_key=_str_or_none(r.get("associated_key")),
                direction=_str_or_none(r.get("direction")),
                call_type=_str_or_none(r.get("call_type")),
                source_file=_str_or_none(r.get("source_file")),
            )
        )

    resp = StatsResponse(
        total_calls=int(payload.get("total_calls", 0)),
        percent_of_total=(None if payload.get("percent_of_total") is None else float(payload.get("percent_of_total"))),
        contacts_overview=payload.get("contacts_overview"),
        calls=calls,
        direction_counts=[DirectionCount(**d) for d in payload.get("direction_counts", [])],
        call_type_counts=[CallTypeCount(**d) for d in payload.get("call_type_counts", [])],
        duration_summary=DurationSummary(**payload.get("duration_summary")),
        hourly_counts={str(k): int(v) for k, v in payload.get("hourly_counts", {}).items()},
        per_date_summary=payload.get("per_date_summary", []),
        error=_str_or_none(payload.get("error")),
    )

    safe = _json_sanitize(resp.model_dump())
    return JSONResponse(content=jsonable_encoder(safe), status_code=200)



@app.get("/insights_ai")
def insights_ai(main: str, associated: Optional[str] = None):
    simple_payload = simple_model(CALLS_DF, main, associated)  # <-- dict
    insights_md = ai_insights_from_simple(simple_payload, main, associated)
    resp = {"simple_payload": simple_payload, "insights_md": insights_md, "error": None}
    safe = sanitize(resp)
    return JSONResponse(content=jsonable_encoder(safe), status_code=200)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    os.makedirs(DATA_DIR, exist_ok=True)
    dest = os.path.join(DATA_DIR, file.filename)
    with open(dest, "wb") as f:
        f.write(await file.read())
    return {"status": "saved", "path": dest}

@app.post("/reload")
def reload_data():
    global CALLS_DF
    CALLS_DF = load_calls(DATA_DIR)
    return {"status": "reloaded", "rows": int(len(CALLS_DF))}

