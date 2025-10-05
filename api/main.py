from typing import Optional, List, Any, Dict
import os
from collections import Counter, defaultdict

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd  # for isna & quick stats
# add near top
from fastapi import UploadFile, File

from app.dataloader import load_calls, compute_stats

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))

# ------------------------------------------------------------------------------
# App
# ------------------------------------------------------------------------------
app = FastAPI(title="Call Analytics API", version="2.1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load once; for hot reload add a /reload or startup event
CALLS_DF = load_calls(DATA_DIR)

# ------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------
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
    total_sec: float
    avg_sec: float
    median_sec: float
    max_sec: float
    min_sec: float
    total_hms: str
    avg_hms: str
    median_hms: str
    max_hms: str
    min_hms: str

class StatsResponse(BaseModel):
    total_calls: int
    percent_of_total: Optional[float] = None
    contacts: List[Any]               # {'associated_key','call_count','percent'}
    calls: List[CallRecord]
    direction_counts: List[DirectionCount] | None = None
    call_type_counts: List[CallTypeCount] | None = None
    duration_summary: DurationSummary | None = None
    hourly_counts: Dict[int, int] | None = None

# ------------------------------------------------------------------------------
# Helpers (NaN-safe coercion & aggregates)
# ------------------------------------------------------------------------------
def _str_or_none(v) -> Optional[str]:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    s = str(v).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    return s

def _float_or_none(v) -> Optional[float]:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        return None

def _fmt_hms(seconds: float | int | None) -> str:
    if seconds is None or not pd.notna(seconds):
        return "0:00:00"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:d}:{m:02d}:{s:02d}"

def _aggregate_stats(calls: List[dict]) -> dict:
    if not calls:
        return {
            "direction_counts": [],
            "call_type_counts": [],
            "duration_summary": {
                "total_sec": 0.0, "avg_sec": 0.0, "median_sec": 0.0, "max_sec": 0.0, "min_sec": 0.0,
                "total_hms": "0:00:00", "avg_hms": "0:00:00", "median_hms": "0:00:00",
                "max_hms": "0:00:00", "min_hms": "0:00:00",
            },
            "hourly_counts": {i: 0 for i in range(24)},
        }

    dir_counter = Counter()
    type_counter = Counter()
    durations = []
    hours = []

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
            if len(parts) >= 1:
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
        total = float(s.sum())
        avg = float(s.mean())
        med = float(s.median())
        mx = float(s.max())
        mn = float(s.min())
    else:
        total = avg = med = mx = mn = 0.0

    duration_summary = {
        "total_sec": total, "avg_sec": avg, "median_sec": med, "max_sec": mx, "min_sec": mn,
        "total_hms": _fmt_hms(total), "avg_hms": _fmt_hms(avg), "median_hms": _fmt_hms(med),
        "max_hms": _fmt_hms(mx), "min_hms": _fmt_hms(mn),
    }

    hourly = defaultdict(int)
    for h in hours:
        hourly[h] += 1
    hourly_counts = {i: int(hourly.get(i, 0)) for i in range(24)}

    return {
        "direction_counts": direction_counts,
        "call_type_counts": call_type_counts,
        "duration_summary": duration_summary,
        "hourly_counts": hourly_counts,
    }

# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "rows": int(len(CALLS_DF))}

@app.get("/contacts", response_model=ContactsResponse)
def contacts(main: str = Query(..., description="Main contact number (e.g., 923009605339)")):
    if not main:
        raise HTTPException(400, "Query param 'main' is required")

    stats = compute_stats(CALLS_DF, main_number=main, associated=None)

    contacts_payload = [
        ContactsOverview(
            associated_key=_str_or_none(c.get("associated_key")) or "",
            call_count=int(c.get("call_count", 0)),
            percent=float(c.get("percent", 0.0)),
        )
        for c in stats.get("contacts", [])
    ]

    return ContactsResponse(
        total_calls=int(stats.get("total_calls", 0)),
        contacts=contacts_payload
    )

@app.get("/stats", response_model=StatsResponse)
def stats(
    main: str = Query(..., description="Main contact number (e.g., 923009605339)"),
    associated: Optional[str] = Query(
        None,
        description="Either a phone (e.g., 923001234567) or a label (e.g., 'Jazz'). "
                    "Matched against the unified `associated_key`."
    )
):
    if not main:
        raise HTTPException(400, "Query param 'main' is required")

    result = compute_stats(CALLS_DF, main_number=main, associated=associated)

    # Build NaN-safe call records
    calls: List[CallRecord] = []
    for r in result.get("calls", []):
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

    agg = _aggregate_stats([c.model_dump() for c in calls])

    return StatsResponse(
        total_calls=int(result.get("total_calls", 0)),
        percent_of_total=(None if result.get("percent_of_total") is None else float(result.get("percent_of_total"))),
        contacts=result.get("contacts", []),
        calls=calls,
        direction_counts=[DirectionCount(**d) for d in agg["direction_counts"]],
        call_type_counts=[CallTypeCount(**d) for d in agg["call_type_counts"]],
        duration_summary=DurationSummary(**agg["duration_summary"]),
        hourly_counts=agg["hourly_counts"],
    )


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    dest = os.path.join(DATA_DIR, file.filename)
    with open(dest, "wb") as f:
        f.write(await file.read())
    return {"status": "saved", "path": dest}

@app.post("/reload")
def reload_data():
    global CALLS_DF
    CALLS_DF = load_calls(DATA_DIR)
    return {"status": "reloaded", "rows": int(len(CALLS_DF))}
