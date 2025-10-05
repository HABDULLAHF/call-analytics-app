# models/analytics.py

from __future__ import annotations
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd

# Reuse utilities from your dataloader
from app.dataloader import normalize_number

# ---------- small helpers ----------
def _secs_to_hms(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return ""
    try:
        x = int(float(x))
    except Exception:
        return ""
    h = x // 3600
    m = (x % 3600) // 60
    s = x % 60
    return f"{h:d}:{m:02d}:{s:02d}"

def _sort_time_key(t: Any) -> Tuple[int, int, int]:
    # sort "HH:MM[:SS]" safely
    if not isinstance(t, str):
        return (99, 99, 99)
    parts = t.split(":")
    try:
        h = int(parts[0]) if len(parts) > 0 else 99
        m = int(parts[1]) if len(parts) > 1 else 99
        s = int(parts[2]) if len(parts) > 2 else 0
        return (h, m, s)
    except Exception:
        return (99, 99, 99)

# ---------- aggregates for charts/kpis ----------

def build_direction_counts(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty or "direction" not in df.columns:
        return []
    vc = (
        df["direction"]
        .fillna("Unknown")
        .astype(str)
        .value_counts(dropna=False)
    )
    c = vc.reset_index()
    # Force canonical 2-column shape and names: ['direction', 'count']
    if c.shape[1] >= 2:
        c = c.iloc[:, :2]
        c.columns = ["direction", "count"]
    else:
        return []
    c["count"] = pd.to_numeric(c["count"], errors="coerce").fillna(0).astype(int)
    # Optional stable ordering
    c = c.sort_values(["count", "direction"], ascending=[False, True], kind="stable").reset_index(drop=True)
    return c.to_dict(orient="records")


def build_call_type_counts(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty or "call_type" not in df.columns:
        return []
    vc = (
        df["call_type"]
        .fillna("Unknown")
        .astype(str)
        .value_counts(dropna=False)
    )
    c = vc.reset_index()
    # Force canonical 2-column shape and names: ['call_type', 'count']
    if c.shape[1] >= 2:
        c = c.iloc[:, :2]
        c.columns = ["call_type", "count"]
    else:
        return []
    c["count"] = pd.to_numeric(c["count"], errors="coerce").fillna(0).astype(int)
    c = c.sort_values(["count", "call_type"], ascending=[False, True], kind="stable").reset_index(drop=True)
    return c.to_dict(orient="records")

    if df is None or df.empty or "call_type" not in df.columns:
        return []
    c = (
        df["call_type"]
        .fillna("Unknown")
        .astype(str)
        .value_counts()
        .reset_index()
        .rename(columns={"index": "call_type", "call_type": "count"})
    )
    c["count"] = pd.to_numeric(c["count"], errors="coerce").fillna(0).astype(int)
    return c.to_dict(orient="records")

def build_hourly_counts(df: pd.DataFrame) -> Dict[str, int]:
    # prefer start_datetime hour; fallback to 'time' column
    if df is None or df.empty:
        return {str(h): 0 for h in range(24)}
    hours = None
    if "start_datetime" in df.columns and df["start_datetime"].notna().any():
        try:
            hours = pd.to_datetime(df["start_datetime"], errors="coerce").dt.hour
        except Exception:
            hours = None
    if hours is None or hours.dropna().empty:
        if "time" in df.columns and df["time"].notna().any():
            try:
                hours = pd.to_datetime(df["time"].astype(str), errors="coerce").dt.hour
            except Exception:
                hours = None
    if hours is None or hours.dropna().empty:
        return {str(h): 0 for h in range(24)}

    counts = hours.dropna().astype(int).value_counts()
    # normalize to 0..23
    out = {str(h): 0 for h in range(24)}
    for h, v in counts.items():
        if 0 <= int(h) <= 23:
            out[str(int(h))] = int(v)
    return out

def build_duration_summary(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty or "duration_seconds" not in df.columns:
        return {
            "total_sec": 0, "avg_sec": 0, "median_sec": 0, "min_sec": 0, "max_sec": 0,
            "total_hms": "0:00:00", "avg_hms": "0:00:00", "median_hms": "0:00:00",
            "min_hms": "0:00:00", "max_hms": "0:00:00",
        }
    s = pd.to_numeric(df["duration_seconds"], errors="coerce").dropna()
    if s.empty:
        return {
            "total_sec": 0, "avg_sec": 0, "median_sec": 0, "min_sec": 0, "max_sec": 0,
            "total_hms": "0:00:00", "avg_hms": "0:00:00", "median_hms": "0:00:00",
            "min_hms": "0:00:00", "max_hms": "0:00:00",
        }
    total = float(s.sum())
    avg = float(s.mean()) if len(s) else 0.0
    med = float(s.median()) if len(s) else 0.0
    minv = float(s.min())
    maxv = float(s.max())
    return {
        "total_sec": int(total),
        "avg_sec": int(avg),
        "median_sec": int(med),
        "min_sec": int(minv),
        "max_sec": int(maxv),
        "total_hms": _secs_to_hms(total),
        "avg_hms": _secs_to_hms(avg),
        "median_hms": _secs_to_hms(med),
        "min_hms": _secs_to_hms(minv),
        "max_hms": _secs_to_hms(maxv),
    }

# ---------- public API ----------
def build_contacts_overview(df_main: pd.DataFrame) -> pd.DataFrame:
    """
    For a given main's dataframe (already filtered by main_number),
    return per-associated counts and % of total.
    Always returns columns: ['associated_key', 'call_count', 'percent'].
    """
    total_calls = int(len(df_main))
    if total_calls == 0:
        return pd.DataFrame(columns=["associated_key", "call_count", "percent"])

    # Value counts on unified key
    vc = (
        df_main.get("associated_key", pd.Series(dtype=object))
        .fillna("Unknown")
        .astype(str)
        .value_counts(dropna=False)
    )

    # Reset & force canonical column names regardless of pandas version
    g = vc.reset_index()
    if g.shape[1] >= 2:
        g = g.iloc[:, :2]
        g.columns = ["associated_key", "call_count"]
    else:
        g = pd.DataFrame(columns=["associated_key", "call_count"])

    # Ensure numeric counts
    g["call_count"] = pd.to_numeric(g["call_count"], errors="coerce").fillna(0).astype(int)

    # Compute percent safely
    denom = float(total_calls) if total_calls > 0 else 1.0
    g["percent"] = (g["call_count"] / denom * 100.0).round(2)

    # Sort stable: most-called first, then key asc for determinism
    g = g.sort_values(["call_count", "associated_key"], ascending=[False, True], kind="stable").reset_index(drop=True)

    return g[["associated_key", "call_count", "percent"]]

def per_date_summary(df_filtered: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Build a per-date table with:
    - Call Count
    - Call Timing (list)
    - Call Duration (each)
    - Date and Day
    - Sorted by date (and calls already chronologically sorted)
    - Also include Associated (counts) for that date
    """
    rows: List[Dict[str, Any]] = []
    if df_filtered.empty or "date" not in df_filtered.columns:
        return rows

    # Ensure chronological rows by start time if present
    if "start_datetime" in df_filtered.columns:
        df_filtered = df_filtered.sort_values("start_datetime", kind="stable")

    for date_val, g in df_filtered.groupby("date", dropna=False):
        day_name = ""
        if "day_name" in g.columns and g["day_name"].notna().any():
            day_name = g["day_name"].dropna().astype(str).mode().iloc[0]

        # Associated counts (per-date)
        if "associated_key" in g.columns:
            assoc_counts = (
                g["associated_key"].fillna("Unknown").astype(str).value_counts()
            )
            assoc_str = ", ".join([f"{k} ({v})" for k, v in assoc_counts.items()])
        else:
            assoc_str = ""

        # times
        if "time" in g.columns:
            times = sorted(
                g["time"].dropna().astype(str).tolist(),
                key=_sort_time_key
            )
            times_str = ", ".join(times)
        else:
            times_str = ""

        # duration list + stats
        if "duration_seconds" in g.columns:
            durs = g["duration_seconds"].tolist()
            durs_hms = [_secs_to_hms(x) for x in durs if pd.notna(x)]
            durs_str = ", ".join(durs_hms)

            total_s = float(g["duration_seconds"].fillna(0).sum())
            avg_s = float(g["duration_seconds"].dropna().mean()) if g["duration_seconds"].notna().any() else 0.0
        else:
            durs_str, total_s, avg_s = "", 0.0, 0.0

        rows.append({
            "Date": str(date_val),
            "Day": day_name,
            "Associated (counts)": assoc_str,
            "Call Count": int(len(g)),
            "Call Timing": times_str,
            "Call Durations (each)": durs_str,
            "Total Duration": _secs_to_hms(total_s),
            "Avg Duration": _secs_to_hms(avg_s),
        })

    rows.sort(key=lambda r: r["Date"])
    return rows

def compute_contact_percentage(contacts_df: pd.DataFrame, associated: Optional[str]) -> Optional[float]:
    """
    Return the % of calls for the requested 'associated' (can be phone or label).
    If associated is None → return None.
    """
    if associated is None or contacts_df.empty:
        return None

    assoc_norm = normalize_number(associated)
    key = assoc_norm if assoc_norm else associated.strip()

    row = contacts_df[contacts_df["associated_key"] == key]
    if row.empty:
        return 0.0
    return float(row["percent"].iloc[0])

def simple_model(
    df: pd.DataFrame,
    main_number: str,
    associated: Optional[str] = None
) -> Dict[str, Any]:
    """
    Pure-pandas analytics for the 6 requested items (+ extra aggregates for UI).
    Output keys:
      - total_calls
      - contacts_overview [{associated_key, call_count, percent}]
      - percent_of_total (for selected associated or None)
      - calls (chronological list)
      - per_date_summary (list of dicts)
      - direction_counts (list of {direction, count})
      - call_type_counts (list of {call_type, count})
      - hourly_counts (dict '0'..'23' -> int)
      - duration_summary (dict)
    """
    empty = {
        "total_calls": 0,
        "contacts_overview": [],
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
    }

    if df is None or df.empty:
        return empty

    main_norm = normalize_number(main_number)
    if not main_norm:
        return empty

    df_main = df[df["main_number"] == main_norm].copy()
    if df_main.empty:
        return empty

    total_calls = int(len(df_main))
    contacts = build_contacts_overview(df_main)

    # Filter by associated (phone or label)
    assoc_norm = normalize_number(associated) if associated else None
    if assoc_norm:
        df_fil = df_main[
            (df_main["associated_number"] == assoc_norm) | (df_main["associated_key"] == assoc_norm)
        ].copy()
        key_for_percent = assoc_norm
    elif associated:
        df_fil = df_main[df_main["associated_key"] == associated.strip()].copy()
        key_for_percent = associated.strip()
    else:
        df_fil = df_main.copy()
        key_for_percent = None

    # Sort chronologically
    if "start_datetime" in df_fil.columns:
        df_fil = df_fil.sort_values("start_datetime", kind="stable")

    # Calls list (for detail table)
    calls = []
    for _, r in df_fil.iterrows():
        calls.append({
            "date": str(r.get("date", "")),
            "day": str(r.get("day_name", "")),
            "time": str(r.get("time", "")),
            "start_datetime": r["start_datetime"].isoformat() if pd.notna(r.get("start_datetime")) else None,
            "end_datetime": r["end_datetime"].isoformat() if pd.notna(r.get("end_datetime")) else None,
            "duration_seconds": float(r.get("duration_seconds")) if pd.notna(r.get("duration_seconds")) else None,
            "associated_number": r.get("associated_number"),
            "associated_label": r.get("associated_label"),
            "associated_key": r.get("associated_key"),
            "direction": r.get("direction"),
            "call_type": r.get("call_type"),
            "source_file": r.get("source_file"),
        })

    # Percent for requested associated
    pct = compute_contact_percentage(contacts, key_for_percent) if key_for_percent is not None else None

    # Per-date rollup
    per_date = per_date_summary(df_fil)

    # Extra aggregates for UI
    direction_counts = build_direction_counts(df_fil)
    call_type_counts = build_call_type_counts(df_fil)
    hourly_counts = build_hourly_counts(df_fil)
    duration_summary = build_duration_summary(df_fil)

    return {
        "total_calls": total_calls,
        "contacts_overview": contacts.to_dict(orient="records"),
        "percent_of_total": pct,
        "calls": calls,
        "per_date_summary": per_date,
        "direction_counts": direction_counts,
        "call_type_counts": call_type_counts,
        "hourly_counts": hourly_counts,
        "duration_summary": duration_summary,
    }

# Optional: an "AI model" hook — for now just delegates to simple_model
def ai_model(
    df: pd.DataFrame,
    main_number: str,
    associated: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Placeholder AI path. Replace internals to call OpenAI and return the SAME structure.
    """
    return simple_model(df, main_number, associated)
