import re
import string
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import pytz

# Timezone
KARACHI_TZ = pytz.timezone("Asia/Karachi")


# ==============================
# Helpers
# ==============================
def _digits(s: str) -> str:
    return "".join(ch for ch in str(s) if ch.isdigit())


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names: lowercase + underscores."""
    df.columns = (
        df.columns.str.strip()
        .str.replace(r"[\s\-_]+", "_", regex=True)
        .str.lower()
    )
    return df


def normalize_number(num: Optional[str]) -> Optional[str]:
    """
    Normalize to a reasonable MSISDN; prefer Pakistan format when possible.
    Rules:
      - strip non-digits
      - if starts '92' and len>=12 → keep
      - if starts '0' and len>=11 → '92' + local without leading 0
      - if starts '3' and len>=10 → '92' + number
      - else if len>=10 → keep digits (do NOT force 92 to avoid false positives)
      - else None
    """
    if num is None or (isinstance(num, float) and pd.isna(num)):
        return None
    d = _digits(num)
    if not d:
        return None
    if d.startswith("92") and len(d) >= 12:
        return d
    if d.startswith("0") and len(d) >= 11:
        return "92" + d[1:]
    if d.startswith("3") and len(d) >= 10:
        return "92" + d
    if len(d) >= 10:
        return d
    return None


def maybe_decode_hex_token(s: Optional[str]) -> Optional[str]:
    """
    If s looks like pure hex (even length) and decodes to printable text, return decoded text; else None.
    Example: '4A617A7A' -> 'Jazz'
    """
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    s = str(s).strip()
    if not s:
        return None
    # must be pure hex
    if any(ch not in "0123456789abcdefABCDEF" for ch in s):
        return None
    if len(s) % 2 != 0:
        return None
    try:
        txt = bytes.fromhex(s).decode("utf-8", errors="strict")
        if txt and all((c in string.printable) for c in txt):
            return txt.strip()
    except Exception:
        return None
    return None


def normalize_party_value(v: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (phone_number_norm, label_norm).
      - Try hex-decode first (e.g. 4A617A7A -> 'Jazz')
      - Else try to normalize as phone
      - Else keep as label if has letters
    """
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return (None, None)
    raw = str(v).strip()
    if not raw:
        return (None, None)

    decoded = maybe_decode_hex_token(raw)
    if decoded:
        return (None, decoded)

    num = normalize_number(raw)
    if num:
        return (num, None)

    if any(ch.isalpha() for ch in raw):
        return (None, raw)

    return (None, None)


def _parse_datetime(series: pd.Series) -> pd.Series:
    """Parse datetimes and localize to Asia/Karachi (assume UTC when naive)."""
    dt = pd.to_datetime(series, errors="coerce")
    if getattr(dt.dt, "tz", None) is None:
        dt = dt.dt.tz_localize("UTC", nonexistent="NaT", ambiguous="NaT")
    return dt.dt.tz_convert(KARACHI_TZ)


def _parse_duration(series: pd.Series) -> pd.Series:
    """Convert durations to seconds (supports 00:03:22, 03:22, '45 sec', raw seconds)."""
    s = series.astype(str).str.strip()
    out = pd.to_numeric(s, errors="coerce")

    # HH:MM:SS or MM:SS
    mask_time = s.str.contains(r"^\d{1,2}:\d{2}(?::\d{2})?$", regex=True, na=False)
    if mask_time.any():
        def hms_to_sec(x: str):
            parts = x.split(":")
            if len(parts) == 2:
                m, sec = parts
                return int(m) * 60 + int(sec)
            if len(parts) == 3:
                h, m, sec = parts
                return int(h) * 3600 + int(m) * 60 + int(sec)
            return None
        out.loc[mask_time] = s.loc[mask_time].map(hms_to_sec)

    # "45 sec", "30 seconds"
    mask_sec = s.str.contains(r"\d+\s*(?:s|sec|secs|second|seconds)$", case=False, na=False)
    if mask_sec.any():
        out.loc[mask_sec] = s.loc[mask_sec].str.extract(r"(\d+)", expand=False).astype(float)

    return pd.to_numeric(out, errors="coerce")


def _find_col(df: pd.DataFrame, candidates: List[str], exclude: List[str] = None) -> Optional[str]:
    """Find first column that contains any candidate token (substring match) and not excluded tokens."""
    exclude = exclude or []
    cols = list(df.columns)
    for cand in candidates:
        for c in cols:
            if any(bad in c for bad in exclude):
                continue
            if cand in c:
                return c
    return None


# ==============================
# Loader
# ==============================
def load_calls(data_path: str) -> pd.DataFrame:
    """
    Load all CSV/XLS/XLSX files in data_path and normalize into unified schema.

    Unified schema (always returned in this order):
      main_number, associated_number, associated_label, associated_key,
      start_datetime, end_datetime, duration_seconds,
      date, day_name, time, direction, call_type, source_file
    """
    data_path = Path(data_path)
    all_data: List[pd.DataFrame] = []
    valid_exts = [".csv", ".xlsx", ".xls"]

    for file in data_path.glob("*"):
        if file.suffix.lower() not in valid_exts:
            continue
        print(f"📂 Loading file: {file.name}")

        # Read
        try:
            if file.suffix.lower() == ".csv":
                df = pd.read_csv(file, encoding="utf-8", engine="python")
            else:
                df = pd.read_excel(file)
        except Exception as e:
            print(f"❌ Failed to load {file.name}: {e}")
            continue

        if df.empty:
            continue

        df = _normalize_columns(df)

        # Build norm with matching index FIRST so scalars broadcast correctly
        norm = pd.DataFrame(index=df.index)
        norm["source_file"] = file.name

        # Main number from filename (broadcast)
        main_from_filename = normalize_number(file.stem)
        norm["file_main"] = main_from_filename

        # Broad-but-safe column detection
        exclude_noise = [
            "imsi", "imei", "cell", "site", "sector", "node", "ip",
            "lac", "tac", "mcc", "mnc", "cid", "lon", "lat", "longitude", "latitude"
        ]

        a_col = (
            _find_col(df, ["a_number", "anumber", "a_party", "aparty", "calling", "caller", "orig", "source", "from"], exclude_noise)
            or _find_col(df, ["a"], exclude_noise)
        )
        b_col = (
            _find_col(df, ["b_number", "bnumber", "b_party", "bparty", "called", "callee", "destination", "receiver", "to"], exclude_noise)
            or _find_col(df, ["b"], exclude_noise)
        )

        start_col = _find_col(df, ["datetime", "date_time", "start_time", "call_time", "start"])
        end_col   = _find_col(df, ["end_time", "end"])
        dur_col   = _find_col(df, ["duration", "call_duration", "talk_time"])
        dir_col   = _find_col(df, ["direction", "call_direction"])
        type_col  = _find_col(df, ["call_type", "type", "category"])

        # Parties: capture BOTH phone and label (hex-decoded or raw)
        if a_col in df:
            a_pairs = df[a_col].map(normalize_party_value)
            norm[["A_num", "A_label"]] = pd.DataFrame(a_pairs.tolist(), index=df.index)
        if b_col in df:
            b_pairs = df[b_col].map(normalize_party_value)
            norm[["B_num", "B_label"]] = pd.DataFrame(b_pairs.tolist(), index=df.index)

        # Time fields
        if start_col in df:
            norm["start_datetime"] = _parse_datetime(df[start_col])
        if end_col in df:
            norm["end_datetime"] = _parse_datetime(df[end_col])
        else:
            # ensure the column exists for later filling
            norm["end_datetime"] = pd.NaT

        # Duration
        if dur_col in df:
            norm["duration_seconds"] = _parse_duration(df[dur_col])
        else:
            norm["duration_seconds"] = pd.to_numeric(pd.Series([None] * len(df)), errors="coerce")

        # ---- NEW: back-fill missing duration or end time
        # If duration missing but both start & end exist → compute duration
        if "start_datetime" in norm.columns and "end_datetime" in norm.columns:
            mask_need_duration = norm["duration_seconds"].isna() & norm["start_datetime"].notna() & norm["end_datetime"].notna()
            if mask_need_duration.any():
                norm.loc[mask_need_duration, "duration_seconds"] = (
                    norm.loc[mask_need_duration, "end_datetime"] - norm.loc[mask_need_duration, "start_datetime"]
                ).dt.total_seconds()

        # If end time missing but start & duration exist → compute end = start + duration
        if "start_datetime" in norm.columns and "duration_seconds" in norm.columns:
            mask_need_end = norm["end_datetime"].isna() & norm["start_datetime"].notna() & norm["duration_seconds"].notna()
            if mask_need_end.any():
                norm.loc[mask_need_end, "end_datetime"] = (
                    norm.loc[mask_need_end, "start_datetime"] + pd.to_timedelta(norm.loc[mask_need_end, "duration_seconds"], unit="s")
                )

        # Other fields (keep raw; API sanitizes NaNs to None)
        if dir_col in df:
            norm["direction"] = df[dir_col]
        if type_col in df:
            norm["call_type"] = df[type_col]

        # Build (main, associated) with a unified associated_key (number if present else label)
        def choose_pair(row):
            main = row.get("file_main")
            a_num, b_num = row.get("A_num"), row.get("B_num")
            a_lab, b_lab = row.get("A_label"), row.get("B_label")

            if main and a_num and main == a_num:
                assoc_num, assoc_lab = b_num, b_lab
            elif main and b_num and main == b_num:
                assoc_num, assoc_lab = a_num, a_lab
            else:
                # fallback: keep filename main; choose B side first if available
                if b_num is not None:
                    assoc_num, assoc_lab = b_num, b_lab
                elif a_num is not None:
                    assoc_num, assoc_lab = a_num, a_lab
                else:
                    assoc_num, assoc_lab = None, (b_lab or a_lab)

            associated_key = assoc_num if assoc_num else assoc_lab
            return pd.Series({
                "main_number": main,
                "associated_number": assoc_num,
                "associated_label": assoc_lab,
                "associated_key": associated_key,
            })

        pairs = norm.apply(choose_pair, axis=1)
        norm = pd.concat([norm, pairs], axis=1)

        # Derived date fields
        if "start_datetime" in norm.columns:
            dt = pd.to_datetime(norm["start_datetime"], errors="coerce").dt.tz_convert(KARACHI_TZ)
            norm["date"] = dt.dt.date
            norm["day_name"] = dt.dt.day_name()
            norm["time"] = dt.dt.time

        all_data.append(norm)

    if not all_data:
        print("⚠️ No valid files found.")
        return pd.DataFrame(columns=[
            "main_number","associated_number","associated_label","associated_key",
            "start_datetime","end_datetime","duration_seconds",
            "date","day_name","time","direction","call_type","source_file"
        ])

    combined = pd.concat(all_data, ignore_index=True)

    # Require at least main and a usable associated_key (number or label)
    combined = combined.dropna(subset=["main_number", "associated_key"], how="any")

    # Ensure all expected columns exist
    expected_cols = [
        "main_number","associated_number","associated_label","associated_key",
        "start_datetime","end_datetime","duration_seconds",
        "date","day_name","time","direction","call_type","source_file",
    ]
    for col in expected_cols:
        if col not in combined.columns:
            combined[col] = None

    # Deduplicate
    combined = combined.drop_duplicates(
        subset=["main_number", "associated_key", "start_datetime"], keep="first"
    )

    print(f"✅ Loaded {len(combined)} records across {len(all_data)} files.")
    return combined[expected_cols]


# ==============================
# Stats
# ==============================
def compute_stats(df: pd.DataFrame, main_number: str, associated: Optional[str] = None) -> dict:
    """
    Compute stats for a given main number and (optionally) a specific associated value.
    'associated' may be a phone number or a label (e.g., hex-decoded text).
    """
    if df.empty:
        return {"total_calls": 0, "contacts": [], "calls": [], "percent_of_total": 0.0}

    main_norm = normalize_number(main_number)
    if not main_norm:
        return {"total_calls": 0, "contacts": [], "calls": [], "percent_of_total": 0.0}

    df_main = df[df["main_number"] == main_norm].copy()
    if df_main.empty:
        return {"total_calls": 0, "contacts": [], "calls": [], "percent_of_total": 0.0}

    total_calls = len(df_main)

    # Group by unified key (phone if present else label)
    df_main["__key"] = df_main["associated_key"].astype(str)
    top = (
        df_main.groupby("__key")
        .size()
        .sort_values(ascending=False)
        .reset_index(name="call_count")
        .rename(columns={"__key": "associated_key"})
    )
    top["percent"] = (top["call_count"] / total_calls * 100).round(2)

    # Filtering
    assoc_norm = normalize_number(associated) if associated else None
    assoc_text = (associated or "").strip() if associated else None

    if assoc_norm:
        df_filtered = df_main[
            (df_main["associated_number"] == assoc_norm) | (df_main["associated_key"] == assoc_norm)
        ]
        key_to_match = assoc_norm
    elif assoc_text:
        df_filtered = df_main[df_main["associated_key"] == assoc_text]
        key_to_match = assoc_text
    else:
        df_filtered = df_main
        key_to_match = None

    df_filtered = df_filtered.sort_values("start_datetime")

    # Calls list
    calls = []
    for _, r in df_filtered.iterrows():
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

    # Percent for the requested associated
    percent = None
    if key_to_match is not None:
        row = top[top["associated_key"] == key_to_match]
        percent = float(row["percent"].iloc[0]) if not row.empty else 0.0

    return {
        "total_calls": int(total_calls),
        "contacts": top.to_dict(orient="records"),
        "calls": calls,
        "percent_of_total": percent,
    }
