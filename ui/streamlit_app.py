import os
import io
import math
import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =========================================
# Config
# =========================================
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="📞 Call Analytics Dashboard", layout="wide")
st.title("📞 Call Analytics Dashboard")

st.caption(
    "Type a **Main Contact Number** (e.g., `923000000000`) and optionally choose an "
    "**Associated** (phone or label like `Jazz`) to see detailed activity and analytics."
)

# =========================================
# Helpers
# =========================================
def fetch_json(url: str, params: dict | None = None, timeout: int = 60):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code != 200:
            st.error(f"HTTP {r.status_code}: {r.text}")
            return None
        return r.json()
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

def fmt_hms(seconds: float | int | None) -> str:
    if seconds is None or not pd.notna(seconds):
        return "0:00:00"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:d}:{m:02d}:{s:02d}"

def safe_series_counts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if df is None or df.empty or col not in df.columns:
        return pd.DataFrame({col: [], "count": []})
    counts = (
        df[col]
        .fillna("Unknown")
        .astype(str)
        .value_counts()
        .reset_index()
        .rename(columns={"index": col, col: "count"})
    )
    return counts

def counts_df_from_api_or_local(stats_json: dict, local_df: pd.DataFrame, api_key: str, col_name: str) -> pd.DataFrame:
    """
    Try API-provided aggregates (list of dicts); fallback to local counts on the calls df.
    Always return DataFrame with [<col_name>, 'count'] (even if empty).
    """
    api_list = stats_json.get(api_key, None) if isinstance(stats_json, dict) else None
    if isinstance(api_list, list) and len(api_list) > 0:
        df = pd.DataFrame(api_list)
        # normalize columns to lowercase
        df.columns = [str(c).lower() for c in df.columns]
        # map possible fields defensively
        if col_name not in df.columns:
            if "direction" in df.columns and col_name != "direction":
                df[col_name] = df["direction"]
            elif "call_type" in df.columns and col_name != "call_type":
                df[col_name] = df["call_type"]
            else:
                df[col_name] = []
        if "count" not in df.columns:
            df["count"] = 0
        return df[[col_name, "count"]].copy()

    # fallback: local
    return safe_series_counts(local_df, col_name)

def normalize_hour_series(obj) -> pd.Series:
    """
    Return a Series indexed 0..23 in ascending order with integer counts.
    Accepts a dict/Series-like with hour keys that might be strings; fills missing hours with 0.
    """
    if obj is None:
        return pd.Series([0]*24, index=range(24))
    try:
        s = pd.Series(obj)
    except Exception:
        return pd.Series([0]*24, index=range(24))

    # Keys may be strings; coerce to numeric hours
    try:
        s.index = pd.to_numeric(s.index, errors="coerce")
    except Exception:
        # if index isn't numeric-convertible, bail to zeros
        return pd.Series([0]*24, index=range(24))

    s = s.dropna()
    if s.empty:
        return pd.Series([0]*24, index=range(24))

    s.index = s.index.astype(int)
    # If there are any duplicate hour keys (shouldn't be, but just in case)
    s = s.groupby(s.index).sum()
    # Ensure full 0..23 coverage and sorted order
    s = s.reindex(range(24), fill_value=0).astype(int)
    return s

# Helpers for Per-Date Summary
def _secs_to_hms(x):
    try:
        x = float(x)
    except Exception:
        return ""
    x = int(x)
    h = x // 3600
    m = (x % 3600) // 60
    s = x % 60
    return f"{h:d}:{m:02d}:{s:02d}"

def _sort_time_str(t):
    # Robust sort key for "HH:MM:SS" / "HH:MM" strings
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

# -------- New: API upload/reload helpers --------
def api_post_file(api_base: str, endpoint: str, filename: str, data_bytes: bytes, mime: str) -> tuple[bool, str]:
    url = f"{api_base.rstrip('/')}/{endpoint.lstrip('/')}"
    try:
        files = {"file": (filename, data_bytes, mime)}
        r = requests.post(url, files=files, timeout=120)
        if r.status_code in (200, 201):
            return True, "Uploaded to API successfully."
        return False, f"API {endpoint} returned {r.status_code}: {r.text}"
    except Exception as e:
        return False, f"Upload failed: {e}"

def api_reload(api_base: str) -> tuple[bool, str]:
    """Try POST /reload; if not allowed, try GET /reload; if both fail, return False."""
    for method in ("post", "get"):
        try:
            fn = getattr(requests, method)
            r = fn(f"{api_base.rstrip('/')}/reload", timeout=60)
            if r.status_code in (200, 204):
                return True, "Reloaded API dataset."
            # 405 means endpoint exists with other method; keep trying
            if r.status_code == 405:
                continue
        except Exception:
            continue
    return False, "Reload endpoint not available; restart API to load new files."

# =========================================
# Sidebar — settings + data preview + UPLOAD
# =========================================
with st.sidebar:
    st.header("⚙️ Settings")
    API_URL = st.text_input("API Base URL", value=API_URL, help="e.g., http://127.0.0.1:8000")

    st.divider()
    st.subheader("⬆️ Upload Contacts (CSV/XLSX)")
    uploaded = st.file_uploader(
        "Drop a .csv or .xlsx/.xls here to add to the dataset",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=False
    )

    colu1, colu2 = st.columns([1, 1])
    target_mode = colu1.radio(
        "Save to",
        options=["API (/upload)"],#, "Local data folder"],
        index=0,
        help="Prefer API upload if your backend exposes POST /upload. Otherwise, save to the app's ../data folder."
    )
    # local_folder_hint = colu2.text_input(
    #     "Local folder",
    #     value=os.path.abspath(os.path.join( "..", "data")),
    #     help="Used only if 'Local data folder' is selected."
    # )

    if uploaded is not None:
        fname = uploaded.name
        data_bytes = uploaded.getvalue()
        ext = os.path.splitext(fname)[1].lower()
        mime = "text/csv" if ext == ".csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

        if st.button("Add File to Dataset", use_container_width=True):
            if target_mode == "API (/upload)":
                ok, msg = api_post_file(API_URL, "/upload", fname, data_bytes, mime)
                if ok:
                    st.success(msg)
                    rok, rmsg = api_reload(API_URL)
                    if rok:
                        st.success(rmsg)
                    else:
                        st.info(rmsg)
                else:
                    st.warning(msg)
                    st.info("Falling back to saving locally…")
                    # Fall through to local save
                    #target_mode = "Local data folder"

            # if target_mode == "Local data folder":
            #     try:
            #         os.makedirs(local_folder_hint, exist_ok=True)
            #         dest_path = os.path.join(local_folder_hint, fname)
            #         with open(dest_path, "wb") as f:
            #             f.write(data_bytes)
            #         st.success(f"Saved to local folder: {dest_path}")
            #         rok, rmsg = api_reload(API_URL)
            #         if rok:
            #             st.success(rmsg)
            #         else:
            #             st.info(rmsg)
            #     except Exception as e:
            #         st.error(f"Local save failed: {e}")

    st.divider()
    st.subheader("👁️ Data Preview")
    health = fetch_json(f"{API_URL}/health", timeout=15)
    if health:
        st.success("API Connected ✅")
    else:
        st.warning("Cannot reach API. Check that FastAPI is running.")
    st.caption("Tip: use a main number, e.g., `923000000000`.")

# =========================================
# Main controls
# =========================================
col1, col2 = st.columns([2, 1])
with col1:
    main = st.text_input("Main Contact Number", value="", placeholder="e.g., 923000000000")
with col2:
    fetch_btn = st.button("Fetch", use_container_width=True)

# =========================================
# Interaction
# =========================================
if main or fetch_btn:
    # 1) Overview for main
    contacts_data = fetch_json(f"{API_URL}/contacts", params={"main": main}, timeout=60)
    if not contacts_data:
        st.stop()

    total_main_calls = int(contacts_data.get("total_calls", 0))
    contacts_df = pd.DataFrame(contacts_data.get("contacts", []))

    c1, c2 = st.columns(2)
    c1.metric("Total Calls (Main)", f"{total_main_calls:,}")
    associated = None

    if not contacts_df.empty:
        top_row = contacts_df.iloc[0]
        c2.metric("Top Associated %", f"{top_row.get('percent', 0):.2f}%", help=f"Top: {top_row.get('associated_key')}")

        st.subheader("📋 Associated Contacts")
        st.dataframe(
            contacts_df.rename(columns={
                "associated_key": "Associated (phone or label)",
                "call_count": "Calls",
                "percent": "% of Main"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Associated Overview (CSV)",
            data=df_to_csv_bytes(contacts_df),
            file_name=f"{main}_associated_overview.csv",
            mime="text/csv",
            use_container_width=True
        )

        assoc_choices = ["-- All --"] + contacts_df["associated_key"].astype(str).tolist()
        with st.expander("Filter by Associated", expanded=True):
            assoc_pick = st.selectbox("Pick an Associated (optional)", assoc_choices, index=0)
            assoc_manual = st.text_input("…or type Associated manually (phone or label like 'Jazz')", value="")
        if assoc_pick and assoc_pick != "-- All --":
            associated = assoc_pick
        if assoc_manual.strip():
            associated = assoc_manual.strip()

        with st.expander("Top Associated Shares (Top 10)", expanded=False):
            top10 = contacts_df.head(10).copy()
            if not top10.empty:
                top10 = top10.rename(columns={"associated_key": "Associated"})
                st.bar_chart(top10.set_index("Associated")["percent"])
            else:
                st.caption("No associated data to chart.")
    else:
        st.info("No calls found for this main number.")

    st.divider()
    st.subheader("📅 Detailed Activity")

    # 2) Detailed stats (calls list + analytics)
    params = {"main": main}
    if associated:
        params["associated"] = associated

    stats_data = fetch_json(f"{API_URL}/stats", params=params, timeout=120)
    if not stats_data:
        st.stop()

    calls_df = pd.DataFrame(stats_data.get("calls", []))
    percent = stats_data.get("percent_of_total")

    # ======== Metrics row ========
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Filtered Calls", f"{0 if calls_df.empty else len(calls_df):,}")
    m2.metric("% of Main (Selected)", f"{(percent if percent is not None else 100):.2f}%")

    # Duration KPIs (prefer API duration_summary if present)
    dur = stats_data.get("duration_summary") or {}
    total_sec = dur.get("total_sec")
    avg_sec = dur.get("avg_sec")
    med_sec = dur.get("median_sec")

    # Fallbacks if API didn’t compute
    if (total_sec is None) and ("duration_seconds" in calls_df.columns):
        total_sec = float(calls_df["duration_seconds"].fillna(0).sum())
    if (avg_sec is None) and ("duration_seconds" in calls_df.columns):
        avg_sec = float(calls_df["duration_seconds"].dropna().mean()) if calls_df["duration_seconds"].dropna().shape[0] > 0 else 0.0
    if (med_sec is None) and ("duration_seconds" in calls_df.columns):
        med_sec = float(calls_df["duration_seconds"].dropna().median()) if calls_df["duration_seconds"].dropna().shape[0] > 0 else 0.0

    m3.metric("Total Duration", fmt_hms(total_sec or 0))
    m4.metric("Avg / Median", f"{fmt_hms(avg_sec or 0)} / {fmt_hms(med_sec or 0)}")

    # ======== Selected share vs others ========
    if associated and (percent is not None) and total_main_calls > 0:
        st.markdown("### Share of Calls: Selected vs Others")
        fig, ax = plt.subplots()
        others = max(0.0, 100.0 - float(percent))
        ax.pie([float(percent), others], labels=["Selected", "Others"], autopct="%1.1f%%")
        ax.axis("equal")
        st.pyplot(fig)

    # ======== Calls table ========
    if not calls_df.empty:
        display_cols = [
            "date", "day", "time",
            "associated_key", "associated_number", "associated_label",
            "duration_seconds", "direction", "call_type",
            "source_file", "start_datetime", "end_datetime"
        ]
        display_cols = [c for c in display_cols if c in calls_df.columns]
        st.dataframe(
            calls_df[display_cols],
            use_container_width=True,
            hide_index=True
        )

        # ======== 📅 Per-Date Summary ========
        st.markdown("### 📅 Per-Date Summary")
        if percent is not None:
            st.metric("Contact Percentage (of main)", f"{float(percent):.2f}%")

        # Ensure chronological order by start_datetime
        if "start_datetime" in calls_df.columns:
            calls_df = calls_df.sort_values("start_datetime", kind="stable")

        grouped_rows = []
        if "date" in calls_df.columns:
            for date_val, g in calls_df.groupby("date", dropna=False):
                day_name = g["day"].dropna().astype(str).mode().iloc[0] if "day" in g.columns and g["day"].notna().any() else ""

                # Associated contacts (counts) for this date
                if "associated_key" in g.columns:
                    assoc_counts = (
                        g["associated_key"]
                        .fillna("Unknown")
                        .astype(str)
                        .value_counts()
                    )
                    assoc_str = ", ".join([f"{k} ({v})" for k, v in assoc_counts.items()])
                else:
                    assoc_str = ""

                # times (sorted)
                times_list = sorted(g["time"].dropna().astype(str).tolist(), key=_sort_time_str) if "time" in g.columns else []
                times_str = ", ".join(times_list)

                # durations for each call (in original chronological order)
                durs = g["duration_seconds"].tolist() if "duration_seconds" in g.columns else []
                durs_hms = [_secs_to_hms(x) for x in durs]
                durs_str = ", ".join([d for d in durs_hms if d])

                # quick duration stats
                total_s = float(g["duration_seconds"].fillna(0).sum()) if "duration_seconds" in g.columns else 0.0
                avg_s = float(g["duration_seconds"].dropna().mean()) if "duration_seconds" in g.columns and g["duration_seconds"].dropna().shape[0] > 0 else 0.0

                grouped_rows.append({
                    "Date": str(date_val),
                    "Day": day_name,
                    "Associated (counts)": assoc_str,   # shows per-date associated contacts & counts
                    "Call Count": len(g),
                    "Call Timing": times_str,
                    "Call Durations (each)": durs_str,
                    "Total Duration": _secs_to_hms(total_s),
                    "Avg Duration": _secs_to_hms(avg_s),
                })

            per_date_df = pd.DataFrame(grouped_rows).sort_values("Date")
            st.dataframe(per_date_df, use_container_width=True, hide_index=True)

            st.download_button(
                "Download Per-Date Summary (CSV)",
                data=df_to_csv_bytes(per_date_df),
                file_name=f"{main}_{(associated or 'ALL').replace(' ', '_')}_per_date_summary.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.caption("No 'date' column available to summarize per date.")

        # ======== Distribution charts ========
        st.markdown("### Distribution by Direction & Call Type")
        dir_counts = counts_df_from_api_or_local(stats_data, calls_df, "direction_counts", "direction")
        type_counts = counts_df_from_api_or_local(stats_data, calls_df, "call_type_counts", "call_type")

        cdir, ctype = st.columns(2)
        with cdir:
            if not dir_counts.empty and "direction" in dir_counts.columns and "count" in dir_counts.columns:
                st.bar_chart(dir_counts.set_index("direction")["count"])
            else:
                st.caption("No direction data available.")
        with ctype:
            if not type_counts.empty and "call_type" in type_counts.columns and "count" in type_counts.columns:
                st.bar_chart(type_counts.set_index("call_type")["count"])
            else:
                st.caption("No call type data available.")

        # ======== Duration Histogram ========
        if "duration_seconds" in calls_df.columns and calls_df["duration_seconds"].dropna().shape[0] > 0:
            st.markdown("### Duration Histogram (seconds)")
            vals = calls_df["duration_seconds"].dropna().astype(float)
            fig2, ax2 = plt.subplots()
            bins = min(50, max(10, int(math.sqrt(len(vals)))))
            ax2.hist(vals, bins=bins)
            ax2.set_xlabel("Duration (seconds)")
            ax2.set_ylabel("Calls")
            st.pyplot(fig2)

        # ======== Time of Day (by hour) — ALWAYS SORTED 0..23 ========
        st.markdown("### Time of Day (by hour)")
        hourly = stats_data.get("hourly_counts")
        if isinstance(hourly, dict) and len(hourly) > 0:
            hourly_series = normalize_hour_series(hourly)
            st.bar_chart(hourly_series)
        else:
            if "time" in calls_df.columns and calls_df["time"].notna().any():
                hours = (
                    pd.to_datetime(calls_df["time"].astype(str), errors="coerce")
                    .dt.hour
                    .dropna()
                    .astype(int)
                )
                if not hours.empty:
                    hour_counts = hours.value_counts()
                    hourly_series = normalize_hour_series(hour_counts)
                    st.bar_chart(hourly_series)
                else:
                    st.caption("No valid time values to chart.")
            else:
                st.caption("No time field available.")

        # ======== Download filtered calls ========
        st.download_button(
            "Download Filtered Calls (CSV)",
            data=df_to_csv_bytes(calls_df),
            file_name=f"{main}_{(associated or 'ALL').replace(' ', '_')}_calls.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        if associated:
            st.info("No matching call records for this associated key.")
        else:
            st.info("No call records found.")

else:
    st.info("Enter a **Main Contact Number** above to begin.")
