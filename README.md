
# 📞 Call Analytics App

FastAPI + Streamlit app to analyze call logs by **Main Contact** and **Associated Contact**.
Designed for Pakistani numbers (but robust to mixed formats). Timezone: **Asia/Karachi**.

---

## ✨ Features

* **Data ingestion (CSV/XLS/XLSX)** with automatic normalization:

  * Column auto-detection for A/B parties, start/end times, duration, direction, call type.
  * Phone normalization to **MSISDN** (e.g., `0xxx…` → `92xxx…`, `3xxx…` → `92xxx…`).
  * **Hex tokens → labels** (e.g., `4A617A7A` → `Jazz`) for associated labels.
  * **Karachi timezone** applied; naive timestamps are assumed **UTC** then converted.
  * If `end_datetime` is missing but `start_datetime` and `duration` exist → **computed**.
  * If `duration` is missing but `start/end` exist → **computed**.
* **Association logic**

  * `main_number` inferred from **filename** (e.g., `923007087230.xlsx`).
  * `associated_key` unified as **phone (preferred)** or **label** (fallback).
* **API (FastAPI)**

  * `GET /contacts?main=…` → list of associated contacts with call counts + % share.
  * `GET /stats?main=…&associated=…` → filtered calls + aggregates.
  * Optional: `POST /upload` to add a file + `POST /reload` to refresh in-memory dataset.
* **UI (Streamlit)**

  * Enter **Main**; optionally filter by an **Associated** (phone or label).
  * **Per-Date Summary** (sorted): call count, timings, durations (each), date & day, associated per-date.
  * Charts: **Direction**, **Call Type**, **Hourly (0→23)**, **Duration histogram**.
  * KPI tiles: total/filtered calls, **% of main**, **total/avg/median duration**.
  * **Upload** CSV/XLSX from the sidebar (to API or save locally) + trigger reload.
  * Download CSVs (associated overview, per-date summary, filtered calls).

---

## 🗂️ Project Structure

```
call-analytics-app/
├─ api/
│  └─ main.py                # FastAPI app (endpoints, models)
├─ app/
│  └─ dataloader.py          # Loader + compute_stats
├─ ui/
│  └─ streamlit_app.py       # Streamlit UI
├─ data/                     # (ignored) Put your real datasets here
├─ data-sample/              # (optional) Small anonymized samples for demo
├─ requirements.txt
├─ README.md
└─ .gitignore
```

---

## 🧩 Requirements

* Windows 10/11
* Python **3.11** (recommended)
* Git (for version control)
* (Optional) GitHub Desktop or GitHub CLI

---

## 🛠️ Setup (Windows)

### 1) Clone or create the folder

```powershell
cd "C:\Users\centu\OneDrive\Desktop"
git clone https://github.com/<you>/call-analytics-app.git
cd call-analytics-app
```

*(or just create the folder and add files if not cloned yet)*

### 2) Create & activate virtual environment

```powershell
python -m venv .venv
# If you see an Execution Policy error enabling scripts, run:
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# then:
. .\.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

> If you don’t have `requirements.txt` yet, use:
>
> ```powershell
> pip install fastapi uvicorn[standard] pandas openpyxl streamlit pytz requests pydantic
> ```

---

## ▶️ Run the Apps

### Backend (FastAPI)

```powershell
# from project root (venv active)
uvicorn api.main:app --reload --port 8000
```

* API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* Health: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### Frontend (Streamlit)

Open a **new terminal** with venv active:

```powershell
streamlit run ui/streamlit_app.py
```

* UI: [http://127.0.0.1:8501](http://127.0.0.1:8501)
* Ensure **API_URL** in the sidebar points to your backend (default is `http://127.0.0.1:8000`).

---

## 📥 Adding Data

* Put `.csv`, `.xlsx`, `.xls` files into `data/`.
  **Name each file with the main number** (e.g., `923007087230.xlsx`).
* Or use the **Streamlit sidebar → Upload Contacts**:

  * **API (/upload)**: posts the file to your backend then triggers `/reload`.
  * **Local data folder**: saves to `../data` and attempts `/reload`.

### Optional API endpoints for upload/reload

Add these to `api/main.py` if not present:

```python
from fastapi import UploadFile, File

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
CALLS_DF = load_calls(DATA_DIR)

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
```

---

## 🧠 Data Rules (Normalization)

* **Phone numbers** (`normalize_number`)

  * Strip non-digits.
  * If starts with `92` and length ≥ 12 → keep.
  * If starts with `0` and length ≥ 11 → `92` + local without `0`.
  * If starts with `3` and length ≥ 10 → `92` + number.
  * Else if length ≥ 10 → keep digits.
* **Labels from hex** (e.g., `4A617A7A` → `Jazz`) via `maybe_decode_hex_token`.
* **Datetime**

  * Parse any standard format; if **naive**, assume UTC → convert to **Asia/Karachi**.
* **Duration**: supports `HH:MM:SS`, `MM:SS`, `"45 sec"`, raw seconds.
* **End time inference**:

  * If `end_datetime` missing but `start_datetime` & `duration` exist → compute.
  * If `duration` missing but both start & end exist → compute.
* **Unified schema** (returned):

  ```
  main_number, associated_number, associated_label, associated_key,
  start_datetime, end_datetime, duration_seconds,
  date, day_name, time, direction, call_type, source_file
  ```

---

## 🌐 API Endpoints

### `GET /health`

```json
{"status":"ok","rows":12345}
```

### `GET /contacts?main=<msisdn>`

* Returns grouped associated contacts for the given `main`.

```json
{
  "total_calls": 120,
  "contacts": [
    {"associated_key":"923001234567","call_count":38,"percent":31.67},
    {"associated_key":"Jazz","call_count":25,"percent":20.83}
  ]
}
```

### `GET /stats?main=<msisdn>[&associated=<phone-or-label>]`

* Returns the filtered call list (sorted), KPIs & aggregates.
* Includes direction/type counts, duration summary, **hourly (0–23)**.

```json
{
  "total_calls": 120,
  "percent_of_total": 31.67,
  "contacts": [...],
  "calls": [
    {
      "date":"2025-01-10","day":"Friday","time":"09:15:22",
      "start_datetime":"2025-01-10T04:15:22+05:00",
      "end_datetime":"2025-01-10T04:20:22+05:00",
      "duration_seconds":300.0,
      "associated_key":"923001234567","direction":"Outgoing","call_type":"Voice",
      "source_file":"923007087230.xlsx"
    }
  ],
  "direction_counts":[{"direction":"Outgoing","count":70},{"direction":"Incoming","count":50}],
  "call_type_counts":[{"call_type":"Voice","count":110},{"call_type":"SMS","count":10}],
  "duration_summary":{
    "total_sec":54000,"avg_sec":450,"median_sec":300,
    "max_sec":2400,"min_sec":5,
    "total_hms":"15:00:00","avg_hms":"0:07:30","median_hms":"0:05:00","max_hms":"0:40:00","min_hms":"0:00:05"
  },
  "hourly_counts":{"0":0,"1":1,...,"23":2}
}
```

---

## 🖥️ UI Highlights (Streamlit)

* **Associated Overview**: calls and % share for each associated.
* **Per-Date Summary** (sorted):

  * **Associated (counts)** for that date
  * **Call Count**, **Call Timing** (list of times), **Call Durations (each)**
  * **Total** & **Average Duration**
* **Charts**:

  * Direction & Call Type (bar)
  * Duration histogram (auto bins)
  * **Time of Day (hourly 0→23)** — always sorted, fills missing hours with 0
* **Downloads**:

  * Associated overview CSV
  * Per-date summary CSV
  * Filtered calls CSV
* **Upload Contacts**:

  * Upload a new CSV/XLSX to the backend or local data folder and **reload**.

---

## 🧪 Example File Naming & Columns

* File name: `923007087230.xlsx` → used as `main_number`.
* Typical columns (auto-detected regardless of exact names/case/spaces):

  * A-party: `a_number`, `a party`, `calling`, `source`, `from`, …
  * B-party: `b_number`, `b party`, `called`, `destination`, `to`, …
  * Start time: `datetime`, `date_time`, `start_time`, `call_time`, `start`, …
  * End time: `end_time`, `end`
  * Duration: `duration`, `call_duration`, `talk_time`
  * Direction: `direction`, `call_direction`
  * Type: `call_type`, `type`, `category`

> The loader ignores noisy columns like IMSI/IMEI/LAC/Cell/Site/Lat/Lon, etc.

---

## 🪛 Troubleshooting

* **PowerShell “running scripts is disabled”**
  Run **as admin** once:

  ```powershell
  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
  ```
* **Streamlit can’t reach API**

  * Start FastAPI first: `uvicorn api.main:app --reload --port 8000`
  * In Streamlit sidebar, set `API_URL` to `http://127.0.0.1:8000`.
* **Empty charts / KeyError in charts**
  The UI guards against empties; if you still see issues, your dataset may have no matching rows for that filter.
* **Karachi timezone**
  Timestamps with no timezone are assumed **UTC** then converted to `Asia/Karachi`.
* **End time missing in source file**
  Loader computes `end_datetime = start_datetime + duration` if possible.
* **Associated appears as gibberish**
  It might be **hex-encoded**; loader tries to decode to a human-readable label.

---

## 🔒 Notes on Data Privacy

* The `data/` folder is **.gitignored** by default to avoid pushing sensitive call records.
* Use `data-sample/` with tiny anonymized files for demos.

---

## 📦 Dependencies

* `fastapi`, `uvicorn[standard]`
* `pandas`, `openpyxl`, `pytz`
* `pydantic`
* `requests`
* `streamlit`

Install all with:

```powershell
pip install -r requirements.txt
```

---

## 🧩 Extensibility

* Add additional column heuristics in `app/dataloader.py::_find_col`.
* Add new aggregates in API and surface them in the UI.
* Swap Karachi TZ by changing `KARACHI_TZ` in `dataloader.py`.

---

## ✅ Done!

* Start **API** → `uvicorn api.main:app --reload --port 8000`
* Start **UI** → `streamlit run ui/streamlit_app.py`
* Drop files into **data/** or **Upload** via the sidebar → **Analyze** 🎉

If you want, I can also generate a **`requirements.txt`** or a minimal **GitHub Actions** workflow for linting/tests.
