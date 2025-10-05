
# 📞 Call Analytics App

FastAPI + Streamlit app to analyze call logs by **Main Contact** and **Associated Contact**.
Designed for Pakistani numbers (but robust to mixed formats). Timezone: **Asia/Karachi**.

---

## ✨ Features

**Data ingestion (CSV/XLS/XLSX)** with automatic normalization:

* Column auto-detection for A/B parties, start/end times, duration, direction, call type.
* Phone normalization to **MSISDN** (e.g., `0xxx…` → `92xxx…`, `3xxx…` → `92xxx…`).
* **Hex tokens → labels** (e.g., `4A617A7A` → `Jazz`) for associated labels.
* **Karachi timezone** applied; naive timestamps assumed **UTC** then converted.
* If `end_datetime` is missing but `start_datetime` + `duration` exist → **computed**.
* If `duration` is missing but `start_datetime` + `end_datetime` exist → **computed**.

**Association logic**

* `main_number` inferred from **filename** (e.g., `923007087230.xlsx`).
* `associated_key` unified as **phone (preferred)** or **label** (fallback).

**API (FastAPI)**

* `GET /contacts?main=…` → associated contacts with call counts + % share.
* `GET /stats?main=…&associated=…` → **simple (pandas) analytics** + aggregates.
* `GET /stats_ai?main=…&associated=…` → **AI analytics (OpenAI)** on **main-only** rows.
* `POST /upload` to add a file + `POST /reload` to refresh in-memory dataset.

**UI (Streamlit)**

* Enter **Main**; optionally filter by an **Associated** (phone or label).
* **Per-Date Summary** (sorted): call count, timings, durations (each), date & day, per-date associated counts.
* Charts: **Direction**, **Call Type**, **Hourly (0→23)**, **Duration histogram**.
* KPI tiles: total/filtered calls, **% of main**, **total/avg/median duration**.
* **Upload** CSV/XLSX from the sidebar (to API) + trigger reload.
* **AI / Simple** toggle to switch between OpenAI and local (pandas) analytics.
* Downloads: associated overview, per-date summary, filtered calls.

---

## 🗂️ Project Structure

```
call-analytics-app/
├─ api/
│  └─ main.py                  # FastAPI app (endpoints, models, upload/reload)
├─ app/
│  └─ dataloader.py            # Loader + normalize + compute_stats (simple path)
├─ models/
│  ├─ analytics.py             # Simple (pandas) analytics helpers
│  └─ ai_openai.py             # AI analytics via OpenAI (main-only slice + schema)
├─ config/
│  ├─ settings.py              # Centralized config (reads from .env)
│  └─ __init__.py
├─ ui/
│  └─ streamlit_app.py         # Streamlit UI with AI/Simple toggle & uploads
├─ data/                       # (ignored) Put your real datasets here
├─ data-sample/                # (optional) Small anonymized samples for demo
├─ .env.example                # Sample env with OPENAI_*
├─ requirements.txt
├─ README.md
└─ .gitignore
```

---


## 🛠️ Setup (Windows)

### 1) Clone or create the folder

```powershell
cd "C:\Users\centu\OneDrive\Desktop"
git clone https://github.com/<you>/call-analytics-app.git
cd call-analytics-app
```

*(or create the folder manually and add files if not cloned yet)*

### 2) Create & activate virtual environment

```powershell
python -m venv .venv
# If PowerShell blocks activation:
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# Then:
. .\.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

> If you don’t have `requirements.txt` yet:
>
> ```powershell
> pip install fastapi uvicorn[standard] pandas openpyxl pytz requests pydantic streamlit python-dotenv
> ```

---

## 🔐 Configure Environment (OpenAI for AI mode)

Copy `.env.example` to `.env` and fill in your key/model:

```
# .env
OPENAI_API_KEY=sk-********************************
OPENAI_MODEL=gpt-4o-mini    # or gpt-4.1-mini / gpt-4o etc.
```

The app reads these via `config/settings.py` (using `python-dotenv`).
**Note:** `.env` is **.gitignored**—keep your key private.

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

Open a **new terminal** (venv still active):

```powershell
streamlit run ui/streamlit_app.py
```

* UI: [http://127.0.0.1:8501](http://127.0.0.1:8501)
* Ensure **API_URL** in the sidebar points to your backend (default `http://127.0.0.1:8000`).
* Use the **AI / Simple** toggle at the top of the page:

  * **Simple** → local pandas analytics (fastest, offline).
  * **AI** → OpenAI-powered analysis using **only MAIN rows**.

---

## 📥 Adding Data

* Put `.csv`, `.xlsx`, `.xls` files into `data/`.
  **Name each file with the main number** (e.g., `923007087230.xlsx`).
* Or use the **Streamlit sidebar → Upload Contacts**:

  * **API (/upload)**: posts the file to your backend then triggers `/reload`.

### Optional API endpoints for upload/reload

(Already included in `api/main.py`—shown for reference.)

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

## 🧠 Analytics (Simple vs AI)

Both paths compute the same outputs for the selected **MAIN** (and optional **ASSOCIATED**):

1. **Call Count** (number of times contacted)
2. **Call Timing** (times per call)
3. **Call Duration** (per call)
4. **Date & Day** (per call)
5. **Date-wise Sorting** (chronological)
6. **Contact Percentage** (calls to the selected associated ÷ all main calls)

### Simple (Pandas) path

* Implemented in `app/dataloader.py::compute_stats` and `models/analytics.py`.
* Fast, offline, deterministic.

### AI (OpenAI) path

* Implemented in `models/ai_openai.py` with **structured JSON schema**.
* **Always sends only MAIN’s rows** to the model; if `associated` is provided, the model computes its stats **within the main-only slice**.
* Falls back gracefully (with an `error` field) if the API key/model is missing.

---

## 🧠 Data Rules (Normalization)

* **Phone numbers** (`normalize_number`)

  * Strip non-digits.
  * If starts with `92` and len ≥ 12 → keep.
  * If starts with `0` and len ≥ 11 → `92` + local without `0`.
  * If starts with `3` and len ≥ 10 → `92` + number.
  * Else if len ≥ 10 → keep digits.

* **Labels from hex** (e.g., `4A617A7A` → `Jazz`) via `maybe_decode_hex_token`.

* **Datetime**

  * Parse any standard format; if **naive**, assume UTC → convert to **Asia/Karachi**.

* **Duration**: supports `HH:MM:SS`, `MM:SS`, `"45 sec"`, raw seconds.

* **End time inference**

  * If `end_datetime` missing but `start_datetime` & `duration` exist → compute.
  * If `duration` missing but `start_datetime` & `end_datetime` exist → compute.

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

Returns grouped associated contacts for the given `main`.

```json
{
  "total_calls": 120,
  "contacts": [
    {"associated_key":"923001234567","call_count":38,"percent":31.67},
    {"associated_key":"Jazz","call_count":25,"percent":20.83}
  ]
}
```

### `GET /stats?main=<msisdn>[&associated=<phone-or-label>]` (Simple)

Returns the filtered call list (sorted), KPIs & aggregates.

### `GET /stats_ai?main=<msisdn>[&associated=<phone-or-label>]` (AI)

Same structure as `/stats`, but computed by OpenAI using **main-only** rows.

Example (truncated):

```json
{
  "total_calls": 120,
  "percent_of_total": 31.67,
  "contacts": [...],
  "calls": [...],
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

  * **Associated (counts)** per date
  * **Call Count**, **Call Timing** (list), **Call Durations (each)**
  * **Total** & **Average Duration**
* **Charts**: Direction / Call Type (bar), Duration histogram, **Time of Day (0→23)**.
* **Downloads**: associated overview, per-date summary, filtered calls.
* **Uploads**: add CSV/XLSX to backend and reload.

---

## 🪛 Troubleshooting

**PowerShell “running scripts is disabled”**

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**Streamlit can’t reach API**

* Start FastAPI first (`uvicorn api.main:app --reload --port 8000`).
* In Streamlit sidebar, set `API_URL` to `http://127.0.0.1:8000`.

**Empty charts / KeyError**

* The UI protects against empty frames, but if filters return no rows you’ll see empty sections.

**Karachi timezone**

* Naive timestamps assumed UTC → converted to `Asia/Karachi`.

**End time missing**

* Loader computes `end_datetime = start_datetime + duration` if possible.

**AI mode errors**

* Ensure `.env` has `OPENAI_API_KEY` and `OPENAI_MODEL`.
* If OpenAI returns schema errors, we fall back to a safe payload with an `error` note.

---

## 🔒 Privacy

* `data/` is **.gitignored** to avoid pushing sensitive call records.
* Use `data-sample/` with tiny anonymized files for demos.

---

## 📦 Dependencies

* `fastapi`, `uvicorn[standard]`
* `pandas`, `openpyxl`, `pytz`
* `pydantic`
* `requests`
* `streamlit`
* `python-dotenv`
* `openai` (for AI mode)

Install with:

```powershell
pip install -r requirements.txt
```

---

## 🧩 Extensibility

* Add column heuristics in `app/dataloader.py::_find_col`.
* Add new aggregates in API and surface in the UI.
* Switch timezone by editing `KARACHI_TZ` in `dataloader.py`.
* Replace AI model/version via `.env` without code changes.

---

## ✅ Quick Start

1. **API** → `uvicorn api.main:app --reload --port 8000`
2. **UI** → `streamlit run ui/streamlit_app.py`
3. Drop files into **data/** or **Upload** via the sidebar → **Analyze** 🎉

*Use the **AI/Simple** toggle at the top to switch engines.*


flowchart LR
  subgraph User["User (Browser)"]
    UI[Streamlit UI]
  end

  subgraph Backend["Backend (FastAPI)"]
    API[REST Endpoints<br/>(/health, /contacts, /stats, /stats_ai, /upload, /reload)]
    DL[Dataloader & Normalizer<br/>(CSV/XLSX → unified schema)]
    SIMPLE[Simple Analytics<br/>(Pandas)]
    AI[AI Analytics<br/>(OpenAI Structured JSON)]
  end

  subgraph Storage["Local Files"]
    DATA[data/ *.csv *.xlsx]
    CFG[.env / config/settings.py]
  end

  subgraph OpenAI["OpenAI API"]
    GPT[Model<br/>(OPENAI_MODEL)]
  end

  UI -- API_URL --> API
  API <-- upload/reload --> DATA
  API --> DL
  DL --> SIMPLE
  DL --> AI
  SIMPLE -- /stats --> UI
  AI -- /stats_ai --> UI
  CFG -. reads keys/models .-> AI
  AI -- calls --> GPT
