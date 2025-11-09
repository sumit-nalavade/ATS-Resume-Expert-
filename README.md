# ATS-Resume-Expert-
AI-powered Streamlit app that analyzes and scores resumes against job descriptions using keyword, experience, and education matching, with Google Gemini integration for ATS-optimized rewrites and smart HR-style evaluations.


# ATS Resume Expert — Enhanced

**One-page overview:**
A Streamlit app that scores and evaluates PDF resumes against a pasted Job Description (JD). It uses deterministic keyword, experience, and education heuristics plus (optional) Google Generative AI (Gemini) for rewrite and human-style evaluation. Designed to work on Windows and *nix with Poppler (pdf2image) and/or `pymupdf` fallback.

---

## Features

* Single resume evaluation with:

  * First-page preview (image) rendered from PDF
  * Extracted text preview (first ~2000 chars)
  * Deterministic scoring: keyword match, years experience, education match, overall weighted score
  * Optional LLM-driven resume rewrite (ATS-optimised)
  * Optional LLM-driven detailed evaluation
* Batch processing:

  * Upload multiple PDFs or a ZIP of PDFs
  * Deterministic scoring leaderboard and downloadable JSON
  * Optional per-resume LLM evaluation (costly)
* Admin page:

  * Environment & model visibility
  * Poppler path shown for debugging
* Robust PDF rendering: `pdf2image` + Poppler preferred; `pymupdf` (fitz) fallback supported for both rendering and text extraction

---

## Repo layout (relevant)

```
.
├─ app.py                # Streamlit web app (main)
├─ requirements.txt      # Python deps (suggested)
├─ README.md             # This file
├─ .env.example          # Example environment variables
└─ assets/               # (optional) logos, sample JDs, sample PDFs
```

---

## Quick start (recommended)

### 1) Clone & create environment

```bash
git clone <repo-url>
cd <repo-dir>

# create virtualenv (example)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Configure environment variables

Create a `.env` file in the project root (or set OS env vars). Use `.env.example` as a template.

`.env` example:

```
GOOGLE_API_KEY=ya29....            # required only if you use LLM features
STREAMLIT_SERVER_PORT=8501         # optional
```

> **Important:** LLM features (rewrite, evaluation) call Google Generative API. Ensure your key has the right project/billing/permissions. If you do not have a key, leave it blank — deterministic scoring still works.

### 3) (Windows) Install Poppler (recommended for pdf2image rendering)

* Download Poppler for Windows (a pre-built binary distribution).
* Unzip and note the `bin` folder path, e.g.:
  `C:\Program Files\poppler-25.07.0\Library\bin`
* You can either:

  * Add that `bin` folder to your PATH, **or**
  * Edit the top of `app.py` and set `PREFERRED_POPPLER_DIR` to the `bin` path (the app already includes a variable for this).

On macOS:

```bash
brew install poppler
```

On Linux (Debian/Ubuntu):

```bash
sudo apt-get update
sudo apt-get install -y poppler-utils
```

### 4) (Optional) Install pymupdf fallback

If you can't install Poppler or want a fallback for rendering/extraction:

```bash
pip install pymupdf
```

The app will use `pymupdf` if Poppler/pdf2image fails.

### 5) Run Streamlit

```bash
streamlit run app.py
```

Open the URL the command prints (default `http://localhost:8501`).

---

## Usage notes

### UI flow

* Use the **sidebar** to paste a Job Description (JD) — required for scoring.
* Choose `Mode` from the sidebar: `Single Resume`, `Batch Processing`, `Admin Dashboard`.
* Single Resume:

  * Upload a PDF, preview, then press:

    * **Compute Scoring Breakdown** — deterministic scoring only
    * **Rewrite for ATS** — uses LLM to rewrite (requires `GOOGLE_API_KEY`)
    * **Run LLM Full Evaluation** — LLM produces human-like evaluation (requires key)
* Batch Processing:

  * Upload multiple PDFs or a single ZIP.
  * Optionally enable LLM per resume (can be slow and cost money).
  * Download results as `batch_results.json`.

### Scoring breakdown (deterministic)

* `keyword_score` — top keyword matches between JD and resume
* `years_score` — compares years found in resume vs years requested in JD (if present)
* `education_score` — checks for presence of degree keywords requested by JD
* `overall_score` — weighted average: 60% keywords, 25% experience, 15% education

---

## Configuration & debugging tips

### Poppler / pdf2image errors

* Common error: `pdfinfo`/Poppler not found. Fix by:

  * Installing Poppler and ensuring `poppler/bin` is in PATH; or
  * Set `PREFERRED_POPPLER_DIR` at top of `app.py` to the `bin` folder path (Windows users).
* If pdf2image still fails and you have `pymupdf` installed, the app will attempt `pymupdf` fallback automatically.

### If text extraction is empty

* PDFs that are scanned images may not contain selectable text. Consider OCRing them first (not included). Rendered preview still shows first page image.
* `pymupdf` text extraction is generally reliable for text PDFs; install it for better results.

### LLM issues (Gemini / Google Generative)

* If `_list_visible_model_ids()` fails or returns empty:

  * Check `GOOGLE_API_KEY` in `.env` and that it is active + has billing/project access.
  * Ensure the key can call the Generative API (API enabled in GCP).
* If `get_gemini_response` raises an error about embeddings-only models:

  * The code attempts multiple models and will surface a helpful message; review logs shown on the Admin page.
* LLM usage is **not** free. Monitor your GCP billing.

### Logging

* `app.py` configures a logger named `ats_app`. Watch console output for warnings and helpful debug messages.

---

## Security & privacy

* Uploaded resumes and JDs are handled in memory by the Streamlit process in this implementation. If you deploy publicly:

  * Implement authentication on the app.
  * Persist files only if necessary and with encryption/retention policies.
  * Review privacy rules and obtain candidate consent before sending files to external LLM APIs.

---

## Dependencies (suggested `requirements.txt`)

```
streamlit>=1.25
pdf2image>=1.16
pillow>=9.0
python-dotenv>=0.21
google-generativeai>=0.2
pymupdf>=1.22   # optional fallback: pip install pymupdf
pandas>=1.5
```

> Pin versions as appropriate for your environment.

---

## Key functions (quick dev reference)

* `input_pdf_setup(uploaded_file, poppler_path=POPLER_BIN)`
  Renders the first PDF page to JPEG bytes (prefers `pdf2image`+Poppler; falls back to `pymupdf`).
* `extract_text_from_pdf_bytes(pdf_bytes)`
  Extracts text from PDF bytes using `pymupdf` when available; otherwise returns empty string with warning.
* `compute_overall_score(resume_text, jd_text)`
  Returns breakdown: keyword score, years, education, overall.
* `get_gemini_response(system_prompt, pdf_content, user_prompt, explicit_model_id=None)`
  Tries to call Google Generative models to produce text (auto-selects preferred models when available).

---

## Troubleshooting quick checklist

* No preview / `pdf2image` errors: confirm Poppler `bin` in PATH or set `PREFERRED_POPPLER_DIR` in `app.py`.
* Empty text extraction: install `pymupdf` (`pip install pymupdf`) or OCR scanned PDFs.
* LLM/Model listing errors: validate `GOOGLE_API_KEY` and GCP project/billing + API enabled.
* `streamlit` startup issues: ensure virtualenv is activated and correct Python version (3.8–3.11 recommended).

---

## Example `.env.example`

```
# Add your Google Generative API key (if you intend to use LLM features)
GOOGLE_API_KEY=

# Optional Streamlit overrides
STREAMLIT_SERVER_PORT=8501
```

---

## Caveats & Future improvements

* OCR pipeline for image-only PDFs (Tesseract or cloud OCR).
* Better keyword extraction (lemmatization / synonyms) and configurable weighting.
* Role-based admin: persist batch run history, allow job templates per role.
* Rate-limiting and queuing for large batch runs with LLM usage to control cost.


