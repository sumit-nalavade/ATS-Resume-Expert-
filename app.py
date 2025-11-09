# app.py
from dotenv import load_dotenv
load_dotenv()

import os
import io
import json
import base64
import zipfile
import streamlit as st
from PIL import Image
import pdf2image
import google.generativeai as genai
import logging
import re
from datetime import datetime

# ---- POPPLER / rendering config (insert near top of file) ----
import os

# If you have Poppler installed, set this to the bin folder path (Windows example).
# You mentioned earlier: "C:\Program Files (x86)\Release-25.07.0-0\poppler-25.07.0\Library"
# The actual executables live in the 'bin' subfolder — usually:
PREFERRED_POPPLER_DIR = r"C:\Program Files (x86)\Release-25.07.0-0\poppler-25.07.0\Library\bin"

# Make POPPLER_BIN either the valid path or None (so code falls back to pymupdf if available).
if os.path.isdir(PREFERRED_POPPLER_DIR):
    POPPLER_BIN = PREFERRED_POPPLER_DIR
else:
    POPPLER_BIN = None

# Friendly admin-display value for the UI (avoids NameError)
POPPLER_DISPLAY = POPPLER_BIN if POPPLER_BIN else "<not configured; will try pymupdf fallback>"

# ---- End POPPLER / rendering config ----


# Optional fallback renderer & text extractor
try:
    import fitz  # pymupdf
    _HAVE_PYMUPDF = True
except Exception:
    _HAVE_PYMUPDF = False

# Configure Google API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Poppler bin (Windows) - adjust if needed
POPLER_BIN = r"C:\Program Files (x86)\Release-25.07.0-0\poppler-25.07.0\Library\bin"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ats_app")

# ---------------- Model helpers (kept from your prior version) ----------------
def _list_visible_model_ids():
    ids = []
    try:
        for m in genai.list_models():
            mid = getattr(m, "name", None) or getattr(m, "id", None) or None
            if mid:
                ids.append(mid)
    except Exception as e:
        raise RuntimeError(f"Failed to call list_models(): {e}")
    return ids

def _filter_candidate_generative_models(all_model_ids):
    low = [m.lower() for m in all_model_ids]
    preferred_substrings = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini"
    ]
    candidates = []
    for pref in preferred_substrings:
        for idx, mid in enumerate(low):
            if pref in mid and all_model_ids[idx] not in candidates:
                candidates.append(all_model_ids[idx])
    for idx, mid in enumerate(low):
        if any(x in mid for x in ["embedding", "imagen", "veo", "aqa"]):
            continue
        if all_model_ids[idx] not in candidates:
            candidates.append(all_model_ids[idx])
    return candidates

def get_model_for_generate(preferred_models=None):
    if preferred_models is None:
        preferred_models = [
            "models/gemini-2.5-pro",
            "models/gemini-2.5-flash",
            "models/gemini-2.0-flash",
        ]
    available = _list_visible_model_ids()
    if not available:
        raise RuntimeError("No models returned by list_models(); check API key, billing, and permissions.")
    for pref in preferred_models:
        for a in available:
            if pref.lower() in a.lower():
                return a
    candidates = _filter_candidate_generative_models(available)
    if candidates:
        return candidates[0]
    return available[0]

def get_gemini_response(system_prompt, pdf_content, user_prompt, explicit_model_id=None):
    """Attempts multiple candidate models (keeps your prior shape of generate_content call)."""
    try:
        visible = _list_visible_model_ids()
    except Exception as e:
        raise RuntimeError(f"Could not list models: {e}")
    if not visible:
        raise RuntimeError("No models visible to your key. Check API key, project, billing, and permissions.")
    candidates = []
    if explicit_model_id:
        candidates.append(explicit_model_id)
    try:
        auto = get_model_for_generate()
        if auto and auto not in candidates:
            candidates.append(auto)
    except Exception as e:
        logger.warning("Auto selection failed: %s", e)
    candidates.extend(_filter_candidate_generative_models(visible))
    seen = set()
    final_candidates = []
    for c in candidates:
        if c and c not in seen:
            final_candidates.append(c)
            seen.add(c)
    last_error = None
    for model_id in final_candidates:
        try:
            logger.info("Trying model: %s", model_id)
            model = genai.GenerativeModel(model_id)
            response = model.generate_content([system_prompt, pdf_content[0], user_prompt])
            return response.text
        except Exception as e:
            logger.warning("Model %s failed: %s", model_id, e)
            last_error = (model_id, e)
            continue
    debug_sample = visible[:40]
    msg_lines = []
    if last_error:
        mid, exc = last_error
        msg_lines.append(f"Last attempt: model '{mid}' failed with: {exc}")
    msg_lines.append(f"Models visible to your key (sample): {debug_sample}")
    msg_lines.append("Common causes: attempted model is embeddings-only or not supported for generate_content; your API key lacks permission; SDK/API version mismatch.")
    raise RuntimeError("\n".join(msg_lines))

# ---------------- PDF rendering helpers (kept) ----------------
def render_pdf_first_page_with_pymupdf(pdf_bytes):
    if not _HAVE_PYMUPDF:
        raise RuntimeError("pymupdf is not installed. Install with: pip install pymupdf")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap()
    jpeg_bytes = pix.tobytes("jpeg")
    return jpeg_bytes

def input_pdf_setup(uploaded_file, poppler_path=POPLER_BIN):
    if uploaded_file is None:
        raise FileNotFoundError("No file uploaded")
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
    pdf_bytes = uploaded_file.read()
    if not pdf_bytes:
        raise ValueError("Uploaded file appears empty")
    try:
        if poppler_path:
            images = pdf2image.convert_from_bytes(pdf_bytes, poppler_path=poppler_path)
        else:
            images = pdf2image.convert_from_bytes(pdf_bytes)
        first_page = images[0]
        img_buf = io.BytesIO()
        first_page.save(img_buf, format="JPEG")
        img_bytes = img_buf.getvalue()
    except Exception as e:
        msg = str(e).lower()
        if ("pdfinfo" in msg or "poppler" in msg or isinstance(e, pdf2image.PDFInfoNotInstalledError)) and _HAVE_PYMUPDF:
            try:
                img_bytes = render_pdf_first_page_with_pymupdf(pdf_bytes)
            except Exception as e2:
                raise RuntimeError(f"Both pdf2image+poppler and pymupdf rendering failed: {e2}") from e2
        else:
            if _HAVE_PYMUPDF:
                try:
                    img_bytes = render_pdf_first_page_with_pymupdf(pdf_bytes)
                except Exception as e2:
                    raise RuntimeError(f"Rendering failed with pdf2image and pymupdf fallback also failed: {e2}") from e2
            else:
                raise RuntimeError(
                    "pdf2image failed to render the PDF. Ensure Poppler is installed and the 'bin' folder is "
                    "reachable (either in PATH or passed as poppler_path). Alternatively install pymupdf "
                    "(`pip install pymupdf`) to use the fallback renderer."
                ) from e
    pdf_parts = [
        {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(img_bytes).decode()
        }
    ]
    return pdf_parts

# ---------------- Text extraction helper ----------------
def extract_text_from_pdf_bytes(pdf_bytes):
    """
    Extract plain text from PDF bytes. Uses PyMuPDF (fitz) if available.
    If not available, returns an empty string and a warning message.
    """
    if _HAVE_PYMUPDF:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = []
            for page in doc:
                text.append(page.get_text("text"))
            return "\n".join(text)
        except Exception as e:
            logger.warning("pymupdf text extraction failed: %s", e)
            return ""
    else:
        logger.warning("pymupdf not installed; extract_text_from_pdf_bytes will return empty string. Install pymupdf for full functionality.")
        return ""

def extract_text_from_uploaded_file(uploaded_file):
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
    pdf_bytes = uploaded_file.read()
    return extract_text_from_pdf_bytes(pdf_bytes)

# ---------------- Scoring module ----------------
STOPWORDS = set(["the","and","a","an","in","on","for","with","to","of","by","is","are","as","at","from","that","this","it","be","or"])

def simple_keyword_extractor(text, top_n=40):
    tokens = re.findall(r"[a-zA-Z0-9\+\#\-]+", text.lower())
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1 and not t.isdigit()]
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    sorted_tokens = sorted(freq.items(), key=lambda x: -x[1])
    return [t for t, _ in sorted_tokens[:top_n]]

def compute_keyword_score(resume_text, jd_text):
    jd_keywords = simple_keyword_extractor(jd_text, top_n=60)
    if not jd_keywords:
        return 0.0, []
    resume_lower = resume_text.lower()
    matched = [kw for kw in jd_keywords if re.search(r"\b" + re.escape(kw) + r"\b", resume_lower)]
    score = len(matched) / len(jd_keywords) * 100.0
    return score, matched

def parse_years_experience(resume_text):
    # 1) look for explicit 'X years' patterns
    m = re.findall(r"(\d{1,2})\s*\+?\s*(?:years|yrs|yr)\b", resume_text.lower())
    if m:
        years = max([int(x) for x in m])
        return years
    # 2) look for date ranges like 2018-2022 or Jan 2018 - Mar 2022
    ranges = re.findall(r"(\b\d{4})\s*[-–]\s*(\d{4}\b|present)", resume_text.lower())
    total = 0
    for start, end in ranges:
        try:
            s = int(start)
            e = datetime.now().year if end.strip() in ("present", "current") else int(end)
            if e >= s:
                total += (e - s)
        except Exception:
            continue
    if total > 0:
        return total
    return 0

def education_match_score(resume_text, jd_text):
    edu_keywords = ["bachelor","b.sc","btech","b.e","master","m.sc","mtech","mba","phd","doctor"]
    jd_lower = jd_text.lower()
    resume_lower = resume_text.lower()
    needed = [k for k in edu_keywords if k in jd_lower]
    if not needed:
        # no education mention in JD -> full score
        return 100.0, []
    matched = [k for k in needed if k in resume_lower]
    score = len(matched) / len(needed) * 100.0
    return score, matched

def compute_overall_score(resume_text, jd_text):
    kw_score, matched_keywords = compute_keyword_score(resume_text, jd_text)
    years = parse_years_experience(resume_text)
    # target years: try extract from JD (e.g., '3+ years')
    jd_years = 0
    m = re.search(r"(\d{1,2})\s*\+?\s*(?:years|yrs|yr)\b", jd_text.lower())
    if m:
        jd_years = int(m.group(1))
    # compute years score: if jd_years ==0 -> full points
    if jd_years == 0:
        years_score = 100.0
    else:
        years_score = min(100.0, (years / jd_years) * 100.0)
    edu_score, matched_edu = education_match_score(resume_text, jd_text)
    overall = 0.6 * kw_score + 0.25 * years_score + 0.15 * edu_score
    return {
        "keyword_score": round(kw_score, 1),
        "matched_keywords": matched_keywords,
        "years_found": years,
        "years_score": round(years_score, 1),
        "education_score": round(edu_score, 1),
        "matched_education": matched_edu,
        "overall_score": round(overall, 1)
    }

# ---------------- Batch processing helpers ----------------
def process_single_resume_file(file_like, jd_text, run_llm=False, explicit_model=None):
    """
    Returns a dict with filename, scores, extracted_text, and optionally llm_response.
    file_like: Streamlit UploadedFile or (filename, bytes)
    """
    if hasattr(file_like, 'read'):
        try:
            file_like.seek(0)
        except Exception:
            pass
        raw_bytes = file_like.read()
        filename = getattr(file_like, 'name', 'uploaded.pdf')
    else:
        filename, raw_bytes = file_like

    text = extract_text_from_pdf_bytes(raw_bytes)
    scores = compute_overall_score(text, jd_text)
    result = {
        "filename": filename,
        "scores": scores,
        "text_snippet": (text[:1000] + '...') if text else "",
    }
    if run_llm:
        try:
            img_part = input_pdf_setup(io.BytesIO(raw_bytes))
            system_prompt = "You are an experienced Technical Human Resource Manager. Provide a concise evaluation of the resume vs job description."
            user_prompt = jd_text
            llm_resp = get_gemini_response(system_prompt, img_part, user_prompt, explicit_model_id=explicit_model)
            result['llm_response'] = llm_resp
        except Exception as e:
            result['llm_error'] = str(e)
    return result

# ---------------- Streamlit UI -------------------
st.set_page_config(page_title="ATS Resume EXpert", layout="wide")
st.title("ATS Resume Expert — Enhanced")

page = st.sidebar.selectbox("Mode", ["Single Resume", "Batch Processing", "Admin Dashboard"]) 

jd_text = st.sidebar.text_area("Job Description (or paste JD)", height=200)
model_choice = st.sidebar.checkbox("Let app choose model automatically (recommended)", value=True)
custom_model = None
if not model_choice:
    try:
        models = _list_visible_model_ids()
    except Exception:
        models = []
    custom_model = st.sidebar.selectbox("Explicit model (optional)", [None] + models)

st.sidebar.markdown("---")


# -------- Single Resume Page --------
if page == "Single Resume":
    st.header("Single Resume Evaluation")
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"]) 
    col1, col2 = st.columns([1, 1])

    if uploaded_file is not None:
        with col1:
            st.success("PDF uploaded")
            st.write("**Preview (first page)**")
            try:
                img = input_pdf_setup(uploaded_file)
                img_data = base64.b64decode(img[0]['data'])
                st.image(Image.open(io.BytesIO(img_data)), caption="First page")
            except Exception as e:
                st.error(f"Could not render preview: {e}")

        with col2:
            st.write("**Extracted text (first 1000 chars)**")
            text = extract_text_from_uploaded_file(uploaded_file)
            st.text_area("Resume text snippet", value=text[:2000], height=200)

        run_score = st.button("Compute Scoring Breakdown")
        run_rewrite = st.button("Rewrite for ATS")
        run_llm_eval = st.button("Run LLM Full Evaluation")

        if run_score:
            resume_text = extract_text_from_uploaded_file(uploaded_file)
            scores = compute_overall_score(resume_text, jd_text)
            st.subheader("Scoring Breakdown")
            st.metric("Overall Score", f"{scores['overall_score']}%")
            st.progress(int(scores['overall_score']))
            st.write(f"Keyword score: {scores['keyword_score']}% — Matched: {', '.join(scores['matched_keywords'][:20])}")
            st.write(f"Experience found: {scores['years_found']} years — Score: {scores['years_score']}%")
            st.write(f"Education score: {scores['education_score']}% — Matched: {', '.join(scores['matched_education'])}")

        if run_rewrite:
            resume_text = extract_text_from_uploaded_file(uploaded_file)
            prompt = (
                "You are an expert resume writer. Rewrite and optimize the following resume content for ATS and recruiters. "
                "Keep bullets concise, use strong action verbs, preserve facts but improve clarity. Return only the rewritten resume (no commentary).\n\n"
                f"JOB DESCRIPTION:\n{jd_text}\n\nRESUME_TEXT:\n{resume_text[:6000]}"
            )
            try:
                img_part = input_pdf_setup(uploaded_file)
            except Exception:
                img_part = [{"mime_type":"text/plain","data":base64.b64encode(b"").decode()}]
            try:
                explicit_model = custom_model if custom_model else None
                rewritten = get_gemini_response(prompt, img_part, "", explicit_model_id=explicit_model)
                st.subheader("ATS-Optimized Resume (LLM)")
                st.text_area("Rewritten Resume", value=rewritten, height=400)
                st.download_button("Download rewritten (txt)", data=rewritten, file_name="rewritten_resume.txt")
            except Exception as e:
                st.error(f"Rewrite failed: {e}")

        if run_llm_eval:
            try:
                img_part = input_pdf_setup(uploaded_file)
                explicit_model = custom_model if custom_model else None
                system_prompt = "You are an experienced Technical Human Resource Manager. Provide a detailed evaluation of the resume vs the job description. Include strengths, weaknesses, missing keywords, and a short scoreout of 100."
                llm_out = get_gemini_response(system_prompt, img_part, jd_text, explicit_model_id=explicit_model)
                st.subheader("LLM Evaluation")
                st.write(llm_out)
            except Exception as e:
                st.error(f"LLM evaluation failed: {e}")

# -------- Batch Processing Page --------
elif page == "Batch Processing":
    st.header("Batch Resume Processing")
    st.write("Upload multiple PDF files or a ZIP file containing PDFs. The system will run deterministic scoring for each file and show a leaderboard.")
    uploaded_files = st.file_uploader("Upload PDFs or a single ZIP", accept_multiple_files=True)
    run_batch = st.button("Run Batch Processing")
    run_llm_in_batch = st.checkbox("Also run LLM per resume (costly)", value=False)

    results = []
    if run_batch and uploaded_files:
        files_to_process = []
        for uf in uploaded_files:
            if uf.name.lower().endswith('.zip'):
                with zipfile.ZipFile(io.BytesIO(uf.read())) as z:
                    for name in z.namelist():
                        if name.lower().endswith('.pdf'):
                            data = z.read(name)
                            files_to_process.append((name, data))
            elif uf.name.lower().endswith('.pdf'):
                files_to_process.append(uf)
        if not files_to_process:
            st.warning("No PDF files found in the uploaded items.")
        else:
            progress_bar = st.progress(0)
            for i, f in enumerate(files_to_process):
                try:
                    res = process_single_resume_file(f, jd_text, run_llm=run_llm_in_batch, explicit_model=custom_model)
                    results.append(res)
                except Exception as e:
                    results.append({"filename": getattr(f, 'name', f[0]), "error": str(e)})
                progress_bar.progress(int((i+1)/len(files_to_process)*100))
            st.success("Batch processing completed.")
            leaderboard = []
            for r in results:
                scores = r.get('scores', {})
                overall = scores.get('overall_score', 0) if scores else 0
                leaderboard.append({
                    'filename': r.get('filename',''),
                    'overall_score': overall,
                    'keyword_score': scores.get('keyword_score',0),
                    'years_found': scores.get('years_found',0),
                    'education_score': scores.get('education_score',0)
                })
            import pandas as pd
            df = pd.DataFrame(leaderboard).sort_values('overall_score', ascending=False)
            st.subheader("Leaderboard")
            st.dataframe(df)
            st.download_button("Download results (JSON)", data=json.dumps(results, indent=2), file_name="batch_results.json")

# -------- Admin Dashboard Page --------
else:
    st.header("Admin Dashboard (summary)")
    st.write("This page will surface recent batch runs and top candidates. For now it shows quick model and environment info.")
    st.subheader("Environment & Models")
    st.write("pymupdf installed: ", _HAVE_PYMUPDF)
    try:
        visible = _list_visible_model_ids()
        st.write("Visible models (sample):", visible[:20])
    except Exception as e:
        st.error(f"Could not list models: {e}")

    st.subheader("Quick checks")
    st.write("Poppler path used:", POPPLER_BIN)
    st.write("Tips: Use Batch Processing to run mass-evaluations. Download results as JSON for further processing.")

# EOF
