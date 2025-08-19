# app.py
"""
Ascendion-ready • Agentic RAG MVP (Free OSS)
- TF-IDF based RAG (scikit-learn) with chunking
- Agentic planner -> tools: retriever, calculator, url_reader (stub)
- Guardrails: profanity, PII, jailbreak detection; RBAC tool gating
- Feedback loop saved to JSONL; Observability via JSONL logs and inline counters
- Designed to run on Streamlit Community Cloud (CPU-only)
"""

import os
import re
import time
import json
import uuid
import math
from datetime import datetime
from typing import List, Dict, Any

import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional PDF reading (streamlit environment may not always have pypdf; optional)
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

# --------- CONFIG/FILES ----------
APP_TITLE = "Ascendion • Agentic RAG MVP"
DATA_DIR = "data"
LOG_DIR = "logs"
EVENT_LOG = os.path.join(LOG_DIR, f"{datetime.utcnow().date()}.jsonl")
FEEDBACK_LOG = os.path.join(LOG_DIR, "feedback.jsonl")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

DEMO_FILES = {
    "demo_policy.txt": """Company Policy Alpha (v1.0)
- Customer data must not be shared outside tenant boundary.
- PII (IDs, card numbers) must be redacted before export.
- Critical incidents require response within 1 business day.
""",
    "demo_faq.txt": """FAQ
Q: Where is the Eiffel Tower?
A: The Eiffel Tower is located in Paris, France, built in 1889.

Q: What powers life on Earth?
A: The Sun provides energy necessary for life on Earth.
"""
}

# RBAC
POLICY = {
    "roles": ["viewer", "analyst", "admin"],
    "tools_allowed": {
        "viewer": ["retriever"],
        "analyst": ["retriever", "calculator"],
        "admin": ["retriever", "calculator", "url_reader"]
    },
    "max_answer_chars_by_role": {"viewer": 700, "analyst": 1400, "admin": 3000}
}

# Guardrail regexes
PROFANITY = re.compile(r"\b(fuck|shit|bitch|asshole|bastard)\b", re.I)
PII = re.compile(r"\b(\d{12}|\d{16}|\d{3}-\d{2}-\d{4}|[0-9]{4}\s?[0-9]{4}\s?[0-9]{4}\s?[0-9]{4})\b")
JAIL = re.compile(r"(ignore (all|previous) instructions|bypass|disable safety|jailbreak|prompt injection)", re.I)

# --------- HELPERS ----------
def ensure_demo_files():
    for fn, content in DEMO_FILES.items():
        path = os.path.join(DATA_DIR, fn)
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

def log_event(kind: str, payload: Dict[str, Any]):
    p = dict(payload)
    p["kind"] = kind
    p["ts"] = datetime.utcnow().isoformat()
    with open(EVENT_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(p, ensure_ascii=False) + "\n")

def append_feedback(rec: Dict[str, Any]):
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

# --------- Guardrails ----------
def validate_prompt(text: str) -> List[str]:
    errs = []
    if len(text.strip()) < 3:
        errs.append("Prompt too short.")
    if PROFANITY.search(text):
        errs.append("Contains profanity.")
    if PII.search(text):
        errs.append("Possible PII detected.")
    if JAIL.search(text):
        errs.append("Potential jailbreak attempt.")
    return errs

def moderate_output(text: str) -> str:
    text = PROFANITY.sub("[REDACTED]", text)
    text = PII.sub("[REDACTED_ID]", text)
    return text

# --------- Chunking & TF-IDF index ----------
def chunk_text(text: str, chunk_size: int = 450, overlap: int = 50) -> List[str]:
    text = text.replace("\r\n", "\n")
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(L, start + chunk_size)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if start >= L:
            break
    return chunks

@st.cache_resource(show_spinner=False)
def build_index_from_docs(docs: List[Dict[str, str]]):
    chunks = []
    for doc in docs:
        src = doc.get("source", "unknown")
        for i, ch in enumerate(chunk_text(doc.get("text", ""))):
            chunks.append({"id": f"{src}#c{i}", "text": ch, "source": src})
    texts = [c["text"] for c in chunks] or [""]
    vect = TfidfVectorizer(stop_words="english")
    mat = vect.fit_transform(texts)
    return {"chunks": chunks, "vectorizer": vect, "matrix": mat}

def retrieve_tfidf(query: str, index: Dict[str, Any], k: int = 3):
    if not index or not index.get("chunks"):
        return []
    qv = index["vectorizer"].transform([query])
    sims = cosine_similarity(qv, index["matrix"])[0]
    idx = np.argsort(-sims)[:k]
    results = []
    for i in idx:
        results.append({"score": float(sims[i]), **index["chunks"][i]})
    return results

# --------- Tools ----------
def tool_retriever(q: str, index: Dict[str, Any], top_k: int):
    if not index:
        return "Knowledge base is empty. Upload docs and rebuild."
    docs = retrieve_tfidf(q, index, k=top_k)
    if not docs:
        return "No relevant passages."
    out = "\n\n".join([f"[{i+1}] ({d['source']}) {d['text'][:400].replace('\\n',' ')}" for i, d in enumerate(docs)])
    return out

def tool_calculator(expr: str):
    try:
        allowed = {"__builtins__": {}}
        local = {"math": math}
        if re.search(r"[A-Za-z]", expr) and not expr.strip().startswith("math."):
            return "Calculator: unsupported expression (only math.* allowed)."
        res = eval(expr, allowed, local)
        return str(res)
    except Exception as e:
        return f"Calculator error: {e}"

def tool_url_reader(url: str):
    return "[URL reader disabled in free demo]"

# --------- Planner / Agent ----------
def plan_steps(query: str) -> List[Dict[str, str]]:
    steps = [{"tool": "retriever", "arg": query}]
    if re.search(r"calculate|compute|\d+\s*[\+\-\*/]\s*\d+", query.lower()):
        steps.append({"tool": "calculator", "arg": query})
    if re.search(r"https?://", query.lower()) or query.lower().startswith("read "):
        steps.append({"tool": "url_reader", "arg": query})
    steps.append({"tool": "synthesize", "arg": query})
    return steps

def synthesize_answer(query: str, retrieved_text: str, calc_note: str | None):
    if not retrieved_text or retrieved_text.strip() == "":
        base = "I could not find relevant context in the knowledge base."
        if calc_note:
            base += f"\nCalculator note: {calc_note}"
        return base
    first = retrieved_text.split("\n\n")[0]
    summary = first.split(".")[0].strip()
    ans = f"Based on retrieved passages: {summary}. (See [1])"
    if calc_note:
        ans += f"\nCalculator: {calc_note}"
    return ans

def verify_answer(answer: str, retrieved_text: str):
    ok = True
    reasons = []
    if len(answer) > 150 and "[1]" not in answer:
        ok = False
        reasons.append("No citation in a long answer.")
    ctx_terms = set(re.findall(r"[a-zA-Z]{4,}", retrieved_text.lower()))
    ans_terms = set(re.findall(r"[a-zA-Z]{4,}", answer.lower()))
    overlap = len(ctx_terms & ans_terms)
    if len(answer) > 150 and overlap < 3:
        ok = False
        reasons.append("Low overlap with context; possible hallucination.")
    return {"ok": ok, "reasons": reasons, "overlap": overlap}

# --------- UI ----------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Free & Open Source • Streamlit Community Cloud friendly • CPU-only demo")

ensure_demo_files()

with st.sidebar:
    st.header("Knowledge Base / Ingest")
    uploads = st.file_uploader("Upload .txt/.md/.pdf (optional)", type=["txt", "md", "pdf"], accept_multiple_files=True)
    rebuild = st.button("Build / Rebuild KB")
    st.markdown("---")
    st.header("Settings")
    top_k = st.slider("Top-K passages", 1, 6, 3)
    role = st.selectbox("Role (RBAC demo)", POLICY["roles"], index=1)
    st.info("Tools allowed for this role: " + ", ".join(POLICY["tools_allowed"][role]))
    show_logs = st.checkbox("Show recent logs", value=False)

# initialize index on first run
if "index" not in st.session_state:
    docs = []
    for fname in os.listdir(DATA_DIR):
        if fname.lower().endswith((".txt", ".md")):
            with open(os.path.join(DATA_DIR, fname), "r", encoding="utf-8") as fh:
                docs.append({"source": fname, "text": fh.read()})
    st.session_state.index = build_index_from_docs(docs)
    log_event("startup", {"msg": "index initialized", "docs": len(docs)})

# rebuild index if requested
if rebuild:
    docs = []
    for fname in os.listdir(DATA_DIR):
        if fname.lower().endswith((".txt", ".md")):
            with open(os.path.join(DATA_DIR, fname), "r", encoding="utf-8") as fh:
                docs.append({"source": fname, "text": fh.read()})
    if uploads:
        for up in uploads:
            try:
                name = up.name
                if name.lower().endswith((".txt", ".md")):
                    txt = up.getvalue().decode("utf-8", errors="ignore")
                elif name.lower().endswith(".pdf") and PdfReader is not None:
                    with open("tmp_upload.pdf", "wb") as tmpf:
                        tmpf.write(up.getvalue())
                    r = PdfReader("tmp_upload.pdf")
                    pages = [p.extract_text() or "" for p in r.pages]
                    txt = "\n".join(pages)
                    os.remove("tmp_upload.pdf")
                else:
                    txt = up.getvalue().decode("utf-8", errors="ignore")
            except Exception as e:
                txt = f"[Error reading {up.name}: {e}]"
            docs.append({"source": up.name, "text": txt})
    st.session_state.index = build_index_from_docs(docs)
    st.success(f"Rebuilt KB with {len(docs)} documents.")
    log_event("kb_rebuild", {"doc_count": len(docs)})

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Agentic Query")
    query = st.text_area("Enter enterprise question", height=160, placeholder="e.g., Summarize policy and list customer-facing risks.")
    run = st.button("Run")
    st.markdown("### Feedback")
    upv = st.button("👍 Useful")
    downv = st.button("👎 Needs work")
    fb_note = st.text_input("Optional feedback note")

with col2:
    st.subheader("Answer & Evidence")
    if run and query:
        errs = validate_prompt(query)
        if errs:
            st.error("Input blocked by guardrails: " + "; ".join(errs))
            log_event("blocked_input", {"prompt": query, "errors": errs, "role": role})
        else:
            # planning + execute
            steps = plan_steps(query)
            ctx_text = ""
            calc_note = None
            used_tools = []
            start = time.time()
            for s in steps:
                tool = s["tool"]; arg = s["arg"]
                allowed = (tool in POLICY["tools_allowed"][role]) or tool == "synthesize"
                if not allowed:
                    ctx_text += f"\n[policy] Tool '{tool}' blocked for role '{role}'.\n"
                    continue
                used_tools.append(tool)
                if tool == "retriever":
                    out = tool_retriever(arg, st.session_state.index, top_k)
                    ctx_text += out + "\n\n"
                elif tool == "calculator":
                    calc_note = tool_calculator(arg)
                elif tool == "url_reader":
                    ctx_text += tool_url_reader(arg) + "\n\n"
                elif tool == "synthesize":
                    pass
            latency = int((time.time() - start) * 1000)

            # synthesize & verify
            answer_raw = synthesize_answer(query, ctx_text, calc_note)
            ver = verify_answer(answer_raw, ctx_text)
            safe = moderate_output(answer_raw)
            cap = POLICY["max_answer_chars_by_role"][role]
            clipped = safe[:cap] + (" … [truncated by role]" if len(safe) > cap else "")

            badge = "✅ Verified" if ver["ok"] else "⚠️ Needs Review"
            st.markdown(f"**{badge}**  •  latency: **{latency} ms**  •  tools: {', '.join(used_tools)}")
            st.markdown(clipped)
            with st.expander("Context / Execution Trace"):
                st.code(ctx_text)

            log_event("inference", {"id": str(uuid.uuid4()), "role": role, "prompt": query, "tools": used_tools, "latency_ms": latency, "verification": ver})

    if upv or downv:
        rec = {"id": str(uuid.uuid4()), "score": 1 if upv else 0, "note": fb_note, "ts": datetime.utcnow().isoformat()}
        append_feedback(rec)
        st.success("Thank you — feedback recorded.")
        log_event("feedback", rec)

# Observability
st.markdown("---")
st.subheader("Observability")
inf = blocked = 0
if os.path.exists(EVENT_LOG):
    with open(EVENT_LOG, "r", encoding="utf-8") as fh:
        for ln in fh.readlines():
            try:
                o = json.loads(ln)
                if o.get("kind") == "inference":
                    inf += 1
                if o.get("kind") == "blocked_input":
                    blocked += 1
            except:
                pass
st.write(f"Total inferences today: **{inf}** • Blocked inputs: **{blocked}**")
if show_logs:
    if os.path.exists(EVENT_LOG):
        with open(EVENT_LOG, "r", encoding="utf-8") as fh:
            st.code("".join(fh.readlines()[-400:]))
    else:
        st.write("No logs yet.")
