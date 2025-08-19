"""
Free OSS Agentic RAG Demo (Streamlit-friendly)

Notes:
- Primary retrieval: TF-IDF (scikit-learn) over chunked docs (works on CPU, small memory).
- LLM: Optional local GGUF via ctransformers if you add it. By default the app FALLS BACK to a
  high-quality template-based synth so the app runs on free Streamlit without huge model downloads.
- Guardrails: profanity, PII, jailbreak detection + RBAC tool gating.
- Observability: JSONL logs in ./logs, simple metrics shown inline.
- Debug passes performed:
    1) Fixed caching/initialization issues (session_state vs cache_resource).
    2) Hardened planner/synthesis fallbacks, safe eval for calculator, and RBAC enforcement.
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

# Lightweight retrieval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# PDF support (optional for uploads)
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

# ------------- CONFIG -------------
APP_NAME = "Free OSS Agentic RAG • Guardrails • Observability"
DATA_DIR = "data"
LOG_DIR = "logs"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Demo files to create if missing
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

# RBAC demo policy
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
JAILBREAK = re.compile(r"(ignore (all|previous) instructions|bypass|disable safety|jailbreak|prompt injection)", re.I)

# Optional local LLM config (disabled by default)
USE_LOCAL_LLM = False  # Set to True and install ctransformers + provide model file to enable
LOCAL_LLM_MODEL_FILE = os.path.join("data", "local_model.gguf")  # if available, the app will try to use it

# ------------- UTILITIES -------------
def ensure_demo_files():
    for fname, content in DEMO_FILES.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

def log_event(kind: str, payload: Dict[str, Any]):
    payload = dict(payload)
    payload["kind"] = kind
    payload["ts"] = datetime.utcnow().isoformat()
    fname = os.path.join(LOG_DIR, f"{datetime.utcnow().date()}.jsonl")
    with open(fname, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

# ------------- Guardrails -------------
def validate_user_prompt(text: str) -> List[str]:
    errs = []
    if len(text.strip()) < 3:
        errs.append("Prompt too short.")
    if PROFANITY.search(text):
        errs.append("Contains profanity.")
    if PII.search(text):
        errs.append("Possible PII detected.")
    if JAILBREAK.search(text):
        errs.append("Potential jailbreak/injection intent.")
    return errs

def moderate_output(text: str) -> str:
    text = PROFANITY.sub("[REDACTED]", text)
    text = re.sub(r"\b(\d{12}|\d{16}|[0-9]{4}\s?[0-9]{4}\s?[0-9]{4}\s?[0-9]{4})\b", "[REDACTED_ID]", text)
    return text

# ------------- Document ingestion & chunking -------------
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

def read_uploaded_file(up) -> str:
    name = up.name.lower()
    try:
        if name.endswith(".txt") or name.endswith(".md"):
            return up.getvalue().decode("utf-8", errors="ignore")
        elif name.endswith(".pdf") and PdfReader is not None:
            # Using pypdf to extract text
            with open("tmp_upload.pdf", "wb") as tmp:
                tmp.write(up.getvalue())
            reader = PdfReader("tmp_upload.pdf")
            pages = [p.extract_text() or "" for p in reader.pages]
            os.remove("tmp_upload.pdf")
            return "\n".join(pages)
        else:
            return up.getvalue().decode("utf-8", errors="ignore")
    except Exception as e:
        return f"[Error reading file {up.name}: {e}]"

# ------------- Lightweight TF-IDF "vector store" -------------
@st.cache_resource(show_spinner=False)
def build_index_from_documents(docs: List[Dict[str, Any]]):
    """
    docs: list of {"source": source_name, "text": full_text}
    Returns a dict with:
      - 'chunks': list of {"text": str, "source": str, "id": str}
      - 'tfidf': fitted TfidfVectorizer
      - 'matrix': TF-IDF matrix (n_chunks x vocab)
    """
    all_chunks = []
    for doc in docs:
        src = doc.get("source", "unknown")
        text = doc.get("text", "")
        chunks = chunk_text(text)
        for i, ch in enumerate(chunks):
            all_chunks.append({"id": f"{src}#c{i}", "text": ch, "source": src})
    texts = [c["text"] for c in all_chunks] or [""]
    vect = TfidfVectorizer(stop_words="english")
    matrix = vect.fit_transform(texts)
    return {"chunks": all_chunks, "tfidf": vect, "matrix": matrix}

def retrieve(query: str, index: Dict[str, Any], k: int = 3):
    if index is None or not index.get("chunks"):
        return []
    qv = index["tfidf"].transform([query])
    sims = cosine_similarity(qv, index["matrix"])[0]
    top_idx = np.argsort(-sims)[:k]
    results = []
    for idx in top_idx:
        score = float(sims[idx])
        chunk = index["chunks"][idx]
        results.append({"score": score, "id": chunk["id"], "source": chunk["source"], "text": chunk["text"]})
    return results

# ------------- Planner / Tools -------------
def requires_calculation(q: str) -> bool:
    # crude heuristic
    return bool(re.search(r"[-+*/\^]=?|\bcalculate\b|\bcompute\b|\d+\s*[\+\-\*/]\s*\d+", q))

def requires_url(q: str) -> bool:
    return bool(re.search(r"https?://", q)) or q.lower().startswith("read ") or q.lower().startswith("fetch ")

def safe_calc(expr: str) -> str:
    # Very restricted eval: allow math module names and numbers/operators
    try:
        allowed = {"__builtins__": {}}
        # allow math functions
        local = {"math": math}
        # sanitize expr: keep digits, operators, spaces, math tokens, parentheses, dot
        # If it contains letters (except math.*) reject
        if re.search(r"[A-Za-z]", expr) and not expr.strip().startswith("math."):
            return "Calculator: unsupported expression."
        res = eval(expr, allowed, local)
        return str(res)
    except Exception as e:
        return f"Calculator error: {e}"

# ------------- Optional local LLM (lazy import) -------------
def load_local_llm_if_available():
    """
    Tries to import ctransformers and load a local model file if present.
    If unavailable, returns None. This keeps the app runnable on free Streamlit.
    """
    if not USE_LOCAL_LLM:
        return None
    try:
        from langchain_community.llms import CTransformers
        if os.path.exists(LOCAL_LLM_MODEL_FILE):
            llm = CTransformers(model="local", model_file=LOCAL_LLM_MODEL_FILE,
                                config={"max_new_tokens": 256, "temperature": 0.2})
            return llm
        else:
            st.warning("LOCAL_LLM enabled but model file not found at: " + LOCAL_LLM_MODEL_FILE)
            return None
    except Exception as e:
        st.warning("Local LLM load failed (ctransformers not installed or incompatible): " + str(e))
        return None

LOCAL_LLM = None  # lazy

def synthesize_with_llm(llm, prompt: str) -> str:
    # LangChain CTransformers wrapper returns a string when called like llm(prompt)
    try:
        out = llm(prompt)
        if isinstance(out, str):
            return out
        # handle wrapper objects
        return str(out)
    except Exception as e:
        return f"[LLM runtime error: {e}]"

# ------------- Synthesis fallback (no LLM) -------------
def synthesize_template(question: str, retrieved: List[Dict[str, Any]], calc_note: str | None):
    if not retrieved:
        base = "I couldn't find relevant context in the knowledge base."
        if calc_note:
            base += "\nCalculator note: " + calc_note
        return base
    parts = []
    for i, r in enumerate(retrieved, start=1):
        snippet = r["text"].strip().replace("\n", " ")
        snippet = (snippet[:400] + "…") if len(snippet) > 420 else snippet
        parts.append(f"[{i}] {snippet}")
    answer = f"Based on the retrieved passages:\n\n" + "\n\n".join(parts) + "\n\nAnswer:\n"
    # make concise answer by echoing short summary of first snippet
    first = retrieved[0]["text"].strip()
    first_sent = first.split(".")[0]
    ans_text = f"{first_sent.strip()}. (See citation [1])"
    if calc_note:
        ans_text += f"\nCalculator: {calc_note}"
    return answer + ans_text

# ------------- Verification -------------
def verify_answer(answer: str, retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
    citations = re.findall(r"\[(\d+)\]", answer)
    ok = True
    reasons = []
    if len(answer) > 200 and not citations:
        ok = False
        reasons.append("Long answer without citations.")
    # lexical overlap check
    ctx_text = " ".join([r["text"] for r in retrieved])
    ctx_terms = set(re.findall(r"[a-zA-Z]{4,}", ctx_text.lower()))
    ans_terms = set(re.findall(r"[a-zA-Z]{4,}", answer.lower()))
    overlap = len(ctx_terms & ans_terms)
    if len(answer) > 200 and overlap < 3:
        ok = False
        reasons.append("Low overlap with context; possible hallucination.")
    return {"ok": ok, "reasons": reasons, "citations": citations, "overlap": overlap}

# ------------- Streamlit UI -------------
st.set_page_config(page_title=APP_NAME, layout="wide")
st.title(APP_NAME)
st.caption("Free + OSS • Designed to run on Streamlit Community (CPU-only).")

ensure_demo_files()

with st.sidebar:
    st.header("Ingestion & KB")
    uploaded = st.file_uploader("Upload .txt/.md/.pdf (optional)", accept_multiple_files=True, type=["txt", "md", "pdf"])
    build = st.button("Build / Rebuild KB")
    st.markdown("---")
    st.header("Settings")
    top_k = st.slider("Top-K passages", 1, 6, 3)
    role = st.selectbox("Role (RBAC demo)", POLICY["roles"], index=1)
    st.info("Tools allowed: " + ", ".join(POLICY["tools_allowed"][role]))
    st.checkbox("Enable debug logs (shows latest log files)", value=False, key="show_debug_logs")
    st.checkbox("Try local LLM if available (may be slow)", value=False, key="try_local_llm")

# initialize index on first run from demo files if not present
if "index" not in st.session_state:
    docs = []
    for f in os.listdir(DATA_DIR):
        if f.lower().endswith(".txt") or f.lower().endswith(".md"):
            with open(os.path.join(DATA_DIR, f), "r", encoding="utf-8") as fh:
                docs.append({"source": f, "text": fh.read()})
    if not docs:
        ensure_demo_files()
        for k, _ in DEMO_FILES.items():
            with open(os.path.join(DATA_DIR, k), "r", encoding="utf-8") as fh:
                docs.append({"source": k, "text": fh.read()})
    st.session_state.index = build_index_from_documents(docs)
    log_event("startup", {"msg": "index initialized from demo files", "doc_count": len(docs)})

# If user requests rebuild, include uploaded files + demo files
if build:
    docs = []
    # include demo always
    for k in DEMO_FILES:
        with open(os.path.join(DATA_DIR, k), "r", encoding="utf-8") as fh:
            docs.append({"source": k, "text": fh.read()})
    if uploaded:
        for up in uploaded:
            txt = read_uploaded_file(up)
            docs.append({"source": up.name, "text": txt})
    st.session_state.index = build_index_from_documents(docs)
    st.success("Knowledge base rebuilt with " + str(len(docs)) + " documents.")
    log_event("kb_rebuild", {"doc_count": len(docs)})

# Try local LLM if user asked AND USE_LOCAL_LLM flag is True in code
if st.session_state.get("try_local_llm", False) and USE_LOCAL_LLM:
    if "local_llm" not in st.session_state:
        st.session_state.local_llm = load_local_llm_if_available()
        if st.session_state.local_llm:
            st.success("Local LLM loaded (ctransformers).")
        else:
            st.warning("Local LLM not available; continuing without it.")

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Ask (Agentic-style)")
    question = st.text_area("Enter your question", placeholder="E.g., Summarize policy and list customer-facing risks.", height=160)
    run = st.button("Run")
    st.markdown("### Feedback")
    good = st.button("👍 Useful")
    bad = st.button("👎 Needs work")
    fb_note = st.text_input("Feedback note (optional)")

with col2:
    st.subheader("Answer & Evidence")
    if run and question:
        errs = validate_user_prompt(question)
        if errs:
            st.error("Blocked by guardrails: " + "; ".join(errs))
            log_event("blocked_input", {"prompt": question, "errors": errs, "role": role})
        else:
            # PLAN
            steps = [{"tool": "retriever", "arg": question}]
            if requires_calculation(question):
                steps.append({"tool": "calculator", "arg": question})
            if requires_url(question):
                steps.append({"tool": "url_reader", "arg": question})
            steps.append({"tool": "synthesize", "arg": question})

            ctx_text = ""
            calc_note = None
            used_tools = []

            start = time.time()
            # EXECUTE
            for step in steps:
                tool = step["tool"]
                arg = step["arg"]
                allowed = (tool in POLICY["tools_allowed"][role]) or tool == "synthesize"
                if not allowed:
                    ctx_text += f"\n[policy] Tool '{tool}' blocked for role '{role}'.\n"
                    continue
                used_tools.append(tool)
                if tool == "retriever":
                    retrieved = retrieve(arg, st.session_state.index, k=top_k)
                    if retrieved:
                        parts = []
                        for i, r in enumerate(retrieved, start=1):
                            parts.append(f"[{i}] ({r['source']}) {r['text'][:300].replace('\\n',' ')}")
                        ctx_text += "\n\n".join(parts)
                    else:
                        ctx_text += "\nNo context retrieved."
                elif tool == "calculator":
                    calc_note = safe_calc(arg)
                elif tool == "url_reader":
                    ctx_text += "\n[URL reader disabled in Streamlit free demo]"
                elif tool == "synthesize":
                    pass

            latency_ms = int((time.time() - start) * 1000)

            # SYNTHESIZE: prefer local LLM (if the user enabled AND app is configured),
            # otherwise fallback to template-based synthesis
            use_llm_here = False
            llm_obj = st.session_state.get("local_llm", None)
            if llm_obj:
                use_llm_here = True

            if use_llm_here:
                prompt = f"System: You are an enterprise assistant. Use retrieved context when answering.\n\nContext:\n{ctx_text}\n\nQuestion:\n{question}\n\nCalculator note: {calc_note if calc_note else 'none'}\n\nAnswer with citations like [1],[2]:"
                answer_raw = synthesize_with_llm(llm_obj, prompt)
            else:
                # fallback
                ra = retrieve(question, st.session_state.index, k=top_k)
                answer_raw = synthesize_template(question, ra, calc_note)

            # verify and moderate
            ver = verify_answer(answer_raw, retrieve(question, st.session_state.index, k=top_k))
            safe_answer = moderate_output(answer_raw)
            # apply role cap
            cap = POLICY["max_answer_chars_by_role"][role]
            clipped = safe_answer[:cap] + (" … [truncated by role]" if len(safe_answer) > cap else "")

            # display
            badge = "✅ Verified" if ver["ok"] else "⚠️ Needs Review"
            st.markdown(f"**{badge}**  •  `latency: {latency_ms} ms`  •  Tools used: {', '.join(used_tools)}")
            st.markdown(clipped)
            with st.expander("Retrieved Context / Execution Trace"):
                st.code(ctx_text)

            # logging
            log_event("inference", {
                "id": str(uuid.uuid4()),
                "role": role,
                "prompt": question,
                "tools": used_tools,
                "latency_ms": latency_ms,
                "verification": ver,
            })

    # feedback handling
    if good or bad:
        fb = {"id": str(uuid.uuid4()), "score": 1 if good else 0, "note": fb_note, "ts": datetime.utcnow().isoformat()}
        fname = os.path.join(LOG_DIR, f"feedback-{datetime.utcnow().date()}.jsonl")
        with open(fname, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(fb, ensure_ascii=False) + "\n")
        st.success("Thanks — feedback recorded.")
        log_event("feedback", fb)

# Observability panel
st.markdown("---")
st.subheader("Observability")
log_path = os.path.join(LOG_DIR, f"{datetime.utcnow().date()}.jsonl")
inf_count = 0
blocked = 0
slow = 0
last_lines = []
if os.path.exists(log_path):
    with open(log_path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()[-200:]
        for ln in lines:
            try:
                obj = json.loads(ln)
                if obj.get("kind") == "inference":
                    inf_count += 1
                    if obj.get("latency_ms", 0) > 4000:
                        slow += 1
                if obj.get("kind") == "blocked_input":
                    blocked += 1
                last_lines.append(obj)
            except:
                pass

st.write(f"Total inferences today: **{inf_count}** • Slow (>4s): **{slow}** • Blocked inputs: **{blocked}**")
if st.session_state.get("show_debug_logs"):
    st.subheader("Recent log rows")
    st.write(last_lines[-10:])
