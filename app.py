import os, time, re, json, uuid, math, tempfile
from datetime import datetime
from typing import List, Dict, Any

import streamlit as st

# ---- LLM backends (choose one) ----
USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() == "true"
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini" if not USE_OLLAMA else "llama3")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ---- LangChain & RAG stack ----
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

# LLM wrappers
if USE_OLLAMA:
    from langchain_community.llms import Ollama
    llm = Ollama(model=MODEL_NAME, temperature=0.2)
else:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0.2, api_key=OPENAI_API_KEY)

# ---- Simple moderation / guardrails ----
PROFANITY = re.compile(r"\b(fuck|shit|bitch|asshole)\b", re.I)
PII = re.compile(r"\b(\d{12}|\d{16}|\d{3}-\d{2}-\d{4})\b")  # ad-hoc IDs/SSN-ish
JAILBREAK = re.compile(r"ignore previous|bypass|disable safety|jailbreak", re.I)

def validate_user_prompt(text: str) -> List[str]:
    errs = []
    if len(text.strip()) < 3:
        errs.append("Prompt too short.")
    if PROFANITY.search(text):
        errs.append("Contains profanity.")
    if PII.search(text):
        errs.append("Contains possible PII.")
    if JAILBREAK.search(text):
        errs.append("Jailbreak intent detected.")
    return errs

def moderate_output(text: str) -> str:
    # simple redaction demo
    text = PROFANITY.sub("[REDACTED]", text)
    text = re.sub(r"\b(\d{12}|\d{16})\b", "[REDACTED_ID]", text)
    return text

# ---- Observability ----
LOG_DIR = os.getenv("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

def log_event(kind: str, payload: Dict[str, Any]):
    payload = dict(payload)
    payload["kind"] = kind
    payload["ts"] = datetime.utcnow().isoformat()
    with open(os.path.join(LOG_DIR, f"{datetime.utcnow().date()}.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

# ---- Build RAG store ----
@st.cache_resource(show_spinner=False)
def get_embedder():
    # Fast, CPU-friendly
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def chunk_docs(docs: List[Document]):
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    return splitter.split_documents(docs)

def load_files(files) -> List[Document]:
    docs = []
    for up in files:
        suffix = os.path.splitext(up.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(up.read()); tmp.flush()
            path = tmp.name
        if suffix in [".pdf"]:
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        else:
            loader = TextLoader(path, encoding="utf-8")
            docs.extend(loader.load())
    return docs

@st.cache_resource(show_spinner=False)
def build_vector_store(_docs: List[Document], use_faiss: bool = False):
    chunks = chunk_docs(_docs)
    embedder = get_embedder()
    if use_faiss:
        return FAISS.from_documents(chunks, embedder)
    return Chroma.from_documents(chunks, embedder)

# ---- Tools for the Agent ----
def calc_tool(q: str) -> str:
    try:
        return str(eval(q, {"__builtins__": {}}, {"math": math}))
    except Exception as e:
        return f"Calculator error: {e}"

def url_reader(url: str) -> str:
    # Stub for free deployments without outbound internet.
    # In prod: replace with requests+Readability or a crawler.
    return f"[URL fetch disabled in demo] You asked me to read: {url}"

# ---- Streamlit UI ----
st.set_page_config(page_title="Enterprise RAG + Agentic AI (Guardrailed)", layout="wide")
st.title("🏢 Enterprise RAG + Agentic AI • Guardrails • Observability (MVP)")

with st.sidebar:
    st.markdown("### Ingestion")
    uploads = st.file_uploader("Upload PDFs or .txt", type=["pdf", "txt", "md"], accept_multiple_files=True)
    use_faiss = st.toggle("Use FAISS (CPU-fast alt.)", value=False)
    build_btn = st.button("Build Knowledge Base")
    st.markdown("---")
    st.markdown("### Settings")
    top_k = st.slider("Top-K passages", 1, 10, 4)
    route_mode = st.selectbox("Routing", ["Auto (Agent)", "Direct RAG"])
    st.caption(f"Model: {'Ollama/'+MODEL_NAME if USE_OLLAMA else MODEL_NAME}")

if "vs" not in st.session_state:
    st.session_state.vs = None
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "feedback" not in st.session_state:
    st.session_state.feedback = []

col1, col2 = st.columns([1,2])

with col1:
    st.subheader("Chat")
    user_prompt = st.text_area("Ask a question (enterprise context)", height=120, placeholder="E.g., Summarize policy ABC and list customer-visible risks.")
    run_btn = st.button("Run")
    st.markdown("### Feedback")
    fb_col1, fb_col2 = st.columns(2)
    with fb_col1:
        if st.button("👍 Good"):
            st.session_state.feedback.append({"id": str(uuid.uuid4()), "score": 1})
    with fb_col2:
        if st.button("👎 Needs work"):
            st.session_state.feedback.append({"id": str(uuid.uuid4()), "score": 0})

with col2:
    st.subheader("Answer")

    # Build KB
    if build_btn and uploads:
        with st.spinner("Building vector store..."):
            docs = load_files(uploads)
            st.session_state.vs = build_vector_store(docs, use_faiss=use_faiss)
        st.success("Knowledge base ready.")

    # Define retriever tool (binds current vector store at call time)
    def retrieve(q: str) -> str:
        if st.session_state.vs is None:
            return "No KB yet. Upload docs and click Build."
        docs = st.session_state.vs.similarity_search(q, k=top_k)
        joined = "\n\n".join([f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)])
        return f"Top {len(docs)} context passages:\n{joined}"

    tools = [
        Tool(name="retriever", func=retrieve, description="Retrieve top passages from enterprise KB for a question."),
        Tool(name="calculator", func=calc_tool, description="Evaluate math expressions using Python math."),
        Tool(name="url_reader", func=url_reader, description="Fetch and summarize a URL (disabled in demo)."),
    ]

    # Router: Agent vs Direct RAG
    def run_direct_rag(q: str) -> str:
        ctx = retrieve(q)
        system = (
            "You are an enterprise assistant. Use ONLY retrieved context. "
            "If unsure or missing context, say you lack sufficient information."
        )
        prompt = f"{system}\n\nContext:\n{ctx}\n\nUser question:\n{q}\n\nAnswer:"
        start = time.time()
        ans = llm.invoke(prompt) if hasattr(llm, "invoke") else llm(prompt)
        latency = round((time.time() - start) * 1000)
        text = ans.content if hasattr(ans, "content") else str(ans)
        return text, ctx, latency

    def run_agentic(q: str) -> str:
        memory = st.session_state.memory
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=6,
        )
        start = time.time()
        resp = agent.run(q)
        latency = round((time.time() - start) * 1000)
        return resp, "(agent used tools; see logs)", latency

    # Execute
    if run_btn and user_prompt:
        errors = validate_user_prompt(user_prompt)
        if errors:
            st.error("Input blocked by guardrails: " + "; ".join(errors))
            log_event("blocked_input", {"prompt": user_prompt, "errors": errors})
        else:
            route = "agent" if route_mode.startswith("Auto") else "rag"
            try:
                if route == "agent":
                    out, ctx, latency = run_agentic(user_prompt)
                else:
                    out, ctx, latency = run_direct_rag(user_prompt)

                safe_out = moderate_output(out)
                st.markdown(safe_out)
                with st.expander("Context / Tool traces"):
                    st.code(ctx)

                # Metrics + logs
                st.caption(f"Latency: {latency} ms • Route: {route}")
                log_event("inference", {
                    "id": str(uuid.uuid4()),
                    "route": route,
                    "latency_ms": latency,
                    "prompt": user_prompt,
                    "output": safe_out,
                })
            except Exception as e:
                st.error(f"Runtime error: {e}")
                log_event("error", {"prompt": user_prompt, "error": str(e)})

    # Feedback export
    if st.session_state.feedback:
        st.caption(f"Feedback count: {len(st.session_state.feedback)}")
        if st.button("Export feedback to JSON"):
            fname = os.path.join(LOG_DIR, f"feedback-{datetime.utcnow().date()}.json")
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(st.session_state.feedback, f, ensure_ascii=False, indent=2)
            st.success(f"Saved: {fname}")
