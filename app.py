import os
import tempfile
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline

load_dotenv()

st.set_page_config(
    page_title="Legal Aid Q&A Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a3a5c 0%, #2d6a9f 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .main-header p  { color: #cce0f5; margin: 0.3rem 0 0 0; font-size: 0.95rem; }
    .source-badge {
        background: #e8f0fe;
        border: 1px solid #1a3a5c;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.78rem;
        color: #1a3a5c;
        margin-right: 4px;
    }
    .status-ready   { color: #1e7e34; font-weight: 600; }
    .status-loading { color: #856404; font-weight: 600; }
    .chat-user { background:#f0f4ff; border-radius:8px; padding:10px 14px; margin:4px 0; }
    .chat-bot  { background:#f9f9f9; border-radius:8px; padding:10px 14px; margin:4px 0; border-left: 3px solid #1a3a5c; }
    .stButton > button {
        background-color: #1a3a5c;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.4rem 1rem;
    }
    .stButton > button:hover { background-color: #2d6a9f; }
</style>
""", unsafe_allow_html=True)


# ── Session state ────────────────────────────────────────
if "pipeline"      not in st.session_state: st.session_state.pipeline      = None
if "chat_history"  not in st.session_state: st.session_state.chat_history  = []
if "docs_loaded"   not in st.session_state: st.session_state.docs_loaded   = False
if "doc_names"     not in st.session_state: st.session_state.doc_names     = []


# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Royal_Coat_of_Arms_of_the_United_Kingdom_%28HM_Government%29.svg/200px-Royal_Coat_of_Arms_of_the_United_Kingdom_%28HM_Government%29.svg.png", width=60)
    st.title("Legal Aid Assistant")
    st.caption("Powered by RAG + GPT-4o-mini")
    st.divider()

    # API Key
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        st.caption("Your key is never stored.")
    else:
        st.markdown('<p class="status-ready">✓ API key loaded from environment</p>', unsafe_allow_html=True)

    st.divider()

    # Load sample docs
    st.subheader("Documents")
    if st.button("Load Sample Legal Docs", use_container_width=True):
        if not api_key:
            st.error("Please enter your OpenAI API key first.")
        else:
            with st.spinner("Indexing sample documents..."):
                try:
                    pipeline = RAGPipeline(openai_api_key=api_key)
                    docs = pipeline.load_sample_docs("./sample_docs")
                    if not docs:
                        st.error("No sample documents found in ./sample_docs")
                    else:
                        pipeline.index_documents(docs)
                        st.session_state.pipeline    = pipeline
                        st.session_state.docs_loaded = True
                        st.session_state.doc_names   = list({
                            Path(d.metadata.get("source", "Unknown")).name for d in docs
                        })
                        st.success(f"Loaded {len(docs)} document(s).")
                except Exception as e:
                    st.error(f"Error: {e}")

    # Upload docs
    uploaded = st.file_uploader(
        "Or upload your own (PDF / TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )
    if uploaded and st.button("Index Uploaded Docs", use_container_width=True):
        if not api_key:
            st.error("Please enter your OpenAI API key first.")
        else:
            with st.spinner("Indexing uploaded documents..."):
                try:
                    if st.session_state.pipeline is None:
                        st.session_state.pipeline = RAGPipeline(openai_api_key=api_key)
                    pipeline = st.session_state.pipeline
                    for f in uploaded:
                        suffix = ".pdf" if f.type == "application/pdf" else ".txt"
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(f.read())
                            tmp_path = tmp.name
                        docs = pipeline.load_uploaded_file(tmp_path)
                        pipeline.index_documents(docs)
                        os.unlink(tmp_path)
                        if f.name not in st.session_state.doc_names:
                            st.session_state.doc_names.append(f.name)
                    st.session_state.docs_loaded = True
                    st.success(f"Indexed {len(uploaded)} file(s).")
                except Exception as e:
                    st.error(f"Error: {e}")

    # Indexed docs list
    if st.session_state.doc_names:
        st.divider()
        st.subheader("Indexed Documents")
        for name in st.session_state.doc_names:
            st.markdown(f"📄 {name}")

    # Clear chat
    st.divider()
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.caption("Built by Aysha Nasim · Azure AI & LLMOps Engineer")


# ── Main panel ───────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>⚖️ Legal Aid Q&A Assistant</h1>
  <p>Ask questions about UK legal aid eligibility, rights, and procedures — powered by RAG over real legal documents.</p>
</div>
""", unsafe_allow_html=True)

# Status banner
col1, col2, col3 = st.columns(3)
with col1:
    status = "🟢 Ready" if st.session_state.docs_loaded else "🟡 Load documents to begin"
    st.metric("System Status", status)
with col2:
    st.metric("Documents Indexed", len(st.session_state.doc_names))
with col3:
    st.metric("Questions Asked", len(st.session_state.chat_history))

st.divider()

# Example questions
if not st.session_state.chat_history:
    st.subheader("Try asking...")
    examples = [
        "Am I eligible for legal aid?",
        "What is the means test for legal aid?",
        "What types of cases does legal aid cover?",
        "How do I apply for legal aid in the UK?",
        "What is the merits test for legal aid?",
    ]
    cols = st.columns(len(examples))
    for i, (col, q) in enumerate(zip(cols, examples)):
        with col:
            if st.button(q, key=f"ex_{i}", use_container_width=True):
                st.session_state._example_q = q
                st.rerun()

# Display conversation
for turn in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant", avatar="⚖️"):
        st.markdown(turn["answer"])
        if turn.get("sources"):
            st.markdown(
                "**Sources:** " + " ".join(
                    f'<span class="source-badge">{s}</span>' for s in turn["sources"]
                ),
                unsafe_allow_html=True
            )

# Handle example question click
if hasattr(st.session_state, "_example_q"):
    question = st.session_state._example_q
    del st.session_state._example_q

    if not st.session_state.docs_loaded:
        st.warning("Please load documents first using the sidebar.")
    else:
        with st.spinner("Thinking..."):
            answer, sources = st.session_state.pipeline.query(question)
        st.session_state.chat_history.append({
            "question": question, "answer": answer, "sources": sources
        })
        st.rerun()

# Chat input
question = st.chat_input(
    "Ask a question about UK legal aid...",
    disabled=not st.session_state.docs_loaded
)
if question:
    with st.spinner("Searching documents and generating answer..."):
        answer, sources = st.session_state.pipeline.query(question)
    st.session_state.chat_history.append({
        "question": question, "answer": answer, "sources": sources
    })
    st.rerun()

if not st.session_state.docs_loaded:
    st.info("👈 Load the sample legal documents from the sidebar to get started.")
