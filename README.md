# ⚖️ Legal Aid Q&A Assistant

A Retrieval-Augmented Generation (RAG) chatbot for UK legal aid and rights information. Built with LangChain, ChromaDB, OpenAI GPT-4o-mini, and Streamlit.

**Live Demo:** [Add your Streamlit Cloud URL here after deploying]

---

## What It Does

- Answers questions about UK legal aid eligibility, employment rights, and housing law
- Uses RAG architecture to retrieve relevant passages from legal documents before generating answers
- Cites source documents for every answer
- Supports uploading custom PDF or TXT legal documents
- Maintains conversation history for follow-up questions

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-ada-002 |
| Vector Store | ChromaDB (local) |
| RAG Framework | LangChain |
| Frontend | Streamlit |
| Document Loaders | LangChain PyPDF + TextLoader |

---

## Architecture

```
User Question
      │
      ▼
┌─────────────────┐
│  Query Embedder │  (OpenAI Embeddings)
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│  ChromaDB       │◄────│  Legal Documents  │
│  Vector Store   │     │  (chunked + indexed)│
└────────┬────────┘     └──────────────────┘
         │ Top-k similar chunks
         ▼
┌─────────────────┐
│  GPT-4o-mini    │  (answer + cite sources)
└────────┬────────┘
         │
         ▼
    Answer + Sources
```

---

## Running Locally

### 1. Clone the repo
```bash
git clone https://github.com/Aysha1984/legal-rag-chatbot.git
cd legal-rag-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key
```bash
cp .env.example .env
# Edit .env and add your key: OPENAI_API_KEY=sk-...
```

### 4. Run the app
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Deploying to Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `OPENAI_API_KEY` in **Settings → Secrets**:
   ```toml
   OPENAI_API_KEY = "sk-your-key-here"
   ```
5. Deploy — you'll get a public URL to share

---

## Sample Documents Included

- **Legal Aid Eligibility** — means test, merits test, scope, how to apply
- **Employment Rights UK** — minimum wage, unfair dismissal, redundancy, discrimination
- **Housing Rights UK** — tenancy deposits, repairs, eviction, tenant protections

You can also upload your own PDF or TXT legal documents via the sidebar.

---

## About

Built by **Aysha Nasim** — AI Engineer with experience building LLM-based legal AI systems at the UK Ministry of Justice, including Legal Aid Agent and OPG Financial Analysis Agent using Azure AI Studio and Azure Foundry.

- LinkedIn: [linkedin.com/in/aysha-nasim-072abb131](https://www.linkedin.com/in/aysha-nasim-072abb131/)
- GitHub: [github.com/Aysha1984](https://github.com/Aysha1984)
