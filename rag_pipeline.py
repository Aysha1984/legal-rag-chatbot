import os
from pathlib import Path
from typing import List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate


SYSTEM_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful UK legal aid assistant. Use the provided legal documents
to answer questions clearly and accurately. Always cite which document or section your
answer comes from. If the answer is not in the documents, say so clearly — do not guess.

Context from legal documents:
{context}

Question: {question}

Answer (cite your sources):"""
)


class RAGPipeline:
    def __init__(self, openai_api_key: str, persist_dir: str = "./chroma_db"):
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=openai_api_key
        )
        self.persist_dir = persist_dir
        self.vectorstore = None
        self.chain = None
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " "]
        )

    def load_sample_docs(self, docs_dir: str = "./sample_docs") -> List[Document]:
        docs = []
        path = Path(docs_dir)
        for f in path.glob("*.txt"):
            loader = TextLoader(str(f), encoding="utf-8")
            docs.extend(loader.load())
        for f in path.glob("*.pdf"):
            loader = PyPDFLoader(str(f))
            docs.extend(loader.load())
        return docs

    def load_uploaded_file(self, file_path: str) -> List[Document]:
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()

    def index_documents(self, docs: List[Document]):
        chunks = self.splitter.split_documents(docs)
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                chunks,
                self.embeddings,
                persist_directory=self.persist_dir
            )
        else:
            self.vectorstore.add_documents(chunks)
        self._build_chain()

    def _build_chain(self):
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            ),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": SYSTEM_PROMPT},
            return_source_documents=True,
            verbose=False
        )

    def query(self, question: str) -> Tuple[str, List[str]]:
        if not self.chain:
            return "Please upload documents first.", []
        result = self.chain({"question": question})
        answer = result["answer"]
        sources = list({
            Path(doc.metadata.get("source", "Unknown")).name
            for doc in result.get("source_documents", [])
        })
        return answer, sources

    def is_ready(self) -> bool:
        return self.chain is not None
