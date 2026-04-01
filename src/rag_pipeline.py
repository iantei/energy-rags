"""
Energy Policy RAG Pipeline
--------------------------
Ingests NREL/DOE PDF reports, builds a ChromaDB vector store,
and answers researcher questions with source citations.
"""

import os
from pathlib import Path
from typing import Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ── Configuration ──────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data" / "pdfs"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
RETRIEVER_K = 5          # number of chunks to retrieve per query

SYSTEM_PROMPT = """You are a research assistant specializing in energy systems, \
EV infrastructure, and transportation electrification. You answer questions \
using only the provided context from NREL and DOE technical reports.

Guidelines:
- Be precise and cite the source document and page for every claim.
- If the context does not contain enough information, say so clearly.
- Use technical language appropriate for energy researchers.
- Prefer quantitative findings where available.

Context:
{context}
"""

# ── Document ingestion ─────────────────────────────────────────────────────────

def load_pdfs(pdf_dir: Path = DATA_DIR) -> list[Document]:
    """Load all PDFs from the data directory."""
    docs = []
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in {pdf_dir}. "
            "Add NREL/DOE report PDFs to data/pdfs/ before ingesting."
        )
    for pdf_path in pdf_files:
        print(f"  Loading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        docs.extend(loader.load())
    print(f"Loaded {len(docs)} pages from {len(pdf_files)} PDF(s).")
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")
    return chunks


def build_vectorstore(chunks: list[Document]) -> Chroma:
    """Embed chunks and persist to ChromaDB."""
    print("Building vector store (this may take a minute)...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name="energy_reports",
    )
    print(f"Vector store saved to {CHROMA_DIR}")
    return vectorstore


def ingest(pdf_dir: Path = DATA_DIR) -> Chroma:
    """Full ingestion pipeline: load → chunk → embed → persist."""
    print("\n── Ingestion ──────────────────────────────────────────")
    docs = load_pdfs(pdf_dir)
    chunks = chunk_documents(docs)
    return build_vectorstore(chunks)


# ── RAG chain ──────────────────────────────────────────────────────────────────

def load_vectorstore() -> Chroma:
    """Load an existing persisted ChromaDB vector store."""
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            "No vector store found. Run ingest() first."
        )
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name="energy_reports",
    )


def format_context(docs: list[Document]) -> str:
    """Format retrieved chunks with source metadata for the prompt."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = Path(doc.metadata.get("source", "unknown")).name
        page = doc.metadata.get("page", "?")
        parts.append(
            f"[{i}] Source: {source}, Page {page}\n{doc.page_content.strip()}"
        )
    return "\n\n".join(parts)


def build_rag_chain(vectorstore: Optional[Chroma] = None):
    """Build and return the LangChain RAG chain."""
    if vectorstore is None:
        vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="mmr",           # Max Marginal Relevance for diversity
        search_kwargs={"k": RETRIEVER_K, "fetch_k": 15},
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    chain = (
        {
            "context": retriever | format_context,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def query(question: str, chain=None, retriever=None):
    """
    Run a question through the RAG chain.
    Returns (answer, sources) tuple.
    """
    if chain is None:
        chain, retriever = build_rag_chain()

    answer = chain.invoke(question)

    # Fetch source docs separately for the UI citation panel
    source_docs = retriever.invoke(question)
    sources = []
    seen = set()
    for doc in source_docs:
        src = Path(doc.metadata.get("source", "unknown")).name
        page = doc.metadata.get("page", "?")
        key = f"{src}::{page}"
        if key not in seen:
            seen.add(key)
            sources.append({"file": src, "page": page, "snippet": doc.page_content[:200]})

    return answer, sources


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        ingest()
    else:
        print("Energy RAG — interactive mode")
        print("Type 'quit' to exit.\n")
        chain, retriever = build_rag_chain()
        while True:
            q = input("Question: ").strip()
            if q.lower() in ("quit", "exit", "q"):
                break
            if not q:
                continue
            answer, sources = query(q, chain, retriever)
            print(f"\nAnswer:\n{answer}\n")
            print("Sources:")
            for s in sources:
                print(f"  - {s['file']}, p.{s['page']}")
            print()
