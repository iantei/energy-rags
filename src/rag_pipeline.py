"""
Energy Policy RAG Pipeline
--------------------------
Ingests NREL/DOE PDF reports and RouteE Compass docs, builds a ChromaDB
vector store, and answers researcher questions with source citations.

Pydantic models enforce schema integrity at query input and response output,
mirroring the data quality patterns used in production NREL analytical workflows.
"""

from pathlib import Path
from typing import Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field, field_validator, model_validator

# ── Configuration ──────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data" / "pdfs"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"

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


# ── Pydantic models ────────────────────────────────────────────────────────────


class PipelineConfig(BaseModel):
    """Validated pipeline configuration. Fails fast on invalid values."""

    chunk_size: int = Field(default=800, gt=0, description="Token chunk size for splitting")
    chunk_overlap: int = Field(default=150, ge=0, description="Overlap between chunks")
    embedding_model: str = Field(default="text-embedding-3-small")
    llm_model: str = Field(default="gpt-4o-mini")
    retriever_k: int = Field(default=5, ge=1, le=20, description="Chunks to retrieve per query")
    retriever_fetch_k: int = Field(default=15, ge=1, description="Candidates for MMR reranking")

    @model_validator(mode="after")
    def overlap_less_than_chunk(self) -> "PipelineConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        return self

    @model_validator(mode="after")
    def fetch_k_gte_k(self) -> "PipelineConfig":
        if self.retriever_fetch_k < self.retriever_k:
            raise ValueError(
                f"retriever_fetch_k ({self.retriever_fetch_k}) must be >= "
                f"retriever_k ({self.retriever_k})"
            )
        return self


class RAGQuery(BaseModel):
    """Validated query input."""

    question: str = Field(..., min_length=1, description="Research question")
    k: int | None = Field(default=None, ge=1, le=20, description="Override retriever_k")

    @field_validator("question", mode="before")
    @classmethod
    def strip_and_check_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question must not be empty or whitespace only")
        return v


class SourceChunk(BaseModel):
    """A single retrieved source chunk with metadata."""

    file: str = Field(..., description="Source filename")
    page: str = Field(..., description="Page number or '?' for non-PDF sources")
    snippet: str = Field(..., description="Short excerpt from the chunk")
    project: str = Field(default="unknown", description="Project tag from metadata")


class RAGResponse(BaseModel):
    """Structured RAG response with answer and cited sources."""

    question: str
    answer: str
    sources: list[SourceChunk]

    @property
    def source_count(self) -> int:
        return len(self.sources)

    def format_sources_text(self) -> str:
        lines = []
        for i, s in enumerate(self.sources, 1):
            lines.append(f"  [{i}] {s.file}, p.{s.page} ({s.project})")
        return "\n".join(lines)


# ── Pipeline config singleton ──────────────────────────────────────────────────

# Validated once at import time — raises immediately if misconfigured
CONFIG = PipelineConfig()


# ── Document ingestion ─────────────────────────────────────────────────────────


def load_pdfs(pdf_dir: Path = DATA_DIR) -> list[Document]:
    """Load all PDFs from the data directory."""
    docs = []
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in {pdf_dir}. Add NREL/DOE report PDFs to data/pdfs/ before ingesting."
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
        chunk_size=CONFIG.chunk_size,
        chunk_overlap=CONFIG.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")
    return chunks


def build_vectorstore(chunks: list[Document]) -> Chroma:
    """Embed chunks and persist to ChromaDB."""
    print("Building vector store (this may take a minute)...")
    embeddings = OpenAIEmbeddings(model=CONFIG.embedding_model)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name="energy_reports",
    )
    print(f"Vector store saved to {CHROMA_DIR}")
    return vectorstore


def ingest(pdf_dir: Path = DATA_DIR) -> Chroma:
    """Full ingestion pipeline: load -> chunk -> embed -> persist."""
    print("\n── Ingestion ──────────────────────────────────────────")
    docs = load_pdfs(pdf_dir)
    chunks = chunk_documents(docs)
    return build_vectorstore(chunks)


# ── RAG chain ──────────────────────────────────────────────────────────────────


def load_vectorstore() -> Chroma:
    """Load an existing persisted ChromaDB vector store."""
    if not CHROMA_DIR.exists():
        raise FileNotFoundError("No vector store found. Run ingest() first.")
    embeddings = OpenAIEmbeddings(model=CONFIG.embedding_model)
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
        parts.append(f"[{i}] Source: {source}, Page {page}\n{doc.page_content.strip()}")
    return "\n\n".join(parts)


def build_rag_chain(vectorstore: Chroma | None = None):
    """Build and return the LangChain RAG chain."""
    if vectorstore is None:
        vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": CONFIG.retriever_k,
            "fetch_k": CONFIG.retriever_fetch_k,
        },
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )

    llm = ChatOpenAI(model=CONFIG.llm_model, temperature=0)

    chain: Any = (
        {
            "context": retriever | format_context,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def query(raw_question: str, chain=None, retriever=None) -> RAGResponse:
    """
    Run a validated question through the RAG chain.
    Returns a RAGResponse with structured answer and sources.

    Raises:
        pydantic.ValidationError: if question is empty or invalid.
    """
    # Validate input
    validated = RAGQuery(question=raw_question)

    if chain is None:
        chain, retriever = build_rag_chain()

    answer = chain.invoke(validated.question)

    # Fetch source docs for citation panel
    source_docs = retriever.invoke(validated.question)
    sources: list[SourceChunk] = []
    seen: set[str] = set()
    for doc in source_docs:
        src = Path(doc.metadata.get("source", "unknown")).name
        page = str(doc.metadata.get("page", "?"))
        project = doc.metadata.get("project", "nrel")
        key = f"{src}::{page}"
        if key not in seen:
            seen.add(key)
            sources.append(
                SourceChunk(
                    file=src,
                    page=page,
                    snippet=doc.page_content[:200],
                    project=project,
                )
            )

    return RAGResponse(
        question=validated.question,
        answer=answer,
        sources=sources,
    )


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
            response = query(q, chain, retriever)
            print(f"\nAnswer:\n{response.answer}\n")
            print("Sources:")
            print(response.format_sources_text())
            print()
