"""
src/rag_pipeline.py — Core RAG chain and query logic.

Responsibilities:
    - Load ChromaDB vectorstore
    - Build LangChain RAG chain (Cloud or Local backend)
    - Execute validated queries and return RAGResponse

Pydantic models live in src/models.py and are re-exported here
for backward compatibility with existing imports.
"""

import logging
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Re-export all models for backward compatibility
# (existing code that does `from src.rag_pipeline import PipelineConfig` still works)
from src.models import (  # noqa: F401
    BronzeRecord,
    PipelineConfig,
    RAGQuery,
    RAGResponse,
    SilverRecord,
    SourceChunk,
)

# ── Logging ────────────────────────────────────────────────────────────────────

log = logging.getLogger(__name__)

# ── Path configuration ─────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
CHROMA_COLLECTION = "energy_reports_v2"

# ── Validated singleton config ─────────────────────────────────────────────────
# Validated at import time — raises immediately if misconfigured

CONFIG = PipelineConfig()

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a research assistant specializing in energy systems, \
EV infrastructure, and transportation electrification at a national laboratory. \
You answer questions using ONLY the provided context from NREL and DOE technical \
reports and the RouteE Compass codebase.

Guidelines:
- Be precise and cite the source document and section for every claim.
- If the context does not contain enough information, say so clearly.
- Use technical language appropriate for energy researchers and engineers.
- Prefer quantitative findings where available.
- For code questions, include relevant code snippets from the context.

Context:
{context}
"""

# ── Vectorstore ────────────────────────────────────────────────────────────────

def load_vectorstore(embedding_fn) -> Chroma:
    """
    Load ChromaDB with the given embedding function.
    Called per query since Cloud and Local use different embedding models.
    """
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"No vector store found at {CHROMA_DIR}. "
            "Run `python ingest_routee.py` to build the corpus first."
        )
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embedding_fn,
        collection_name=CHROMA_COLLECTION,
    )


def vectorstore_ready() -> bool:
    """Check if ChromaDB has been built."""
    return CHROMA_DIR.exists()


# ── Context formatter ─────────────────────────────────────────────────────────

def format_context(docs: list[Document]) -> str:
    """Format retrieved chunks with source metadata for the LLM prompt."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = Path(doc.metadata.get("source", "unknown")).name
        section = doc.metadata.get("page", "?")
        parts.append(
            f"[{i}] Source: {source}, Section: {section}\n"
            f"{doc.page_content.strip()}"
        )
    return "\n\n".join(parts)


# ── LLM factory ───────────────────────────────────────────────────────────────

def _get_cloud_llm_and_embeddings():
    """Instantiate OpenAI LLM + embeddings. Requires OPENAI_API_KEY."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    llm = ChatOpenAI(model=CONFIG.llm_model, temperature=0)
    embeddings = OpenAIEmbeddings(model=CONFIG.embedding_model)
    return llm, embeddings


def _get_local_llm_and_embeddings():
    """
    Instantiate Ollama LLM + embeddings.
    Connects to Ollama service at CONFIG.ollama_base_url.
    Model runs entirely on GPU VRAM — zero system RAM overhead.
    """
    from langchain_ollama import ChatOllama, OllamaEmbeddings

    llm = ChatOllama(
        model=CONFIG.local_llm_model,
        base_url=CONFIG.ollama_base_url,
        temperature=0,
    )
    embeddings = OllamaEmbeddings(
        model=CONFIG.local_embedding_model,
        base_url=CONFIG.ollama_base_url,
    )
    return llm, embeddings


# ── Chain builder ──────────────────────────────────────────────────────────────

def build_rag_chain(vectorstore: Chroma, llm: Any):
    """Build LangChain RAG chain with MMR retrieval."""
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": CONFIG.retriever_k,
            "fetch_k": CONFIG.retriever_fetch_k,
        },
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

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


# ── Public query API ──────────────────────────────────────────────────────────

def query(
    raw_question: str,
    backend: str = "cloud",
    chain: Any = None,
    retriever: Any = None,
) -> RAGResponse:
    """
    Run a validated question through the RAG chain.

    Dynamically instantiates the correct LLM and embedding model based
    on the backend parameter — no global model state maintained.

    Args:
        raw_question: The user's research question.
        backend: "cloud" for OpenAI or "local" for Ollama.
        chain: Optional pre-built chain (for reuse across requests).
        retriever: Optional pre-built retriever.

    Returns:
        RAGResponse with structured answer and source citations.

    Raises:
        pydantic.ValidationError: if question is empty or invalid.
        FileNotFoundError: if ChromaDB has not been built.
    """
    validated = RAGQuery(question=raw_question)

    if chain is None or retriever is None:
        if backend == "cloud":
            llm, embedding_fn = _get_cloud_llm_and_embeddings()
        else:
            llm, embedding_fn = _get_local_llm_and_embeddings()

        vectorstore = load_vectorstore(embedding_fn)
        chain, retriever = build_rag_chain(vectorstore, llm)

    log.info(f"Query [{backend}]: {validated.question[:80]}")
    answer = chain.invoke(validated.question)

    # Build structured source citations
    source_docs = retriever.invoke(validated.question)
    sources: list[SourceChunk] = []
    seen: set[str] = set()
    for doc in source_docs:
        src = Path(doc.metadata.get("source", "unknown")).name
        section = str(doc.metadata.get("page", "?"))
        project = doc.metadata.get("project", "unknown")
        key = f"{src}::{section}"
        if key not in seen:
            seen.add(key)
            sources.append(
                SourceChunk(
                    file=src,
                    page=section,
                    snippet=doc.page_content[:200],
                    project=project,
                )
            )

    return RAGResponse(
        question=validated.question,
        answer=answer,
        sources=sources,
        backend=backend,
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Energy RAG — interactive mode (cloud backend)")
    print("Type 'quit' to exit.\n")
    while True:
        q = input("Question: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue
        response = query(q, backend="cloud")
        print(f"\nAnswer:\n{response.answer}\n")
        print("Sources:")
        print(response.format_sources_text())
        print()
