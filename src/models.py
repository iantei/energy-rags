"""
src/models.py — Pydantic schema definitions for the Energy RAG pipeline.

Centralised here so that ingest_routee.py, rag_pipeline.py, and app.py
all share the same validated types. Mirrors the schema-integrity pattern
used in production NREL analytical workflows.

Imported by:
    src/rag_pipeline.py  — query validation and response typing
    ingest_routee.py     — pipeline configuration validation
    app.py               — response rendering
    tests/test_models.py — unit tests (no API calls required)
"""

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Pipeline Configuration ─────────────────────────────────────────────────────

class PipelineConfig(BaseModel):
    """
    Validated pipeline configuration.
    Instantiated once at import time — fails fast on misconfiguration
    before any API call or file I/O is attempted.
    """

    chunk_size: int = Field(default=800, gt=0, description="Token chunk size")
    chunk_overlap: int = Field(default=150, ge=0, description="Overlap between chunks")
    embedding_model: str = Field(default="text-embedding-3-small")
    llm_model: str = Field(default="gpt-4o-mini")
    retriever_k: int = Field(default=5, ge=1, le=20)
    retriever_fetch_k: int = Field(default=15, ge=1)

    # Local (Ollama) backend config
    local_llm_model: str = Field(default="llama3.1")
    local_embedding_model: str = Field(default="nomic-embed-text")
    ollama_base_url: str = Field(default="http://ollama:11434")

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


# ── Query Input ────────────────────────────────────────────────────────────────

class RAGQuery(BaseModel):
    """Validated query input. Strips whitespace and rejects empty questions."""

    question: str = Field(..., min_length=1, description="Research question")
    k: int | None = Field(default=None, ge=1, le=20, description="Override retriever_k")

    @field_validator("question", mode="before")
    @classmethod
    def strip_and_check_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question must not be empty or whitespace only")
        return v


# ── Response Output ────────────────────────────────────────────────────────────

class SourceChunk(BaseModel):
    """A single retrieved source chunk with provenance metadata."""

    file: str = Field(..., description="Source filename")
    page: str = Field(..., description="Section heading or page number")
    snippet: str = Field(..., description="Short excerpt from the chunk")
    project: str = Field(default="unknown", description="Project tag")


class RAGResponse(BaseModel):
    """Structured RAG response — answer text with cited source chunks."""

    question: str
    answer: str
    sources: list[SourceChunk]
    backend: str = Field(default="cloud", description="'cloud' or 'local'")

    @property
    def source_count(self) -> int:
        return len(self.sources)

    def format_sources_text(self) -> str:
        """Plain-text source list for CLI output."""
        lines = []
        for i, s in enumerate(self.sources, 1):
            lines.append(f"  [{i}] {s.file} — {s.page} ({s.project})")
        return "\n".join(lines)


# ── Ingestion metadata ─────────────────────────────────────────────────────────

class BronzeRecord(BaseModel):
    """Schema for a single raw document in the Bronze Parquet layer."""

    source_file: str
    source_path: str
    file_hash: str
    content_markdown: str
    page_count: int = Field(default=0, ge=0)
    file_type: str = Field(description="pdf | pdf_fallback | markdown | python_example")
    project: str
    parsed_at: float


class SilverRecord(BaseModel):
    """Schema for a single enriched chunk in the Silver Parquet layer."""

    source_file: str
    project: str
    file_type: str
    section: str
    chunk_index: int = Field(ge=0)
    chunk_text: str = Field(min_length=1)
    file_hash: str
    chunk_id: str
