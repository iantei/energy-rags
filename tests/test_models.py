"""
tests/test_models.py — Unit tests for all Pydantic models.

Tests import from src.models directly (canonical location).
src.rag_pipeline re-exports these for backward compatibility.
No API calls made — all tests run offline in CI.
"""

import pytest
from pydantic import ValidationError

from src.models import (
    BronzeRecord,
    PipelineConfig,
    RAGQuery,
    RAGResponse,
    SilverRecord,
    SourceChunk,
)


# ── PipelineConfig ─────────────────────────────────────────────────────────────

class TestPipelineConfig:
    def test_defaults_are_valid(self):
        config = PipelineConfig()
        assert config.chunk_size == 800
        assert config.chunk_overlap == 150
        assert config.retriever_k == 5

    def test_custom_valid_config(self):
        config = PipelineConfig(chunk_size=512, chunk_overlap=50, retriever_k=3)
        assert config.chunk_size == 512

    def test_overlap_must_be_less_than_chunk_size(self):
        with pytest.raises(ValidationError, match="chunk_overlap"):
            PipelineConfig(chunk_size=100, chunk_overlap=100)

    def test_overlap_greater_than_chunk_size_fails(self):
        with pytest.raises(ValidationError):
            PipelineConfig(chunk_size=100, chunk_overlap=200)

    def test_fetch_k_must_be_gte_k(self):
        with pytest.raises(ValidationError, match="retriever_fetch_k"):
            PipelineConfig(retriever_k=10, retriever_fetch_k=5)

    def test_retriever_k_bounds(self):
        with pytest.raises(ValidationError):
            PipelineConfig(retriever_k=0)
        with pytest.raises(ValidationError):
            PipelineConfig(retriever_k=21)

    def test_chunk_size_must_be_positive(self):
        with pytest.raises(ValidationError):
            PipelineConfig(chunk_size=0)

    def test_local_config_defaults(self):
        config = PipelineConfig()
        assert config.local_llm_model == "llama3.1"
        assert config.local_embedding_model == "nomic-embed-text"


# ── RAGQuery ───────────────────────────────────────────────────────────────────

class TestRAGQuery:
    def test_valid_question(self):
        q = RAGQuery(question="What are the grid impacts of EV deployment?")
        assert q.question == "What are the grid impacts of EV deployment?"

    def test_question_is_stripped(self):
        q = RAGQuery(question="  How does RouteE Compass work?  ")
        assert q.question == "How does RouteE Compass work?"

    def test_empty_question_fails(self):
        with pytest.raises(ValidationError, match="empty"):
            RAGQuery(question="")

    def test_whitespace_only_question_fails(self):
        with pytest.raises(ValidationError, match="empty"):
            RAGQuery(question="   ")

    def test_optional_k_override(self):
        q = RAGQuery(question="test question", k=3)
        assert q.k == 3

    def test_k_bounds(self):
        with pytest.raises(ValidationError):
            RAGQuery(question="test", k=0)
        with pytest.raises(ValidationError):
            RAGQuery(question="test", k=21)

    def test_k_defaults_to_none(self):
        q = RAGQuery(question="test question")
        assert q.k is None


# ── SourceChunk ────────────────────────────────────────────────────────────────

class TestSourceChunk:
    def test_valid_source_chunk(self):
        chunk = SourceChunk(file="79093.pdf", page="49", snippet="EV charging data...")
        assert chunk.file == "79093.pdf"
        assert chunk.project == "unknown"

    def test_project_tag(self):
        chunk = SourceChunk(
            file="config.md",
            page="App Config",
            snippet="energy_weight config...",
            project="routee-compass",
        )
        assert chunk.project == "routee-compass"


# ── RAGResponse ────────────────────────────────────────────────────────────────

class TestRAGResponse:
    def _make_response(self) -> RAGResponse:
        return RAGResponse(
            question="What is RouteE Compass?",
            answer="RouteE Compass is an energy-aware routing tool.",
            sources=[
                SourceChunk(
                    file="config.md",
                    page="App Config",
                    snippet="...",
                    project="routee-compass",
                ),
                SourceChunk(
                    file="79093.pdf",
                    page="1",
                    snippet="...",
                    project="nrel",
                ),
            ],
        )

    def test_source_count(self):
        assert self._make_response().source_count == 2

    def test_format_sources_text(self):
        text = self._make_response().format_sources_text()
        assert "config.md" in text
        assert "79093.pdf" in text
        assert "routee-compass" in text

    def test_empty_sources(self):
        response = RAGResponse(question="test", answer="no sources", sources=[])
        assert response.source_count == 0
        assert response.format_sources_text() == ""

    def test_backend_default(self):
        response = RAGResponse(question="test", answer="ans", sources=[])
        assert response.backend == "cloud"

    def test_backend_local(self):
        response = RAGResponse(
            question="test", answer="ans", sources=[], backend="local"
        )
        assert response.backend == "local"


# ── BronzeRecord ───────────────────────────────────────────────────────────────

class TestBronzeRecord:
    def test_valid_bronze_record(self):
        record = BronzeRecord(
            source_file="report.pdf",
            source_path="/data/pdfs/report.pdf",
            file_hash="abc123",
            content_markdown="# Report\nContent here.",
            page_count=10,
            file_type="pdf",
            project="nrel",
            parsed_at=1234567890.0,
        )
        assert record.source_file == "report.pdf"

    def test_negative_page_count_fails(self):
        with pytest.raises(ValidationError):
            BronzeRecord(
                source_file="f.pdf",
                source_path="/f.pdf",
                file_hash="abc",
                content_markdown="content",
                page_count=-1,
                file_type="pdf",
                project="nrel",
                parsed_at=0.0,
            )


# ── SilverRecord ───────────────────────────────────────────────────────────────

class TestSilverRecord:
    def test_valid_silver_record(self):
        record = SilverRecord(
            source_file="config.md",
            project="routee-compass",
            file_type="markdown",
            section="App Config",
            chunk_index=0,
            chunk_text="The config file specifies which traversal model to use.",
            file_hash="def456",
            chunk_id="def456_0000",
        )
        assert record.section == "App Config"

    def test_empty_chunk_text_fails(self):
        with pytest.raises(ValidationError):
            SilverRecord(
                source_file="f.md",
                project="p",
                file_type="markdown",
                section="s",
                chunk_index=0,
                chunk_text="",
                file_hash="abc",
                chunk_id="abc_0000",
            )

    def test_negative_chunk_index_fails(self):
        with pytest.raises(ValidationError):
            SilverRecord(
                source_file="f.md",
                project="p",
                file_type="markdown",
                section="s",
                chunk_index=-1,
                chunk_text="some text",
                file_hash="abc",
                chunk_id="abc_0000",
            )
