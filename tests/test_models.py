"""
Unit tests for Pydantic models in rag_pipeline.
These run without an OpenAI key — no API calls made.
"""

import pytest
from pydantic import ValidationError

from src.rag_pipeline import (
    PipelineConfig,
    RAGQuery,
    RAGResponse,
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
            page="?",
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
                SourceChunk(file="config.md", page="?", snippet="...", project="routee-compass"),
                SourceChunk(file="79093.pdf", page="1", snippet="...", project="nrel"),
            ],
        )

    def test_source_count(self):
        response = self._make_response()
        assert response.source_count == 2

    def test_format_sources_text(self):
        response = self._make_response()
        text = response.format_sources_text()
        assert "config.md" in text
        assert "79093.pdf" in text
        assert "routee-compass" in text

    def test_empty_sources(self):
        response = RAGResponse(
            question="test", answer="no sources", sources=[]
        )
        assert response.source_count == 0
        assert response.format_sources_text() == ""
