"""
tests/test_ingestion.py — Unit tests for Silver layer chunking logic.

Tests the _chunk_record() and _nearest_heading() functions in
ingest_routee.py without touching the filesystem, ChromaDB, or
any API endpoints. All tests run fully offline in CI.
"""

import pytest

from ingest_routee import _chunk_record, _nearest_heading


# ── _nearest_heading ───────────────────────────────────────────────────────────

class TestNearestHeading:
    SAMPLE_MD = """# Motivation

    RouteE Compass was developed at NREL.

    ## Compass Features

    The core features are listed below.

    ### Dynamic Cost Function

    The cost function is dynamic.
    """

    def test_before_first_heading(self):
        # Position 0, before any heading
        result = _nearest_heading(self.SAMPLE_MD, 0)
        assert result == "Motivation"

    def test_in_first_section(self):
        pos = self.SAMPLE_MD.find("RouteE Compass was developed")
        result = _nearest_heading(self.SAMPLE_MD, pos)
        assert result == "Motivation"

    def test_in_second_section(self):
        pos = self.SAMPLE_MD.find("core features")
        result = _nearest_heading(self.SAMPLE_MD, pos)
        assert result == "Compass Features"

    def test_in_third_section(self):
        pos = self.SAMPLE_MD.find("cost function is dynamic")
        result = _nearest_heading(self.SAMPLE_MD, pos)
        assert result == "Dynamic Cost Function"

    def test_empty_content_returns_introduction(self):
        result = _nearest_heading("No headings here.", 0)
        assert result == "Introduction"

    def test_no_headings_returns_introduction(self):
        result = _nearest_heading("Just plain text.", 5)
        assert result == "Introduction"


# ── _chunk_record ──────────────────────────────────────────────────────────────

def _make_bronze_row(
    content: str,
    file_type: str = "markdown",
    source_file: str = "test.md",
    project: str = "test",
) -> dict:
    return {
        "source_file": source_file,
        "source_path": f"/data/{source_file}",
        "file_hash": "testhash123",
        "content_markdown": content,
        "page_count": 0,
        "file_type": file_type,
        "project": project,
        "parsed_at": 0.0,
    }


class TestChunkRecord:

    def test_markdown_produces_chunks(self):
        content = "# Section A\n\nSome content about energy.\n\n## Section B\n\nMore content."
        row = _make_bronze_row(content, file_type="markdown")
        chunks = _chunk_record(row)
        assert len(chunks) > 0

    def test_chunks_have_required_fields(self):
        content = "# Test\n\nContent here.\n"
        row = _make_bronze_row(content)
        chunks = _chunk_record(row)
        required = {"source_file", "project", "file_type", "section", "chunk_index",
                    "chunk_text", "file_hash", "chunk_id"}
        for chunk in chunks:
            assert required.issubset(chunk.keys()), f"Missing keys: {required - chunk.keys()}"

    def test_chunk_ids_are_unique(self):
        content = "# Section\n\n" + "Word " * 500
        row = _make_bronze_row(content)
        chunks = _chunk_record(row)
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"

    def test_chunk_ids_include_file_hash(self):
        content = "# Section\n\nContent."
        row = _make_bronze_row(content)
        chunks = _chunk_record(row)
        for chunk in chunks:
            assert chunk["chunk_id"].startswith("testhash123")

    def test_python_example_section_tagging(self):
        content = '"""\nTime Energy Tradeoff Example\n"""\nimport routee\n\ndef run():\n    pass\n'
        row = _make_bronze_row(
            content,
            file_type="python_example",
            source_file="03_time_energy_tradeoff_example.py",
        )
        chunks = _chunk_record(row)
        assert len(chunks) > 0
        # Section should contain the example name
        for chunk in chunks:
            assert "part" in chunk["section"].lower()

    def test_pdf_fallback_produces_chunks(self):
        content = "Energy research content.\n\nMore content about EV charging."
        row = _make_bronze_row(content, file_type="pdf_fallback", source_file="report.pdf")
        chunks = _chunk_record(row)
        assert len(chunks) > 0

    def test_large_content_splits_into_multiple_chunks(self):
        # Content larger than chunk_size should produce multiple chunks
        content = "# Large Section\n\n" + "Energy data sentence. " * 200
        row = _make_bronze_row(content)
        chunks = _chunk_record(row)
        assert len(chunks) > 1, "Large content should produce multiple chunks"

    def test_chunk_text_is_not_empty(self):
        content = "# Section\n\nSome meaningful content here."
        row = _make_bronze_row(content)
        chunks = _chunk_record(row)
        for chunk in chunks:
            assert chunk["chunk_text"].strip(), "Chunk text must not be empty"

    def test_project_metadata_preserved(self):
        content = "# RouteE Config\n\nSome TOML config."
        row = _make_bronze_row(content, project="routee-compass")
        chunks = _chunk_record(row)
        for chunk in chunks:
            assert chunk["project"] == "routee-compass"
