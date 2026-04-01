"""
ingest_routee.py — Production Ingestion Engine (Medallion Architecture)
========================================================================
Bronze → Silver → Gold ETL pipeline for the Energy Policy RAG system.

    Bronze: Raw extracted text + metadata → data/raw_reports.parquet
    Silver: Chunked, section-tagged text  → data/chunked_reports.parquet
    Gold:   Vector embeddings             → data/chroma_db/

Key design decisions:
    - Docling for layout-aware PDF parsing (preserves tables, headers locally)
    - ProcessPoolExecutor for HPC-style parallel PDF ingestion
    - Polars + Parquet as ETL checkpoints (memory-efficient, reproducible)
    - MarkdownHeaderTextSplitter to keep tables within section context
    - Each layer is independently re-runnable via CLI flags

Usage:
    # Full pipeline (default: CPU count - 1 workers)
    python ingest_routee.py

    # Full pipeline with explicit worker count
    python ingest_routee.py --workers 4

    # Re-run Silver + Gold only (skip PDF re-parse)
    python ingest_routee.py --from-silver

    # Re-run Gold only (skip chunking)
    python ingest_routee.py --from-gold
"""

import argparse
import hashlib
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import polars as pl
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.models import BronzeRecord, PipelineConfig, SilverRecord

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ingest")

# ── Configuration ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
ROUTEE_DOCS_DIR = Path("/mnt/c/Users/peace/Documents/RouteE/routee-compass/docs")

BRONZE_PATH = DATA_DIR / "raw_reports.parquet"
SILVER_PATH = DATA_DIR / "chunked_reports.parquet"
CHROMA_DIR = DATA_DIR / "chroma_db"
CHROMA_COLLECTION = "energy_reports_v2"

# Validated at import time
CONFIG = PipelineConfig()

ROUTEE_MD_FILES = [
    "config.md",
    "query.md",
    "motivation.md",
    "running.md",
    "installation.md",
    "units.md",
    "developers/contributing.md",
    "developers/rust_code_style.md",
]

ROUTEE_EXAMPLE_FILES = [
    "examples/01_open_street_maps_example.py",
    "examples/02_different_powertrains_example.py",
    "examples/03_time_energy_tradeoff_example.py",
    "examples/04_charging_stations_example.py",
    "examples/05_ambient_temperature_example.py",
]

# ── Utilities ──────────────────────────────────────────────────────────────────

def _compute_file_hash(path: Path) -> str:
    """SHA-256 hash for deduplication and cache invalidation."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha.update(block)
    return sha.hexdigest()


# ── Bronze Layer ───────────────────────────────────────────────────────────────

def _parse_pdf_worker(pdf_path_str: str) -> dict | None:
    """
    Worker function executed in a subprocess via ProcessPoolExecutor.

    Docling is imported inside the worker to avoid pickling issues
    across process boundaries. Falls back to PyPDF if Docling is
    unavailable (e.g., during CI where Docling is not installed).
    """
    import logging as _log

    worker_log = _log.getLogger(f"worker.{os.getpid()}")
    pdf_path = Path(pdf_path_str)

    try:
        from docling.document_converter import DocumentConverter

        worker_log.info(f"Docling parsing: {pdf_path.name}")
        start = time.perf_counter()
        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        content = result.document.export_to_markdown()
        page_count = len(result.document.pages) if hasattr(result.document, "pages") else 0
        elapsed = time.perf_counter() - start
        worker_log.info(f"  {pdf_path.name}: {len(content):,} chars, {elapsed:.1f}s")
        file_type = "pdf"

    except ImportError:
        worker_log.warning(
            f"Docling not installed — falling back to PyPDF for {pdf_path.name}. "
            "Install docling for layout-aware table parsing: pip install docling"
        )
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(pdf_path))
            content = "\n\n".join(p.extract_text() or "" for p in reader.pages)
            page_count = len(reader.pages)
            file_type = "pdf_fallback"
        except Exception as exc:
            worker_log.error(f"PyPDF fallback failed for {pdf_path.name}: {exc}")
            return None

    except Exception as exc:
        worker_log.error(f"Failed to parse {pdf_path.name}: {exc}")
        return None

    return {
        "source_file": pdf_path.name,
        "source_path": str(pdf_path),
        "file_hash": _compute_file_hash(pdf_path),
        "content_markdown": content,
        "page_count": page_count,
        "file_type": file_type,
        "project": "nrel",
        "parsed_at": time.time(),
    }


def _parse_text_file(path: Path, project: str) -> dict | None:
    """Parse a Markdown or Python source file into a Bronze record."""
    try:
        file_type = "markdown" if path.suffix == ".md" else "python_example"
        return {
            "source_file": path.name,
            "source_path": str(path),
            "file_hash": _compute_file_hash(path),
            "content_markdown": path.read_text(encoding="utf-8"),
            "page_count": 0,
            "file_type": file_type,
            "project": project,
            "parsed_at": time.time(),
        }
    except Exception as exc:
        log.error(f"Failed to parse {path}: {exc}")
        return None


def build_bronze_layer(workers: int = 1) -> pl.DataFrame:
    """
    BRONZE LAYER: Parse all source files in parallel → raw_reports.parquet

    PDFs are dispatched to a ProcessPoolExecutor — each worker runs
    Docling independently, mirroring HPC batch document processing.
    Markdown/Python files are parsed sequentially (fast, I/O bound).
    """
    log.info("═══ BRONZE LAYER ═══")
    records: list[dict] = []

    # Parallel PDF parsing
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        log.warning(f"No PDFs found in {PDF_DIR}")
    else:
        log.info(f"{len(pdf_files)} PDF(s) → {workers} worker(s)")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_parse_pdf_worker, str(p)): p for p in pdf_files}
            for future in as_completed(futures):
                path = futures[future]
                try:
                    result = future.result(timeout=300)
                    if result:
                        records.append(result)
                        log.info(f"  ✓ {path.name}")
                    else:
                        log.warning(f"  ✗ {path.name} returned no content")
                except Exception as exc:
                    log.error(f"  ✗ {path.name}: {exc}")

    # Sequential markdown + Python parsing
    if ROUTEE_DOCS_DIR.exists():
        for fname in ROUTEE_MD_FILES + ROUTEE_EXAMPLE_FILES:
            path = ROUTEE_DOCS_DIR / fname
            if path.exists():
                record = _parse_text_file(path, "routee-compass")
                if record:
                    records.append(record)
                    log.info(f"  ✓ {fname}")
            else:
                log.warning(f"  ✗ {fname} not found")
    else:
        log.warning(f"RouteE docs not found: {ROUTEE_DOCS_DIR}")

    if not records:
        raise RuntimeError("Bronze: no documents parsed. Check data/pdfs/ for PDFs.")

    df = pl.DataFrame(records)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(BRONZE_PATH)
    log.info(f"Bronze complete: {len(df)} docs → {BRONZE_PATH}")
    return df


# ── Silver Layer ───────────────────────────────────────────────────────────────

def _nearest_heading(content: str, char_pos: int) -> str:
    """Find the nearest markdown heading at or before char_pos."""
    import re
    pattern = re.compile(r"^\s*(#{1,3})\s+(.+)$", re.MULTILINE)
    heading = "Introduction"
    for m in pattern.finditer(content):
        if m.start() <= char_pos:
            heading = m.group(2).strip()
        else:
            break
    return heading


def _chunk_record(row: dict) -> list[dict]:
    """
    Chunk a single Bronze record into Silver records.

    Strategy:
        - Markdown/PDF: MarkdownHeaderTextSplitter first (preserves tables
          within sections), then RecursiveCharacterTextSplitter for oversize
          sections.
        - Python examples: RecursiveCharacterTextSplitter on function
          boundaries, tagged with example name and part number.
    """
    from langchain.text_splitter import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )

    content = row["content_markdown"]
    source_file = row["source_file"]
    project = row["project"]
    file_type = row["file_type"]
    file_hash = row["file_hash"]

    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG.chunk_size,
        chunk_overlap=CONFIG.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []

    if file_type in ("pdf", "pdf_fallback", "markdown"):
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
            strip_headers=False,
        )
        try:
            header_chunks = md_splitter.split_text(content)
        except Exception:
            header_chunks = []

        if header_chunks:
            for hchunk in header_chunks:
                section = (
                    hchunk.metadata.get("h3")
                    or hchunk.metadata.get("h2")
                    or hchunk.metadata.get("h1")
                    or "Introduction"
                )
                for sub in char_splitter.split_text(hchunk.page_content):
                    idx = len(chunks)
                    chunks.append({
                        "source_file": source_file,
                        "project": project,
                        "file_type": file_type,
                        "section": section,
                        "chunk_index": idx,
                        "chunk_text": sub,
                        "file_hash": file_hash,
                        "chunk_id": f"{file_hash}_{idx:04d}",
                    })
        else:
            # Fallback: regex heading detection
            search_start = 0
            for i, text in enumerate(char_splitter.split_text(content)):
                pos = content.find(text[:50], search_start)
                pos = pos if pos != -1 else search_start
                section = _nearest_heading(content, pos)
                search_start = max(0, pos - CONFIG.chunk_overlap)
                chunks.append({
                    "source_file": source_file,
                    "project": project,
                    "file_type": file_type,
                    "section": section,
                    "chunk_index": i,
                    "chunk_text": text,
                    "file_hash": file_hash,
                    "chunk_id": f"{file_hash}_{i:04d}",
                })

    elif file_type == "python_example":
        py_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG.chunk_size,
            chunk_overlap=CONFIG.chunk_overlap,
            separators=["\ndef ", "\nclass ", "\n\n", "\n", " "],
        )
        stem = Path(source_file).stem
        example_name = (
            stem.split("_", 1)[1].replace("_", " ").title()
            if "_" in stem else stem
        )
        for i, text in enumerate(py_splitter.split_text(content)):
            chunks.append({
                "source_file": source_file,
                "project": project,
                "file_type": file_type,
                "section": f"{example_name} (part {i + 1})",
                "chunk_index": i,
                "chunk_text": text,
                "file_hash": file_hash,
                "chunk_id": f"{file_hash}_{i:04d}",
            })

    return chunks


def build_silver_layer(bronze_df: pl.DataFrame | None = None) -> pl.DataFrame:
    """
    SILVER LAYER: Chunk Bronze records → chunked_reports.parquet

    Loads from Bronze Parquet if no DataFrame is provided, enabling
    independent re-runs without re-parsing PDFs.
    """
    log.info("═══ SILVER LAYER ═══")

    if bronze_df is None:
        if not BRONZE_PATH.exists():
            raise FileNotFoundError(f"Bronze not found at {BRONZE_PATH}")
        bronze_df = pl.read_parquet(BRONZE_PATH)
        log.info(f"Loaded Bronze: {len(bronze_df)} docs from {BRONZE_PATH}")

    all_chunks: list[dict] = []
    for row in bronze_df.iter_rows(named=True):
        try:
            doc_chunks = _chunk_record(row)
            all_chunks.extend(doc_chunks)
            log.info(f"  ✓ {row['source_file']}: {len(doc_chunks)} chunks")
        except Exception as exc:
            log.error(f"  ✗ {row['source_file']}: {exc}")

    if not all_chunks:
        raise RuntimeError("Silver: no chunks produced.")

    silver_df = pl.DataFrame(all_chunks)

    # Deduplicate by chunk_id (idempotent re-ingestion)
    before = len(silver_df)
    silver_df = silver_df.unique(subset=["chunk_id"], keep="first")
    if len(silver_df) < before:
        log.info(f"Deduped {before - len(silver_df)} duplicate chunks")

    silver_df.write_parquet(SILVER_PATH)
    log.info(f"Silver complete: {len(silver_df)} chunks → {SILVER_PATH}")
    log.info(
        "Chunks by project:\n"
        + str(silver_df.group_by("project").agg(pl.len().alias("count")))
    )
    return silver_df


# ── Gold Layer ─────────────────────────────────────────────────────────────────

def build_gold_layer(
    silver_df: pl.DataFrame | None = None,
    embedding_model: str = CONFIG.embedding_model,
) -> Chroma:
    """
    GOLD LAYER: Embed Silver chunks → ChromaDB

    Wipes and rebuilds the collection for consistency with Silver layer.
    Batch embeds in groups of 100 to respect OpenAI rate limits.
    """
    log.info("═══ GOLD LAYER ═══")

    if silver_df is None:
        if not SILVER_PATH.exists():
            raise FileNotFoundError(f"Silver not found at {SILVER_PATH}")
        silver_df = pl.read_parquet(SILVER_PATH)
        log.info(f"Loaded Silver: {len(silver_df)} chunks from {SILVER_PATH}")

    documents = [
        Document(
            page_content=row["chunk_text"],
            metadata={
                "source": row["source_file"],
                "page": row["section"],
                "project": row["project"],
                "file_type": row["file_type"],
                "chunk_id": row["chunk_id"],
            },
        )
        for row in silver_df.iter_rows(named=True)
    ]

    # Wipe and rebuild for consistency
    import shutil
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        log.info("Wiped existing ChromaDB for clean rebuild")

    embeddings = OpenAIEmbeddings(model=embedding_model)
    vectorstore: Chroma | None = None
    batch_size = 100

    for i in range(0, len(documents), batch_size):
        batch = documents[i: i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(documents) - 1) // batch_size + 1
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=str(CHROMA_DIR),
                collection_name=CHROMA_COLLECTION,
            )
        else:
            vectorstore.add_documents(batch)
        log.info(f"  Batch {batch_num}/{total_batches} embedded")

    log.info(f"Gold complete: {len(documents)} vectors → {CHROMA_DIR}")
    return vectorstore  # type: ignore[return-value]


# ── Pipeline orchestration ─────────────────────────────────────────────────────

def run_full_pipeline(workers: int = 1) -> None:
    start = time.perf_counter()
    log.info(f"Full pipeline: {PDF_DIR} + {ROUTEE_DOCS_DIR} → ChromaDB ({workers} worker(s))")
    bronze_df = build_bronze_layer(workers=workers)
    silver_df = build_silver_layer(bronze_df=bronze_df)
    build_gold_layer(silver_df=silver_df)
    log.info(f"Pipeline complete in {time.perf_counter() - start:.1f}s")


def run_from_silver() -> None:
    log.info("Re-running from Silver layer")
    silver_df = build_silver_layer()
    build_gold_layer(silver_df=silver_df)


def run_from_gold() -> None:
    log.info("Re-running Gold layer only")
    build_gold_layer()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Energy Policy RAG — Ingestion Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest_routee.py                    # Full pipeline, auto workers
  python ingest_routee.py --workers 4        # Full pipeline, 4 workers
  python ingest_routee.py --from-silver      # Skip PDF parse
  python ingest_routee.py --from-gold        # Skip chunking too
        """,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Parallel PDF parsing workers (default: CPU count - 1)",
    )
    parser.add_argument(
        "--from-silver",
        action="store_true",
        help="Re-run from Silver Parquet (skip Bronze/PDF parsing)",
    )
    parser.add_argument(
        "--from-gold",
        action="store_true",
        help="Re-run Gold layer only (skip Bronze and Silver)",
    )
    args = parser.parse_args()

    if args.from_gold:
        run_from_gold()
    elif args.from_silver:
        run_from_silver()
    else:
        run_full_pipeline(workers=args.workers)


if __name__ == "__main__":
    main()
