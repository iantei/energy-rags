"""
RouteE Compass documentation ingester with section-aware chunking.

Each chunk is tagged with the nearest markdown heading above it,
so the UI shows meaningful section references instead of '?'.

Run once to add RouteE docs to the existing ChromaDB vector store:
    python ingest_routee.py
"""

import re
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# ── Config ─────────────────────────────────────────────────────────────────────

ROUTEE_DOCS_DIR = Path("/mnt/c/Users/peace/Documents/RouteE/routee-compass/docs")

MD_FILES = [
    "config.md",
    "query.md",
    "motivation.md",
    "running.md",
    "installation.md",
    "units.md",
    "developers/contributing.md",
    "developers/rust_code_style.md",
]

EXAMPLE_FILES = [
    "examples/01_open_street_maps_example.py",
    "examples/02_different_powertrains_example.py",
    "examples/03_time_energy_tradeoff_example.py",
    "examples/04_charging_stations_example.py",
    "examples/05_ambient_temperature_example.py",
]

CHROMA_DIR = Path(__file__).parent / "data" / "chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# ── Section-aware markdown loader ──────────────────────────────────────────────

def extract_section_for_position(content: str, char_pos: int) -> str:
    """
    Find the nearest markdown heading at or before char_pos.
    Returns the heading text, or 'Introduction' if none found yet.
    """
    heading_pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
    last_heading = "Introduction"
    for match in heading_pattern.finditer(content):
        if match.start() <= char_pos:
            last_heading = match.group(2).strip()
        else:
            break
    return last_heading


def load_markdown_with_sections(path: Path, source_label: str) -> list[Document]:
    """
    Load a markdown file and split into chunks, tagging each chunk
    with the nearest section heading as metadata.
    """
    if not path.exists():
        print(f"  Skipping (not found): {path}")
        return []

    content = path.read_text(encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )

    # Split raw text first to get chunks with character offsets
    raw_chunks = splitter.create_documents([content])

    # For each chunk, find its position in the original content
    # and tag with the nearest heading
    docs = []
    search_start = 0
    for chunk in raw_chunks:
        chunk_text = chunk.page_content
        # Find where this chunk appears in the original content
        pos = content.find(chunk_text[:50], search_start)
        if pos == -1:
            pos = search_start

        section = extract_section_for_position(content, pos)
        search_start = max(0, pos - CHUNK_OVERLAP)

        docs.append(
            Document(
                page_content=chunk_text,
                metadata={
                    "source": source_label,
                    "page": section,          # section heading replaces page number
                    "project": "routee-compass",
                    "file_type": "markdown",
                },
            )
        )

    return docs


def load_python_example(path: Path, source_label: str) -> list[Document]:
    """
    Load a Python example file, tagging chunks with the example name
    and function/class context where possible.
    """
    if not path.exists():
        print(f"  Skipping (not found): {path}")
        return []

    content = path.read_text(encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\ndef ", "\nclass ", "\n\n", "\n", " ", ""],
    )

    raw_chunks = splitter.create_documents([content])

    # Extract example title from filename e.g. "Time Energy Tradeoff Example"
    example_name = (
        path.stem.split("_", 1)[1]  # strip leading number
        .replace("_", " ")
        .title()
    )

    docs = []
    for i, chunk in enumerate(raw_chunks):
        docs.append(
            Document(
                page_content=chunk.page_content,
                metadata={
                    "source": source_label,
                    "page": f"{example_name} (part {i + 1})",
                    "project": "routee-compass",
                    "file_type": "python_example",
                },
            )
        )

    return docs


# ── Main ingestion ─────────────────────────────────────────────────────────────

def ingest_routee_docs() -> None:
    print("\n── RouteE Compass Documentation Ingestion ─────────────────────────")

    if not ROUTEE_DOCS_DIR.exists():
        print(f"❌ Docs directory not found: {ROUTEE_DOCS_DIR}")
        return

    all_docs: list[Document] = []

    print("\nLoading markdown docs:")
    for fname in MD_FILES:
        path = ROUTEE_DOCS_DIR / fname
        docs = load_markdown_with_sections(path, f"routee-compass/{fname}")
        if docs:
            print(f"  ✓ {fname} ({len(docs)} chunks)")
            all_docs.extend(docs)

    print("\nLoading Python examples:")
    for fname in EXAMPLE_FILES:
        path = ROUTEE_DOCS_DIR / fname
        docs = load_python_example(path, f"routee-compass/{fname}")
        if docs:
            print(f"  ✓ {fname} ({len(docs)} chunks)")
            all_docs.extend(docs)

    if not all_docs:
        print("❌ No documents loaded.")
        return

    print(f"\nTotal chunks to add: {len(all_docs)}")

    # Show a sample of section tags for verification
    print("\nSample section tags:")
    for doc in all_docs[:5]:
        print(f"  [{doc.metadata['source']}] section='{doc.metadata['page']}'")

    # Add to existing ChromaDB collection
    print("\nAdding to vector store...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name="energy_reports",
    )
    vectorstore.add_documents(all_docs)
    print(f"✅ Added {len(all_docs)} RouteE Compass chunks.")
    print("   Restart app.py to use the updated corpus.")


if __name__ == "__main__":
    ingest_routee_docs()
