"""
RouteE Compass documentation ingester.
Run once to add RouteE docs to the existing ChromaDB vector store:
    python ingest_routee.py
"""

from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# ── Config ─────────────────────────────────────────────────────────────────────

ROUTEE_DOCS_DIR = Path("/mnt/c/Users/peace/Documents/RouteE/routee-compass/docs")

# Files to ingest — ordered by relevance
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


# ── Helpers ────────────────────────────────────────────────────────────────────


def load_file(path: Path, source_label: str) -> list[Document]:
    """Load a single file and tag it with source metadata."""
    if not path.exists():
        print(f"  Skipping (not found): {path}")
        return []
    try:
        loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()
        # Override source metadata to a clean label
        for doc in docs:
            doc.metadata["source"] = source_label
            doc.metadata["project"] = "routee-compass"
        return docs
    except Exception as e:
        print(f"  Error loading {path}: {e}")
        return []


def ingest_routee_docs():
    print("\n── RouteE Compass Documentation Ingestion ─────────────────────────")

    if not ROUTEE_DOCS_DIR.exists():
        print(f"❌ Docs directory not found: {ROUTEE_DOCS_DIR}")
        print("   Update ROUTEE_DOCS_DIR in this script to your local path.")
        return

    # Load all docs
    all_docs = []

    print("\nLoading markdown docs:")
    for fname in MD_FILES:
        path = ROUTEE_DOCS_DIR / fname
        docs = load_file(path, f"routee-compass/{fname}")
        if docs:
            print(f"  ✓ {fname} ({len(docs)} page(s))")
            all_docs.extend(docs)

    print("\nLoading Python examples:")
    for fname in EXAMPLE_FILES:
        path = ROUTEE_DOCS_DIR / fname
        docs = load_file(path, f"routee-compass/{fname}")
        if docs:
            print(f"  ✓ {fname} ({len(docs)} page(s))")
            all_docs.extend(docs)

    if not all_docs:
        print("❌ No documents loaded. Check ROUTEE_DOCS_DIR path.")
        return

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)
    print(f"\nSplit into {len(chunks)} chunks.")

    # Add to existing ChromaDB (does not overwrite existing EVI-Pro chunks)
    print("Adding to existing vector store...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name="energy_reports",
    )
    vectorstore.add_documents(chunks)
    print(f"✅ Added {len(chunks)} RouteE Compass chunks to vector store.")
    print("   Restart app.py to use the updated corpus.")


if __name__ == "__main__":
    ingest_routee_docs()
