# Energy Policy RAG Assistant

A retrieval-augmented generation (RAG) pipeline that answers research questions
over NREL and DOE technical reports, with source citations.

Built as a portfolio project demonstrating LLM prototyping for energy systems
research — directly aligned with NLR Data Engineering Group responsibilities.

---

## Architecture

```
PDFs (NREL/DOE reports)
  └── PyPDFLoader
        └── RecursiveCharacterTextSplitter (800 tokens, 150 overlap)
              └── OpenAI text-embedding-3-small
                    └── ChromaDB (local persistent vector store)
                          └── MMR Retriever (k=5)
                                └── GPT-4o-mini
                                      └── Answer + Citations → Gradio UI
```

## Stack

| Component       | Tool                          |
|-----------------|-------------------------------|
| Orchestration   | LangChain 0.3                 |
| Vector Store    | ChromaDB (local)              |
| Embeddings      | OpenAI text-embedding-3-small |
| LLM             | GPT-4o-mini                   |
| PDF Ingestion   | PyPDF                         |
| UI              | Gradio 4                      |

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <your-repo>
cd energy_rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
cp .env.example .env
# Edit .env and add your key: OPENAI_API_KEY=sk-...
```

### 3. Add PDF reports

**Option A — Download sample NREL reports automatically:**
```bash
python scripts/download_sample_pdfs.py
```

**Option B — Add your own:**
Place any NREL or DOE PDF reports into `data/pdfs/`.
Good sources:
- https://www.nrel.gov/publications/
- https://www.osti.gov/

### 4. Ingest and run

```bash
# Ingest PDFs into ChromaDB (one-time, re-run if you add new PDFs)
python -m src.rag_pipeline ingest

# Launch the Gradio UI
python app.py
```

Then open http://localhost:7860 in your browser.

You can also ingest directly from the **Setup tab** in the UI.

---

## Design Decisions

**MMR Retrieval** — Max Marginal Relevance reduces redundant chunks from
the same page and improves answer coverage across the corpus.

**Chunk size 800 / overlap 150** — Technical report prose benefits from
longer chunks to preserve table context and numbered findings. Overlap
prevents key sentences from being split across chunk boundaries.

**Source citations in every answer** — Supports research reproducibility,
a core requirement for energy systems analysis at national labs.

**GPT-4o-mini** — Balances cost and quality for a prototype; swap for
`gpt-4o` for higher-stakes production use.

---

## Example Questions

- What are the key barriers to electrifying on-demand transit fleets?
- How does charging infrastructure availability affect EV adoption rates?
- What grid impacts are projected from large-scale EV fleet deployment?
- What vehicle classes are most suitable for near-term electrification?
- Summarize the findings on charging demand at the national scale.

---

## Project Structure

```
energy_rag/
├── app.py                        # Gradio UI
├── src/
│   └── rag_pipeline.py           # Core RAG pipeline
├── scripts/
│   └── download_sample_pdfs.py   # Fetch sample NREL reports
├── data/
│   ├── pdfs/                     # Place PDF reports here
│   └── chroma_db/                # Auto-generated vector store
├── requirements.txt
├── .env.example
└── README.md
```

---

## Extending This Project

- **Swap ChromaDB for Qdrant or Weaviate** for a production vector store
- **Add reranking** (Cohere Rerank or FlashRank) for better retrieval precision
- **Metadata filtering** — filter by report year, technology area, or author
- **Streaming responses** — `chain.stream()` for real-time token output in Gradio
- **Evaluation** — add RAGAs or TRULENS for systematic answer quality tracking
