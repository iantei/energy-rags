"""
app.py — Gradio UI with dynamic Cloud / Local LLM toggle.

Design principles:
    - No global LLM instantiation — models created per query based on
      the dropdown selection. Avoids holding both backends in memory
      simultaneously (important on 16GB RAM / 8GB VRAM hardware).
    - ChromaDB validity check on startup — surfaces a clear setup message
      if ingestion hasn't been run yet.
    - Pydantic validation via src.rag_pipeline.query() — invalid inputs
      return structured error messages rather than crashing.
"""

import logging

import gradio as gr

from src.rag_pipeline import CONFIG, query, vectorstore_ready

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("app")

# ── Backend mapping ────────────────────────────────────────────────────────────

BACKEND_OPTIONS = {
    "☁️  Cloud (OpenAI GPT-4o-mini)": "cloud",
    "🖥️  Local Air-Gapped (Ollama llama3.1)": "local",
}


def _backend_key(choice: str) -> str:
    return BACKEND_OPTIONS.get(choice, "cloud")


# ── Gradio helpers ─────────────────────────────────────────────────────────────

def _format_sources_md(response) -> str:
    if not response.sources:
        return "_No sources retrieved._"
    lines = [
        "| # | Document | Section | Project | Snippet |",
        "|---|----------|---------|---------|---------|",
    ]
    for i, s in enumerate(response.sources, 1):
        snippet = s.snippet.replace("\n", " ").replace("|", "/")[:100] + "..."
        lines.append(
            f"| {i} | `{s.file}` | {s.page} | {s.project} | {snippet} |"
        )
    return "\n".join(lines)


# ── Gradio callback ────────────────────────────────────────────────────────────

def ask_question(
    question: str,
    llm_choice: str,
    history: list,
) -> tuple:
    history = history or []

    if not question.strip():
        return history, "", "_Please enter a question._", "_Ready_"

    if not vectorstore_ready():
        msg = (
            "⚠️ No vector store found. "
            "Run `python ingest_routee.py` to build the corpus first, "
            "or use the Setup tab instructions."
        )
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": msg})
        return history, "", "_Run ingestion first._", "⚠️ No corpus"

    backend = _backend_key(llm_choice)

    try:
        response = query(question, backend=backend)
        history.append({"role": "user", "content": response.question})
        history.append({"role": "assistant", "content": response.answer})
        sources_md = _format_sources_md(response)
        status = (
            f"✅ {response.source_count} source(s) · "
            f"**{llm_choice.split('(')[1].rstrip(')')}**"
        )
        return history, "", sources_md, status

    except ConnectionError:
        msg = (
            f"❌ Cannot reach Ollama at `{CONFIG.ollama_base_url}`. "
            "Start the service: `docker compose up ollama` "
            "or `ollama serve` locally."
        )
        log.error("Ollama connection error")
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": msg})
        return history, "", "_Connection error._", "❌ Ollama unreachable"

    except Exception as exc:
        log.error(f"Query error: {exc}", exc_info=True)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": f"❌ {type(exc).__name__}: {exc}"})
        return history, "", "_An error occurred._", f"❌ {type(exc).__name__}"


# ── UI layout ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="Energy Policy RAG Assistant") as demo:

    gr.Markdown("""
    # 🔋 Energy Policy RAG Assistant
    **Ask research questions over NREL & DOE technical reports and RouteE Compass docs.**

    Medallion ETL (Bronze/Silver/Gold) · Pydantic schema validation · Cloud or Local inference
    """)

    with gr.Tabs():

        with gr.Tab("💬 Ask a Question"):
            with gr.Row():

                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(label="Conversation", height=480)

                    with gr.Row():
                        question_box = gr.Textbox(
                            placeholder=(
                                "e.g. How does RouteE Compass handle "
                                "battery state for EVs?"
                            ),
                            label="Your question",
                            lines=2,
                            scale=5,
                        )
                        ask_btn = gr.Button("Ask", variant="primary", scale=1)

                    with gr.Row():
                        llm_dropdown = gr.Dropdown(
                            choices=list(BACKEND_OPTIONS.keys()),
                            value=list(BACKEND_OPTIONS.keys())[0],
                            label="Inference Backend",
                            info=(
                                "Cloud: OpenAI API (best quality, requires key). "
                                "Local: Ollama on GPU (air-gapped, free)."
                            ),
                            scale=3,
                        )
                        status_box = gr.Markdown("_Ready_", scale=2)

                with gr.Column(scale=2):
                    gr.Markdown("### 📄 Retrieved Sources")
                    sources_display = gr.Markdown(
                        "_Sources will appear here after your first query._"
                    )

            history_state = gr.State([])

            ask_btn.click(
                ask_question,
                inputs=[question_box, llm_dropdown, history_state],
                outputs=[chatbot, question_box, sources_display, status_box],
            )
            question_box.submit(
                ask_question,
                inputs=[question_box, llm_dropdown, history_state],
                outputs=[chatbot, question_box, sources_display, status_box],
            )

        with gr.Tab("⚙️ Setup"):
            gr.Markdown(f"""
            ### Quick Start

            **Option 1 — Docker (recommended for Local backend)**
            ```bash
            # Start all services (app + Ollama with GPU passthrough)
            docker compose up -d

            # Pull local models into Ollama (first time, ~8GB)
            docker exec -it ollama ollama pull llama3.1
            docker exec -it ollama ollama pull nomic-embed-text

            # Run ingestion pipeline
            docker exec -it energy-rag python ingest_routee.py --workers 4
            ```

            **Option 2 — Local virtualenv (Cloud backend only)**
            ```bash
            ./run.sh   # handles venv activation + API key export
            ```

            **Ingestion options**
            ```bash
            python ingest_routee.py              # Full Bronze → Silver → Gold
            python ingest_routee.py --workers 4  # Parallel PDF parsing
            python ingest_routee.py --from-silver  # Re-chunk only (skip PDF parse)
            python ingest_routee.py --from-gold    # Re-embed only
            ```

            **Current config**
            - Cloud LLM: `{CONFIG.llm_model}` · Embeddings: `{CONFIG.embedding_model}`
            - Local LLM: `{CONFIG.local_llm_model}` · Embeddings: `{CONFIG.local_embedding_model}`
            - Ollama URL: `{CONFIG.ollama_base_url}`
            - Chunk size: {CONFIG.chunk_size} · Overlap: {CONFIG.chunk_overlap}
            - Retriever k: {CONFIG.retriever_k} (MMR, fetch_k={CONFIG.retriever_fetch_k})
            """)

        with gr.Tab("ℹ️ Architecture"):
            gr.Markdown("""
            ### ETL Pipeline (Medallion Architecture)
            ```
            PDFs + Markdown/Python source files
              └── Bronze: Docling (layout-aware) + PyPDF fallback
                    └── raw_reports.parquet
                          └── Silver: MarkdownHeaderTextSplitter + section tagging
                                └── chunked_reports.parquet
                                      └── Gold: OpenAI embeddings → ChromaDB
                                            └── MMR Retriever → LLM → RAGResponse
            ```

            ### Key Design Decisions
            - **No global LLM**: instantiated per-query, avoids holding Cloud + Local
              in memory simultaneously on 16GB RAM hardware
            - **Parquet checkpoints**: each ETL layer persists independently,
              enabling `--from-silver` and `--from-gold` re-runs
            - **Pydantic everywhere**: `PipelineConfig`, `RAGQuery`, `RAGResponse`,
              `BronzeRecord`, `SilverRecord` — all schema-validated
            - **ProcessPoolExecutor**: PDF parsing parallelised across CPU cores,
              mirroring HPC batch processing patterns from NREL
            """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # Bind to all interfaces for Docker
        server_port=7860,
        show_error=True,
    )
