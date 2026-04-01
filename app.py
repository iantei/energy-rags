"""
Energy Policy RAG — Gradio Interface (Gradio 6 compatible)
"""

import gradio as gr
from pathlib import Path
from src.rag_pipeline import build_rag_chain, query, ingest, CHROMA_DIR, DATA_DIR

_chain = None
_retriever = None

EXAMPLE_QUESTIONS = [
    "What are the key barriers to electrifying on-demand transit fleets?",
    "How does charging infrastructure availability affect EV adoption rates?",
    "What grid impacts are projected from large-scale EV fleet deployment?",
    "What vehicle classes are most suitable for near-term electrification?",
    "Summarize the findings on charging demand at the national scale.",
]


def ensure_chain():
    global _chain, _retriever
    if _chain is None:
        if not CHROMA_DIR.exists():
            return False
        _chain, _retriever = build_rag_chain()
    return True


def format_sources_md(sources):
    if not sources:
        return "_No sources retrieved._"
    lines = ["| # | Document | Page | Snippet |", "|---|----------|------|---------|"]
    for i, s in enumerate(sources, 1):
        snippet = s["snippet"].replace("\n", " ").replace("|", "/")[:120] + "..."
        lines.append(f"| {i} | `{s['file']}` | {s['page']} | {snippet} |")
    return "\n".join(lines)


def run_ingest(pdf_dir_str):
    pdf_dir = Path(pdf_dir_str.strip()) if pdf_dir_str.strip() else DATA_DIR
    if not pdf_dir.exists():
        return f"Directory not found: {pdf_dir}"
    try:
        ingest(pdf_dir)
        global _chain, _retriever
        _chain, _retriever = build_rag_chain()
        pdf_count = len(list(pdf_dir.glob("*.pdf")))
        return f"Ingested {pdf_count} PDF(s) from {pdf_dir}. Ready to query."
    except Exception as e:
        return f"Ingestion error: {e}"


def ask_question(question, history):
    history = history or []
    if not question.strip():
        return history, "", "_Please enter a question._"

    if not ensure_chain():
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": "No vector store found. Please ingest PDFs first using the Setup tab."})
        return history, "", "_Run ingestion first._"

    try:
        answer, sources = query(question, _chain, _retriever)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        return history, "", format_sources_md(sources)
    except Exception as e:
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": f"Error: {e}"})
        return history, "", "_Error retrieving sources._"


with gr.Blocks(title="Energy Policy RAG Assistant") as demo:

    gr.Markdown("""
    # Energy Policy RAG Assistant
    **Ask research questions over NREL & DOE technical reports.**
    Powered by OpenAI embeddings + ChromaDB + GPT-4o-mini.
    """)

    with gr.Tabs():

        with gr.Tab("Ask a Question"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(label="Conversation", height=460)
                    with gr.Row():
                        question_box = gr.Textbox(
                            placeholder="e.g. What are the grid impacts of large-scale EV fleet deployment?",
                            label="Your question",
                            lines=2,
                            scale=5,
                        )
                        ask_btn = gr.Button("Ask", variant="primary", scale=1)

                with gr.Column(scale=2):
                    gr.Markdown("### Retrieved Sources")
                    sources_display = gr.Markdown("_Sources will appear here after your first query._")

            history_state = gr.State([])

            ask_btn.click(
                ask_question,
                inputs=[question_box, history_state],
                outputs=[chatbot, question_box, sources_display],
            )
            question_box.submit(
                ask_question,
                inputs=[question_box, history_state],
                outputs=[chatbot, question_box, sources_display],
            )

        with gr.Tab("Setup / Ingest"):
            gr.Markdown("""
            ### Ingest PDF Reports
            1. Place NREL or DOE PDF reports in `data/pdfs/`
            2. Optionally specify a custom directory below
            3. Click **Ingest PDFs**
            """)
            with gr.Row():
                pdf_dir_input = gr.Textbox(
                    label="PDF directory (leave blank for default data/pdfs/)",
                    placeholder=str(DATA_DIR),
                    scale=4,
                )
                ingest_btn = gr.Button("Ingest PDFs", variant="primary", scale=1)
            ingest_status = gr.Textbox(label="Status", interactive=False, lines=3)
            ingest_btn.click(run_ingest, inputs=[pdf_dir_input], outputs=[ingest_status])

        with gr.Tab("About"):
            gr.Markdown("""
            ### Architecture
            ```
            PDFs -> PyPDFLoader -> RecursiveCharacterTextSplitter
                 -> OpenAI text-embedding-3-small -> ChromaDB (local)
                 -> MMR Retriever (k=5) -> GPT-4o-mini -> Answer + Citations
            ```
            ### Stack
            - LangChain: orchestration
            - ChromaDB: local persistent vector store
            - OpenAI: embeddings (text-embedding-3-small) + LLM (GPT-4o-mini)
            - Gradio: UI
            """)


if __name__ == "__main__":
    demo.launch(share=False, show_error=True)
