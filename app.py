"""
Gradio interface for the Industrial Maintenance Knowledge Assistant.

Wraps generate_answer.py's AnswerGenerator in a chat UI showing:
  - the grounded answer
  - a confidence badge (high / medium / low-refused)
  - an expandable panel with the actual retrieved source passages (citations)

Deploy target: Hugging Face Spaces (free CPU tier). See deployment notes
at the bottom of this file.
"""
import gradio as gr
from generate_answer import AnswerGenerator

print("Initializing retrieval + generation pipeline (this runs once at startup) ...")
generator = AnswerGenerator()
print("Ready.")


def format_sources(sources):
    if not sources:
        return "_No sources retrieved (question was refused before retrieval was needed, or deemed out of scope)._"
    lines = []
    for i, r in enumerate(sources, 1):
        code_label = f"**{r['code']}**" if r["code"] else f"**{r['title']}**" if "title" in r else "**Reference**"
        lines.append(
            f"**[Source {i}]** {code_label} — _{r['source_doc']}, page {r['page']}_\n\n"
            f"> {r['text']}"
        )
    return "\n\n---\n\n".join(lines)


def confidence_badge(confidence_str):
    if "high" in confidence_str:
        return f"🟢 **Confidence: {confidence_str}**"
    elif "medium" in confidence_str:
        return f"🟡 **Confidence: {confidence_str}**"
    else:
        return f"🔴 **Confidence: {confidence_str}**"


def respond(query, history):
    if not query.strip():
        return history, "", ""

    result = generator.answer(query)
    badge = confidence_badge(result["confidence"])
    sources_md = format_sources(result["sources"])

    # dict-based message format - required by the Chatbot component in the
    # installed Gradio version (tuple format isn't accepted)
    history = history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": result["answer"]},
    ]
    return history, badge, sources_md


with gr.Blocks(title="Industrial Maintenance Knowledge Assistant") as demo:
    gr.Markdown(
        "# 🏭 Industrial Maintenance Knowledge Assistant\n"
        "Ask about PLC alarm codes, motor/drive faults, or Modbus communication errors. "
        "Answers are grounded in official Siemens S7-1200/S7-1500/SINAMICS G120 documentation, "
        "with citations to the exact source and page.\n\n"
        "_Try:_ `F01000` · `motor overload alarm what should I check` · `Modbus slave address error`"
    )

    chatbot = gr.Chatbot(label="Conversation", height=400)
    query_box = gr.Textbox(
        label="Your question",
        placeholder="e.g. PLC showing motor overload alarm, what should I check?",
    )
    submit_btn = gr.Button("Ask", variant="primary")

    confidence_display = gr.Markdown()
    with gr.Accordion("📄 Retrieved sources (citations)", open=False):
        sources_display = gr.Markdown()

    submit_btn.click(
        respond,
        inputs=[query_box, chatbot],
        outputs=[chatbot, confidence_display, sources_display],
    ).then(lambda: "", outputs=query_box)

    query_box.submit(
        respond,
        inputs=[query_box, chatbot],
        outputs=[chatbot, confidence_display, sources_display],
    ).then(lambda: "", outputs=query_box)

    gr.Examples(
        examples=[
            "F01000",
            "motor is drawing too much current and overload alarm triggered",
            "Modbus slave has an invalid address configured",
            "how many diagnostic events does the CPU log keep",
        ],
        inputs=query_box,
    )

if __name__ == "__main__":
    import os
    # Render (and most hosting platforms) provide the public port via the
    # PORT environment variable, and expect the app to bind to 0.0.0.0 so
    # it's reachable from outside the container. share=True (used for the
    # Colab tunnel) isn't needed here - Render provides its own public URL.
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)

# =============================================================================
# DEPLOYMENT NOTES - Hugging Face Spaces (free)
# =============================================================================
# 1. Go to huggingface.co -> New Space -> choose "Gradio" as the SDK, CPU
#    basic (free) hardware.
# 2. Upload these files to the Space repo:
#      - app.py                  (this file)
#      - generate_answer.py
#      - rerank.py
#      - hybrid_search.py
#      - unified_corpus.json
#      - eval_set.py             (imported by rerank.py's chain of imports)
#      - chroma_db/               (the WHOLE folder, persisted from Colab -
#                                   this avoids re-embedding 255 records on
#                                   every Space restart)
#      - requirements.txt        (see below)
# 3. requirements.txt should contain:
#      gradio
#      chromadb
#      sentence-transformers
#      transformers
#      torch
#      rank_bm25
# 4. First load will be slow (~2-3 min) while models download and load into
#    memory on the free CPU tier. This is normal - mention it in your README
#    demo instructions so reviewers don't think it's broken.
# =============================================================================
