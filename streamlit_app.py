"""
Streamlit interface for the Industrial Maintenance Knowledge Assistant.
Same backend as app.py (generate_answer.AnswerGenerator) - only the UI
layer differs, since Streamlit's programming model (rerun top-to-bottom on
each interaction) is different from Gradio's event-callback model.

Deploy target: Streamlit Community Cloud (share.streamlit.io) - free,
deploys directly from a GitHub repo, no credit card.
"""
import streamlit as st
from generate_answer import AnswerGenerator

st.set_page_config(page_title="Industrial Maintenance Knowledge Assistant", page_icon="🏭")

st.title("🏭 Industrial Maintenance Knowledge Assistant")
st.markdown(
    "Ask about PLC alarm codes, motor/drive faults, or Modbus communication errors. "
    "Answers are grounded in official Siemens S7-1200/S7-1500/SINAMICS G120 "
    "documentation, with citations to the exact source and page."
)


@st.cache_resource(show_spinner="Loading retrieval + generation pipeline (first load only) ...")
def load_generator():
    return AnswerGenerator()


generator = load_generator()

if "messages" not in st.session_state:
    st.session_state.messages = []

# render past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("confidence"):
            badge = {
                "high": "🟢",
                "medium": "🟡",
            }.get(msg["confidence"].split()[0], "🔴")
            st.caption(f"{badge} Confidence: {msg['confidence']}")
        if msg.get("sources"):
            with st.expander("📄 Retrieved sources (citations)"):
                for i, r in enumerate(msg["sources"], 1):
                    label = r["code"] or r.get("title", "Reference")
                    st.markdown(f"**[Source {i}] {label}** — _{r['source_doc']}, page {r['page']}_")
                    st.markdown(f"> {r['text']}")

# example queries
st.markdown("**Try:** `F01000` · `motor overload alarm what should I check` · `Modbus slave address error`")

query = st.chat_input("Your question (e.g. PLC showing motor overload alarm, what should I check?)")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching documentation and generating answer ..."):
            result = generator.answer(query)
        st.markdown(result["answer"])

        badge = {"high": "🟢", "medium": "🟡"}.get(result["confidence"].split()[0], "🔴")
        st.caption(f"{badge} Confidence: {result['confidence']}")

        if result["sources"]:
            with st.expander("📄 Retrieved sources (citations)"):
                for i, r in enumerate(result["sources"], 1):
                    label = r["code"] or r.get("title", "Reference")
                    st.markdown(f"**[Source {i}] {label}** — _{r['source_doc']}, page {r['page']}_")
                    st.markdown(f"> {r['text']}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "confidence": result["confidence"],
        "sources": result["sources"],
    })
