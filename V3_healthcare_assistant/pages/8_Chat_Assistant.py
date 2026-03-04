# =============================================================
# Anil Pathak's Agentic Healthcare Assistant
# Capstone Project — Agentic Healthcare Assistant for Medical
#                    Task Automation
# =============================================================
# Submitted by  : Anil Pathak
# Generated with: Claude (Anthropic) AI Coding Assistant
# File          : pages/8_Chat_Assistant.py
# Purpose       : Top-level — AI Chat Assistant page.
#                 Natural language interface to the Healthcare
#                 Agent. Supports patient lookup, appointment
#                 booking, medical history retrieval, and
#                 medical information queries via GPT-4o-mini
#                 function calling. Displays tool activity and
#                 active patient context in the sidebar panel.
# =============================================================

import streamlit as st
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.sidebar_helper import render_sidebar


st.set_page_config(page_title="Chat Assistant", page_icon="💬", layout="wide")

# Chat assistant — no fixed role, render sidebar with current role
render_sidebar()


# ── Page header ───────────────────────────────────────────────
st.title("💬 AI Healthcare Chat Assistant")
st.caption("Powered by GPT-4o-mini | Agentic AI with Function Calling")
st.divider()


# ── Initialise agent in session state ─────────────────────────
# The agent is initialised once per session and cached in
# st.session_state to preserve conversation memory across reruns.
if "agent" not in st.session_state:
    with st.spinner("Initialising Healthcare Agent..."):
        from agent import HealthcareAgent
        from tools.rag_tool import build_vector_store
        build_vector_store()   # Build/load FAISS vector store for RAG
        st.session_state.agent = HealthcareAgent(
            session_id=f"chat_{int(time.time())}",
            enable_logging=True
        )

if "chat_messages"  not in st.session_state:
    st.session_state.chat_messages = []
if "tool_traces"    not in st.session_state:
    st.session_state.tool_traces = []

# ── Layout: chat (left) and tool activity (right) ─────────────
chat_col, info_col = st.columns([2, 1])

with chat_col:
    # Render full conversation history
    for msg in st.session_state.chat_messages:
        avatar = "🧑" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Chat input box
    if user_input := st.chat_input(
        "Ask me about patients, appointments, or medical information..."
    ):
        # Display user message immediately
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)

        # Call agent and display response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                start_t  = time.time()
                response = st.session_state.agent.chat(user_input, verbose=False)
                elapsed  = round((time.time() - start_t) * 1000)

            st.markdown(response)
            tools_used = st.session_state.agent.last_tools_used
            st.caption(
                f"⏱️ {elapsed}ms  |  "
                f"🔧 Tools: {', '.join(tools_used) if tools_used else 'None'}"
            )

        st.session_state.chat_messages.append({"role": "assistant", "content": response})

        # Store tool trace for sidebar display
        if tools_used:
            st.session_state.tool_traces.append({
                "query":   user_input[:60],
                "tools":   tools_used,
                "time_ms": elapsed
            })

        st.rerun()

# ── Right panel: tool activity and patient context ─────────────
with info_col:
    st.subheader("🔧 Tool Activity")
    if st.session_state.tool_traces:
        for trace in reversed(st.session_state.tool_traces[-5:]):
            with st.expander(f"💬 {trace['query']}...", expanded=False):
                for tool in trace["tools"]:
                    st.markdown(f"• `{tool}`")
                st.caption(f"⏱️ {trace['time_ms']}ms")
    else:
        st.info("Tool usage will appear here as you chat.")

    st.divider()
    st.subheader("🧠 Active Patient Context")
    patient_ctx = st.session_state.agent.memory.get_patient_context()
    if patient_ctx:
        for k, v in patient_ctx.items():
            val = str(v).strip()
            if val and val not in ["N/A", "nan", "None", ""]:
                st.markdown(f"**{k}:** {val[:40]}")
    else:
        st.info("No active patient context.")

    st.divider()

    # Quick query buttons for common tasks
    st.subheader("⚡ Quick Queries")
    quick_queries = [
        "Show all patients",
        "Find a cardiologist",
        "Summarise Anjali's history",
        "Latest diabetes treatments",
        "Ramesh Kulkarni medications",
    ]
    for q in quick_queries:
        if st.button(q, width='stretch', key=f"q_{q}"):
            st.session_state.chat_messages.append({"role": "user", "content": q})
            with st.spinner("Processing..."):
                response = st.session_state.agent.chat(q, verbose=False)
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
            st.rerun()

    st.divider()
    if st.button("🔄 Reset Conversation", width='stretch'):
        st.session_state.agent.reset_session()
        st.session_state.chat_messages = []
        st.session_state.tool_traces   = []
        st.rerun()
