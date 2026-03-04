# =============================================================
# Anil Pathak's Agentic Healthcare Assistant
# Capstone Project — Agentic Healthcare Assistant for Medical
#                    Task Automation
# =============================================================
# Submitted by  : Anil Pathak
# Generated with: Claude (Anthropic) AI Coding Assistant
# File          : pages/0_Doctor_Dashboard.py
# Purpose       : Doctor / Admin Section hub dashboard.
#                 Displays 5 navigation cards: Doctor Schedule,
#                 Medical Records, Model Evaluation, Logs &
#                 Analytics, and Chat Assistant. Sets role to
#                 "admin" so the contextual sidebar expands
#                 the Doctor/Admin section links.
# =============================================================

import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.sidebar_helper import render_sidebar

st.set_page_config(
    page_title="Doctor / Admin Section",
    page_icon="🩺",
    layout="wide"
)

# Set role so sidebar expands admin links
st.session_state["role"] = "admin"

# Render contextual sidebar
render_sidebar()

# ── Page header ───────────────────────────────────────────────
st.title("🩺 Doctor / Admin Section")
st.caption("Select what you would like to do today")
st.divider()

# ── Row 1: 2 cards ───────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div style="
        background-color: #eafaf1;
        border: 2px solid #2ecc71;
        border-radius: 12px;
        padding: 32px 20px 16px 20px;
        text-align: center;
        min-height: 190px;
    ">
        <div style="font-size: 2.8rem; margin-bottom: 10px;">🗓️</div>
        <div style="font-weight: 700; font-size: 1.1rem;
                    margin-bottom: 6px;">Doctor Schedule</div>
        <div style="font-size: 0.88rem; color: #555;">
            View appointment schedules for any doctor by name
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Doctor Schedule →", key="d1",
                 width='stretch', type="primary"):
        st.switch_page("pages/4_Doctor_Schedule.py")

with col2:
    st.markdown("""
    <div style="
        background-color: #eafaf1;
        border: 2px solid #2ecc71;
        border-radius: 12px;
        padding: 32px 20px 16px 20px;
        text-align: center;
        min-height: 190px;
    ">
        <div style="font-size: 2.8rem; margin-bottom: 10px;">📋</div>
        <div style="font-weight: 700; font-size: 1.1rem;
                    margin-bottom: 6px;">Medical Records</div>
        <div style="font-size: 0.88rem; color: #555;">
            Add and update structured and unstructured patient records
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Medical Records →", key="d2",
                 width='stretch', type="primary"):
        st.switch_page("pages/5_Medical_Records.py")

st.markdown(" ")

# ── Row 2: 2 cards ───────────────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.markdown("""
    <div style="
        background-color: #eafaf1;
        border: 2px solid #2ecc71;
        border-radius: 12px;
        padding: 32px 20px 16px 20px;
        text-align: center;
        min-height: 190px;
    ">
        <div style="font-size: 2.8rem; margin-bottom: 10px;">📊</div>
        <div style="font-weight: 700; font-size: 1.1rem;
                    margin-bottom: 6px;">Model Evaluation</div>
        <div style="font-size: 0.88rem; color: #555;">
            Run LLM-as-judge evaluation across 8 test cases
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Model Evaluation →", key="d3",
                 width='stretch', type="primary"):
        st.switch_page("pages/6_Model_Evaluation.py")

with col4:
    st.markdown("""
    <div style="
        background-color: #eafaf1;
        border: 2px solid #2ecc71;
        border-radius: 12px;
        padding: 32px 20px 16px 20px;
        text-align: center;
        min-height: 190px;
    ">
        <div style="font-size: 2.8rem; margin-bottom: 10px;">📈</div>
        <div style="font-weight: 700; font-size: 1.1rem;
                    margin-bottom: 6px;">Logs & Analytics</div>
        <div style="font-size: 0.88rem; color: #555;">
            Monitor tool usage, booking success rates and performance
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Logs & Analytics →", key="d4",
                 width='stretch', type="primary"):
        st.switch_page("pages/7_Logs_Analytics.py")

st.markdown(" ")

# ── Row 3: Chat Assistant centred ─────────────────────────────
col5, col_mid, col6 = st.columns([1, 2, 1])

with col_mid:
    st.markdown("""
    <div style="
        background-color: #eafaf1;
        border: 2px solid #2ecc71;
        border-radius: 12px;
        padding: 32px 20px 16px 20px;
        text-align: center;
        min-height: 190px;
    ">
        <div style="font-size: 2.8rem; margin-bottom: 10px;">💬</div>
        <div style="font-weight: 700; font-size: 1.1rem;
                    margin-bottom: 6px;">Chat Assistant</div>
        <div style="font-size: 0.88rem; color: #555;">
            Natural language AI agent for any healthcare task
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Chat Assistant →", key="d5",
                 width='stretch', type="primary"):
        st.switch_page("pages/8_Chat_Assistant.py")
