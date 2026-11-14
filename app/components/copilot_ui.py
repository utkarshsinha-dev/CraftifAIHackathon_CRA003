# -*- coding: utf-8 -*-
"""
Sentinel Vision — Safety Copilot UI Component
Conversational EHS Copilot chat interface designed to integrate seamlessly
with the backend incident intelligence engine via BackendAdapter.
"""

from typing import Dict, List, Any, Optional
import streamlit as st

from app.components.backend_adapter import backend_adapter


def render_copilot_module(incidents: List[Dict[str, Any]]) -> None:
    """Renders the conversational enterprise Safety Copilot chat interface."""
    st.subheader("🤖 Sentinel Safety Copilot")

    # Detect backend readiness
    backend_status = backend_adapter.get_copilot_status()
    is_real_ai = (backend_status["status"] == "AVAILABLE" and backend_status["has_answer_question"])

    if is_real_ai:
        st.markdown(
            """
            <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 6px; padding: 7px 14px; margin-bottom: 14px; font-size: 11.5px; color: #047857; font-family: 'JetBrains Mono', monospace; display: flex; align-items: center; gap: 8px;">
              <span class="pulse-emerald"></span>
              &bull; BACKEND SAFETY COPILOT CONNECTED (src/incident_service.py &bull; 100% Offline Perception Grounding)
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.25); border-radius: 6px; padding: 7px 14px; margin-bottom: 14px; font-size: 11.5px; color: #b45309; font-family: 'JetBrains Mono', monospace;">
              &bull; DETERMINISTIC FALLBACK MODE (Local Rule-Based Grounding)
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Initialize message history
    if "copilot_messages" not in st.session_state:
        st.session_state["copilot_messages"] = [
            {
                "role": "assistant",
                "content": (
                    "**Welcome to Sentinel Safety Copilot.**\n\n"
                    "I am directly connected to the backend incident intelligence engine. "
                    "I can assist you with perimeter security audits, worker exposure dwell analysis, "
                    "or incident forensics over the active DeepStream perception stream. Ask a question or select a prompt below."
                ),
                "mode": "SYSTEM",
                "evidence": [],
            }
        ]

    # Preset Suggested Questions
    st.markdown("**Suggested Safety Queries:**")
    pill_col1, pill_col2, pill_col3 = st.columns(3)
    with pill_col1:
        if st.button("📋 Summarize all incidents", key="pill_sum"):
            st.session_state["pending_prompt"] = "Summarize all safety incidents"
        if st.button("🚨 Show critical violations", key="pill_crit"):
            st.session_state["pending_prompt"] = "Show critical violations"
    with pill_col2:
        if st.button("⏱️ Longest zone occupancy", key="pill_occ"):
            st.session_state["pending_prompt"] = "Which worker stayed the longest?"
        if st.button("⚠️ Highest-risk incident", key="pill_risk"):
            st.session_state["pending_prompt"] = "Which incident has the highest risk?"
    with pill_col3:
        if st.button("📍 Summarize Zone 1 events", key="pill_z1"):
            st.session_state["pending_prompt"] = "Summarize events in Zone 1"
        if st.button("🛡️ Recommended CAPA measures", key="pill_capa"):
            st.session_state["pending_prompt"] = "What corrective actions could help prevent similar incidents?"

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Display conversation history
    for msg in st.session_state["copilot_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Provenance badge & evidence references
            mode = msg.get("mode", "")
            evidence = msg.get("evidence", [])

            if mode and mode != "SYSTEM":
                badge_color = "#047857" if "BACKEND" in str(mode) else ("#0284c7" if "ADVISORY" in str(mode) else "#b45309")
                st.markdown(
                    f"""
                    <div style="font-size: 10.5px; color: {badge_color}; font-weight: 600; font-family: 'JetBrains Mono', monospace; margin-top: 4px;">
                      MODE: {mode}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if evidence:
                st.caption(f"Referenced Incident IDs: `{', '.join([str(e)[:8] + '...' for e in evidence[:3]])}`")

    # Chat Input Box
    user_input = st.chat_input("Ask a question regarding safety incidents, workers, or zone violations...")
    prompt_to_send = user_input or st.session_state.pop("pending_prompt", None)

    if prompt_to_send:
        # Append User Message
        st.session_state["copilot_messages"].append({
            "role": "user",
            "content": prompt_to_send,
            "mode": "USER",
            "evidence": [],
        })
        with st.chat_message("user"):
            st.markdown(prompt_to_send)

        # Assistant Processing with Loading Spinner
        with st.spinner("Evaluating perception stream and spatial incident records..."):
            result = backend_adapter.answer_question(prompt_to_send, incidents)

        st.session_state["copilot_messages"].append({
            "role": "assistant",
            "content": result["text"],
            "mode": result["mode"],
            "evidence": result.get("evidence", []),
        })

        with st.chat_message("assistant"):
            st.markdown(result["text"])
            badge_color = "#047857" if "BACKEND" in str(result['mode']) else ("#0284c7" if "ADVISORY" in str(result['mode']) else "#b45309")
            st.markdown(
                f"""
                <div style="font-size: 10.5px; color: {badge_color}; font-weight: 600; font-family: 'JetBrains Mono', monospace; margin-top: 4px;">
                  MODE: {result['mode']}
                </div>
                """,
                unsafe_allow_html=True,
            )
            if result.get("evidence"):
                st.caption(f"Referenced Incident IDs: `{', '.join([str(e)[:8] + '...' for e in result['evidence'][:3]])}`")

        st.rerun()
