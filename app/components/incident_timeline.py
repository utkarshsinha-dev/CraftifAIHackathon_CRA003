# -*- coding: utf-8 -*-
"""
Sentinel Vision AI 2.0 — Incident Timeline Component
Chronological incident auditing, event inspection, and frame jumping.
"""

from typing import Dict, List, Any, Optional
import streamlit as st


def render_incident_timeline_module(incidents: List[Dict[str, Any]]) -> None:
    """Renders the chronological incident timeline with interactive inspection."""
    st.subheader("⏱️ Incident Chronology & Alert Timeline")

    if not incidents:
        st.info("No recorded safety incidents available.")
        return

    # Sort chronologically by frame_index
    sorted_incidents = sorted(incidents, key=lambda x: x.get("frame_index", 0))

    st.markdown(
        f"""
        <div style="font-size: 11px; color: #94a3b8; margin-bottom: 12px; font-family: 'JetBrains Mono', monospace;">
          Showing <strong>{len(sorted_incidents)}</strong> verified safety events throttled via anti-spam cooldown state machine.
        </div>
        """,
        unsafe_allow_html=True,
    )

    for idx, inc in enumerate(sorted_incidents):
        frame_idx = inc.get("frame_index", 0)
        time_sec = frame_idx / 30.0016
        track_id = inc.get("track_id", "N/A")
        event_type = inc.get("event_type", "UNKNOWN")
        severity = inc.get("severity", "CRITICAL")
        zone_name = inc.get("zone_name", "Unknown Zone")
        conf = inc.get("confidence", 0.0)
        footprint = inc.get("ground_contact_point", [0, 0])

        # Badge styling classes
        sev_class = "badge-critical" if severity == "CRITICAL" else "badge-warning"
        event_class = "badge-entry" if "ENTRY" in event_type else "badge-occupancy"

        col_time, col_card, col_action = st.columns([1, 4, 1])

        with col_time:
            st.markdown(
                f"""
                <div style="text-align: right; font-family: 'JetBrains Mono', monospace; font-size: 11px; padding-top: 10px;">
                  <strong style="color: #38bdf8;">t={time_sec:.2f}s</strong><br>
                  <span style="color: #64748b;">F#{frame_idx}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_card:
            with st.expander(
                f"🚨 {event_type} — Person #{track_id} in {zone_name} [{severity}]",
                expanded=(idx == 0),
            ):
                st.markdown(
                    f"""
                    <div style="margin-bottom: 8px;">
                      <span class="badge {sev_class}">{severity}</span>
                      <span class="badge {event_class}">{event_type}</span>
                      <span class="badge" style="background:#1e293b; color:#94a3b8;">Person #{track_id}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.caption("Zone Identifier")
                    st.write(f"`{inc.get('zone_id', 'N/A')}`")
                with c2:
                    st.caption("Detection Confidence")
                    st.write(f"**{conf * 100:.1f}%**")
                with c3:
                    st.caption("Ground Contact Datum")
                    st.write(f"({footprint[0]:.4f}, {footprint[1]:.4f})")

                st.caption("Raw JSON Record")
                st.json(inc)

        with col_action:
            if st.button("Jump to F#", key=f"jump_{idx}"):
                st.session_state["scrubber_slider"] = frame_idx
                st.rerun()
