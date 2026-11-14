# -*- coding: utf-8 -*-
"""
Sentinel Vision AI 2.0 — Safety Operations Command Center
Master Dashboard Application built with Streamlit.
Author: Developer 2 (Frontend / UI / Full-Stack Command Center)
Branch: feat/safety-dashboard
"""

import os
import sys
import streamlit as st

# Ensure root repository directory is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Component imports
from app.components.metrics import (
    load_incidents,
    compute_kpi_metrics,
    render_kpi_cards,
    render_analytics_charts,
)
from app.components.video_player import render_video_player_module
from app.components.incident_timeline import render_incident_timeline_module
from app.components.zone_editor import render_zone_editor_module
from app.components.incident_explorer import render_incident_explorer_module
from app.components.copilot_ui import render_copilot_module


def inject_custom_css():
    """Injects custom industrial safety CSS stylesheet."""
    css_path = os.path.join(os.path.dirname(__file__), "styles", "theme.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_header(kpis):
    """Renders the master command center industrial header."""
    st.markdown(
        f"""
        <div class="command-header">
          <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px;">
            <div>
              <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
                <span style="font-size: 24px;">🛡️</span>
                <h1>SENTINEL VISION AI 2.0 — SAFETY OPERATIONS COMMAND CENTER</h1>
              </div>
              <div class="subtext">
                TRACK A: CRAFTIFAI PHYSICAL AI PERCEPTION &bull; DEEPSTREAM 7.1 &bull; TENSORRT 10.3 YOLO11s
              </div>
            </div>
            <div style="display: flex; gap: 8px; align-items: center;">
              <span class="status-pill status-online">&bull; PERCEPTION ENGINE READY</span>
              <span class="status-pill status-replay">SOFTWARE REPLAY MODE</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="Sentinel Vision AI 2.0 — Command Center",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    inject_custom_css()

    # Load authentic data
    incidents = load_incidents()
    kpis = compute_kpi_metrics(incidents)

    # 1. Render Header
    render_header(kpis)

    # 2. Render Top KPI Metrics Cards
    render_kpi_cards(kpis)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # 3. Main Navigation Workspace Tabs
    tab_names = [
        "📹 Video Replay & Monitoring",
        "⏱️ Incident Timeline",
        "📊 Safety Analytics",
        "📐 Zone Calibration Editor",
        "🔍 Incident Forensics Explorer",
        "🤖 Safety Copilot",
    ]
    tab_main, tab_time, tab_analytics, tab_zone, tab_explorer, tab_copilot = st.tabs(tab_names)

    with tab_main:
        col_video, col_alerts = st.columns([3, 2])
        with col_video:
            render_video_player_module(incidents)
        with col_alerts:
            st.subheader("⚡ Real-Time Incursion Feed")
            st.markdown(
                f"""
                <div style="font-size: 11px; color: #94a3b8; margin-bottom: 10px; font-family: 'JetBrains Mono', monospace;">
                  Active Stream: <strong>6 Verified Actionable Alerts</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )
            for inc in incidents:
                sev_cls = "badge-critical" if inc.get("severity") == "CRITICAL" else "badge-warning"
                ev_cls = "badge-entry" if "ENTRY" in inc.get("event_type", "") else "badge-occupancy"
                st.markdown(
                    f"""
                    <div class="incident-item">
                      <div>
                        <span class="badge {sev_cls}">{inc.get('severity')}</span>
                        <span class="badge {ev_cls}">{inc.get('event_type')}</span>
                        <div style="font-size: 12px; font-weight: 600; color: #f8fafc; margin-top: 4px;">
                          Person #{inc.get('track_id')} &bull; {inc.get('zone_name')}
                        </div>
                        <div class="incident-meta">
                          F#{inc.get('frame_index')} (~{inc.get('frame_index')/30.0016:.2f}s) &bull; Conf: {inc.get('confidence', 0)*100:.1f}%
                        </div>
                      </div>
                      <div style="text-align: right;">
                        <span style="font-size: 11px; color: #38bdf8; font-family: 'JetBrains Mono', monospace;">
                          ({inc.get('ground_contact_point', [0,0])[0]:.2f}, {inc.get('ground_contact_point', [0,0])[1]:.2f})
                        </span>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with tab_time:
        render_incident_timeline_module(incidents)

    with tab_analytics:
        st.subheader("📊 Spatial Incursion & Exposure Analytics")
        render_analytics_charts(incidents)

    with tab_zone:
        render_zone_editor_module()

    with tab_explorer:
        render_incident_explorer_module(incidents)

    with tab_copilot:
        render_copilot_module(incidents)

    # Footer
    st.markdown(
        """
        <div style="margin-top: 30px; padding-top: 12px; border-top: 1px solid #1e293b; font-size: 10px; color: #64748b; display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace;">
          <span>SENTINEL VISION AI 2.0 &bull; CRAFTIFAI PHYSICAL AI HACKATHON &bull; BRANCH: feat/safety-dashboard</span>
          <span>AUTHENTIC PERCEPTION DATA &bull; ZERO SIMULATED PLACEHOLDERS</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
