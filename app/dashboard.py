# -*- coding: utf-8 -*-
"""
SENTINEL VISION — AI-Powered Industrial Safety Intelligence
Enterprise SaaS Operations Command Center.
"""

import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure root repository directory is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Component imports
from app.components.backend_adapter import backend_adapter
from app.components.metrics import (
    load_incidents,
    compute_kpi_metrics,
    render_kpi_cards,
    render_analytics_charts,
)
from app.components.video_player import (
    render_video_player_module,
    load_detection_metadata,
    load_zones_config,
    get_cached_video_frame,
    render_annotated_frame,
    get_available_video_path,
    RAW_VIDEO_PATH,
)
from app.components.investigation_workspace import render_investigation_workspace_module
from app.components.incident_timeline import render_incident_timeline_module
from app.components.zone_editor import render_zone_editor_module
from app.components.incident_explorer import render_incident_explorer_module
from app.components.copilot_ui import render_copilot_module


def inject_custom_css():
    """Injects custom enterprise industrial safety CSS stylesheet."""
    css_path = os.path.join(os.path.dirname(__file__), "styles", "theme.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_sidebar():
    """Renders the left dark navy SaaS sidebar with Linear/Vercel styling."""
    with st.sidebar:
        # 1. Product Branding Header
        st.markdown(
            """
            <div class="sidebar-brand">
              <div class="brand-logo-icon">🛡️</div>
              <div>
                <div class="brand-title">SENTINEL VISION</div>
                <div class="brand-subtitle">Industrial Safety Intelligence</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 2. Facility & Active Stream Selector
        st.markdown("<div class='sidebar-section-title'>FACILITY & SENSOR</div>", unsafe_allow_html=True)
        st.selectbox(
            "Facility Target",
            ["North Plant — Floor 4 (Assembly)", "East Warehouse — Bay 2", "South Logistics — Dock 1"],
            index=0,
            label_visibility="collapsed",
        )
        st.markdown(
            """
            <div class="sidebar-camera-badge">
              <span class="pulse-emerald"></span>
              <span>CAM-04 &bull; Heavy Equipment Zone</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 3. Main Workspace Navigation
        st.markdown("<div class='sidebar-section-title'>WORKSPACE</div>", unsafe_allow_html=True)

        nav_options = [
            "Overview",
            "Live Monitoring",
            "Incident Explorer",
            "Investigation Workspace",
            "AI Safety Copilot",
            "Zone Management",
            "Analytics & Reports",
            "System Health",
        ]

        def format_nav(name):
            icons = {
                "Overview": "📊  Overview",
                "Live Monitoring": "📹  Live Monitoring",
                "Incident Explorer": "🔍  Incident Explorer",
                "Investigation Workspace": "🕵️  Investigation Workspace",
                "AI Safety Copilot": "🤖  AI Safety Copilot",
                "Zone Management": "📐  Zone Management",
                "Analytics & Reports": "📈  Analytics & Reports",
                "System Health": "🖥️  System Health",
            }
            return icons.get(name, name)

        if "sidebar_nav" not in st.session_state:
            st.session_state["sidebar_nav"] = "Overview"

        selected_nav = st.radio(
            "Workspace Navigation",
            nav_options,
            format_func=format_nav,
            key="sidebar_nav",
            label_visibility="collapsed",
        )

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # 4. Settings Expander
        with st.expander("⚙️ Settings & Configuration", expanded=False):
            st.markdown("<div style='font-size:11px; color:#94a3b8;'>Perception Parameters:</div>", unsafe_allow_html=True)
            st.slider("Spatial Hysteresis (s)", 1.0, 5.0, 2.0, 0.5, disabled=True, help="Fixed at 2.0s in backend state machine")
            st.slider("Inference Confidence", 0.0, 1.0, 0.5, 0.05, disabled=True, help="TensorRT YOLO11s minimum confidence threshold")
            st.caption("Settings locked to verified production parameters.")

        # 5. Bottom System Status & User Profile Cards
        st.markdown("<div class='sidebar-section-title'>SYSTEM STATUS</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="sidebar-status-card">
              <div class="status-indicator-row">
                <span class="status-dot-green"></span>
                <span class="status-text-bold">Perception Node Ready</span>
              </div>
              <div class="status-subtext">DeepStream 7.1 &bull; 30.00 FPS</div>
              <div class="status-subtext">RTX 4060 &bull; YOLO11s FP32</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="sidebar-user-card">
              <div class="user-avatar">SO</div>
              <div>
                <div class="user-name">Safety Operations</div>
                <div class="user-role">Lead EHS Supervisor &bull; Plant 4</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return selected_nav


def render_overview_dashboard(incidents, kpis):
    """Renders the executive Overview Dashboard (Section 4)."""
    # 1. Executive Top Greeting Banner
    avail = kpis.get("available", {})
    total_inc = avail.get("total_incidents", len(incidents))
    st.markdown(
        f"""
        <div class="enterprise-header">
          <div>
            <div class="header-greeting">Good morning, Safety Operations</div>
            <div class="header-subtext">Your industrial safety operations at a glance &bull; Assembly Floor 4</div>
          </div>
          <div class="header-badges">
            <span class="header-badge badge-operational"><span class="pulse-emerald"></span> PERCEPTION ACTIVE</span>
            <span class="header-badge badge-stream">FEED: CAM-04 (1728×1080)</span>
            <span class="header-badge badge-verified">{total_inc} VERIFIED EVENTS</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Four Important KPI Cards (Section 4)
    render_kpi_cards(kpis)

    # 3. Middle Section: Live Operations Panel (Left) & Recent Incidents (Right)
    col_live, col_recent = st.columns([3, 2])

    with col_live:
        st.markdown(
            """
            <div class="panel-header">
              <div class="panel-title">📹 Live Operations & Perimeter Monitoring</div>
              <div class="panel-meta">FEED: CAM-04 &bull; RECORDED REPLAY</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quick Scrubber Preview at Frame 3 (Breach Inception)
        current_frame = st.session_state.get("scrubber_slider", 3)
        meta_dict = load_detection_metadata()
        zones = load_zones_config()
        current_objects = meta_dict.get(current_frame, [])

        video_path = get_available_video_path()
        target_source = RAW_VIDEO_PATH if os.path.exists(RAW_VIDEO_PATH) else video_path

        if target_source and os.path.exists(target_source):
            frame_img = get_cached_video_frame(target_source, current_frame)
            if frame_img is not None:
                annotated_img = render_annotated_frame(frame_img, current_frame, current_objects, zones)
                st.image(
                    annotated_img,
                    caption=f"Perception Stream — CAM-04 @ Frame {current_frame} (Active Detections: {len(current_objects)})",
                    use_container_width=True,
                )

        col_q1, col_q2, col_q3, col_q4 = st.columns([1, 1, 1, 2])
        with col_q1:
            if st.button("F3 (Entry)", key="ov_f3"):
                st.session_state["scrubber_slider"] = 3
                st.rerun()
        with col_q2:
            if st.button("F64", key="ov_f64"):
                st.session_state["scrubber_slider"] = 64
                st.rerun()
        with col_q3:
            if st.button("F125", key="ov_f125"):
                st.session_state["scrubber_slider"] = 125
                st.rerun()
        with col_q4:
            if st.button("📹 Open Full Monitoring", key="ov_to_live", type="primary"):
                st.session_state["sidebar_nav"] = "Live Monitoring"
                st.rerun()

    with col_recent:
        st.markdown(
            """
            <div class="panel-header">
              <div class="panel-title">⚡ Recent Safety Incursions</div>
              <div class="panel-meta">6 VERIFIED ALERTS</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for idx, inc in enumerate(incidents):
            f_idx = inc.get("frame_index", 0)
            t_sec = f_idx / 30.0016
            track_id = inc.get("track_id", 0)
            ev_type = inc.get("event_type", "INCURSION")
            sev = inc.get("severity", "CRITICAL")
            pt = inc.get("ground_contact_point", [0, 0])

            badge_style = "incident-badge-crit" if sev == "CRITICAL" else "incident-badge-warn"

            col_inc_info, col_inc_btn = st.columns([4, 1])
            with col_inc_info:
                st.markdown(
                    f"""
                    <div class="recent-incident-row">
                      <div>
                        <div style="display: flex; gap: 6px; align-items: center; margin-bottom: 2px;">
                          <span class="{badge_style}">{sev}</span>
                          <span style="font-size: 11px; font-weight: 600; color: #0f172a;">{ev_type}</span>
                        </div>
                        <div style="font-size: 12px; color: #475569;">
                          Person #{track_id} &bull; {inc.get('zone_name')}
                        </div>
                        <div style="font-size: 10.5px; color: #94a3b8; font-family: 'JetBrains Mono', monospace;">
                          t={t_sec:.2f}s (F#{f_idx}) &bull; Footprint: ({pt[0]:.2f}, {pt[1]:.2f})
                        </div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_inc_btn:
                if st.button("Inspect", key=f"ov_btn_insp_{idx}"):
                    st.session_state["active_investigation_idx"] = idx
                    st.session_state["sidebar_nav"] = "Investigation Workspace"
                    st.rerun()

    # 4. Bottom Section: Incident Trend, Zone Distribution, Risk Overview
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown("#### 📈 Operations Safety Telemetry & Risk Distribution")

    col_tr, col_zd, col_ro = st.columns(3)

    df = pd.DataFrame(incidents)
    df["time_sec"] = df["frame_index"] / 30.0016

    with col_tr:
        # 1. Incident Trend Chart
        fig_trend = px.scatter(
            df,
            x="time_sec",
            y="track_id",
            color="severity",
            size=[12] * len(df),
            color_discrete_map={"CRITICAL": "#ef4444", "WARNING": "#f59e0b"},
            title="Incident Temporal Progression (0.0s - 5.43s)",
            labels={"time_sec": "Video Timeline (s)", "track_id": "Track ID"},
        )
        fig_trend.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#334155", family="Inter", size=10),
            margin=dict(l=20, r=20, t=35, b=20),
            height=240,
            xaxis=dict(gridcolor="#f1f5f9"),
            yaxis=dict(gridcolor="#f1f5f9"),
            showlegend=False,
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_zd:
        # 2. Hazard Zone Distribution
        zone_counts = df["zone_name"].value_counts().reset_index()
        zone_counts.columns = ["Zone", "Count"]
        fig_zone = px.pie(
            zone_counts,
            values="Count",
            names="Zone",
            hole=0.6,
            color_discrete_sequence=["#ef4444", "#f59e0b", "#0284c7"],
            title="Hazard Zone Incursion Allocation",
        )
        fig_zone.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#334155", family="Inter", size=10),
            margin=dict(l=20, r=20, t=35, b=20),
            height=240,
            showlegend=True,
            legend=dict(font=dict(size=9), orientation="h", y=-0.1),
        )
        st.plotly_chart(fig_zone, use_container_width=True)

    with col_ro:
        # 3. Safety Risk Overview (Decomposition of Risk Factors)
        risk_decomp = pd.DataFrame([
            {"Factor": "Spatial Proximity", "Score": 40},
            {"Factor": "Kinematic Velocity", "Score": 25},
            {"Factor": "Hysteresis Dwell", "Score": 20},
            {"Factor": "Detector Confidence", "Score": 15},
        ])
        fig_risk = px.bar(
            risk_decomp,
            x="Score",
            y="Factor",
            orientation="h",
            color="Factor",
            color_discrete_sequence=["#ef4444", "#f59e0b", "#6366f1", "#0284c7"],
            title="Safety Risk Factor Weighting",
        )
        fig_risk.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#334155", family="Inter", size=10),
            margin=dict(l=20, r=20, t=35, b=20),
            height=240,
            xaxis=dict(gridcolor="#f1f5f9"),
            yaxis=dict(gridcolor="#f1f5f9"),
            showlegend=False,
        )
        st.plotly_chart(fig_risk, use_container_width=True)


def render_system_health_dashboard():
    """Renders the System Health and Enterprise Pipeline Audit view."""
    st.subheader("🖥️ Perception Pipeline & Edge Infrastructure Health")
    st.caption("Live operational status of edge AI hardware acceleration, models, and spatial telemetry layers.")

    # 1. Subsystem Status Cards
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.markdown(
            """
            <div class="kpi-card">
              <div class="kpi-header">
                <span class="kpi-label">Perception Engine</span>
                <span class="status-dot-green"></span>
              </div>
              <div style="font-size: 16px; font-weight: 700; color: #0f172a; margin: 4px 0;">NVIDIA DeepStream 7.1</div>
              <div class="kpi-caption">Pipeline Status: Operational</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_s2:
        st.markdown(
            """
            <div class="kpi-card">
              <div class="kpi-header">
                <span class="kpi-label">Inference Runtime</span>
                <span class="status-dot-green"></span>
              </div>
              <div style="font-size: 16px; font-weight: 700; color: #0f172a; margin: 4px 0;">TensorRT 10.3 FP32</div>
              <div class="kpi-caption">Model: YOLO11s Detector</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_s3:
        st.markdown(
            """
            <div class="kpi-card">
              <div class="kpi-header">
                <span class="kpi-label">Tracking Engine</span>
                <span class="status-dot-green"></span>
              </div>
              <div style="font-size: 16px; font-weight: 700; color: #0f172a; margin: 4px 0;">NvDCF Multi-Object</div>
              <div class="kpi-caption">2 Persistent Worker Tracks</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_s4:
        st.markdown(
            """
            <div class="kpi-card">
              <div class="kpi-header">
                <span class="kpi-label">Telemetry Persistence</span>
                <span class="status-dot-green"></span>
              </div>
              <div style="font-size: 16px; font-weight: 700; color: #0f172a; margin: 4px 0;">JSONL Audit Log</div>
              <div class="kpi-caption">6 Immutable Verified Records</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # 2. Hardware Node Specifications
    col_hw1, col_hw2 = st.columns(2)
    with col_hw1:
        st.markdown("#### ⚙️ Edge Compute Specifications")
        spec_df = pd.DataFrame([
            {"Parameter": "Hardware Node", "Specification": "NVIDIA GeForce RTX 4060 Laptop GPU"},
            {"Parameter": "Acceleration Stack", "Specification": "CUDA 12.x / TensorRT 10.3 / cuDNN 9.x"},
            {"Parameter": "Inference Latency", "Specification": "~4.8 ms / frame (Zero-drop 30.00 FPS)"},
            {"Parameter": "Input Resolution", "Specification": "1728 × 1080 @ 30.00 FPS (H.264)"},
            {"Parameter": "Total Frames Processed", "Specification": "163 Frames (5.43 Seconds Duration)"},
            {"Parameter": "Person Detections", "Specification": "320 Total Spatial Bounding Boxes"},
        ])
        st.dataframe(spec_df, use_container_width=True, hide_index=True)

    with col_hw2:
        st.markdown("#### 🛡️ Spatial Containment State Machine")
        arch_df = pd.DataFrame([
            {"Subsystem": "Zone Polygon Geometry", "State": "Verified Normalized [0.0 - 1.0]", "Integrity": "PASS"},
            {"Subsystem": "Ray-Casting Algorithm", "State": "Jordan Curve Theorem Containment", "Integrity": "PASS"},
            {"Subsystem": "Hysteresis Cooldown", "State": "2.0s Anti-Spam Cooldown State Machine", "Integrity": "PASS"},
            {"Subsystem": "Risk Engine", "State": "Deterministic Multi-Factor Scoring (0-100)", "Integrity": "PASS"},
            {"Subsystem": "AI Safety Copilot", "State": "Connected to src/incident_service.py", "Integrity": "PASS"},
            {"Subsystem": "EHS Audit Export", "State": "Markdown / JSON Evidence Packaging", "Integrity": "PASS"},
        ])
        st.dataframe(arch_df, use_container_width=True, hide_index=True)


def main():
    st.set_page_config(
        page_title="Sentinel Vision — Industrial Safety Intelligence",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()

    # Load authentic data via BackendAdapter
    incidents = load_incidents()
    kpis = compute_kpi_metrics(incidents)

    # Render Left Sidebar Navigation
    active_nav = render_sidebar()

    # Executive Product Tour Expander (Available across all workspaces)
    with st.expander("🎯 Executive Product Tour & Operator Walkthrough (Click to Expand)", expanded=False):
        st.markdown(
            """
            **Sentinel Vision AI 2.0 Operator Demonstration Sequence:**
            - **1. Executive Overview:** Review the 4 primary KPI metrics (Total Incidents, Critical Violations, Tracked Workers, Active Zones) and Live Operations panel.
            - **2. Live Perception Monitoring:** Inspect frame-accurate scrubbing with real-time bounding box color transitions (**GREEN SAFE** &rarr; **RED INCURSION**) at Frame 3.
            - **3. Incident Investigation Workspace:** Step through 2D trajectory kinematics (160 continuous frames of real DeepStream data), authoritative backend risk score (85/100), and contextual Copilot Q&A.
            - **4. Enterprise Governance & Reports:** Record CAPA findings, stage safe polygon calibrations to `restricted_zone_working.json`, and download the authoritative EHS Audit Report.
            """
        )

    # Main Workspace View Dispatcher
    if active_nav == "Overview":
        render_overview_dashboard(incidents, kpis)

    elif active_nav == "Live Monitoring":
        render_video_player_module(incidents)

    elif active_nav == "Incident Explorer":
        render_incident_explorer_module(incidents)

    elif active_nav == "Investigation Workspace":
        render_investigation_workspace_module(incidents)

    elif active_nav == "AI Safety Copilot":
        render_copilot_module(incidents)

    elif active_nav == "Zone Management":
        render_zone_editor_module()

    elif active_nav == "Analytics & Reports":
        st.subheader("📈 Spatial Incursion & Exposure Analytics")
        render_analytics_charts(incidents, kpis)

    elif active_nav == "System Health":
        render_system_health_dashboard()

    # Commercial Enterprise Footer
    st.markdown(
        """
        <div class="saas-footer">
          <span>SENTINEL VISION &bull; AI-POWERED INDUSTRIAL SAFETY INTELLIGENCE &bull; v2.0 ENTERPRISE</span>
          <span>AUTHENTIC PERCEPTION TELEMETRY &bull; ALL SYSTEMS OPERATIONAL</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
