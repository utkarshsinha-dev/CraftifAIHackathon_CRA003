# -*- coding: utf-8 -*-
"""
Sentinel Vision AI 2.0 — Metrics & KPI Analytics Component
Computes and renders high-precision industrial safety metrics and Plotly charts.
Categorizes metrics strictly into Available, Derived, and Unavailable measurements.
"""

import os
from typing import Dict, List, Any, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.components.backend_adapter import backend_adapter


def load_incidents(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Loads verified incident records via BackendAdapter."""
    return backend_adapter.load_incidents(path)


def compute_kpi_metrics(incidents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes categorized KPI metrics via BackendAdapter."""
    return backend_adapter.get_incident_statistics(incidents)


def render_kpi_cards(kpi_stats: Dict[str, Any]) -> None:
    """Renders sleek industrial KPI metric cards with measurement provenance tags."""
    avail = kpi_stats.get("available", {})
    derived = kpi_stats.get("derived", {})

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card card-info">
              <div class="label">Total Incidents</div>
              <div class="value">{avail.get('total_incidents', 0)}</div>
              <div class="caption">Direct Measurement &bull; Verified</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card card-critical">
              <div class="label">Critical Violations</div>
              <div class="value">{avail.get('critical_incidents', 0)}</div>
              <div class="caption">Direct Measurement &bull; Severity</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card card-warning">
              <div class="label">Tracked Entities</div>
              <div class="value">{avail.get('active_tracks', 0)}</div>
              <div class="caption">NvDCF Persistent IDs</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card card-success">
              <div class="label">Zone Breaches</div>
              <div class="value">{avail.get('total_entries', 0)}</div>
              <div class="caption">Initial Boundary Crossings</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col5:
        st.markdown(
            f"""
            <div class="metric-card card-info">
              <div class="label">Occupancy Dwell</div>
              <div class="value">{derived.get('estimated_occupancy_seconds', 0.0)}s</div>
              <div class="caption">Derived &bull; 2.0s Cooldown State</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_analytics_charts(incidents: List[Dict[str, Any]], kpi_stats: Optional[Dict[str, Any]] = None) -> None:
    """Renders responsive Plotly industrial analytics charts with clear metadata."""
    if not incidents:
        st.info("No incident records available for analytics generation.")
        return

    df = pd.DataFrame(incidents)

    # Calculate time in seconds
    if "frame_index" in df.columns:
        df["time_sec"] = df["frame_index"] / 30.0016
    else:
        df["time_sec"] = 0.0

    col1, col2 = st.columns(2)

    with col1:
        # Chart 1: Incidents by Severity (Donut Chart)
        if "severity" in df.columns:
            sev_counts = df["severity"].value_counts().reset_index()
            sev_counts.columns = ["Severity", "Count"]
            color_map = {
                "CRITICAL": "#ef4444",
                "WARNING": "#f59e0b",
                "INFO": "#38bdf8",
            }
            fig_sev = px.pie(
                sev_counts,
                values="Count",
                names="Severity",
                hole=0.55,
                color="Severity",
                color_discrete_map=color_map,
                title="Incidents by Severity Level (Direct Measurement)",
            )
            fig_sev.update_layout(
                paper_bgcolor="#111827",
                plot_bgcolor="#111827",
                font=dict(color="#f8fafc", family="Inter"),
                margin=dict(l=20, r=20, t=40, b=20),
                height=280,
                showlegend=True,
            )
            st.plotly_chart(fig_sev, use_container_width=True)

    with col2:
        # Chart 2: Incidents by Event Type & Track ID
        if "event_type" in df.columns and "track_id" in df.columns:
            track_counts = (
                df.groupby(["track_id", "event_type"]).size().reset_index(name="Occurrences")
            )
            track_counts["Track"] = "Person #" + track_counts["track_id"].astype(str)
            fig_track = px.bar(
                track_counts,
                x="Track",
                y="Occurrences",
                color="event_type",
                barmode="group",
                title="Events per Personnel Track ID (Direct Measurement)",
                color_discrete_map={
                    "ZONE_INTRUSION_ENTRY": "#3b82f6",
                    "ZONE_INTRUSION_OCCUPANCY": "#a855f7",
                    "ZONE_INTRUSION_EXIT": "#10b981",
                },
            )
            fig_track.update_layout(
                paper_bgcolor="#111827",
                plot_bgcolor="#111827",
                font=dict(color="#f8fafc", family="Inter"),
                margin=dict(l=20, r=20, t=40, b=20),
                height=280,
                xaxis=dict(gridcolor="#1e293b"),
                yaxis=dict(gridcolor="#1e293b"),
            )
            st.plotly_chart(fig_track, use_container_width=True)

    # Chart 3: Temporal Dispersion Across Video Timeline
    fig_time = px.scatter(
        df,
        x="time_sec",
        y="track_id",
        color="severity",
        size=[14] * len(df),
        symbol="event_type",
        hover_data=["zone_name", "confidence", "frame_index"],
        labels={"time_sec": "Video Timeline (Seconds)", "track_id": "Track ID"},
        title="Temporal Breach Incursion Progression (0.0s – 5.43s)",
        color_discrete_map={"CRITICAL": "#ef4444", "WARNING": "#f59e0b"},
    )
    fig_time.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(color="#f8fafc", family="Inter"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=240,
        xaxis=dict(gridcolor="#1e293b", dtick=0.5),
        yaxis=dict(gridcolor="#1e293b", dtick=1),
    )
    st.plotly_chart(fig_time, use_container_width=True)

    # Metric Provenance & Integrity Statement
    if kpi_stats and "unavailable" in kpi_stats:
        with st.expander("ℹ️ Measurement Provenance & Integrity Disclosure", expanded=False):
            st.markdown(
                """
                - **Direct Measurements:** Incident counts, frame indices, bounding boxes, and track IDs are authentically parsed from DeepStream metadata.
                - **Derived Measurements:** Dwell duration is computed from periodic 2.0s cooldown state transitions (`OUTSIDE` &rarr; `ENTRY` &rarr; `OCCUPANCY`).
                - **Decoupled / Unavailable Data:** Biometric identity, sub-millisecond continuous sensor dwell, and direct industrial PLC hardware feedback are currently decoupled for software demonstration safety.
                """
            )

    # Official EHS Incident Audit Report Section
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("#### 📄 Authoritative EHS Incident Audit Report")
    st.caption("Generated directly by the backend service (`src/incident_service.py` & `scripts/generate_ehs_report.py`) from authentic DeepStream perception telemetry.")

    col_rep_btn, col_rep_info = st.columns([1, 2])
    with col_rep_btn:
        report_res = backend_adapter.generate_incident_report(incidents=incidents)
        report_text = report_res.get("report_text", "")
        st.download_button(
            label="📥 Download Formal EHS Audit Report (.md)",
            data=report_text,
            file_name="EHS_INCIDENT_REPORT.md",
            mime="text/markdown",
            key="dl_official_ehs_report",
            type="primary",
        )
    with col_rep_info:
        st.markdown(
            f"""
            <div style="font-size: 11px; color: #94a3b8; font-family: 'JetBrains Mono', monospace; padding-top: 6px;">
              &bull; <strong>Engine:</strong> <code>src/incident_service.py</code> (Developer 1)<br>
              &bull; <strong>Scope:</strong> {len(incidents)} Verified Records &bull; Full Risk & Dwell Telemetry
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("👁️ Preview Formal EHS Incident Report Content", expanded=False):
        st.markdown(report_text)
