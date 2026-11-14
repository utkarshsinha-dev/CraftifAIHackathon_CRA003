# -*- coding: utf-8 -*-
"""
Sentinel Vision AI 2.0 — Metrics & KPI Analytics Component
Computes and renders high-precision industrial safety metrics and Plotly charts.
"""

import json
import os
from typing import Dict, List, Any, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DEFAULT_INCIDENT_PATH = os.path.join("data", "output", "incident_records.jsonl")


def load_incidents(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Loads verified incident records from JSONL.
    Falls back gracefully to default repository path.
    """
    target_path = path or DEFAULT_INCIDENT_PATH
    if not os.path.exists(target_path):
        return []

    incidents = []
    with open(target_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                incidents.append(record)
            except json.JSONDecodeError:
                continue
    return incidents


def compute_kpi_metrics(incidents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes summary KPI indicators strictly from authentic incident records.
    Never fabricates metrics.
    """
    if not incidents:
        return {
            "total_incidents": 0,
            "critical_incidents": 0,
            "warning_incidents": 0,
            "active_tracks": 0,
            "total_entries": 0,
            "total_occupancies": 0,
            "estimated_occupancy_seconds": 0.0,
            "unique_zones": 0,
        }

    total = len(incidents)
    critical = sum(1 for inc in incidents if inc.get("severity") == "CRITICAL")
    warning = sum(1 for inc in incidents if inc.get("severity") == "WARNING")
    unique_tracks = len(set(inc.get("track_id") for inc in incidents if "track_id" in inc))
    unique_zones = len(set(inc.get("zone_id") for inc in incidents if "zone_id" in inc))

    entries = sum(1 for inc in incidents if inc.get("event_type") == "ZONE_INTRUSION_ENTRY")
    occupancies = sum(1 for inc in incidents if inc.get("event_type") == "ZONE_INTRUSION_OCCUPANCY")

    # Estimated occupancy duration based on cooldown cycles (2.0s per alert)
    estimated_occ_sec = round(occupancies * 2.0 + (entries * 0.5), 1)

    return {
        "total_incidents": total,
        "critical_incidents": critical,
        "warning_incidents": warning,
        "active_tracks": unique_tracks,
        "total_entries": entries,
        "total_occupancies": occupancies,
        "estimated_occupancy_seconds": estimated_occ_sec,
        "unique_zones": unique_zones,
    }


def render_kpi_cards(kpis: Dict[str, Any]) -> None:
    """Renders sleek industrial KPI metric cards."""
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card card-info">
              <div class="label">Total Incidents</div>
              <div class="value">{kpis['total_incidents']}</div>
              <div class="caption">Verified Safety Alerts</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card card-critical">
              <div class="label">Critical Violations</div>
              <div class="value">{kpis['critical_incidents']}</div>
              <div class="caption">High-Severity Hazards</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card card-warning">
              <div class="label">Tracked Entities</div>
              <div class="value">{kpis['active_tracks']}</div>
              <div class="caption">Distinct Personnel</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card card-success">
              <div class="label">Zone Breaches</div>
              <div class="value">{kpis['total_entries']}</div>
              <div class="caption">Boundary Incursions</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col5:
        st.markdown(
            f"""
            <div class="metric-card card-info">
              <div class="label">Occupancy Dwell</div>
              <div class="value">{kpis['estimated_occupancy_seconds']}s</div>
              <div class="caption">Hazard Exposure Time</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_analytics_charts(incidents: List[Dict[str, Any]]) -> None:
    """Renders responsive Plotly industrial analytics charts."""
    if not incidents:
        st.info("No incident records available for chart generation.")
        return

    df = pd.DataFrame(incidents)

    # Format timestamp / seconds
    if "timestamp_iso" in df.columns:
        try:
            df["time_sec"] = pd.to_datetime(df["timestamp_iso"]).dt.second + (
                pd.to_datetime(df["timestamp_iso"]).dt.microsecond / 1e6
            )
        except Exception:
            df["time_sec"] = df.get("frame_index", 0) / 30.0
    else:
        df["time_sec"] = df.get("frame_index", 0) / 30.0

    col1, col2 = st.columns(2)

    with col1:
        # Chart 1: Incidents by Severity (Donut Chart)
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
            title="Incidents by Severity Level",
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
            track_counts["Track"] = "Track " + track_counts["track_id"].astype(str)
            fig_track = px.bar(
                track_counts,
                x="Track",
                y="Occurrences",
                color="event_type",
                barmode="group",
                title="Events per Personnel Track ID",
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

    # Chart 3: Temporal Distribution Over Video Timeline
    fig_time = px.scatter(
        df,
        x="time_sec",
        y="track_id",
        color="severity",
        size=[12] * len(df),
        symbol="event_type",
        hover_data=["zone_name", "confidence", "frame_index"],
        labels={"time_sec": "Video Timeline (Seconds)", "track_id": "Track ID"},
        title="Temporal Breach Dispersion Across Video Duration (0.0s – 5.43s)",
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
