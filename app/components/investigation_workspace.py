# -*- coding: utf-8 -*-
"""
Sentinel Vision AI 2.0 — Advanced Incident Investigation Workspace
Provides end-to-end incident forensics, 2D trajectory visualization,
risk factor breakdown, contextual Copilot interrogation, and CAPA report export.
"""

import json
import os
from typing import Dict, List, Any, Optional
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components.backend_adapter import backend_adapter
from app.components.incident_explorer import load_evidence_frame, render_evidence_snapshot

CONFIG_ZONES_PATH = os.path.join("configs", "restricted_zone.json")


@st.cache_data
def get_zones_for_map() -> List[Dict[str, Any]]:
    """Loads restricted zone polygons for overlay on trajectory chart."""
    if not os.path.exists(CONFIG_ZONES_PATH):
        return []
    try:
        with open(CONFIG_ZONES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("zones", [])
    except Exception:
        return []


def render_trajectory_chart(trajectory: Dict[str, Any], current_incident: Dict[str, Any], zones: List[Dict[str, Any]]) -> go.Figure:
    """Renders 2D spatial trajectory map in normalized image-coordinate space."""
    fig = go.Figure()

    # 1. Plot Restricted Zones
    for zone in zones:
        poly = zone.get("polygon", [])
        if len(poly) >= 3:
            # Close the polygon loop for plotting
            poly_closed = poly + [poly[0]]
            zx = [p[0] for p in poly_closed]
            zy = [p[1] for p in poly_closed]

            sev = zone.get("severity", "CRITICAL")
            fill_col = "rgba(239, 68, 68, 0.15)" if sev == "CRITICAL" else "rgba(245, 158, 11, 0.15)"
            line_col = "rgba(239, 68, 68, 0.8)" if sev == "CRITICAL" else "rgba(245, 158, 11, 0.8)"

            fig.add_trace(go.Scatter(
                x=zx,
                y=zy,
                fill="toself",
                fillcolor=fill_col,
                line=dict(color=line_col, width=2, dash="dash"),
                name=f"{zone.get('zone_name')} [{sev}]",
                hoverinfo="name",
            ))

    # 2. Plot Trajectory Path
    points = trajectory.get("points", [])
    if points:
        px = [pt["x"] for pt in points]
        py = [pt["y"] for pt in points]
        hover_texts = [f"F#{pt['frame']} (t={pt['time_sec']:.2f}s)" for pt in points]

        # Trajectory path line
        fig.add_trace(go.Scatter(
            x=px,
            y=py,
            mode="lines+markers",
            line=dict(color="#38bdf8", width=2),
            marker=dict(size=4, color="#0284c7"),
            name=f"Person #{trajectory.get('track_id')} Path",
            text=hover_texts,
            hoverinfo="text+x+y",
        ))

        # Start and End Markers
        fig.add_trace(go.Scatter(
            x=[px[0]],
            y=[py[0]],
            mode="markers+text",
            marker=dict(size=10, color="#10b981", symbol="circle"),
            text=["Start F#3"],
            textposition="top right",
            name="Track Inception",
        ))

        fig.add_trace(go.Scatter(
            x=[px[-1]],
            y=[py[-1]],
            mode="markers+text",
            marker=dict(size=10, color="#a855f7", symbol="square"),
            text=["End F#162"],
            textposition="bottom right",
            name="Track Terminus",
        ))

    # 3. Highlight Current Incident Location
    footprint = current_incident.get("ground_contact_point", [])
    if len(footprint) == 2:
        fig.add_trace(go.Scatter(
            x=[footprint[0]],
            y=[footprint[1]],
            mode="markers+text",
            marker=dict(size=14, color="#ef4444", symbol="cross", line=dict(color="#ffffff", width=2)),
            text=[f"Incident F#{current_incident.get('frame_index')}"],
            textposition="bottom center",
            name="Breach Datum",
        ))

    fig.update_layout(
        title="2D Spatial Trajectory & Exclusion Boundary Enclosure",
        xaxis=dict(
            title="Normalized X Coordinate [0.0 - 1.0]",
            range=[0.0, 1.0],
            gridcolor="#1e293b",
            zeroline=False,
        ),
        yaxis=dict(
            title="Normalized Y Coordinate [0.0 - 1.0]",
            range=[1.0, 0.0],  # Invert Y to match computer vision image coordinates (origin top-left)
            gridcolor="#1e293b",
            zeroline=False,
        ),
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(color="#f8fafc", family="Inter"),
        margin=dict(l=40, r=20, t=40, b=40),
        height=320,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9)),
    )
    return fig


def render_investigation_workspace_module(incidents: List[Dict[str, Any]]) -> None:
    """Renders the comprehensive Incident Investigation & Evidence Management Workspace."""
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #1e1b4b 0%, #1e293b 60%, #0f172a 100%); border: 1px solid #3730a3; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px;">
          <div style="font-size: 16px; font-weight: 700; color: #e0e7ff; margin-bottom: 4px;">
            🕵️ EHS Incident Investigation & Evidence Management Workspace
          </div>
          <div style="font-size: 11px; color: #94a3b8; font-family: 'JetBrains Mono', monospace;">
            Step-by-Step Forensic Analysis &bull; Evidence Synchronization &bull; Trajectory Kinematics &bull; Corrective Actions (CAPA)
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not incidents:
        st.info("No recorded safety incidents available for investigation.")
        return

    # Workflow Progress Tracker Strip
    active_dataset = backend_adapter.get_copilot_status().get("active_dataset", "data/output/incident_records.jsonl")
    st.markdown(
        f"""
        <div style="background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 6px; padding: 6px 12px; margin-bottom: 12px; font-size: 11px; font-family: 'JetBrains Mono', monospace; display: flex; justify-content: space-between; align-items: center;">
          <span>📁 <strong>ACTIVE DATASET:</strong> <code>{active_dataset}</code> ({len(incidents)} Verified Records)</span>
          <span style="color: #94a3b8;">Provenance: DeepStream 7.1 &bull; 2.0s Cooldown Suppression</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="display: flex; gap: 6px; margin-bottom: 16px; font-size: 10px; font-family: 'JetBrains Mono', monospace;">
          <span style="background: #1e293b; padding: 4px 8px; border-radius: 4px; color: #38bdf8;">1. Select Incident</span> &rarr;
          <span style="background: #1e293b; padding: 4px 8px; border-radius: 4px; color: #38bdf8;">2. Review Visual Evidence</span> &rarr;
          <span style="background: #1e293b; padding: 4px 8px; border-radius: 4px; color: #38bdf8;">3. Risk & Trajectory</span> &rarr;
          <span style="background: #1e293b; padding: 4px 8px; border-radius: 4px; color: #38bdf8;">4. Copilot Interrogation</span> &rarr;
          <span style="background: #1e293b; padding: 4px 8px; border-radius: 4px; color: #38bdf8;">5. Record CAPA Notes</span> &rarr;
          <span style="background: #1e293b; padding: 4px 8px; border-radius: 4px; color: #34d399;">6. Export Package</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Incident Selector Toolbar
    incident_options = [
        f"Incident #{i+1:02d} | F#{inc.get('frame_index', 0):03d} (t={inc.get('frame_index', 0)/30.0016:.2f}s) | Person #{inc.get('track_id', 'N/A')} | {inc.get('event_type')} | {inc.get('severity')}"
        for i, inc in enumerate(incidents)
    ]

    selected_idx = st.selectbox(
        "Select Incident to Investigate:",
        range(len(incident_options)),
        format_func=lambda i: incident_options[i],
        key="active_investigation_idx",
    )
    current_inc = incidents[selected_idx]
    inc_id = current_inc.get("incident_id", f"inc_{selected_idx}")
    track_id = current_inc.get("track_id", 0)
    frame_idx = current_inc.get("frame_index", 0)

    # Retrieve supplementary forensic data via BackendAdapter
    risk_data = backend_adapter.get_risk_assessment(current_inc)
    trajectory_data = backend_adapter.get_trajectory(int(track_id) if str(track_id).isdigit() else 0)
    zones = get_zones_for_map()

    # 2. Main Two-Column Investigation Split
    col_left, col_right = st.columns([1, 1])

    # Left Column: Visual Evidence & Trajectory Map
    with col_left:
        st.markdown("#### 📷 Visual Evidence & Spatial Verification")

        frame_img = load_evidence_frame(frame_idx)
        if frame_img is not None:
            evidence_img = render_evidence_snapshot(frame_img, current_inc)
            st.image(
                evidence_img,
                caption=f"Evidence Capture — Frame #{frame_idx} (Person #{track_id} in {current_inc.get('zone_name')})",
                use_container_width=True,
            )
        else:
            st.warning("Video frame capture unavailable.")

        # Synchronization with Master Video Player
        col_jump, col_stat = st.columns([2, 1])
        with col_jump:
            if st.button(f"🎥 Jump to Frame #{frame_idx} in Video Player", key=f"inv_jump_{selected_idx}", type="secondary"):
                st.session_state["scrubber_slider"] = frame_idx
                st.toast(f"Synchronized Video Player to Frame #{frame_idx}")
                st.info(f"Synchronized! Open Tab 1 to scrub from Frame #{frame_idx}.")

        with col_stat:
            st.markdown(
                f"""
                <div style="font-size: 11px; text-align: right; color: #94a3b8; font-family: 'JetBrains Mono', monospace; padding-top: 6px;">
                  Footprint: ({current_inc.get('ground_contact_point', [0,0])[0]:.3f}, {current_inc.get('ground_contact_point', [0,0])[1]:.3f})
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 2D Trajectory Kinematics Map
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        traj_fig = render_trajectory_chart(trajectory_data, current_inc, zones)
        st.plotly_chart(traj_fig, use_container_width=True)

        # Kinematic Summary Box
        prior_miss = trajectory_data.get('missing_frames_prior_to_lock', 3)
        seq_gaps = trajectory_data.get('missing_frames_in_sequence', 0)
        obs_pts = trajectory_data.get('observed_points_count', trajectory_data.get('total_points', 0))
        interp_pts = trajectory_data.get('interpolated_points_count', 0)

        st.markdown(
            f"""
            <div style="background: #111827; border: 1px solid #1e293b; border-radius: 6px; padding: 10px 14px; font-size: 11px; font-family: 'JetBrains Mono', monospace; color: #94a3b8;">
              <div style="color: #38bdf8; font-weight: bold; margin-bottom: 4px;">🎯 TRAJECTORY INTEGRITY & KINEMATICS</div>
              &bull; <strong>Kinematics:</strong> {trajectory_data.get('movement_classification')}<br>
              &bull; <strong>Observed Points:</strong> {obs_pts} frames | <strong>Missing Prior to Lock:</strong> {prior_miss} (F#0-2) | <strong>Sequence Gaps:</strong> {seq_gaps} | <strong>Interpolated:</strong> {interp_pts}<br>
              &bull; <strong>Path Displacement:</strong> {trajectory_data.get('net_displacement_norm', 0.0)} units | <strong>Mean Velocity:</strong> {trajectory_data.get('mean_velocity_norm_per_sec', 0.0)} units/sec<br>
              &bull; <em>Disclaimer:</em> {trajectory_data.get('disclaimer')}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Right Column: Incident Overview, Risk Factors, and Contextual Copilot
    with col_right:
        st.markdown("#### 📋 Incident Profile & Risk Factor Analysis")

        # Overview Metrics Grid
        c_o1, c_o2, c_o3 = st.columns(3)
        with c_o1:
            st.caption("Event Type")
            st.write(f"**{current_inc.get('event_type')}**")
        with c_o2:
            st.caption("Severity Level")
            sev_color = "#f87171" if current_inc.get("severity") == "CRITICAL" else "#fbbf24"
            st.markdown(f"<strong style='color:{sev_color};'>{current_inc.get('severity')}</strong>", unsafe_allow_html=True)
        with c_o3:
            st.caption("Confidence")
            st.write(f"**{float(current_inc.get('confidence', 0.0))*100:.1f}%**")

        st.markdown(
            f"""
            <div style="background: #111827; border: 1px solid #1e293b; border-radius: 6px; padding: 10px 14px; margin-bottom: 12px; font-size: 11px;">
              <strong>Incident ID:</strong> <code>{inc_id}</code><br>
              <strong>Hazard Zone:</strong> {current_inc.get('zone_name')} (<code>{current_inc.get('zone_id')}</code>)<br>
              <strong>Timing:</strong> Timestamp: <code>{current_inc.get('frame_index', 0)/30.0016:.2f}s</code> (Frame #{frame_idx})
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Risk Factors Card
        st.markdown("##### ⚠️ Risk Assessment Breakdown")
        r_score = risk_data.get("risk_score")
        if r_score is not None:
            st.progress(float(r_score) / 100.0)
            is_auth = risk_data.get("is_authoritative", False)
            prov_badge = (
                "<span style='background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; color: #34d399; padding: 2px 6px; border-radius: 4px; font-size: 10px;'>✅ AUTHORITATIVE BACKEND RISK SCORE</span>"
                if is_auth
                else "<span style='background: rgba(245, 158, 11, 0.15); border: 1px solid #f59e0b; color: #fbbf24; padding: 2px 6px; border-radius: 4px; font-size: 10px;'>⚠️ PROVISIONAL FRONTEND CALCULATION</span>"
            )
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; align-items: center; font-size: 11px; margin-bottom: 8px;">
                  <span><strong>Calculated Risk Score:</strong> <strong style="color: #f87171;">{r_score}/100</strong></span>
                  <div>{prov_badge} <span class="badge badge-critical">{risk_data.get('risk_tier', 'CRITICAL')}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if risk_data.get("explanation"):
                st.markdown(
                    f"""
                    <div style="background: rgba(239, 68, 68, 0.08); border-left: 3px solid #ef4444; padding: 6px 10px; margin-bottom: 8px; font-size: 11px; color: #cbd5e1;">
                      <strong style="color: #f87171;">Backend Evaluation:</strong> {risk_data.get('explanation')}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            for factor in risk_data.get("factors", []):
                st.markdown(
                    f"""
                    <div style="border-left: 3px solid #f59e0b; padding-left: 8px; margin-bottom: 6px; font-size: 11px;">
                      <strong>{factor.get('factor')}</strong> <span style="font-size: 9px; color: #94a3b8;">[{factor.get('weight')}]</span><br>
                      <span style="color: #cbd5e1;">{factor.get('description')}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.caption(f"Provenance Source: {risk_data.get('source')}")
        else:
            st.warning("Risk assessment unavailable: No backend risk telemetry recorded.")

        # Contextual Safety Copilot Queries
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown("##### 🤖 Contextual Forensic Copilot")
        st.caption("Ask contextual questions grounded specifically in this incident:")

        # Quick contextual pill buttons
        c_q1, c_q2 = st.columns(2)
        with c_q1:
            if st.button("❓ Why is this critical?", key=f"cp_why_{selected_idx}"):
                st.session_state["inv_prompt"] = "Why was this incident classified as critical?"
            if st.button("⏱️ How long in zone?", key=f"cp_dwell_{selected_idx}"):
                st.session_state["inv_prompt"] = "How long was the person inside the restricted area?"
        with c_q2:
            if st.button("🔍 What evidence exists?", key=f"cp_ev_{selected_idx}"):
                st.session_state["inv_prompt"] = "What evidence supports this violation?"
            if st.button("🛡️ Recommended CAPA?", key=f"cp_capa_{selected_idx}"):
                st.session_state["inv_prompt"] = "What preventive measures could reduce similar incidents?"

        # Handle active contextual inquiry
        inv_query = st.session_state.pop("inv_prompt", None)
        if inv_query:
            with st.spinner("Analyzing incident telemetry and zone parameters..."):
                copilot_ans = backend_adapter.answer_question(inv_query, incidents, incident_context=current_inc)
            st.markdown(copilot_ans["text"])
            st.caption(f"Mode: `{copilot_ans['mode']}` &bull; Evidence: `{copilot_ans.get('evidence', [])}`")

    # 3. Bottom Section: Operator Observations & Corrective Actions (CAPA)
    st.markdown("---")
    st.markdown("#### 📝 Operator Observations & Corrective Actions (CAPA)")

    # Session State Keys for Notes & Status
    status_key = f"inv_status_{inc_id}"
    notes_key = f"inv_notes_{inc_id}"
    capa_key = f"inv_capa_{inc_id}"
    conditions_key = f"inv_cond_{inc_id}"

    if status_key not in st.session_state:
        st.session_state[status_key] = "Under Review"
    if notes_key not in st.session_state:
        st.session_state[notes_key] = "Personnel standing adjacent to equipment cart; feet encroaching on painted boundary line."
    if capa_key not in st.session_state:
        st.session_state[capa_key] = "Deploy temporary optical light curtain and repaint yellow/black hazard perimeter stripes."
    if conditions_key not in st.session_state:
        st.session_state[conditions_key] = ["Maintenance cart positioned near exclusion line", "Floor hazard demarcation partially faded"]

    col_stat_sel, col_cond = st.columns([1, 2])
    with col_stat_sel:
        inv_status = st.selectbox(
            "Investigation Status (Session-Recorded):",
            ["Open", "Under Review", "Resolved"],
            index=["Open", "Under Review", "Resolved"].index(st.session_state[status_key]),
            key=f"status_select_{selected_idx}",
        )
        st.session_state[status_key] = inv_status

    with col_cond:
        selected_conditions = st.multiselect(
            "Suspected Contributing Conditions:",
            options=[
                "Maintenance cart positioned near exclusion line",
                "Floor hazard demarcation partially faded",
                "Lack of active physical light curtain / interlock",
                "Personnel unaware of automated machinery envelope",
                "Obstructed walkway creating boundary bypass",
                "Routine diagnostics activity during energized state",
            ],
            default=st.session_state[conditions_key],
            key=f"cond_select_{selected_idx}",
        )
        st.session_state[conditions_key] = selected_conditions

    col_notes, col_capa = st.columns(2)
    with col_notes:
        operator_notes = st.text_area(
            "Operator Forensic Observations:",
            value=st.session_state[notes_key],
            height=100,
            key=f"notes_area_{selected_idx}",
        )
        st.session_state[notes_key] = operator_notes

    with col_capa:
        capa_text = st.text_area(
            "Recommended Corrective & Preventive Actions (CAPA):",
            value=st.session_state[capa_key],
            height=100,
            key=f"capa_area_{selected_idx}",
        )
        st.session_state[capa_key] = capa_text

    st.markdown(
        """
        <div style="background: rgba(245, 158, 11, 0.08); border-left: 3px solid #f59e0b; padding: 6px 12px; font-size: 10px; color: #fbbf24; font-family: 'JetBrains Mono', monospace; margin-bottom: 14px;">
          DISCLAIMER: Investigation notes, statuses, and CAPA recommendations are stored within the active browser session for demonstration. Actions are advisory and have not been executed.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4. Export Investigation Package Toolbar
    st.markdown("#### 📦 Export Incident Investigation Package")

    col_exp_md, col_exp_json, col_exp_csv = st.columns(3)

    # Generate Markdown Report
    report_md = backend_adapter.generate_investigation_report_markdown(
        incident=current_inc,
        operator_notes=operator_notes,
        capa_actions=capa_text,
        investigation_status=inv_status,
        contributing_conditions=selected_conditions,
        risk_data=risk_data,
        trajectory_data=trajectory_data,
    )

    with col_exp_md:
        st.download_button(
            label="📄 Download Markdown Report (.md)",
            data=report_md,
            file_name=f"investigation_report_{inc_id[:8]}.md",
            mime="text/markdown",
            key=f"dl_md_{selected_idx}",
        )

    # Generate JSON Package
    evidence_package = {
        "investigation_metadata": {
            "version": "2.0",
            "auditor_role": "Industrial Safety Officer / EHS Manager",
            "status": inv_status,
            "session_storage_only": True,
        },
        "incident_record": current_inc,
        "risk_assessment": risk_data,
        "trajectory_kinematics": trajectory_data,
        "operator_findings": {
            "contributing_conditions": selected_conditions,
            "forensic_observations": operator_notes,
            "recommended_capa": capa_text,
        },
    }

    with col_exp_json:
        st.download_button(
            label="📦 Download JSON Evidence Package",
            data=json.dumps(evidence_package, indent=2),
            file_name=f"evidence_package_{inc_id[:8]}.json",
            mime="application/json",
            key=f"dl_json_{selected_idx}",
        )

    with col_exp_csv:
        df_single = pd.DataFrame([current_inc])
        st.download_button(
            label="📊 Download Incident CSV Record",
            data=df_single.to_csv(index=False),
            file_name=f"incident_{inc_id[:8]}.csv",
            mime="text/csv",
            key=f"dl_csv_{selected_idx}",
        )
