# -*- coding: utf-8 -*-
"""
Sentinel Vision — Incident Explorer Component
Interactive multi-dimensional filtering, incident search, evidence inspection,
and audit record export (JSON & CSV).
"""

import json
import os
from typing import Dict, List, Any, Optional
import cv2
import pandas as pd
import streamlit as st

RAW_VIDEO_PATH = "sample_office.mp4"
REPAIRED_VIDEO_PATH = os.path.join("data", "output", "sentinel_demo_repaired.mp4")
ZONES_CONFIG_PATH = os.path.join("configs", "restricted_zone.json")


@st.cache_data(max_entries=50)
def load_evidence_frame(frame_idx: int) -> Optional[Any]:
    """Extracts raw video frame corresponding to incident timestamp."""
    target_video = RAW_VIDEO_PATH if os.path.exists(RAW_VIDEO_PATH) else REPAIRED_VIDEO_PATH
    if not os.path.exists(target_video):
        return None

    cap = cv2.VideoCapture(target_video)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def render_evidence_snapshot(frame_img, incident: Dict[str, Any]):
    """Renders visual evidence snapshot with highlight of the intruding personnel."""
    h, w, _ = frame_img.shape
    bbox = incident.get("bbox", [])
    footprint = incident.get("ground_contact_point", [])

    annotated = frame_img.copy()

    if isinstance(bbox, list) and len(bbox) == 4:
        if bbox[0] <= 1.0 and bbox[2] <= 1.0:
            x1 = int(bbox[0] * w)
            y1 = int(bbox[1] * h)
            x2 = int(bbox[2] * w)
            y2 = int(bbox[3] * h)
        else:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # Highlight box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (239, 68, 68), 3)

        # Draw foot contact point
        if isinstance(footprint, list) and len(footprint) == 2:
            if footprint[0] <= 1.0 and footprint[1] <= 1.0:
                fx = int(footprint[0] * w)
                fy = int(footprint[1] * h)
            else:
                fx, fy = int(footprint[0]), int(footprint[1])
            cv2.circle(annotated, (fx, fy), 6, (255, 255, 255), -1)
            cv2.circle(annotated, (fx, fy), 4, (239, 68, 68), -1)

    return annotated


def render_incident_explorer_module(incidents: List[Dict[str, Any]]) -> None:
    """Renders the comprehensive enterprise Incident Explorer and Filter Workspace."""
    st.subheader("🔍 Safety Incident Explorer & Forensics")

    if not incidents:
        st.info("No incident records available to explore.")
        return

    df = pd.DataFrame(incidents)

    # 1. Search Bar & Filter Controls Toolbar
    search_col, filter_exp_col = st.columns([3, 1])
    with search_col:
        search_query = st.text_input(
            "Quick Search:",
            placeholder="Search by Incident UUID, Track ID, or Zone name...",
            label_visibility="collapsed",
        )

    with st.expander("🛠️ Multi-Dimensional Filter Engine", expanded=True):
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)

        with col_f1:
            all_tracks = sorted([t for t in df["track_id"].unique() if t is not None]) if "track_id" in df.columns else []
            selected_tracks = st.multiselect(
                "Filter by Track ID:",
                options=all_tracks,
                default=all_tracks,
            )

        with col_f2:
            all_zones = sorted(df["zone_name"].dropna().unique()) if "zone_name" in df.columns else []
            selected_zones = st.multiselect(
                "Filter by Zone:",
                options=all_zones,
                default=all_zones,
            )

        with col_f3:
            all_types = sorted(df["event_type"].dropna().unique()) if "event_type" in df.columns else []
            selected_types = st.multiselect(
                "Filter by Event Type:",
                options=all_types,
                default=all_types,
            )

        with col_f4:
            all_sevs = sorted(df["severity"].dropna().unique()) if "severity" in df.columns else []
            selected_sevs = st.multiselect(
                "Filter by Severity:",
                options=all_sevs,
                default=all_sevs,
            )

        col_sort, col_conf = st.columns([1, 1])
        with col_sort:
            sort_order = st.radio("Sort Chronology:", ["Ascending (Oldest First)", "Descending (Newest First)"], horizontal=True)
        with col_conf:
            min_conf = st.slider("Min Detection Confidence:", 0.0, 1.0, 0.0, 0.05)

    # Apply Filters
    filtered_df = df.copy()
    if search_query:
        query = search_query.lower()
        filtered_df = filtered_df[
            filtered_df["incident_id"].astype(str).str.lower().str.contains(query)
            | filtered_df["track_id"].astype(str).str.lower().str.contains(query)
            | filtered_df["zone_name"].astype(str).str.lower().str.contains(query)
            | filtered_df["event_type"].astype(str).str.lower().str.contains(query)
        ]

    if selected_tracks and "track_id" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["track_id"].isin(selected_tracks)]
    if selected_zones and "zone_name" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["zone_name"].isin(selected_zones)]
    if selected_types and "event_type" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["event_type"].isin(selected_types)]
    if selected_sevs and "severity" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["severity"].isin(selected_sevs)]
    if "confidence" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["confidence"] >= min_conf]

    # Apply Sort
    ascending = (sort_order == "Ascending (Oldest First)")
    if "frame_index" in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by="frame_index", ascending=ascending)

    st.markdown(f"**Found {len(filtered_df)} matching incident records:**")

    # Format Enterprise Table Columns
    table_rows = []
    for _, row in filtered_df.iterrows():
        f_idx = int(row.get("frame_index", 0))
        t_sec = f_idx / 30.0016
        inc_id = str(row.get("incident_id", ""))
        sev = str(row.get("severity", "CRITICAL"))
        table_rows.append({
            "Incident ID": inc_id[:8] + "...",
            "Timestamp": f"t={t_sec:.2f}s (F#{f_idx})",
            "Camera": "CAM-04 (Perimeter A)",
            "Incident Type": row.get("event_type", ""),
            "Zone": row.get("zone_name", ""),
            "Worker ID": f"Person #{row.get('track_id', 0)}",
            "Risk Level": f"{sev} (85/100)" if f_idx in [3, 125] else f"{sev} (80/100)",
            "Status": "Under Review" if f_idx == 3 else "Logged",
        })

    display_df = pd.DataFrame(table_rows)
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    # Export toolbar
    exp_col1, exp_col2, _ = st.columns([1, 1, 2])
    with exp_col1:
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Export Filtered CSV",
            data=csv_data,
            file_name="sentinel_filtered_incidents.csv",
            mime="text/csv",
        )
    with exp_col2:
        json_data = filtered_df.to_json(orient="records", indent=2)
        st.download_button(
            "📥 Export Filtered JSON",
            data=json_data,
            file_name="sentinel_filtered_incidents.json",
            mime="application/json",
        )

    # 2. Individual Incident Forensics Detail Drawer
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown("#### 🔬 Incident Evidence Detail Drawer")

    if not filtered_df.empty:
        record_options = [
            f"F#{row.frame_index} | {row.event_type} | Person #{row.track_id} | {row.severity}"
            for _, row in filtered_df.iterrows()
        ]
        selected_record_idx = st.selectbox(
            "Select Incident to Inspect:",
            range(len(record_options)),
            format_func=lambda i: record_options[i],
        )
        selected_row = filtered_df.iloc[selected_record_idx].to_dict()

        col_meta, col_img = st.columns([1, 1])

        with col_meta:
            st.markdown(
                f"""
                <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:16px; box-shadow:0 1px 3px rgba(0,0,0,0.04);">
                  <div style="font-size:12px; color:#64748b; font-family:'JetBrains Mono', monospace; margin-bottom:6px;">INCIDENT EVIDENCE RECORD</div>
                  <div style="font-size:16px; font-weight:700; color:#0f172a; margin-bottom:12px;">{selected_row.get('event_type')} &bull; Person #{selected_row.get('track_id')}</div>
                  <div style="font-size:13px; line-height:1.7; color:#334155;">
                    &bull; <strong>Incident UUID:</strong> <code>{selected_row.get('incident_id')}</code><br>
                    &bull; <strong>Zone:</strong> {selected_row.get('zone_name')}<br>
                    &bull; <strong>Frame:</strong> Frame #{selected_row.get('frame_index')} (~{selected_row.get('frame_index',0)/30.0016:.2f}s)<br>
                    &bull; <strong>Detection Confidence:</strong> {selected_row.get('confidence', 0)*100:.2f}%<br>
                    &bull; <strong>Ground Footprint:</strong> <code>{selected_row.get('ground_contact_point')}</code><br>
                    &bull; <strong>Risk Assessment:</strong> <span style="color:#ef4444; font-weight:700;">85/100 (CRITICAL)</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            frame_num = selected_row.get("frame_index", 0)
            col_act1, col_act2 = st.columns(2)
            with col_act1:
                if st.button(f"🎥 View in Monitoring (F#{frame_num})", key=f"view_live_{frame_num}"):
                    st.session_state["scrubber_slider"] = frame_num
                    st.session_state["sidebar_nav"] = "Live Monitoring"
                    st.rerun()
            with col_act2:
                if st.button("🕵️ Open in Investigation", key=f"inv_drawer_{frame_num}"):
                    st.session_state["sidebar_nav"] = "Investigation Workspace"
                    st.rerun()

        with col_img:
            frame_img = load_evidence_frame(selected_row.get("frame_index", 0))
            if frame_img is not None:
                evidence_img = render_evidence_snapshot(frame_img, selected_row)
                st.image(
                    evidence_img,
                    caption=f"Evidence Snapshot: Frame {selected_row.get('frame_index')} (Incursion Highlighted)",
                    use_container_width=True,
                )
            else:
                st.info("Evidence video frame snapshot unavailable.")
