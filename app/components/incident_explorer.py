# -*- coding: utf-8 -*-
"""
Sentinel Vision AI 2.0 — Incident Explorer Component
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
        # Check normalized vs raw
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
    """Renders the comprehensive Incident Explorer and Filter Workspace."""
    st.subheader("🔍 Safety Incident Explorer & Forensics")

    if not incidents:
        st.info("No incident records available to explore.")
        return

    df = pd.DataFrame(incidents)

    # 1. Filter Controls Toolbar
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

    # Display Data Table
    display_cols = ["frame_index", "event_type", "severity", "zone_name", "track_id", "confidence"]
    cols_to_show = [c for c in display_cols if c in filtered_df.columns]
    st.dataframe(
        filtered_df[cols_to_show],
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

    # 2. Individual Incident Forensics Drilldown
    st.markdown("---")
    st.markdown("#### 🔬 Incident Evidence Drilldown")

    if not filtered_df.empty:
        record_options = [
            f"F#{row.frame_index} | {row.event_type} | Person #{row.track_id} | {row.severity}"
            for _, row in filtered_df.iterrows()
        ]
        selected_record_idx = st.selectbox("Select Incident to Inspect:", range(len(record_options)), format_func=lambda i: record_options[i])
        selected_row = filtered_df.iloc[selected_record_idx].to_dict()

        col_meta, col_img = st.columns([1, 1])

        with col_meta:
            st.markdown(f"**Incident ID:** `{selected_row.get('incident_id', 'N/A')}`")
            st.markdown(f"**Event Type:** `{selected_row.get('event_type')}`")
            st.markdown(f"**Severity Level:** `{selected_row.get('severity')}`")
            st.markdown(f"**Track Identifier:** Person #{selected_row.get('track_id')}")
            st.markdown(f"**Zone Name:** {selected_row.get('zone_name')}")
            frame_num = selected_row.get('frame_index', 0)
            st.markdown(f"**Frame Index:** {frame_num} (~{frame_num/30.0016:.2f}s)")
            st.markdown(f"**Detection Confidence:** {selected_row.get('confidence', 0)*100:.2f}%")
            st.markdown(f"**Ground Contact Datum:** `{selected_row.get('ground_contact_point')}`")

            # Jump to Video Replay
            if st.button(f"🎥 View in Video Replay (Frame #{frame_num})"):
                st.session_state["scrubber_slider"] = frame_num
                st.info(f"Synchronized Video Replay to Frame #{frame_num}. Navigate to Tab 1 to view.")

        with col_img:
            frame_img = load_evidence_frame(selected_row.get("frame_index", 0))
            if frame_img is not None:
                evidence_img = render_evidence_snapshot(frame_img, selected_row)
                st.image(
                    evidence_img,
                    caption=f"Evidence Snapshot: Frame {selected_row.get('frame_index')} (Breach Highlighted)",
                    use_container_width=True,
                )
            else:
                st.info("Evidence video frame snapshot unavailable.")
