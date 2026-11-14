# -*- coding: utf-8 -*-
"""
Sentinel Vision AI 2.0 — Video Player & Perception Replay Component
Provides interactive playback, frame-accurate scrub inspection, zone overlay,
and synchronized DeepStream detection telemetry.
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple
import cv2
import numpy as np
import streamlit as st

REPAIRED_VIDEO_PATH = os.path.join("data", "output", "sentinel_demo_repaired.mp4")
OVERLAY_VIDEO_PATH = os.path.join("data", "output", "sentinel_demo_overlay.mp4")
RAW_VIDEO_PATH = "sample_office.mp4"
METADATA_PATH = os.path.join("artifacts", "pipegen-export", "output", "metadata.json")
ZONES_PATH = os.path.join("configs", "restricted_zone.json")


@st.cache_data
def load_detection_metadata() -> Dict[int, List[Dict[str, Any]]]:
    """Caches and maps metadata.json frames to frame indices."""
    if not os.path.exists(METADATA_PATH):
        return {}

    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        frames_dict = {}
        for f_entry in data.get("frames", []):
            f_idx = f_entry.get("frame", 0)
            frames_dict[f_idx] = f_entry.get("objects", [])
        return frames_dict
    except Exception:
        return {}


@st.cache_data
def load_zones_config() -> List[Dict[str, Any]]:
    """Loads zone definitions from configs/restricted_zone.json."""
    if not os.path.exists(ZONES_PATH):
        return []
    try:
        with open(ZONES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("zones", [])
    except Exception:
        return []


def get_available_video_path() -> Optional[str]:
    """Finds best available video path with fallback."""
    for path in [REPAIRED_VIDEO_PATH, OVERLAY_VIDEO_PATH, RAW_VIDEO_PATH]:
        if os.path.exists(path):
            return path
    return None


def get_video_frame(video_path: str, frame_idx: int) -> Optional[np.ndarray]:
    """Extracts a specific frame index using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def render_annotated_frame(
    frame: np.ndarray,
    frame_idx: int,
    objects: List[Dict[str, Any]],
    zones: List[Dict[str, Any]],
) -> np.ndarray:
    """
    Renders polygon zones, bounding boxes, and ground-contact points
    onto the raw video frame.
    """
    h, w, _ = frame.shape
    annotated = frame.copy()
    overlay = annotated.copy()

    # 1. Draw Zones with Alpha Tint
    for zone in zones:
        poly_norm = zone.get("polygon", [])
        if len(poly_norm) >= 3:
            pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in poly_norm], np.int32)
            pts = pts.reshape((-1, 1, 2))

            severity = zone.get("severity", "CRITICAL")
            color = (239, 68, 68) if severity == "CRITICAL" else (245, 158, 11)  # RGB

            # Fill polygon semi-transparently
            cv2.fillPoly(overlay, [pts], color)
            # Outline
            cv2.polylines(annotated, [pts], True, color, 2, cv2.LINE_AA)

            # Zone Label
            first_pt = pts[0][0]
            cv2.putText(
                annotated,
                f"{zone.get('zone_name', 'Zone')} [{severity}]",
                (first_pt[0] + 10, first_pt[1] + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    # Blend alpha
    cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)

    # 2. Draw Tracked Detections
    for obj in objects:
        bbox = obj.get("bbox", {})
        track_id = obj.get("id", -1)
        conf = obj.get("confidence", 0.0)

        left = int(bbox.get("left", 0))
        top = int(bbox.get("top", 0))
        bw = int(bbox.get("width", 0))
        bh = int(bbox.get("height", 0))
        right = left + bw
        bottom = top + bh

        # Footprint: bottom center (x_center, y_max)
        foot_x = int(left + (bw / 2))
        foot_y = bottom

        # Box outline
        box_color = (14, 165, 233)  # Cyan default
        cv2.rectangle(annotated, (left, top), (right, bottom), box_color, 2)

        # Label tag
        label = f"Person #{track_id} ({conf:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (left, top - 20), (left + tw + 6, top), box_color, -1)
        cv2.putText(
            annotated,
            label,
            (left + 3, top - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (15, 23, 42),
            1,
            cv2.LINE_AA,
        )

        # Ground contact crosshair
        cv2.circle(annotated, (foot_x, foot_y), 5, (239, 68, 68), -1)
        cv2.line(annotated, (foot_x - 10, foot_y), (foot_x + 10, foot_y), (255, 255, 255), 1)
        cv2.line(annotated, (foot_x, foot_y - 10), (foot_x, foot_y + 10), (255, 255, 255), 1)

    return annotated


def render_video_player_module(incidents: List[Dict[str, Any]]) -> None:
    """Renders the comprehensive video playback & frame scrubber module."""
    st.subheader("📹 Perception Stream & Synchronized Replay")

    st.markdown(
        """
        <div style="background: #111827; border: 1px solid #1e293b; border-radius: 6px; padding: 8px 12px; margin-bottom: 12px; font-size: 11px; color: #94a3b8; display: flex; justify-content: space-between;">
          <span><strong>FEED:</strong> sample_office.mp4 (1728&times;1080 @ 30.00 FPS)</span>
          <span class="status-pill status-replay">Recorded Replay Mode</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    video_path = get_available_video_path()
    if not video_path:
        st.error("No demonstration video found in repository output paths.")
        return

    tab1, tab2 = st.tabs(["⚡ Frame-Accurate Scrub Telemetry", "🎬 Full Video Playback"])

    with tab1:
        # Frame scrubber controls
        total_frames = 163
        col_ctrl1, col_ctrl2 = st.columns([3, 1])

        with col_ctrl1:
            frame_slider = st.slider(
                "Timeline Scrubber (Frame Index / Time)",
                min_value=0,
                max_value=total_frames - 1,
                value=st.session_state.get("current_frame", 3),
                step=1,
                format="Frame %d",
                key="scrubber_slider",
            )
            elapsed_sec = frame_slider / 30.0016
            st.caption(f"Elapsed Video Time: **{elapsed_sec:.3f} seconds** (Frame {frame_slider}/{total_frames - 1})")

        with col_ctrl2:
            st.markdown("**Quick Breach Navigation:**")
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                if st.button("F3 (Entry)", key="btn_f3"):
                    st.session_state["scrubber_slider"] = 3
                    st.rerun()
            with col_b2:
                if st.button("F64", key="btn_f64"):
                    st.session_state["scrubber_slider"] = 64
                    st.rerun()
            with col_b3:
                if st.button("F125", key="btn_f125"):
                    st.session_state["scrubber_slider"] = 125
                    st.rerun()

        # Load and render frame
        meta_dict = load_detection_metadata()
        zones = load_zones_config()
        current_objects = meta_dict.get(frame_slider, [])

        # Prefer extracting from raw video to allow dynamic overlay drawing
        frame_img = get_video_frame(RAW_VIDEO_PATH if os.path.exists(RAW_VIDEO_PATH) else video_path, frame_slider)

        if frame_img is not None:
            # Check if this frame has an incident
            frame_incidents = [inc for inc in incidents if inc.get("frame_index") == frame_slider]
            is_breach = len(frame_incidents) > 0

            # Render custom annotated frame
            annotated_img = render_annotated_frame(frame_img, frame_slider, current_objects, zones)
            st.image(
                annotated_img,
                caption=f"Sentinel Vision AI Telemetry — Frame {frame_slider} (Objects: {len(current_objects)}, Breaches: {len(frame_incidents)})",
                use_container_width=True,
            )

            # Telemetry status strip
            t_col1, t_col2, t_col3, t_col4 = st.columns(4)
            with t_col1:
                st.metric("Detected Objects", len(current_objects))
            with t_col2:
                status_txt = "CRITICAL BREACH" if is_breach else "NORMAL"
                st.metric("Zone Status", status_txt)
            with t_col3:
                st.metric("Frame Rate", "30.00 FPS")
            with t_col4:
                st.metric("Inference Engine", "YOLO11s TensorRT")
        else:
            st.warning(f"Could not decode frame {frame_slider}.")

    with tab2:
        st.video(video_path)
        st.caption(f"Playing verified demonstration stream: `{os.path.basename(video_path)}`")
