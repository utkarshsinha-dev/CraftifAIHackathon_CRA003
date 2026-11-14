# -*- coding: utf-8 -*-
"""
Sentinel Vision AI 2.0 — Video Player & Perception Replay Component
Provides interactive playback, cached frame-accurate scrub inspection,
real-time spatial zone containment coloring, and synchronized DeepStream telemetry.
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
    """Finds best available video path with graceful fallback."""
    for path in [REPAIRED_VIDEO_PATH, OVERLAY_VIDEO_PATH, RAW_VIDEO_PATH]:
        if os.path.exists(path):
            return path
    return None


@st.cache_data(max_entries=170)
def get_cached_video_frame(video_path: str, frame_idx: int) -> Optional[np.ndarray]:
    """Caches decoded video frames in memory for zero-latency slider scrubbing."""
    if not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def is_point_in_polygon(x: float, y: float, polygon: List[List[float]]) -> bool:
    """Ray-casting point-in-polygon containment test for normalized coordinates."""
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def render_annotated_frame(
    frame: np.ndarray,
    frame_idx: int,
    objects: List[Dict[str, Any]],
    zones: List[Dict[str, Any]],
) -> np.ndarray:
    """
    Renders polygon zones, bounding boxes, and ground-contact points.
    Dynamically highlights intruders in RED when their footprint breaches a zone.
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

        # Coordinate conversion: handle dict {left, top, width, height} or list [x1, y1, x2, y2]
        if isinstance(bbox, dict):
            raw_left = float(bbox.get("left", 0.0))
            raw_top = float(bbox.get("top", 0.0))
            raw_w = float(bbox.get("width", 0.0))
            raw_h = float(bbox.get("height", 0.0))

            # If normalized (<= 1.0), scale by dimensions
            if raw_left <= 1.0 and raw_w <= 1.0 and raw_w > 0:
                left = int(raw_left * w)
                top = int(raw_top * h)
                bw = int(raw_w * w)
                bh = int(raw_h * h)
            else:
                left = int(raw_left)
                top = int(raw_top)
                bw = int(raw_w)
                bh = int(raw_h)
            right = left + bw
            bottom = top + bh
        elif isinstance(bbox, list) and len(bbox) == 4:
            if bbox[0] <= 1.0 and bbox[2] <= 1.0:
                left = int(bbox[0] * w)
                top = int(bbox[1] * h)
                right = int(bbox[2] * w)
                bottom = int(bbox[3] * h)
                bw = right - left
                bh = bottom - top
            else:
                left, top, right, bottom = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                bw = right - left
                bh = bottom - top
        else:
            continue

        # Footprint: bottom center (x_center, y_max)
        foot_x = int(left + (bw / 2))
        foot_y = bottom
        norm_foot_x = foot_x / float(w)
        norm_foot_y = foot_y / float(h)

        # Check containment against zones
        breaching_zone = None
        for z in zones:
            poly = z.get("polygon", [])
            if len(poly) >= 3 and is_point_in_polygon(norm_foot_x, norm_foot_y, poly):
                breaching_zone = z
                break

        # Dynamic box color based on intrusion state
        if breaching_zone:
            box_color = (239, 68, 68)  # Red for Incursion
            status_tag = f"Person #{track_id} [INTRUSION]"
        else:
            box_color = (16, 185, 129)  # Green for Safe
            status_tag = f"Person #{track_id} [SAFE]"

        cv2.rectangle(annotated, (left, top), (right, bottom), box_color, 2)

        # Label tag
        label = f"{status_tag} ({conf:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (left, top - 20), (left + tw + 6, top), box_color, -1)
        cv2.putText(
            annotated,
            label,
            (left + 3, top - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255) if breaching_zone else (15, 23, 42),
            1,
            cv2.LINE_AA,
        )

        # Ground contact crosshairs
        crosshair_color = (239, 68, 68) if breaching_zone else (56, 189, 248)
        cv2.circle(annotated, (foot_x, foot_y), 5, crosshair_color, -1)
        cv2.line(annotated, (foot_x - 12, foot_y), (foot_x + 12, foot_y), (255, 255, 255), 1)
        cv2.line(annotated, (foot_x, foot_y - 12), (foot_x, foot_y + 12), (255, 255, 255), 1)

    return annotated


def render_video_player_module(incidents: List[Dict[str, Any]]) -> None:
    """Renders the comprehensive video playback & frame scrubber module."""
    st.subheader("📹 Perception Stream & Synchronized Replay")

    st.markdown(
        """
        <div style="background: #111827; border: 1px solid #1e293b; border-radius: 6px; padding: 8px 12px; margin-bottom: 12px; font-size: 11px; color: #94a3b8; display: flex; justify-content: space-between; align-items: center;">
          <span><strong>FEED:</strong> sample_office.mp4 (1728&times;1080 @ 30.00 FPS)</span>
          <span class="status-pill status-replay">Recorded Perception Replay</span>
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
        total_frames = 163

        # Synchronize session state key
        if "scrubber_slider" not in st.session_state:
            st.session_state["scrubber_slider"] = 3

        col_ctrl1, col_ctrl2 = st.columns([3, 1])

        with col_ctrl1:
            frame_slider = st.slider(
                "Timeline Scrubber (Frame Index / Time)",
                min_value=0,
                max_value=total_frames - 1,
                value=st.session_state["scrubber_slider"],
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

        target_source = RAW_VIDEO_PATH if os.path.exists(RAW_VIDEO_PATH) else video_path
        frame_img = get_cached_video_frame(target_source, frame_slider)

        if frame_img is not None:
            frame_incidents = [inc for inc in incidents if inc.get("frame_index") == frame_slider]
            is_breach = len(frame_incidents) > 0

            # Render custom annotated frame
            annotated_img = render_annotated_frame(frame_img, frame_slider, current_objects, zones)
            st.image(
                annotated_img,
                caption=f"Sentinel Vision AI Telemetry — Frame {frame_slider} (Objects: {len(current_objects)}, Active Breaches: {len(frame_incidents)})",
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
                st.metric("Source Framerate", "30.00 FPS")
            with t_col4:
                st.metric("Inference Engine", "YOLO11s TensorRT FP32")
        else:
            st.warning(f"Could not decode frame {frame_slider} from `{target_source}`.")

    with tab2:
        st.video(video_path)
        st.caption(f"Playing verified demonstration stream: `{os.path.basename(video_path)}`")
