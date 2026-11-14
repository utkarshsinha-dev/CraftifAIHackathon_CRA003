# -*- coding: utf-8 -*-
"""
Sentinel Vision AI 2.0 — Interactive Zone Editor Component
Provides real-time 2D polygon zone calibration, vertex manipulation,
normalization validation, working-copy staging, and safe configuration persistence.
"""

import json
import os
import shutil
from typing import Dict, List, Any, Optional, Tuple
import cv2
import numpy as np
import streamlit as st

CONFIG_PATH = os.path.join("configs", "restricted_zone.json")
WORKING_CONFIG_PATH = os.path.join("configs", "restricted_zone_working.json")
BACKUP_PATH = os.path.join("configs", "restricted_zone.json.bak")
RAW_VIDEO_PATH = "sample_office.mp4"
REPAIRED_VIDEO_PATH = os.path.join("data", "output", "sentinel_demo_repaired.mp4")


def load_raw_calibration_frame() -> np.ndarray:
    """Extracts reference calibration frame (Frame 0) for zone overlays."""
    target_video = RAW_VIDEO_PATH if os.path.exists(RAW_VIDEO_PATH) else REPAIRED_VIDEO_PATH
    if not os.path.exists(target_video):
        return np.zeros((1080, 1728, 3), dtype=np.uint8)

    cap = cv2.VideoCapture(target_video)
    if not cap.isOpened():
        return np.zeros((1080, 1728, 3), dtype=np.uint8)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return np.zeros((1080, 1728, 3), dtype=np.uint8)


def load_zones_data() -> Dict[str, Any]:
    """Loads zone configuration payload from working copy if available, else master config."""
    path_to_read = WORKING_CONFIG_PATH if os.path.exists(WORKING_CONFIG_PATH) else CONFIG_PATH
    if not os.path.exists(path_to_read):
        return {"version": "1.0", "coordinate_system": "normalized", "zones": []}
    try:
        with open(path_to_read, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"version": "1.0", "coordinate_system": "normalized", "zones": []}


def validate_polygon(vertices: List[List[float]]) -> Tuple[bool, str]:
    """Validates polygon geometry and normalized coordinate space."""
    if len(vertices) < 3:
        return False, "A valid 2D polygon requires a minimum of 3 distinct vertices."

    for i, pt in enumerate(vertices):
        if len(pt) != 2:
            return False, f"Vertex #{i+1} must be an [x, y] coordinate pair."
        x, y = pt[0], pt[1]
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return False, f"Vertex #{i+1} ({x:.4f}, {y:.4f}) is outside normalized [0.0, 1.0] range."

    return True, "Polygon coordinates verified valid and normalized."


def render_zone_preview(
    base_frame: np.ndarray,
    zones: List[Dict[str, Any]],
    active_zone_id: Optional[str] = None,
) -> np.ndarray:
    """Renders all zones with highlighted active calibration polygon."""
    h, w, _ = base_frame.shape
    preview = base_frame.copy()
    overlay = preview.copy()

    for zone in zones:
        poly_norm = zone.get("polygon", [])
        if len(poly_norm) >= 3:
            pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in poly_norm], np.int32)
            pts = pts.reshape((-1, 1, 2))

            z_id = zone.get("zone_id")
            severity = zone.get("severity", "CRITICAL")
            is_active = (z_id == active_zone_id)

            color = (239, 68, 68) if severity == "CRITICAL" else (245, 158, 11)  # RGB
            if is_active:
                color = (56, 189, 248)  # Active Cyan highlight

            # Fill
            cv2.fillPoly(overlay, [pts], color)
            # Outline
            thick = 3 if is_active else 2
            cv2.polylines(preview, [pts], True, color, thick, cv2.LINE_AA)

            # Vertex Points & Labels
            for idx, pt in enumerate(pts):
                px, py = pt[0][0], pt[0][1]
                cv2.circle(preview, (px, py), 6, (255, 255, 255), -1)
                cv2.circle(preview, (px, py), 4, color, -1)
                cv2.putText(
                    preview,
                    str(idx + 1),
                    (px + 8, py - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

    # Blend alpha
    cv2.addWeighted(overlay, 0.30, preview, 0.70, 0, preview)
    return preview


def render_zone_editor_module() -> None:
    """Renders the comprehensive interactive zone calibration workspace with safety guards."""
    st.subheader("📐 Restricted Zone Spatial Calibration & Polygon Editor")

    st.markdown(
        """
        <div style="font-size: 11px; color: #94a3b8; margin-bottom: 12px;">
          Calibrate 2D polygonal Exclusion & Warning Envelopes directly over genuine reference footage.
          Coordinates are normalized strictly to <code>[0.0, 1.0]</code> relative to stream dimensions (1728&times;1080).
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Working copy status
    has_working_copy = os.path.exists(WORKING_CONFIG_PATH)
    if has_working_copy:
        st.markdown(
            """
            <div style="background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 6px; padding: 6px 12px; margin-bottom: 12px; font-size: 11px; color: #fbbf24; font-family: 'JetBrains Mono', monospace;">
              ⚠️ STAGED WORKING COPY ACTIVE (configs/restricted_zone_working.json). Production config unmodified.
            </div>
            """,
            unsafe_allow_html=True,
        )

    config_data = load_zones_data()
    zones_list = config_data.get("zones", [])

    zone_names = [f"{z.get('zone_name')} ({z.get('zone_id')})" for z in zones_list]
    zone_names.append("+ Create New Zone")

    col_select, col_preview_btn = st.columns([3, 1])
    with col_select:
        selected_zone_opt = st.selectbox("Select Zone to Edit / Calibrate:", zone_names, index=0)

    # Manage Zone In-Memory State
    is_new_zone = (selected_zone_opt == "+ Create New Zone")
    current_zone_idx = None if is_new_zone else zone_names.index(selected_zone_opt)

    if not is_new_zone:
        target_zone = zones_list[current_zone_idx]
    else:
        target_zone = {
            "zone_id": f"zone_custom_{len(zones_list) + 1:02d}",
            "zone_name": "New Safety Perimeter",
            "severity": "WARNING",
            "polygon": [[0.30, 0.30], [0.70, 0.30], [0.70, 0.70], [0.30, 0.70]],
            "cooldown_seconds": 2.5,
        }

    # Form & Vertex Calibration Area
    col_props, col_visual = st.columns([1, 1])

    with col_props:
        st.markdown("#### 1. Zone Attributes")
        z_id = st.text_input("Zone Identifier (ID)", value=target_zone.get("zone_id", ""))
        z_name = st.text_input("Zone Name", value=target_zone.get("zone_name", ""))
        z_sev = st.selectbox(
            "Severity Level",
            ["CRITICAL", "WARNING", "INFO"],
            index=["CRITICAL", "WARNING", "INFO"].index(target_zone.get("severity", "CRITICAL")),
        )
        z_cooldown = st.slider(
            "Alert Cooldown (Seconds)",
            min_value=0.5,
            max_value=10.0,
            value=float(target_zone.get("cooldown_seconds", 2.0)),
            step=0.5,
        )

        st.markdown("#### 2. Polygon Vertices (Normalized [0.0 - 1.0])")
        current_poly = target_zone.get("polygon", [])

        # Manage vertices via text or coordinate inputs
        poly_str = "\n".join([f"{pt[0]:.4f}, {pt[1]:.4f}" for pt in current_poly])
        updated_poly_text = st.text_area(
            "Vertices (one per line: X, Y):",
            value=poly_str,
            height=120,
            help="Enter normalized coordinates between 0.0 and 1.0, e.g., 0.20, 0.45",
        )

        # Parse text into float pairs
        parsed_vertices = []
        parse_error = None
        for line in updated_poly_text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) == 2:
                try:
                    px, py = float(parts[0].strip()), float(parts[1].strip())
                    parsed_vertices.append([px, py])
                except ValueError:
                    parse_error = f"Malformed float value in line: '{line}'"
            else:
                parse_error = f"Line must contain X, Y separated by comma: '{line}'"

        if parse_error:
            st.error(parse_error)
            valid_geom = False
        else:
            valid_geom, geom_msg = validate_polygon(parsed_vertices)
            if valid_geom:
                st.success(geom_msg)
            else:
                st.error(geom_msg)

    # Render Visual Preview
    with col_visual:
        st.markdown("#### 3. Real-Time Spatial Overlay Preview")
        base_img = load_raw_calibration_frame()

        # Temporary display zones
        preview_zones = [z.copy() for z in zones_list]
        temp_zone = {
            "zone_id": z_id,
            "zone_name": z_name,
            "severity": z_sev,
            "polygon": parsed_vertices if valid_geom else current_poly,
            "cooldown_seconds": z_cooldown,
        }
        if not is_new_zone:
            preview_zones[current_zone_idx] = temp_zone
        else:
            preview_zones.append(temp_zone)

        preview_img = render_zone_preview(base_img, preview_zones, active_zone_id=z_id)
        st.image(
            preview_img,
            caption=f"Calibration View — Target: {z_name} ({len(parsed_vertices)} Vertices)",
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("#### 4. Safe Staging & Persistence Operations")

    col_stage, col_commit, col_cancel = st.columns(3)

    with col_stage:
        st.markdown("**Step A: Save to Working Copy**")
        if st.button("📝 Stage to Working Copy", type="secondary"):
            if not valid_geom:
                st.error("Cannot stage invalid polygon geometry.")
            elif not z_id or not z_name:
                st.error("Zone ID and Name must not be empty.")
            else:
                if not is_new_zone:
                    zones_list[current_zone_idx] = temp_zone
                else:
                    zones_list.append(temp_zone)
                config_data["zones"] = zones_list

                # Save to working copy only
                with open(WORKING_CONFIG_PATH, "w", encoding="utf-8") as f:
                    json.dump(config_data, f, indent=2)
                st.success(f"Staged `{z_name}` to `{WORKING_CONFIG_PATH}`. Production config preserved.")
                st.rerun()

    with col_commit:
        st.markdown("**Step B: Commit to Production Config**")
        confirm_replace = st.checkbox("Confirm overwriting `configs/restricted_zone.json`")
        if st.button("🚀 Commit to Master Config", type="primary", disabled=not confirm_replace):
            if not valid_geom:
                st.error("Cannot commit invalid geometry.")
            else:
                # 1. Automated Backup
                if os.path.exists(CONFIG_PATH):
                    shutil.copy2(CONFIG_PATH, BACKUP_PATH)

                # 2. Update and write
                if not is_new_zone:
                    zones_list[current_zone_idx] = temp_zone
                else:
                    zones_list.append(temp_zone)
                config_data["zones"] = zones_list

                with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                    json.dump(config_data, f, indent=2)

                # Remove working copy once committed
                if os.path.exists(WORKING_CONFIG_PATH):
                    os.remove(WORKING_CONFIG_PATH)

                st.success(f"Committed changes to production config! Backup created at `{BACKUP_PATH}`.")
                st.rerun()

    with col_cancel:
        st.markdown("**Step C: Discard & Revert**")
        if st.button("❌ Discard Staged Changes"):
            if os.path.exists(WORKING_CONFIG_PATH):
                os.remove(WORKING_CONFIG_PATH)
                st.info("Discarded staged working copy.")
                st.rerun()
            else:
                st.info("No active working copy to discard.")

        if st.button("↺ Restore Master from Backup"):
            if os.path.exists(BACKUP_PATH):
                shutil.copy2(BACKUP_PATH, CONFIG_PATH)
                st.success("Restored master configuration from `.bak` backup.")
                st.rerun()
            else:
                st.warning("No `.bak` backup file found.")

    # Backend Reprocessing Pipeline Explanation
    st.markdown("---")
    with st.expander("🔄 How Zone Modifications are Reprocessed by the Backend", expanded=False):
        st.markdown(
            """
            > **Integration Notice:** Editing or saving polygon definitions updates configuration files (`configs/restricted_zone.json`).
            > To re-evaluate the raw DeepStream perception metadata against the newly saved boundaries:
            >
            > ```powershell
            > # Execute the spatial reasoning bridge with updated zones:
            > python scripts/pipegen_to_sentinel.py --zones configs/restricted_zone.json
            > ```
            >
            > This re-runs the ray-casting state machine and outputs updated incident records to `data/output/incident_records.jsonl`.
            """
        )
