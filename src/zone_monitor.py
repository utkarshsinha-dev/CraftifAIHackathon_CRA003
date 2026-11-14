"""
Sentinel Vision AI - Restricted Zone Intrusion Monitor

Evaluates person bounding boxes and tracking IDs against configurable restricted
zones, emitting structured JSON incident records upon zone breaches.

Features:
- Ray-casting 2D polygonal containment against arbitrary convex/concave zones.
- Ground-contact footprint projection: ((x_min + x_max)/2, y_max).
- Stateful transitions: OUTSIDE -> ZONE_INTRUSION_ENTRY -> ZONE_INTRUSION_OCCUPANCY -> ZONE_INTRUSION_EXIT.
- Anti-spam cooldown alert throttling.
- Session-grounded ISO timestamps & video-relative elapsed timing (P0).
- Conservative spatial association for untracked personnel stabilization (P0).
- Precise cumulative dwell-time analytics (P1-C).
- Kinematic motion vectors and zone-relative trajectory classification (P1-D).
- Deterministic explainable risk scoring via risk_engine (P1-E).
"""

import json
import math
import os
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

from src.risk_engine import assess_incident_risk, calculate_polygon_centroid


def point_in_polygon(point: Tuple[float, float], polygon: List[List[float]]) -> bool:
    """
    Ray-casting algorithm to test whether a 2D point (x, y) is inside a polygon.
    All coordinates should be in the same space (e.g., normalized [0.0, 1.0]).
    """
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False

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


class ZoneIntrusionMonitor:
    def __init__(
        self,
        config_path: str,
        output_log_path: str = "data/output/incident_records.jsonl",
        session_start_epoch: Optional[float] = None,
    ):
        self.config_path = config_path
        self.output_log_path = output_log_path
        self.zones: List[Dict[str, Any]] = []

        # Session timing calibration
        self.session_start_epoch = session_start_epoch or time.time()
        self.session_start_iso = datetime.fromtimestamp(
            self.session_start_epoch, tz=timezone.utc
        ).isoformat()
        self._first_seen_time: Optional[float] = None

        # State tracking: (entity_id, zone_id) -> state ("INSIDE" | "OUTSIDE")
        self.tracker_states: Dict[Tuple[Any, str], str] = {}
        self.last_alert_time: Dict[Tuple[Any, str], float] = {}

        # Dwell time tracking: (entity_id, zone_id) -> entry_video_time_s
        self.entry_times: Dict[Tuple[Any, str], float] = {}
        self.entry_frames: Dict[Tuple[Any, str], int] = {}

        # Kinematic trajectory tracking: entity_id -> deque of (video_time_s, ground_point)
        self.entity_trajectories: Dict[Any, deque] = {}

        # Conservative untracked identity stabilization
        self.untracked_registry: Dict[str, Dict[str, Any]] = {}
        self.next_untracked_seq: int = 1
        self.max_untracked_dist: float = 0.08  # Max 8% normalized distance for identity continuity
        self.untracked_expiry_frames: int = 10
        self.untracked_expiry_seconds: float = 1.0

        self._load_config()
        os.makedirs(os.path.dirname(self.output_log_path), exist_ok=True)

    def _load_config(self) -> None:
        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(f"Zone config not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.zones = data.get("zones", [])

    @staticmethod
    def compute_ground_contact(bbox: List[float]) -> Tuple[float, float]:
        """
        Calculates ground contact point (bottom-center of bounding box).
        bbox: [x_min, y_min, x_max, y_max] in normalized coordinates [0.0, 1.0].
        """
        x_min, y_min, x_max, y_max = bbox
        return ((x_min + x_max) / 2.0, y_max)

    def _resolve_entity_id(
        self,
        track_id: Optional[int],
        ground_point: Tuple[float, float],
        frame_index: Optional[int],
        video_time: float,
    ) -> Any:
        """
        Provides conservative identity stabilization.
        - If valid track_id exists: returns track_id.
        - If untracked: uses conservative spatial association against active untracked tracks
          to maintain identity continuity and prevent alert spam.
        """
        if track_id is not None:
            return track_id

        # Prune expired untracked entities
        expired_keys = []
        for uid, udata in self.untracked_registry.items():
            if frame_index is not None and udata.get("last_frame") is not None:
                if frame_index - udata["last_frame"] > self.untracked_expiry_frames:
                    expired_keys.append(uid)
            elif (video_time - udata.get("last_time", 0.0)) > self.untracked_expiry_seconds:
                expired_keys.append(uid)
        for k in expired_keys:
            del self.untracked_registry[k]

        # Greedy nearest-neighbor matching
        best_id = None
        min_dist = float("inf")
        gx, gy = ground_point

        for uid, udata in self.untracked_registry.items():
            ux, uy = udata["last_point"]
            dist = math.sqrt((gx - ux) ** 2 + (gy - uy) ** 2)
            if dist <= self.max_untracked_dist and dist < min_dist:
                min_dist = dist
                best_id = uid

        if best_id is not None:
            # Update matched untracked track
            self.untracked_registry[best_id]["last_point"] = ground_point
            self.untracked_registry[best_id]["last_frame"] = frame_index
            self.untracked_registry[best_id]["last_time"] = video_time
            return best_id

        # Allocate new stabilized untracked ID
        new_uid = f"untracked_{self.next_untracked_seq}"
        self.next_untracked_seq += 1
        self.untracked_registry[new_uid] = {
            "last_point": ground_point,
            "last_frame": frame_index,
            "last_time": video_time,
        }
        return new_uid

    def _compute_kinematics(
        self,
        entity_id: Any,
        ground_point: Tuple[float, float],
        video_time_s: float,
        zone_polygon: List[List[float]],
    ) -> Tuple[Dict[str, Any], List[List[float]]]:
        """
        Calculates image-coordinate movement, heading, and hazard-zone-relative kinematics.
        Normalized units per second (screen units / s).
        """
        if entity_id not in self.entity_trajectories:
            self.entity_trajectories[entity_id] = deque(maxlen=8)

        traj = self.entity_trajectories[entity_id]
        recent_pts = [[round(p[1][0], 4), round(p[1][1], 4)] for p in traj]

        if not traj:
            traj.append((video_time_s, ground_point))
            return {
                "velocity_normalized": [0.0, 0.0],
                "speed_normalized": 0.0,
                "direction": "STATIONARY",
                "motion_state": "stationary",
                "reliability": "INITIALIZING",
            }, recent_pts

        prev_time, prev_pt = traj[-1]
        dt = video_time_s - prev_time

        # Update trajectory buffer
        traj.append((video_time_s, ground_point))
        recent_pts = [[round(p[1][0], 4), round(p[1][1], 4)] for p in traj]

        if dt <= 0.0001:
            return {
                "velocity_normalized": [0.0, 0.0],
                "speed_normalized": 0.0,
                "direction": "STATIONARY",
                "motion_state": "stationary",
                "reliability": "INITIALIZING",
            }, recent_pts

        dx = ground_point[0] - prev_pt[0]
        dy = ground_point[1] - prev_pt[1]
        vx = dx / dt
        vy = dy / dt
        speed = math.sqrt(vx * vx + vy * vy)

        # Cardinal direction estimation
        if speed < 0.015:
            cardinal = "STATIONARY"
        else:
            angle_deg = (math.degrees(math.atan2(dy, dx)) + 360) % 360
            dirs = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]
            idx = int((angle_deg + 22.5) / 45.0) % 8
            cardinal = dirs[idx]

        # Trajectory relative to hazard zone centroid
        centroid = calculate_polygon_centroid(zone_polygon)
        prev_dist = math.sqrt((prev_pt[0] - centroid[0]) ** 2 + (prev_pt[1] - centroid[1]) ** 2)
        curr_dist = math.sqrt((ground_point[0] - centroid[0]) ** 2 + (ground_point[1] - centroid[1]) ** 2)

        if speed < 0.015:
            motion_state = "stationary"
        elif curr_dist < (prev_dist - 0.002):
            motion_state = "approaching_hazard_core"
        elif curr_dist > (prev_dist + 0.002):
            motion_state = "receding_from_hazard_core"
        else:
            motion_state = "lateral_movement"

        reliability = "HIGH" if len(traj) >= 3 and (0.01 <= dt <= 1.0) else "INITIALIZING"

        kinematics = {
            "velocity_normalized": [round(vx, 4), round(vy, 4)],
            "speed_normalized": round(speed, 4),
            "direction": cardinal,
            "motion_state": motion_state,
            "reliability": reliability,
        }
        return kinematics, recent_pts

    def process_detection(
        self,
        bbox: List[float],
        confidence: float,
        track_id: Optional[int] = None,
        frame_index: Optional[int] = None,
        timestamp_epoch: Optional[float] = None,
        video_time_s: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Processes a single person detection against all configured zones.
        Returns a list of generated incident records (if any).
        """
        # 1. Temporal grounding
        if video_time_s is not None:
            v_time = float(video_time_s)
        elif timestamp_epoch is not None:
            # Handle synthetic test timestamps vs relative PTS
            if timestamp_epoch < 1e8:
                v_time = float(timestamp_epoch)
            else:
                if self._first_seen_time is None:
                    self._first_seen_time = timestamp_epoch
                v_time = float(timestamp_epoch - self._first_seen_time)
        else:
            if self._first_seen_time is None:
                self._first_seen_time = time.time()
            v_time = float(time.time() - self._first_seen_time)

        # Wall clock ISO computation
        if timestamp_epoch is not None and timestamp_epoch >= 1e8:
            wall_clock_epoch = float(timestamp_epoch)
        else:
            wall_clock_epoch = self.session_start_epoch + max(0.0, v_time)

        timestamp_iso = datetime.fromtimestamp(wall_clock_epoch, tz=timezone.utc).isoformat()
        ground_point = self.compute_ground_contact(bbox)
        generated_incidents = []

        # 2. Conservative entity resolution
        entity_id = self._resolve_entity_id(track_id, ground_point, frame_index, v_time)

        # 3. Spatial evaluation per zone
        for zone in self.zones:
            zone_id = zone["zone_id"]
            polygon = zone["polygon"]
            severity = zone.get("severity", "WARNING")
            zone_name = zone.get("zone_name", zone_id)
            cooldown = zone.get("cooldown_seconds", 2.0)

            state_key = (entity_id, zone_id)
            is_inside = point_in_polygon(ground_point, polygon)
            previous_state = self.tracker_states.get(state_key, "OUTSIDE")

            # Compute kinematics & history
            kinematics, trajectory_history = self._compute_kinematics(
                entity_id, ground_point, v_time, polygon
            )

            if is_inside:
                last_alert = self.last_alert_time.get(state_key, 0.0)
                event_type = None

                if previous_state == "OUTSIDE":
                    event_type = "ZONE_INTRUSION_ENTRY"
                    self.entry_times[state_key] = v_time
                    self.entry_frames[state_key] = frame_index if frame_index is not None else 0
                    dwell_time = 0.0
                else:
                    entry_t = self.entry_times.get(state_key, v_time)
                    dwell_time = max(0.0, v_time - entry_t)
                    if (v_time - last_alert) >= cooldown:
                        event_type = "ZONE_INTRUSION_OCCUPANCY"

                if event_type:
                    risk = assess_incident_risk(
                        zone_severity=severity,
                        event_type=event_type,
                        dwell_time_s=dwell_time,
                        ground_point=ground_point,
                        zone_polygon=polygon,
                        kinematics=kinematics,
                        confidence=confidence,
                    )

                    incident = {
                        "incident_id": str(uuid.uuid4()),
                        "timestamp_iso": timestamp_iso,
                        "frame_index": frame_index,
                        "video_time_s": round(v_time, 4),
                        "dwell_time_s": round(dwell_time, 2),
                        "event_type": event_type,
                        "zone_id": zone_id,
                        "zone_name": zone_name,
                        "severity": severity,
                        "track_id": track_id,
                        "confidence": round(float(confidence), 4),
                        "bbox": [round(float(c), 4) for c in bbox],
                        "ground_contact_point": [round(ground_point[0], 4), round(ground_point[1], 4)],
                        "kinematics": kinematics,
                        "risk_assessment": risk,
                        "trajectory_history": trajectory_history,
                        "session_start_iso": self.session_start_iso,
                    }
                    self._record_incident(incident)
                    generated_incidents.append(incident)
                    self.last_alert_time[state_key] = v_time

                self.tracker_states[state_key] = "INSIDE"

            else:
                if previous_state == "INSIDE":
                    # Entity exited the zone
                    entry_t = self.entry_times.get(state_key, v_time)
                    total_dwell = max(0.0, v_time - entry_t)

                    risk = assess_incident_risk(
                        zone_severity=severity,
                        event_type="ZONE_INTRUSION_EXIT",
                        dwell_time_s=total_dwell,
                        ground_point=ground_point,
                        zone_polygon=polygon,
                        kinematics=kinematics,
                        confidence=confidence,
                    )

                    exit_incident = {
                        "incident_id": str(uuid.uuid4()),
                        "timestamp_iso": timestamp_iso,
                        "frame_index": frame_index,
                        "video_time_s": round(v_time, 4),
                        "dwell_time_s": round(total_dwell, 2),
                        "event_type": "ZONE_INTRUSION_EXIT",
                        "zone_id": zone_id,
                        "zone_name": zone_name,
                        "severity": "INFO",
                        "track_id": track_id,
                        "confidence": round(float(confidence), 4),
                        "bbox": [round(float(c), 4) for c in bbox],
                        "ground_contact_point": [round(ground_point[0], 4), round(ground_point[1], 4)],
                        "kinematics": kinematics,
                        "risk_assessment": risk,
                        "trajectory_history": trajectory_history,
                        "session_start_iso": self.session_start_iso,
                    }
                    self._record_incident(exit_incident)
                    generated_incidents.append(exit_incident)
                    self.tracker_states[state_key] = "OUTSIDE"

                    # Clean up entry reference
                    if state_key in self.entry_times:
                        del self.entry_times[state_key]
                    if state_key in self.entry_frames:
                        del self.entry_frames[state_key]

        return generated_incidents

    def _record_incident(self, incident: Dict[str, Any]) -> None:
        with open(self.output_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(incident) + "\n")
