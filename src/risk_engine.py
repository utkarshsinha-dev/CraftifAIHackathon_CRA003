"""
Sentinel Vision AI - Industrial Safety Risk Assessment Engine

Provides deterministic, explainable risk scoring and severity classification
for workplace restricted zone intrusions based on:
1. Zone hazard level (CRITICAL vs WARNING vs INFO)
2. Intrusion event state (ENTRY vs continuous OCCUPANCY vs EXIT)
3. Cumulative dwell duration inside hazardous perimeters
4. Proximity / trajectory relative to hazardous machinery core
5. Motion kinematics (approaching vs stationary loitering vs receding)
"""

from typing import Dict, Any, List, Optional, Tuple
import math


def calculate_polygon_centroid(polygon: List[List[float]]) -> Tuple[float, float]:
    """Calculates the geometric centroid (x, y) of a 2D polygon."""
    if not polygon:
        return (0.5, 0.5)
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def assess_incident_risk(
    zone_severity: str,
    event_type: str,
    dwell_time_s: float,
    ground_point: Tuple[float, float],
    zone_polygon: List[List[float]],
    kinematics: Optional[Dict[str, Any]] = None,
    confidence: float = 1.0,
) -> Dict[str, Any]:
    """
    Evaluates safety risk deterministically and returns structured risk indicators
    with clear, verifiable explanations.

    Returns:
    {
        "risk_score": int (0 - 100),
        "assessed_severity": str ("CRITICAL" | "WARNING" | "INFO"),
        "explanation": str,
        "contributing_factors": Dict[str, Any]
    }
    """
    contributing_factors = {}
    score = 0

    # 1. Base score by configured zone hazard level
    zone_severity_upper = (zone_severity or "WARNING").upper()
    if zone_severity_upper == "CRITICAL":
        base_zone_pts = 50
    elif zone_severity_upper == "WARNING":
        base_zone_pts = 30
    else:
        base_zone_pts = 10
    score += base_zone_pts
    contributing_factors["zone_severity_weight"] = base_zone_pts

    # 2. Event type weighting
    if event_type == "ZONE_INTRUSION_ENTRY":
        event_pts = 20
        event_reason = "Immediate perimeter breach into restricted boundary"
    elif event_type == "ZONE_INTRUSION_OCCUPANCY":
        # Cumulative dwell risk escalation
        if dwell_time_s >= 6.0:
            dwell_penalty = 25
            event_reason = f"Severe prolonged loitering ({dwell_time_s:.1f}s) inside hazard area"
        elif dwell_time_s >= 4.0:
            dwell_penalty = 20
            event_reason = f"Substantial occupancy duration ({dwell_time_s:.1f}s) exceeding safety threshold"
        elif dwell_time_s >= 2.0:
            dwell_penalty = 15
            event_reason = f"Continued occupancy ({dwell_time_s:.1f}s) past baseline safety cooldown"
        else:
            dwell_penalty = 10
            event_reason = f"Active occupancy ({dwell_time_s:.1f}s) inside restricted boundary"
        event_pts = dwell_penalty
    elif event_type == "ZONE_INTRUSION_EXIT":
        event_pts = -15
        event_reason = f"Personnel successfully exited zone after {dwell_time_s:.1f}s dwell"
    else:
        event_pts = 0
        event_reason = f"Standard event: {event_type}"

    score += event_pts
    contributing_factors["event_type_weight"] = event_pts
    contributing_factors["dwell_time_seconds"] = round(float(dwell_time_s), 2)

    # 3. Kinematic motion direction & proximity factor
    motion_pts = 0
    motion_reason = "Stationary posture"
    if kinematics:
        motion_state = kinematics.get("motion_state", "stationary")
        speed = kinematics.get("speed_normalized", 0.0)
        direction = kinematics.get("direction", "STATIONARY")

        if motion_state == "approaching_hazard_core":
            motion_pts = 10
            motion_reason = f"Actively moving towards hazard core (heading: {direction}, speed: {speed:.3f})"
        elif motion_state == "stationary":
            motion_pts = 5
            motion_reason = "Stationary within active hazard zone"
        elif motion_state == "receding_from_hazard_core":
            motion_pts = -5
            motion_reason = f"Moving away from hazard core towards exit (heading: {direction})"
        else:
            motion_pts = 0
            motion_reason = f"Lateral motion (heading: {direction})"

    score += motion_pts
    contributing_factors["motion_dynamics_weight"] = motion_pts

    # 4. Centroid proximity (penetration depth)
    centroid = calculate_polygon_centroid(zone_polygon)
    dist_to_center = math.sqrt((ground_point[0] - centroid[0]) ** 2 + (ground_point[1] - centroid[1]) ** 2)
    contributing_factors["distance_to_zone_centroid"] = round(dist_to_center, 4)
    if dist_to_center < 0.15:
        proximity_pts = 10
        proximity_desc = "deep penetration near hazard core"
    elif dist_to_center < 0.25:
        proximity_pts = 5
        proximity_desc = "moderate zone penetration"
    else:
        proximity_pts = 0
        proximity_desc = "near boundary edge"
    score += proximity_pts
    contributing_factors["proximity_weight"] = proximity_pts

    # 5. Model confidence dampening (slight penalty if detector confidence is marginal)
    if confidence < 0.50:
        conf_pts = -5
    else:
        conf_pts = 0
    score += conf_pts
    contributing_factors["confidence_factor"] = conf_pts

    # Final score clamping to [0, 100]
    final_score = max(0, min(100, score))

    # Severity tiering
    if final_score >= 70:
        assessed_severity = "CRITICAL"
    elif final_score >= 35:
        assessed_severity = "WARNING"
    else:
        assessed_severity = "INFO"

    # Human-readable explanation
    explanation = (
        f"Assessed Risk: {assessed_severity} (Score {final_score}/100). "
        f"Base zone hazard is {zone_severity_upper}. "
        f"{event_reason} with {proximity_desc}. "
        f"Motion dynamics: {motion_reason}."
    )

    return {
        "risk_score": final_score,
        "assessed_severity": assessed_severity,
        "explanation": explanation,
        "contributing_factors": contributing_factors,
    }
