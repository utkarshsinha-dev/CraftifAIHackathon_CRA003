"""
Sentinel Vision AI - Backend Intelligence Test Suite

Comprehensive tests for:
1. Timestamp correctness & video-relative timing (P0-A)
2. Conservative untracked identity stabilization & jitter resilience (P0-B)
3. Cumulative dwell-time analytics & exit retention (P1-C)
4. Movement direction, velocity vectors, and trajectory kinematics (P1-D)
5. Deterministic risk engine scoring & explainability (P1-E)
6. Safety Copilot intent parsing & evidence grounding (P1 Copilot)
7. Empty and malformed incident file resilience
8. Incident service contract interfaces
9. EHS investigation report generation
"""

import json
import os
import tempfile
import pytest

from src.zone_monitor import ZoneIntrusionMonitor, point_in_polygon
from src.deepstream_adapter import DeepStreamMetadataAdapter, DEEPSTREAM_UNTRACKED_ID
from src.risk_engine import assess_incident_risk, calculate_polygon_centroid
from src.safety_copilot import SafetyCopilot
from src import incident_service
from scripts.generate_ehs_report import create_ehs_report


@pytest.fixture
def temp_output_log():
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    if os.path.exists(path):
        os.remove(path)
    yield path
    if os.path.exists(path):
        os.remove(path)


def test_timestamp_correctness_and_video_relative_timing(temp_output_log):
    """
    Verifies that video PTS seconds are not mapped to 1970 Unix epoch,
    and video_time_s is explicitly tracked alongside session ISO.
    """
    monitor = ZoneIntrusionMonitor(
        config_path="configs/restricted_zone.json",
        output_log_path=temp_output_log,
        session_start_epoch=1770000000.0,  # Explicit session epoch
    )

    # Process detection at video time 1.5 seconds
    incidents = monitor.process_detection(
        bbox=[0.30, 0.50, 0.50, 0.80],
        confidence=0.92,
        track_id=10,
        frame_index=45,
        video_time_s=1.50,
    )
    assert len(incidents) == 1
    rec = incidents[0]

    # Timestamp verification
    assert rec["video_time_s"] == 1.50
    assert rec["frame_index"] == 45
    # Wall clock epoch = 1770000000.0 + 1.50 -> year 2026, NOT 1970
    assert not rec["timestamp_iso"].startswith("1970")
    assert "T" in rec["timestamp_iso"]
    assert "session_start_iso" in rec


def test_untracked_identity_stabilization_and_jitter(temp_output_log):
    """
    Verifies that consecutive frames with missing tracker IDs and minor coordinate
    jitter maintain identity continuity and DO NOT flood multiple ENTRY alerts.
    """
    monitor = ZoneIntrusionMonitor(
        config_path="configs/restricted_zone.json",
        output_log_path=temp_output_log,
    )

    # 5 frames of an untracked person inside Zone 1 with slight jitter
    # Normalized coords inside zone: x around 0.35, y around 0.65
    jitter_boxes = [
        [0.300, 0.450, 0.400, 0.650],  # Frame 1: ground = (0.350, 0.650)
        [0.302, 0.451, 0.401, 0.652],  # Frame 2: ground = (0.3515, 0.652) (small jitter)
        [0.299, 0.449, 0.398, 0.648],  # Frame 3: ground = (0.3485, 0.648)
        [0.301, 0.450, 0.400, 0.651],  # Frame 4
        [0.300, 0.450, 0.400, 0.650],  # Frame 5
    ]

    total_entry_alerts = 0
    total_incidents = []

    for idx, bbox in enumerate(jitter_boxes, 1):
        incs = monitor.process_detection(
            bbox=bbox,
            confidence=0.88,
            track_id=None,  # Missing / untracked ID
            frame_index=idx,
            video_time_s=idx * 0.033,  # 30 fps
        )
        if incs:
            for inc in incs:
                total_incidents.append(inc)
                if inc["event_type"] == "ZONE_INTRUSION_ENTRY":
                    total_entry_alerts += 1

    # Conservative stabilization MUST emit exactly 1 ENTRY alert, not 5!
    assert total_entry_alerts == 1, f"Expected 1 stabilized ENTRY, got {total_entry_alerts}"
    assert len(total_incidents) == 1  # Other frames suppressed by cooldown (dt < 2.0s)


def test_untracked_distinct_entities_separated(temp_output_log):
    """
    Verifies that two distinct untracked entities in separate locations
    receive different stabilized identities and do not erroneously merge.
    """
    monitor = ZoneIntrusionMonitor(
        config_path="configs/restricted_zone.json",
        output_log_path=temp_output_log,
    )

    # Entity 1 at x ~ 0.25 (inside Zone 1)
    e1_incs = monitor.process_detection(
        bbox=[0.20, 0.50, 0.30, 0.70],
        confidence=0.90,
        track_id=None,
        frame_index=1,
        video_time_s=0.1,
    )
    # Entity 2 at x ~ 0.55 (inside Zone 1, far from Entity 1: dx = 0.30 > 0.08 max dist)
    e2_incs = monitor.process_detection(
        bbox=[0.50, 0.50, 0.60, 0.70],
        confidence=0.91,
        track_id=None,
        frame_index=1,
        video_time_s=0.1,
    )

    assert len(e1_incs) == 1
    assert len(e2_incs) == 1
    assert e1_incs[0]["event_type"] == "ZONE_INTRUSION_ENTRY"
    assert e2_incs[0]["event_type"] == "ZONE_INTRUSION_ENTRY"
    # Entities must be tracked under separate keys in tracker_states
    keys = list(monitor.tracker_states.keys())
    assert len(keys) == 2
    assert keys[0][0] != keys[1][0]


def test_dwell_time_calculation_entry_and_exit(temp_output_log):
    """
    Verifies cumulative dwell duration calculation from entry, through occupancy, to exit.
    """
    monitor = ZoneIntrusionMonitor(
        config_path="configs/restricted_zone.json",
        output_log_path=temp_output_log,
    )

    # 1. Entry at t=1.0s
    inc_entry = monitor.process_detection(
        bbox=[0.30, 0.50, 0.40, 0.70],
        confidence=0.90,
        track_id=42,
        frame_index=30,
        video_time_s=1.0,
    )
    assert len(inc_entry) == 1
    assert inc_entry[0]["event_type"] == "ZONE_INTRUSION_ENTRY"
    assert inc_entry[0]["dwell_time_s"] == 0.0

    # 2. Occupancy alert at t=3.5s (2.5s dwell > 2.0s cooldown)
    inc_occ = monitor.process_detection(
        bbox=[0.30, 0.50, 0.40, 0.70],
        confidence=0.91,
        track_id=42,
        frame_index=105,
        video_time_s=3.5,
    )
    assert len(inc_occ) == 1
    assert inc_occ[0]["event_type"] == "ZONE_INTRUSION_OCCUPANCY"
    assert inc_occ[0]["dwell_time_s"] == 2.50

    # 3. Exit at t=5.0s (moves outside Zone 1: polygon is x 0.20-0.65, y 0.45-0.90)
    inc_exit = monitor.process_detection(
        bbox=[0.05, 0.10, 0.15, 0.30],  # outside
        confidence=0.89,
        track_id=42,
        frame_index=150,
        video_time_s=5.0,
    )
    assert len(inc_exit) == 1
    assert inc_exit[0]["event_type"] == "ZONE_INTRUSION_EXIT"
    assert inc_exit[0]["dwell_time_s"] == 4.0  # 5.0 - 1.0 = 4.0s total dwell


def test_movement_and_trajectory_kinematics(temp_output_log):
    """
    Verifies kinematics calculations: velocity, speed, direction, and motion classification.
    """
    monitor = ZoneIntrusionMonitor(
        config_path="configs/restricted_zone.json",
        output_log_path=temp_output_log,
    )

    # Step 1: Initial detection at t=0.0s
    monitor.process_detection(
        bbox=[0.20, 0.50, 0.30, 0.70],
        confidence=0.90,
        track_id=99,
        video_time_s=0.0,
    )

    # Step 2: Step eastward at t=0.5s (dx = +0.05)
    incs = monitor.process_detection(
        bbox=[0.25, 0.50, 0.35, 0.70],
        confidence=0.90,
        track_id=99,
        video_time_s=0.5,
    )

    # Step 3: Call compute_kinematics with next position at t=1.0s (dx = +0.05)
    kin, traj = monitor._compute_kinematics(99, (0.35, 0.70), 1.0, monitor.zones[0]["polygon"])
    assert kin["speed_normalized"] > 0.0
    assert kin["direction"] in ["E", "SE", "NE"]


def test_risk_engine_classification_and_factors():
    """
    Verifies deterministic risk scoring, score bounds [0, 100], and explanations.
    """
    polygon = [[0.2, 0.4], [0.6, 0.4], [0.6, 0.8], [0.2, 0.8]]
    ground_pt = (0.4, 0.6)

    # Critical zone entry
    risk_entry = assess_incident_risk(
        zone_severity="CRITICAL",
        event_type="ZONE_INTRUSION_ENTRY",
        dwell_time_s=0.0,
        ground_point=ground_pt,
        zone_polygon=polygon,
    )
    assert 0 <= risk_entry["risk_score"] <= 100
    assert risk_entry["assessed_severity"] in ["CRITICAL", "WARNING"]
    assert "CRITICAL" in risk_entry["explanation"]
    assert "zone_severity_weight" in risk_entry["contributing_factors"]

    # Exit event should reduce risk score
    risk_exit = assess_incident_risk(
        zone_severity="CRITICAL",
        event_type="ZONE_INTRUSION_EXIT",
        dwell_time_s=3.0,
        ground_point=ground_pt,
        zone_polygon=polygon,
    )
    assert risk_exit["risk_score"] < risk_entry["risk_score"]


def test_safety_copilot_intents_and_evidence(temp_output_log):
    """
    Verifies natural-language question answering backed by exact evidence citations.
    """
    monitor = ZoneIntrusionMonitor(
        config_path="configs/restricted_zone.json",
        output_log_path=temp_output_log,
    )
    # Generate 2 incidents
    monitor.process_detection(
        bbox=[0.30, 0.50, 0.40, 0.70],
        confidence=0.95,
        track_id=0,
        frame_index=10,
        video_time_s=0.33,
    )
    monitor.process_detection(
        bbox=[0.30, 0.50, 0.40, 0.70],
        confidence=0.95,
        track_id=0,
        frame_index=80,
        video_time_s=2.66,
    )

    copilot = SafetyCopilot(incident_log_path=temp_output_log)
    stats = copilot.get_summary_statistics()
    assert stats["total_incidents"] == 2
    assert "0" in stats["unique_tracks"]

    # Test summary query
    res_summary = copilot.answer_question("Give me a safety summary of the session")
    assert res_summary["intent"] == "incident_summary"
    assert "2 safety events" in res_summary["answer"]
    assert len(res_summary["evidence"]) > 0

    # Test dwell query
    res_dwell = copilot.answer_question("Who stayed the longest in the restricted zone?")
    assert res_dwell["intent"] == "longest_occupancy"
    assert "Track 0" in res_dwell["answer"]

    # Test critical query
    res_crit = copilot.answer_question("Show me critical incidents")
    assert res_crit["intent"] == "critical_incidents"

    # Test worker query
    res_worker = copilot.answer_question("What did Worker #0 do?")
    assert res_worker["intent"] == "worker_investigation"
    assert "Track 0" in res_worker["answer"]


def test_safety_copilot_empty_and_missing_log():
    """
    Verifies that querying with an empty or non-existent log runs safely without crashing.
    """
    copilot = SafetyCopilot(incident_log_path="data/output/non_existent_file.jsonl")
    res = copilot.answer_question("What happened?")
    assert res["intent"] == "empty_log"
    assert "No incidents" in res["answer"]


def test_incident_service_frontend_contract(temp_output_log):
    """
    Verifies that incident_service provides clean, stable, JSON-serializable functions.
    """
    monitor = ZoneIntrusionMonitor(
        config_path="configs/restricted_zone.json",
        output_log_path=temp_output_log,
    )
    incs = monitor.process_detection(
        bbox=[0.30, 0.50, 0.40, 0.70],
        confidence=0.92,
        track_id=7,
        frame_index=15,
        video_time_s=0.50,
    )
    event_id = incs[0]["incident_id"]

    # Contract 1: load_incidents
    loaded = incident_service.load_incidents(path=temp_output_log)
    assert len(loaded) == 1

    # Contract 2: get_incident_statistics
    stats = incident_service.get_incident_statistics(incidents=loaded)
    assert stats["total_incidents"] == 1
    assert "7" in stats["unique_tracks"]

    # Contract 3: get_incident_details
    details = incident_service.get_incident_details(event_id, incidents=loaded)
    assert details is not None
    assert details["incident_id"] == event_id

    # Contract 4: answer_question
    ans = incident_service.answer_question("Summary", incidents=loaded)
    assert "answer" in ans
    assert "evidence" in ans

    # Contract 5: generate_incident_report
    report_md = incident_service.generate_incident_report(
        output_path=temp_output_log.replace(".jsonl", ".md"),
        incidents=loaded,
    )
    assert len(report_md) > 100
    assert "SENTINEL VISION AI" in report_md


def test_ehs_report_generation(temp_output_log):
    """
    Verifies that create_ehs_report produces a valid markdown document
    with KPIs, incident tables, and disclaimers.
    """
    out_md = temp_output_log.replace(".jsonl", "_report.md")
    report = create_ehs_report(
        incidents=[{
            "incident_id": "test-uuid-1234",
            "timestamp_iso": "2026-11-14T12:00:00+00:00",
            "frame_index": 5,
            "video_time_s": 0.17,
            "dwell_time_s": 0.0,
            "event_type": "ZONE_INTRUSION_ENTRY",
            "zone_id": "zone_restricted_01",
            "zone_name": "Heavy Equipment Perimeter A",
            "severity": "CRITICAL",
            "track_id": 0,
            "confidence": 0.65,
            "bbox": [0.5, 0.5, 0.6, 0.7],
            "ground_contact_point": [0.55, 0.7],
            "kinematics": {"direction": "STATIONARY", "motion_state": "stationary", "speed_normalized": 0.0},
            "risk_assessment": {"risk_score": 85, "assessed_severity": "CRITICAL", "explanation": "Test alert."},
        }],
        output_path=out_md,
    )
    assert os.path.isfile(out_md)
    assert "EHS INCIDENT INVESTIGATION REPORT" in report
    assert "Track 0" in report
    assert "Disclaimer" in report
    os.remove(out_md)
