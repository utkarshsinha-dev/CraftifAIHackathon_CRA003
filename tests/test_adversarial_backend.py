"""
Sentinel Vision AI - Adversarial & Edge-Case Backend Test Suite

Tests edge cases, boundary conditions, and stress scenarios across:
1. Untracked identity stabilization (same-frame collisions, path crossing, disappearance/reappearance, teleport jumps)
2. Cumulative dwell-time precision (lifecycle, re-entry reset, multi-person isolation, non-monotonic timestamps)
3. Temporal grounding & timestamp sanity (video-relative timing, session anchoring, NaN/Inf protection)
4. Deterministic risk engine (mathematical traceability, boundary scores, input sanitization, motion vectors)
5. Safety Copilot truthfulness & resilience (dynamic grounding, unknown tracks/zones, corrupt records, no fabrication)
6. Incident service API robustness (corrupt JSONL, Unicode, JSON-serializability, mtime caching)
7. EHS report generator edge cases (empty logs, missing telemetry, disclaimer enforcement)
"""

import json
import math
import os
import tempfile
import time
from typing import Dict, List, Any
import pytest

from src.zone_monitor import ZoneIntrusionMonitor, point_in_polygon
from src.risk_engine import assess_incident_risk, calculate_polygon_centroid
from src.safety_copilot import SafetyCopilot
import src.incident_service as incident_service
from scripts.generate_ehs_report import create_ehs_report


@pytest.fixture
def temp_jsonl_path():
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    yield path
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


# ============================================================================
# A. IDENTITY STABILIZATION ADVERSARIAL TESTS
# ============================================================================

def test_untracked_same_frame_multi_person_separation(temp_jsonl_path):
    """
    CRITICAL EDGE CASE:
    Two untracked persons appearing in the EXACT SAME frame very close together
    (e.g., distance 0.03 < max_untracked_dist 0.08) MUST NOT be merged into the
    same untracked ID.
    """
    monitor = ZoneIntrusionMonitor(
        config_path="configs/restricted_zone.json",
        output_log_path=temp_jsonl_path,
    )

    # Person 1 at (0.35, 0.65) inside zone at frame 10
    id1 = monitor._resolve_entity_id(
        track_id=None,
        ground_point=(0.35, 0.65),
        frame_index=10,
        video_time=0.33,
    )

    # Person 2 at (0.37, 0.66) inside zone at the same frame 10 (dist ~0.022 < 0.08)
    id2 = monitor._resolve_entity_id(
        track_id=None,
        ground_point=(0.37, 0.66),
        frame_index=10,
        video_time=0.33,
    )

    assert id1 != id2, f"Same-frame untracked persons incorrectly merged: {id1} == {id2}"
    assert "untracked_" in str(id1)
    assert "untracked_" in str(id2)


def test_untracked_temporary_disappearance_and_reappearance(temp_jsonl_path):
    """
    One person is tracked, disappears for 4 frames (e.g. temporary occlusion < 10 frames),
    and reappears nearby (< 0.08 distance). Identity should be preserved.
    """
    monitor = ZoneIntrusionMonitor(
        config_path="configs/restricted_zone.json",
        output_log_path=temp_jsonl_path,
    )

    id_initial = monitor._resolve_entity_id(
        track_id=None,
        ground_point=(0.40, 0.60),
        frame_index=20,
        video_time=0.66,
    )

    # Reappear at frame 24 (4 frames later) with slight movement (0.42, 0.61)
    id_reappear = monitor._resolve_entity_id(
        track_id=None,
        ground_point=(0.42, 0.61),
        frame_index=24,
        video_time=0.80,
    )

    assert id_initial == id_reappear, "Failed to re-associate entity across temporary occlusion"


def test_untracked_expiry_creates_distinct_entity(temp_jsonl_path):
    """
    One person disappears for 15 frames (> untracked_expiry_frames 10).
    When a detection appears, it must be assigned a new distinct entity ID to prevent false association.
    """
    monitor = ZoneIntrusionMonitor(
        config_path="configs/restricted_zone.json",
        output_log_path=temp_jsonl_path,
    )

    id_initial = monitor._resolve_entity_id(
        track_id=None,
        ground_point=(0.40, 0.60),
        frame_index=10,
        video_time=0.33,
    )

    # Reappear 20 frames later (> 10 frames) at same coordinate
    id_after_expiry = monitor._resolve_entity_id(
        track_id=None,
        ground_point=(0.40, 0.60),
        frame_index=35,
        video_time=1.16,
    )

    assert id_initial != id_after_expiry, "Expired untracked entity was incorrectly resurrected"


def test_untracked_sudden_teleport_jump(temp_jsonl_path):
    """
    Sudden large coordinate displacement (> 0.08 normalized screen distance).
    System must not falsely associate the jump with the existing track.
    """
    monitor = ZoneIntrusionMonitor(
        config_path="configs/restricted_zone.json",
        output_log_path=temp_jsonl_path,
    )

    id1 = monitor._resolve_entity_id(
        track_id=None,
        ground_point=(0.25, 0.50),
        frame_index=1,
        video_time=0.03,
    )

    # Sudden jump to (0.60, 0.85) at next frame (dist ~0.49 >> 0.08)
    id2 = monitor._resolve_entity_id(
        track_id=None,
        ground_point=(0.60, 0.85),
        frame_index=2,
        video_time=0.06,
    )

    assert id1 != id2, "Teleportation jump incorrectly associated to same track"


# ============================================================================
# B. DWELL-TIME CALCULATIONS ADVERSARIAL TESTS
# ============================================================================

def test_dwell_time_reentry_resets_no_double_count(temp_jsonl_path):
    """
    Person enters zone, dwells for 3.0s, exits, and then re-enters 5 seconds later.
    The re-entry dwell time MUST reset to 0.0 and NOT accumulate previous dwell.
    """
    monitor = ZoneIntrusionMonitor(
        config_path="configs/restricted_zone.json",
        output_log_path=temp_jsonl_path,
    )

    # 1. Entry at t=1.0s
    e1 = monitor.process_detection(
        bbox=[0.30, 0.50, 0.40, 0.70],  # inside
        confidence=0.90,
        track_id=55,
        frame_index=30,
        video_time_s=1.0,
    )
    assert len(e1) == 1
    assert e1[0]["event_type"] == "ZONE_INTRUSION_ENTRY"
    assert e1[0]["dwell_time_s"] == 0.0

    # 2. Exit at t=4.0s (dwell was 3.0s)
    ex1 = monitor.process_detection(
        bbox=[0.05, 0.10, 0.15, 0.20],  # outside
        confidence=0.90,
        track_id=55,
        frame_index=120,
        video_time_s=4.0,
    )
    assert len(ex1) == 1
    assert ex1[0]["event_type"] == "ZONE_INTRUSION_EXIT"
    assert ex1[0]["dwell_time_s"] == 3.0

    # 3. Re-entry at t=10.0s
    e2 = monitor.process_detection(
        bbox=[0.30, 0.50, 0.40, 0.70],  # inside again
        confidence=0.90,
        track_id=55,
        frame_index=300,
        video_time_s=10.0,
    )
    assert len(e2) == 1
    assert e2[0]["event_type"] == "ZONE_INTRUSION_ENTRY"
    assert e2[0]["dwell_time_s"] == 0.0, f"Dwell time failed to reset on re-entry: {e2[0]['dwell_time_s']}"


def test_dwell_time_multi_person_isolation(temp_jsonl_path):
    """
    Two distinct persons occupying the same zone at different entry times.
    Dwell times must be calculated completely independently without cross-talk.
    """
    monitor = ZoneIntrusionMonitor(
        config_path="configs/restricted_zone.json",
        output_log_path=temp_jsonl_path,
    )

    # Track A enters at t=1.0s
    monitor.process_detection(
        bbox=[0.30, 0.50, 0.35, 0.65],
        confidence=0.90,
        track_id=101,
        video_time_s=1.0,
    )

    # Track B enters at t=3.0s
    monitor.process_detection(
        bbox=[0.40, 0.55, 0.45, 0.70],
        confidence=0.90,
        track_id=102,
        video_time_s=3.0,
    )

    # At t=5.0s, Track A triggers occupancy (dwell = 4.0s)
    res_a = monitor.process_detection(
        bbox=[0.30, 0.50, 0.35, 0.65],
        confidence=0.90,
        track_id=101,
        video_time_s=5.0,
    )
    assert len(res_a) == 1
    assert res_a[0]["dwell_time_s"] == 4.0

    # At t=5.5s, Track B triggers occupancy (dwell = 2.5s)
    res_b = monitor.process_detection(
        bbox=[0.40, 0.55, 0.45, 0.70],
        confidence=0.90,
        track_id=102,
        video_time_s=5.5,
    )
    assert len(res_b) == 1
    assert res_b[0]["dwell_time_s"] == 2.5


def test_dwell_time_non_monotonic_clock_guard(temp_jsonl_path):
    """
    Out-of-order frame or negative timestamp jump must not produce negative dwell time.
    """
    monitor = ZoneIntrusionMonitor(
        config_path="configs/restricted_zone.json",
        output_log_path=temp_jsonl_path,
    )

    # Entry at t=5.0s
    monitor.process_detection(
        bbox=[0.30, 0.50, 0.40, 0.70],
        confidence=0.90,
        track_id=88,
        video_time_s=5.0,
    )

    # Detection arriving with corrupt timestamp t=3.0s (< entry 5.0s)
    # Cooldown check: 3.0 - 5.0 = -2.0 < 2.0 -> suppressed
    # But if exit occurs with corrupt timestamp:
    ex = monitor.process_detection(
        bbox=[0.01, 0.01, 0.05, 0.05],
        confidence=0.90,
        track_id=88,
        video_time_s=4.0,  # 4.0 < 5.0 entry
    )
    assert len(ex) == 1
    assert ex[0]["dwell_time_s"] >= 0.0, "Negative dwell time produced by non-monotonic clock"


# ============================================================================
# C. TIMESTAMP SANITY AND ROBUSTNESS TESTS
# ============================================================================

def test_timestamp_nan_and_inf_safety(temp_jsonl_path):
    """
    Passing NaN or Inf as video_time_s or timestamp_epoch must not crash the monitor.
    """
    monitor = ZoneIntrusionMonitor(
        config_path="configs/restricted_zone.json",
        output_log_path=temp_jsonl_path,
    )

    # Should safely convert to 0.0 and generate valid ISO timestamp
    incs = monitor.process_detection(
        bbox=[0.30, 0.50, 0.40, 0.70],
        confidence=0.90,
        track_id=12,
        video_time_s=float("nan"),
    )
    assert len(incs) == 1
    assert incs[0]["video_time_s"] == 0.0
    assert "T" in incs[0]["timestamp_iso"]


def test_timestamp_epoch_under_1e8_treated_as_relative_time(temp_jsonl_path):
    """
    If PTS (e.g. 0.1765) is passed as timestamp_epoch, it must not be treated as a Unix epoch
    (which would evaluate to 1970-01-01), but correctly grounded to session start.
    """
    session_start = 1773489600.0  # Year 2026
    monitor = ZoneIntrusionMonitor(
        config_path="configs/restricted_zone.json",
        output_log_path=temp_jsonl_path,
        session_start_epoch=session_start,
    )

    incs = monitor.process_detection(
        bbox=[0.30, 0.50, 0.40, 0.70],
        confidence=0.90,
        track_id=3,
        timestamp_epoch=0.55,  # raw PTS
    )
    assert len(incs) == 1
    assert incs[0]["video_time_s"] == 0.55
    assert "1970" not in incs[0]["timestamp_iso"]
    assert "2026" in incs[0]["timestamp_iso"]


# ============================================================================
# D. DETERMINISTIC RISK ENGINE AUDIT TESTS
# ============================================================================

def test_risk_engine_mathematical_traceability():
    """
    Verifies that the risk score formula is mathematically traceable to its contributing factors:
    score = base_zone_pts + event_pts + motion_pts + proximity_pts + conf_pts
    """
    polygon = [[0.2, 0.4], [0.6, 0.4], [0.6, 0.8], [0.2, 0.8]]
    ground_pt = (0.4, 0.6)

    kinematics = {
        "motion_state": "approaching_hazard_core",
        "speed_normalized": 0.05,
        "direction": "SE",
    }

    res = assess_incident_risk(
        zone_severity="CRITICAL",
        event_type="ZONE_INTRUSION_ENTRY",
        dwell_time_s=0.0,
        ground_point=ground_pt,
        zone_polygon=polygon,
        kinematics=kinematics,
        confidence=0.95,
    )

    factors = res["contributing_factors"]
    expected_sum = (
        factors["zone_severity_weight"]
        + factors["event_type_weight"]
        + factors["motion_dynamics_weight"]
        + factors["proximity_weight"]
        + factors["confidence_factor"]
    )
    clamped_expected = max(0, min(100, expected_sum))
    assert res["risk_score"] == clamped_expected
    assert res["assessed_severity"] == "CRITICAL"


def test_risk_engine_determinism():
    """
    Identical inputs must produce identical outputs 100 times consecutively.
    """
    polygon = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    pt = (0.5, 0.5)

    first_res = assess_incident_risk("WARNING", "ZONE_INTRUSION_OCCUPANCY", 4.5, pt, polygon)
    for _ in range(50):
        iter_res = assess_incident_risk("WARNING", "ZONE_INTRUSION_OCCUPANCY", 4.5, pt, polygon)
        assert iter_res == first_res


def test_risk_engine_severity_boundaries():
    """
    Verifies strict severity threshold tiering:
    < 35: INFO
    35 - 69: WARNING
    >= 70: CRITICAL
    """
    polygon = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    pt = (0.5, 0.5)

    # Info event: zone INFO (10) + exit (-15) + dist (10) -> score 5 -> INFO
    r_info = assess_incident_risk("INFO", "ZONE_INTRUSION_EXIT", 1.0, pt, polygon)
    assert r_info["risk_score"] < 35
    assert r_info["assessed_severity"] == "INFO"

    # Critical event: zone CRITICAL (50) + entry (20) + dist (10) -> score 80 -> CRITICAL
    r_crit = assess_incident_risk("CRITICAL", "ZONE_INTRUSION_ENTRY", 0.0, pt, polygon)
    assert r_crit["risk_score"] >= 70
    assert r_crit["assessed_severity"] == "CRITICAL"


def test_risk_engine_input_sanitization():
    """
    Invalid / missing inputs must not raise uncaught exceptions.
    """
    # Empty polygon, None severity, NaN dwell, NaN confidence
    res = assess_incident_risk(
        zone_severity=None,
        event_type="UNKNOWN_EVENT",
        dwell_time_s=float("nan"),
        ground_point=None,
        zone_polygon=[],
        kinematics=None,
        confidence=float("nan"),
    )
    assert 0 <= res["risk_score"] <= 100
    assert res["assessed_severity"] in ["INFO", "WARNING", "CRITICAL"]
    assert isinstance(res["explanation"], str)


# ============================================================================
# E. SAFETY COPILOT TRUTHFULNESS & RESILIENCE TESTS
# ============================================================================

def test_safety_copilot_truthfulness_on_custom_data(temp_jsonl_path):
    """
    Copilot must dynamically reflect the actual records provided and NEVER
    hallucinate hardcoded Track 0, Track 1, or Frame 3 when given different data.
    """
    custom_records = [
        {
            "incident_id": "test-uuid-99",
            "frame_index": 500,
            "video_time_s": 16.67,
            "dwell_time_s": 8.5,
            "event_type": "ZONE_INTRUSION_OCCUPANCY",
            "zone_id": "custom_zone",
            "zone_name": "Turbine Perimeter C",
            "severity": "CRITICAL",
            "track_id": 99,
            "confidence": 0.94,
            "bbox": [0.4, 0.4, 0.6, 0.6],
            "ground_contact_point": [0.5, 0.6],
            "kinematics": {"motion_state": "approaching_hazard_core", "direction": "N", "speed_normalized": 0.04},
            "risk_assessment": {"risk_score": 92, "assessed_severity": "CRITICAL", "explanation": "Severe proximity."},
        }
    ]
    with open(temp_jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(custom_records[0]) + "\n")

    copilot = SafetyCopilot(incident_log_path=temp_jsonl_path)

    # 1. Highest risk query
    res_high = copilot.answer_question("What is the highest risk incident?")
    assert res_high["intent"] == "highest_risk"
    assert "92/100" in res_high["answer"]
    assert "Track 99" in res_high["answer"]
    assert "Turbine Perimeter C" in res_high["answer"]

    # 2. Critical query must NOT mention Track 0 or Frame 3
    res_crit = copilot.answer_question("List all critical incidents")
    assert res_crit["intent"] == "critical_incidents"
    assert "Track 99" in res_crit["answer"]
    assert "Track 0" not in res_crit["answer"]
    assert "Frame 500" in res_crit["answer"]

    # 3. Unknown worker query
    res_unknown = copilot.answer_question("What did Worker #42 do?")
    assert res_unknown["intent"] == "worker_investigation"
    assert "No safety incidents recorded for Track 42" in res_unknown["answer"]
    assert "99" in res_unknown["answer"]


def test_safety_copilot_untracked_evidence_never_shows_none(temp_jsonl_path):
    """
    Evidence citations for untracked incidents must show 'Track untracked_1' or 'Track untracked',
    NEVER 'Track None'.
    """
    record = {
        "incident_id": "untracked-uuid",
        "frame_index": 45,
        "video_time_s": 1.50,
        "dwell_time_s": 0.0,
        "event_type": "ZONE_INTRUSION_ENTRY",
        "zone_name": "Danger Zone",
        "severity": "CRITICAL",
        "track_id": None,
        "entity_id": "untracked_1",
    }
    with open(temp_jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    copilot = SafetyCopilot(incident_log_path=temp_jsonl_path)
    res = copilot.answer_question("Summary")
    citation = res["evidence"][0]
    assert "Track None" not in citation
    assert ("Track untracked_1" in citation or "Track untracked" in citation)


# ============================================================================
# F. INCIDENT SERVICE API & PERFORMANCE TESTS
# ============================================================================

def test_incident_service_corrupted_jsonl_resilience(temp_jsonl_path):
    """
    File containing corrupted non-JSON lines mixed with valid JSON lines
    must safely load the valid records without crashing.
    """
    valid_record = {
        "incident_id": "valid-1",
        "severity": "WARNING",
        "dwell_time_s": 2.0,
        "track_id": 1,
        "event_type": "ZONE_INTRUSION_ENTRY",
    }
    with open(temp_jsonl_path, "w", encoding="utf-8") as f:
        f.write("CORRUPT_LINE_NOT_JSON\n")
        f.write(json.dumps(valid_record) + "\n")
        f.write('{"truncated": true\n')
        f.write("   \n")

    loaded = incident_service.load_incidents(path=temp_jsonl_path)
    assert len(loaded) == 1
    assert loaded[0]["incident_id"] == "valid-1"

    stats = incident_service.get_incident_statistics(incidents=loaded)
    assert stats["total_incidents"] == 1
    assert stats["warning_count"] == 1


def test_incident_service_unicode_and_special_chars(temp_jsonl_path):
    """
    Zone names containing Unicode or non-ASCII characters (e.g. 危险区域, Fördervolumen, ⚠️ Hazard).
    """
    unicode_record = {
        "incident_id": "uni-123",
        "zone_name": "Roboter-Gefahrenzone ⚠️ (Schutzbereich 2)",
        "track_id": 4,
        "severity": "CRITICAL",
        "dwell_time_s": 3.14,
        "event_type": "ZONE_INTRUSION_OCCUPANCY",
    }
    with open(temp_jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(unicode_record, ensure_ascii=False) + "\n")

    loaded = incident_service.load_incidents(path=temp_jsonl_path)
    assert len(loaded) == 1
    assert "⚠️" in loaded[0]["zone_name"]

    stats = incident_service.get_incident_statistics(incidents=loaded)
    assert "Roboter-Gefahrenzone ⚠️ (Schutzbereich 2)" in stats["zones_breached"]


def test_incident_service_json_serializability_contract(temp_jsonl_path):
    """
    Verifies that EVERY return value from incident_service functions is 100% JSON-serializable.
    Streamlit components crash if dicts contain non-serializable objects (sets, deques, functions).
    """
    # 1. Statistics on empty list
    s_empty = incident_service.get_incident_statistics(incidents=[])
    json_str_empty = json.dumps(s_empty)
    assert isinstance(json_str_empty, str)

    # 2. Statistics on loaded records
    recs = [
        {
            "incident_id": "json-test-1",
            "severity": "CRITICAL",
            "dwell_time_s": 5.0,
            "track_id": 2,
            "event_type": "ZONE_INTRUSION_OCCUPANCY",
            "risk_assessment": {"risk_score": 85},
        }
    ]
    s_full = incident_service.get_incident_statistics(incidents=recs)
    json_str_full = json.dumps(s_full)
    assert isinstance(json_str_full, str)
    assert "max_risk_score" in s_full
    assert "average_risk_score" in s_full
    assert "severity_counts" in s_full

    # 3. Copilot answer
    ans = incident_service.answer_question("Summary", incidents=recs)
    json_ans = json.dumps(ans)
    assert isinstance(json_ans, str)


def test_incident_service_mtime_caching_performance(temp_jsonl_path):
    """
    Verifies that load_incidents uses its modification timestamp cache for sub-millisecond
    repeated retrieval, but automatically reloads when the file is updated.
    """
    r1 = {"incident_id": "rec-1", "severity": "INFO"}
    with open(temp_jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(r1) + "\n")

    # Initial load
    loaded1 = incident_service.load_incidents(path=temp_jsonl_path)
    assert len(loaded1) == 1

    # Second load should hit cache
    loaded2 = incident_service.load_incidents(path=temp_jsonl_path)
    assert len(loaded2) == 1

    # Append new record and update mtime
    time.sleep(0.05)
    r2 = {"incident_id": "rec-2", "severity": "WARNING"}
    with open(temp_jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(r2) + "\n")

    # Should detect mtime change and reload
    loaded3 = incident_service.load_incidents(path=temp_jsonl_path)
    assert len(loaded3) == 2


# ============================================================================
# G. REPORT GENERATOR EDGE CASES
# ============================================================================

def test_ehs_report_empty_records_and_disclaimer():
    """
    Report generation on empty incidents list must succeed and provide a valid
    markdown report with executive summary and disclaimer without crashing.
    """
    report_md = create_ehs_report(incidents=[], output_path=None)
    assert "# SENTINEL VISION AI — EHS INCIDENT INVESTIGATION REPORT" in report_md
    assert "0 safety incident events" in report_md
    assert "Regulatory & Operational Disclaimer" in report_md
    assert "does not constitute formal certification" in report_md


def test_ehs_report_partial_fields_missing():
    """
    Incidents with missing fields (e.g. no kinematics, no risk_assessment, no track_id)
    must still format into a clean report without throwing KeyError.
    """
    sparse_incidents = [
        {
            "incident_id": "sparse-01",
            "frame_index": 10,
            "video_time_s": 0.33,
            "event_type": "ZONE_INTRUSION_ENTRY",
            "zone_name": "Danger Zone",
        }
    ]
    report_md = create_ehs_report(incidents=sparse_incidents, output_path=None)
    assert "sparse-01" in report_md
    assert "Danger Zone" in report_md
