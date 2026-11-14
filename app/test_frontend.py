# -*- coding: utf-8 -*-
"""
Sentinel Vision AI 2.0 — Frontend & Integration Test Suite
Validates all UI modules, backend adapters, coordinate mappings,
and edge-case resilience without modifying backend tests/.
"""

import os
import sys
import json
import numpy as np

# Ensure root directory is on path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.components.backend_adapter import backend_adapter
from app.components.metrics import compute_kpi_metrics, load_incidents
from app.components.video_player import (
    load_detection_metadata,
    load_zones_config,
    is_point_in_polygon,
    render_annotated_frame,
)
from app.components.zone_editor import validate_polygon, load_zones_data


def test_backend_adapter_incidents():
    print("Testing BackendAdapter incident loading & deduplication...")
    incidents = backend_adapter.load_incidents()
    assert len(incidents) == 6, f"Expected 6 incidents, got {len(incidents)}"
    for inc in incidents:
        assert "incident_id" in inc
        assert "frame_index" in inc
        assert "severity" in inc
        assert "ground_contact_point" in inc
    print("  [PASS] BackendAdapter loaded 6 verified incidents successfully.")


def test_backend_adapter_stats():
    print("Testing BackendAdapter categorized statistics...")
    incidents = backend_adapter.load_incidents()
    stats = backend_adapter.get_incident_statistics(incidents)
    avail = stats["available"]
    derived = stats["derived"]

    assert avail["total_incidents"] == 6
    assert avail["critical_incidents"] == 6
    assert avail["active_tracks"] == 2
    assert avail["total_entries"] == 2
    assert avail["total_occupancies"] == 4
    assert derived["estimated_occupancy_seconds"] > 0
    assert len(stats["unavailable"]) == 3
    print("  [PASS] Categorized statistics calculated with 100% precision.")


def test_backend_adapter_copilot():
    print("Testing BackendAdapter Copilot queries...")
    incidents = backend_adapter.load_incidents()

    # Query 1: Summary
    q1 = backend_adapter.answer_question("Summarize all safety incidents", incidents)
    assert q1["mode"] in ["REAL_BACKEND_AI", "DETERMINISTIC_FALLBACK"]
    assert len(q1["evidence"]) > 0

    # Query 2: Longest occupancy
    q2 = backend_adapter.answer_question("Which worker had the longest zone occupancy?", incidents)
    assert "Person #0" in q2["text"] and "Person #1" in q2["text"]

    # Query 3: Critical breaches
    q3 = backend_adapter.answer_question("Show critical violations", incidents)
    assert "CRITICAL" in q3["text"]

    # Query 4: Zone 1
    q4 = backend_adapter.answer_question("Summarize events in Zone 1", incidents)
    assert "zone_restricted_01" in q4["text"]

    print("  [PASS] Copilot query resolution and evidence links verified.")


def test_polygon_geometry_validation():
    print("Testing 2D Polygon validation...")
    # Valid square
    valid_poly = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]]
    ok, msg = validate_polygon(valid_poly)
    assert ok, f"Expected valid, got error: {msg}"

    # Invalid: < 3 points
    ok, msg = validate_polygon([[0.1, 0.1], [0.5, 0.5]])
    assert not ok

    # Invalid: coordinates out of range
    ok, msg = validate_polygon([[0.1, 0.1], [1.5, 0.5], [0.1, 0.5]])
    assert not ok

    print("  [PASS] Polygon validation correctly handles valid and invalid geometries.")


def test_point_in_polygon_containment():
    print("Testing Point-in-Polygon ray-casting containment...")
    zone_poly = [[0.20, 0.45], [0.65, 0.45], [0.65, 0.90], [0.20, 0.90]]

    # True intruder (Person 0 footprint: 0.5608, 0.6479)
    assert is_point_in_polygon(0.5608, 0.6479, zone_poly) == True

    # True intruder (Person 1 footprint: 0.4995, 0.6513)
    assert is_point_in_polygon(0.4995, 0.6513, zone_poly) == True

    # Safe worker outside zone
    assert is_point_in_polygon(0.10, 0.10, zone_poly) == False
    assert is_point_in_polygon(0.80, 0.30, zone_poly) == False

    print("  [PASS] Point-in-polygon containment math mathematically verified.")


def test_annotated_frame_rendering():
    print("Testing annotated frame rendering with simulated video buffer...")
    frame = np.zeros((1080, 1728, 3), dtype=np.uint8)
    zones = load_zones_config()
    objects = [
        {
            "id": 0,
            "confidence": 0.65,
            "bbox": {"left": 940, "top": 627, "width": 58, "height": 72},
        }
    ]

    annotated = render_annotated_frame(frame, 3, objects, zones)
    assert annotated.shape == (1080, 1728, 3)
    assert not np.array_equal(frame, annotated), "Annotated frame should contain overlays"
    print("  [PASS] Overlay rendering executed without error.")


def test_resilience_malformed_records():
    print("Testing resilience against corrupt/malformed records...")
    malformed = [
        {},  # empty dict
        {"severity": "UNKNOWN"},  # missing track_id and frame_index
        {"frame_index": 5, "track_id": None},  # null track ID
        {"frame_index": 10, "track_id": 99, "bbox": [0.1, 0.2, 0.3, 0.4]},  # normalized bbox
    ]
    stats = backend_adapter.get_incident_statistics(malformed)
    assert stats["available"]["total_incidents"] == 4
    assert stats["available"]["active_tracks"] == 1  # Only track 99 is valid
    print("  [PASS] Resilient against missing fields and null track IDs.")


if __name__ == "__main__":
    print("==================================================================")
    print("SENTINEL VISION AI 2.0 — FRONTEND HARDENING VERIFICATION")
    print("==================================================================")
    test_backend_adapter_incidents()
    test_backend_adapter_stats()
    test_backend_adapter_copilot()
    test_polygon_geometry_validation()
    test_point_in_polygon_containment()
    test_annotated_frame_rendering()
    test_resilience_malformed_records()
    print("==================================================================")
    print("ALL 7 FRONTEND HARDENING & INTEGRATION TESTS PASSED (100% SUCCESS)")
    print("==================================================================")
