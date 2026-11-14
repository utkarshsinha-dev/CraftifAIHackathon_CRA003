"""
Sentinel Vision AI - Performance & Stress Benchmarks

Measures latency and throughput under realistic and stress scale:
- Small logs (10 incidents)
- Medium logs (100 incidents)
- Large logs (1,000 incidents)
- Repeated Copilot Q&A latency
- Report generation speed
"""

import json
import os
import tempfile
import time
import pytest

import src.incident_service as incident_service
from scripts.generate_ehs_report import create_ehs_report


def generate_synthetic_incidents(count: int) -> list:
    incidents = []
    for i in range(count):
        incidents.append({
            "incident_id": f"bench-uuid-{i:05d}",
            "frame_index": i * 5,
            "video_time_s": round(i * 0.166, 2),
            "dwell_time_s": round((i % 20) * 0.5, 2),
            "event_type": "ZONE_INTRUSION_ENTRY" if (i % 20 == 0) else "ZONE_INTRUSION_OCCUPANCY",
            "zone_id": f"zone_{i % 3}",
            "zone_name": f"Hazard Perimeter {chr(65 + (i % 3))}",
            "severity": "CRITICAL" if (i % 2 == 0) else "WARNING",
            "track_id": i % 10,
            "confidence": 0.88 + ((i % 10) * 0.01),
            "bbox": [0.2, 0.4, 0.4, 0.6],
            "ground_contact_point": [0.3, 0.6],
            "kinematics": {
                "motion_state": "approaching_hazard_core" if (i % 3 == 0) else "stationary",
                "direction": "S",
                "speed_normalized": 0.02,
            },
            "risk_assessment": {
                "risk_score": 50 + (i % 45),
                "assessed_severity": "CRITICAL" if (i % 2 == 0) else "WARNING",
                "explanation": f"Synthetic benchmark risk record {i}",
            },
        })
    return incidents


def test_performance_small_scale(tmp_path):
    """Small incident scale (10 records): sub-millisecond expected."""
    log_file = str(tmp_path / "small.jsonl")
    records = generate_synthetic_incidents(10)
    with open(log_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    t0 = time.perf_counter()
    loaded = incident_service.load_incidents(path=log_file)
    load_time = time.perf_counter() - t0
    assert len(loaded) == 10
    assert load_time < 0.05  # < 50ms

    t0 = time.perf_counter()
    stats = incident_service.get_incident_statistics(incidents=loaded)
    stats_time = time.perf_counter() - t0
    assert stats["total_incidents"] == 10
    assert stats_time < 0.01  # < 10ms


def test_performance_medium_scale(tmp_path):
    """Medium scale (100 records): high responsiveness."""
    log_file = str(tmp_path / "medium.jsonl")
    records = generate_synthetic_incidents(100)
    with open(log_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    t0 = time.perf_counter()
    loaded = incident_service.load_incidents(path=log_file)
    load_time = time.perf_counter() - t0
    assert len(loaded) == 100
    assert load_time < 0.10

    # Copilot query latency on 100 records
    t0 = time.perf_counter()
    ans = incident_service.answer_question("Summary", incidents=loaded)
    copilot_time = time.perf_counter() - t0
    assert ans["intent"] == "incident_summary"
    assert copilot_time < 0.05  # < 50ms


def test_performance_large_scale_stress(tmp_path):
    """Large scale stress test (1,000 records)."""
    log_file = str(tmp_path / "large.jsonl")
    records = generate_synthetic_incidents(1000)
    with open(log_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # 1. Uncached initial load
    t0 = time.perf_counter()
    loaded = incident_service.load_incidents(path=log_file)
    uncached_time = time.perf_counter() - t0
    assert len(loaded) == 1000
    assert uncached_time < 0.50  # Under 500ms for 1,000 JSON lines disk parse

    # 2. Cached subsequent load
    t0 = time.perf_counter()
    loaded_cached = incident_service.load_incidents(path=log_file)
    cached_time = time.perf_counter() - t0
    assert len(loaded_cached) == 1000
    assert cached_time < 0.02  # Cache hits under 20ms

    # 3. Statistics calculation
    t0 = time.perf_counter()
    stats = incident_service.get_incident_statistics(incidents=loaded)
    stats_time = time.perf_counter() - t0
    assert stats["total_incidents"] == 1000
    assert stats_time < 0.05

    # 4. Copilot query on 1,000 records
    t0 = time.perf_counter()
    ans = incident_service.answer_question("What is the highest risk incident?", incidents=loaded)
    ans_time = time.perf_counter() - t0
    assert ans["intent"] == "highest_risk"
    assert ans_time < 0.05

    # 5. Report generation on 1,000 records
    out_report = str(tmp_path / "LARGE_REPORT.md")
    t0 = time.perf_counter()
    rep = create_ehs_report(incidents=loaded, output_path=out_report)
    rep_time = time.perf_counter() - t0
    assert len(rep) > 5000
    assert rep_time < 0.50  # Under 500ms for 1,000 record report
