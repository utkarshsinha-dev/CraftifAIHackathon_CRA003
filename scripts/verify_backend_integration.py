"""
Sentinel Vision AI 2.0 - Backend Integration Smoke Test & Verification Script

Independently validates backend contracts, authoritative perception telemetry,
Safety Copilot grounding, and EHS report generation against real runtime data.

Usage:
    python scripts/verify_backend_integration.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.incident_service import (
    load_incidents,
    get_incident_statistics,
    get_incident_details,
    answer_question,
    generate_incident_report,
)


def run_smoke_tests() -> bool:
    print("=" * 70)
    print("SENTINEL VISION AI 2.0 - BACKEND LIVE INTEGRATION SMOKE TEST")
    print("=" * 70)

    all_passed = True
    checks_total = 0
    checks_passed = 0

    def check(name: str, condition: bool, details: str = ""):
        nonlocal all_passed, checks_total, checks_passed
        checks_total += 1
        if condition:
            checks_passed += 1
            print(f" [PASS] {name} {details}")
        else:
            all_passed = False
            print(f" [FAIL] {name} {details}")

    # 1. Authoritative Perception Metadata Inspection
    meta_path = PROJECT_ROOT / "artifacts" / "pipegen-export" / "output" / "metadata.json"
    if meta_path.is_file():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        frames = meta.get("frames", [])
        total_dets = sum(len(f.get("objects", [])) for f in frames)
        check("Perception Frames Count", len(frames) == 163, f"({len(frames)} / 163 frames)")
        check("Perception Person Detections", total_dets == 320, f"({total_dets} / 320 detections)")
    else:
        check("Perception Metadata Found", False, f"Missing {meta_path}")

    # 2. Authoritative Incident Records Loading
    log_path = PROJECT_ROOT / "data" / "output" / "incident_records.jsonl"
    incidents = load_incidents(str(log_path))
    check("Incident Log Loading", len(incidents) == 6, f"(Loaded {len(incidents)} authentic records)")

    # 3. Temporal & Event Integrity
    events_by_frame = {}
    for inc in incidents:
        f_idx = inc.get("frame_index")
        evt = inc.get("event_type")
        tid = inc.get("track_id")
        events_by_frame.setdefault(f_idx, []).append((evt, tid))

    f3_evts = events_by_frame.get(3, [])
    f64_evts = events_by_frame.get(64, [])
    f125_evts = events_by_frame.get(125, [])

    check(
        "Frame 3 Entry Events",
        len(f3_evts) == 2 and all(e[0] == "ZONE_INTRUSION_ENTRY" for e in f3_evts),
        f"(2 entry events: tracks {[e[1] for e in f3_evts]})"
    )
    check(
        "Frame 64 Occupancy Events",
        len(f64_evts) == 2 and all(e[0] == "ZONE_INTRUSION_OCCUPANCY" for e in f64_evts),
        f"(2 occupancy events: tracks {[e[1] for e in f64_evts]})"
    )
    check(
        "Frame 125 Occupancy Events",
        len(f125_evts) == 2 and all(e[0] == "ZONE_INTRUSION_OCCUPANCY" for e in f125_evts),
        f"(2 occupancy events: tracks {[e[1] for e in f125_evts]})"
    )

    # 4. KPI Statistics Reproducibility & JSON Serializability
    stats = get_incident_statistics(incidents=incidents)
    check("Statistics Total Incidents", stats.get("total_incidents") == 6, f"(Total: {stats.get('total_incidents')})")
    check("Statistics Unique Tracks", stats.get("unique_tracks") == ["0", "1"], f"(Tracks: {stats.get('unique_tracks')})")
    check("Statistics Critical Count", stats.get("critical_count") == 6, f"(Critical: {stats.get('critical_count')})")
    check("Statistics Max Dwell", stats.get("max_dwell_s") == 4.07, f"(Max Dwell: {stats.get('max_dwell_s')}s)")

    try:
        json_stats = json.dumps(stats)
        check("Statistics JSON Serializability", isinstance(json_stats, str), "(100% JSON-compliant)")
    except Exception as e:
        check("Statistics JSON Serializability", False, f"Serialization error: {e}")

    # 5. Incident Details Retrieval
    first_id = incidents[0].get("incident_id")
    details = get_incident_details(event_id=first_id, incidents=incidents)
    check("Incident Details Lookup by ID", details is not None and details.get("incident_id") == first_id, f"(ID: {first_id})")

    # Keyword alias check: incident_id keyword
    details_kw = get_incident_details(incident_id=first_id, incidents=incidents)
    check("Incident Details Keyword Alias Support", details_kw is not None and details_kw.get("incident_id") == first_id, "(incident_id keyword supported)")

    # 6. Safety Copilot Representative Inquiries
    queries = [
        ("Summarize the recorded incidents.", "incident_summary"),
        ("Which incident has the highest risk?", "highest_risk"),
        ("Which worker stayed the longest?", "longest_occupancy"),
        ("What evidence supports the zone violation?", "zone_investigation"),
        ("Show critical breaches.", "critical_incidents"),
        ("What did Worker #0 do?", "worker_investigation"),
        ("What did Worker #999 do?", "worker_investigation"),
        ("What information is unavailable?", "limitations_inquiry"),
    ]

    for q_text, expected_intent in queries:
        ans = answer_question(question=q_text, incidents=incidents)
        has_answer = bool(ans.get("answer"))
        intent = ans.get("intent")
        evidence = ans.get("evidence", [])
        check(
            f"Copilot Query: '{q_text[:32]}...'",
            has_answer and intent == expected_intent,
            f"-> intent: {intent}, evidence count: {len(evidence)}"
        )

    # 7. EHS Report Generation into Isolated Temporary Path
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp_file:
        tmp_report_path = tmp_file.name

    try:
        report_content = generate_incident_report(output_path=tmp_report_path, incidents=incidents)
        check("EHS Report Generation", bool(report_content) and len(report_content) > 1000, f"({len(report_content)} chars)")
        check("EHS Report File Created", os.path.isfile(tmp_report_path), f"({tmp_report_path})")
        check("EHS Report Disclaimer Present", "Regulatory & Operational Disclaimer" in report_content, "(Disclaimer verified)")
    finally:
        if os.path.exists(tmp_report_path):
            try:
                os.remove(tmp_report_path)
            except OSError:
                pass

    print("=" * 70)
    print(f"SMOKE TEST SUMMARY: {checks_passed} / {checks_total} CHECKS PASSED")
    if all_passed:
        print("RESULT: ALL BACKEND CONTRACTS & VERIFICATIONS PASSED (READY FOR FRONTEND)")
    else:
        print("RESULT: INTEGRATION SMOKE TEST DETECTED DEFECTS")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = run_smoke_tests()
    sys.exit(0 if success else 1)
