"""
Sentinel Vision AI - Incident Service & Dashboard Integration Contract

Provides clean, stable, JSON-serializable Python interfaces for frontend dashboards,
command-line utilities, and external security integrations.

Contract Interfaces:
- load_incidents(path=None) -> List[Dict[str, Any]]
- get_incident_statistics(incidents=None, path=None) -> Dict[str, Any]
- get_incident_details(event_id, incidents=None, path=None) -> Optional[Dict[str, Any]]
- answer_question(question, incidents=None, path=None) -> Dict[str, Any]
- generate_incident_report(output_path=None, incidents=None, path=None) -> str
"""

import json
import os
from typing import Dict, List, Any, Optional

from src.safety_copilot import SafetyCopilot

DEFAULT_LOG_PATH = "data/output/incident_records.jsonl"


def load_incidents(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Loads all structured incident records from a JSON Lines file.
    Returns empty list safely if file is missing or empty.
    """
    target_path = path or DEFAULT_LOG_PATH
    incidents = []
    if not os.path.isfile(target_path):
        return incidents

    with open(target_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                incidents.append(record)
            except json.JSONDecodeError:
                continue
    return incidents


def get_incident_statistics(
    incidents: Optional[List[Dict[str, Any]]] = None,
    path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Computes exact summary metrics for dashboard KPI cards.
    """
    records = incidents if incidents is not None else load_incidents(path)
    total = len(records)
    if total == 0:
        return {
            "total_incidents": 0,
            "unique_tracks": [],
            "critical_count": 0,
            "warning_count": 0,
            "info_count": 0,
            "max_dwell_s": 0.0,
            "zones_breached": [],
            "event_breakdown": {},
        }

    unique_tracks = sorted(
        list(set(str(i.get("track_id")) for i in records if i.get("track_id") is not None))
    )
    zones = sorted(list(set(i.get("zone_name", i.get("zone_id", "Unknown")) for i in records)))

    critical_count = sum(1 for i in records if i.get("severity") == "CRITICAL")
    warning_count = sum(1 for i in records if i.get("severity") == "WARNING")
    info_count = sum(1 for i in records if i.get("severity") == "INFO")

    event_breakdown = {}
    for inc in records:
        evt = inc.get("event_type", "UNKNOWN")
        event_breakdown[evt] = event_breakdown.get(evt, 0) + 1

    max_dwell = max([float(i.get("dwell_time_s", 0.0)) for i in records], default=0.0)

    return {
        "total_incidents": total,
        "unique_tracks": unique_tracks,
        "critical_count": critical_count,
        "warning_count": warning_count,
        "info_count": info_count,
        "max_dwell_s": round(max_dwell, 2),
        "zones_breached": zones,
        "event_breakdown": event_breakdown,
    }


def get_incident_details(
    event_id: str,
    incidents: Optional[List[Dict[str, Any]]] = None,
    path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Retrieves full telemetry, bounding box, kinematics, and risk breakdown
    for a specific incident by its incident_id.
    """
    records = incidents if incidents is not None else load_incidents(path)
    for inc in records:
        if inc.get("incident_id") == event_id:
            return inc
    return None


def answer_question(
    question: str,
    incidents: Optional[List[Dict[str, Any]]] = None,
    path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Passes a natural language inquiry to the Safety Copilot and returns
    an evidence-grounded answer.
    """
    log_file = path or DEFAULT_LOG_PATH
    copilot = SafetyCopilot(incident_log_path=log_file)
    if incidents is not None:
        copilot.incidents = incidents
    return copilot.answer_question(question)


def generate_incident_report(
    output_path: Optional[str] = None,
    incidents: Optional[List[Dict[str, Any]]] = None,
    path: Optional[str] = None,
    format: str = "markdown",
) -> str:
    """
    Generates a structured EHS incident audit report.
    Returns the report text and writes to output_path if provided.
    """
    from scripts.generate_ehs_report import create_ehs_report
    records = incidents if incidents is not None else load_incidents(path)
    target_out = output_path or "data/output/EHS_INCIDENT_REPORT.md"
    return create_ehs_report(incidents=records, output_path=target_out, format=format)
