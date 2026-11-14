# Sentinel Vision AI 2.0 — Backend Integration Contract

**Document Target:** Developer 2 (Frontend / Dashboard Owner)  
**Author:** Developer 1 (Backend / Intelligence Owner)  
**Source Module:** `src/incident_service.py`  
**Status:** Stable, Tested & Verified (20/20 Pytest Passing)

---

## 1. Overview & Import

All dashboard-facing functionality is consolidated into a single high-level service module: `src.incident_service`.

```python
from src.incident_service import (
    load_incidents,
    get_incident_statistics,
    get_incident_details,
    answer_question,
    generate_incident_report,
)
```

All functions return standard Python primitive types (`dict`, `list`, `str`, `float`, `int`, `bool`, `None`) that are **100% JSON-serializable** via `json.dumps()`.

---

## 2. Interface Specifications

### A. `load_incidents(path: Optional[str] = None) -> List[Dict[str, Any]]`
Loads all structured incident records from disk.

- **Parameters:**
  - `path` *(str, optional)*: Path to the JSONL log file. Defaults to `"data/output/incident_records.jsonl"`.
- **Returns:**
  - `List[Dict[str, Any]]`: List of incident dictionaries sorted chronologically.
  - If the file is missing or empty, returns an empty list `[]` (no exception raised).
- **Incident Record Schema:**
  ```json
  {
    "incident_id": "b60605a2-b05c-49d9-a5c2-91cc516d91c5",
    "timestamp_iso": "2026-11-14T06:37:45.669389+00:00",
    "frame_index": 3,
    "video_time_s": 0.1765,
    "dwell_time_s": 0.0,
    "event_type": "ZONE_INTRUSION_ENTRY",
    "zone_id": "zone_restricted_01",
    "zone_name": "Heavy Equipment Perimeter A",
    "severity": "CRITICAL",
    "track_id": 0,
    "confidence": 0.5233,
    "bbox": [0.5439, 0.5806, 0.5777, 0.6479],
    "ground_contact_point": [0.5608, 0.6479],
    "kinematics": {
      "velocity_normalized": [0.0, 0.0],
      "speed_normalized": 0.0,
      "direction": "STATIONARY",
      "motion_state": "stationary",
      "reliability": "INITIALIZING"
    },
    "risk_assessment": {
      "risk_score": 85,
      "assessed_severity": "CRITICAL",
      "explanation": "Assessed Risk: CRITICAL (Score 85/100)...",
      "contributing_factors": {
        "zone_severity_weight": 50,
        "event_type_weight": 20,
        "dwell_time_seconds": 0.0,
        "motion_dynamics_weight": 5,
        "distance_to_zone_centroid": 0.1385,
        "proximity_weight": 10,
        "confidence_factor": 0
      }
    },
    "trajectory_history": [[0.561, 0.6478], ...],
    "session_start_iso": "2026-11-14T06:37:45.492856+00:00"
  }
  ```

---

### B. `get_incident_statistics(incidents: Optional[List[Dict[str, Any]]] = None, path: Optional[str] = None) -> Dict[str, Any]`
Computes aggregate metrics for dashboard KPI cards.

- **Parameters:**
  - `incidents` *(list, optional)*: Pre-loaded list of incident dictionaries. If omitted, loads from `path`.
  - `path` *(str, optional)*: Path to incident file.
- **Returns:**
  ```json
  {
    "total_incidents": 6,
    "unique_tracks": ["0", "1"],
    "critical_count": 6,
    "warning_count": 0,
    "info_count": 0,
    "max_dwell_s": 4.07,
    "zones_breached": ["Heavy Equipment Perimeter A"],
    "event_breakdown": {
      "ZONE_INTRUSION_ENTRY": 2,
      "ZONE_INTRUSION_OCCUPANCY": 4
    }
  }
  ```
  If no incidents exist, returns zeroed fields and empty lists safely.

---

### C. `get_incident_details(event_id: str, incidents: Optional[List[Dict[str, Any]]] = None, path: Optional[str] = None) -> Optional[Dict[str, Any]]`
Looks up full telemetry, bounding box, kinematics, and risk breakdown for a single incident.

- **Parameters:**
  - `event_id` *(str)*: UUID string matching `incident["incident_id"]`.
  - `incidents` *(list, optional)*: Pre-loaded incident list.
  - `path` *(str, optional)*: Path to file.
- **Returns:**
  - `dict` with complete incident payload if found.
  - `None` if the incident ID is not found.

---

### D. `answer_question(question: str, incidents: Optional[List[Dict[str, Any]]] = None, path: Optional[str] = None) -> Dict[str, Any]`
Natural language Q&A engine powered by `SafetyCopilot`.

- **Parameters:**
  - `question` *(str)*: Natural language inquiry (e.g., *"What happened?"*, *"Who stayed the longest?"*, *"Show critical breaches"*, *"What did Track 0 do?"*, *"What happened at frame 64?"*).
  - `incidents` *(list, optional)*: Pre-loaded incident list.
- **Returns:**
  ```json
  {
    "answer": "Sentinel Vision AI captured a total of 6 safety events across 2 tracked personnel...",
    "intent": "incident_summary",
    "evidence": [
      "[Frame 3, t=0.18s | Track 0 | Heavy Equipment Perimeter A | ZONE_INTRUSION_ENTRY (initial entry)]",
      "[Frame 64, t=2.21s | Track 0 | Heavy Equipment Perimeter A | ZONE_INTRUSION_OCCUPANCY (dwell: 2.0s)]"
    ],
    "confidence": 1.0
  }
  ```
  Runs 100% offline with zero external network or API key requirements.

---

### E. `generate_incident_report(output_path: Optional[str] = None, incidents: Optional[List[Dict[str, Any]]] = None, path: Optional[str] = None, format: str = "markdown") -> str`
Generates a structured EHS (Environmental Health and Safety) investigation report.

- **Parameters:**
  - `output_path` *(str, optional)*: File path to save the generated report (e.g., `"data/output/EHS_INCIDENT_REPORT.md"`). If provided, saves to disk.
  - `incidents` *(list, optional)*: Incident records to include.
  - `format` *(str)*: Output format (`"markdown"`).
- **Returns:**
  - `str`: Complete markdown report content ready for display or download.

---

## 3. Example Dashboard Integration (Streamlit)

```python
import streamlit as st
from src.incident_service import (
    load_incidents,
    get_incident_statistics,
    answer_question,
    generate_incident_report,
)

# 1. Load Data
incidents = load_incidents()
stats = get_incident_statistics(incidents)

# 2. Render KPI Metric Cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Incidents", stats["total_incidents"])
col2.metric("Critical Alerts", stats["critical_count"])
col3.metric("Peak Dwell Time", f"{stats['max_dwell_s']}s")
col4.metric("Active Personnel", len(stats["unique_tracks"]))

# 3. Interactive Safety Copilot Chat
query = st.text_input("Ask Safety Copilot:", placeholder="e.g. Who stayed longest in the hazard zone?")
if query:
    response = answer_question(query, incidents=incidents)
    st.markdown(f"**Copilot:** {response['answer']}")
    with st.expander("Grounded Evidence Citations"):
        for ev in response["evidence"]:
            st.caption(ev)

# 4. Download EHS Report
if st.button("Generate EHS Audit Report"):
    report_text = generate_incident_report()
    st.download_button("Download Report (Markdown)", report_text, file_name="EHS_AUDIT_REPORT.md")
```

---

## 4. Verification Notes

- All 5 functions are verified by automated tests in `tests/test_backend_intelligence.py::test_incident_service_frontend_contract`.
- Both `data/output/incident_records.jsonl` and `data/output/EHS_INCIDENT_REPORT.md` are actively generated and available in the workspace.
- The interface will remain strictly stable during frontend integration.
