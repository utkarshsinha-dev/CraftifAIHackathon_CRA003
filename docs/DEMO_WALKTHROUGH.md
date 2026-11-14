# Sentinel Vision AI 2.0 — Safety Operations Command Center
## Comprehensive Demonstration Walkthrough & Operator Guide

> **Project:** Sentinel Vision AI 2.0  
> **Track:** Track A — CraftifAI Physical AI Hackathon (PipeGen Perception)  
> **Branch:** `feat/safety-dashboard`  
> **Role:** Developer 2 (Frontend / Full-Stack Operations Command Center)  
> **System Architecture:** NVIDIA DeepStream 7.1 &bull; TensorRT 10.3 YOLO11s &bull; NvDCF Multi-Object Tracking &bull; Streamlit Interactive Command Center  

---

## 1. Executive Overview

**Sentinel Vision AI 2.0** transforms raw edge perception into an interactive, industrial-grade **Safety Operations Command Center**. Built for EHS (Environmental Health and Safety) managers, security dispatchers, and plant supervisors, the platform delivers continuous spatial monitoring across hazardous machinery envelopes without human fatigue.

### Core Capabilities
- **Synchronized Video & Telemetry Replay:** Inspect surveillance footage frame-by-frame with dynamic bounding boxes, persistent Track IDs, and bottom-center ground contact points $(x_{\text{center}}, y_{\text{max}})$.
- **Dynamic 2D Polygon Zone Calibration:** Calibrate and edit restricted exclusion zones directly over authentic reference video frames with normalized $[0.0, 1.0]$ coordinate validation and automatic configuration backups (`.bak`).
- **Throttled Incident Auditing:** Review machine-readable incident records emitted by DeepStream, filtering 320 raw detections down to 6 actionable alerts conforming strictly to JSON Schema.
- **Forensic Investigation & Evidence Inspection:** Filter by severity, zone, and personnel track ID, with instant visual snapshots of security breaches.
- **Conversational EHS Safety Copilot:** Ask natural language questions regarding safety metrics, risk assessments, and worker dwell time with contract-ready integration to `src/safety_copilot.py`.

---

## 2. Quickstart & Installation Instructions

The application is engineered for zero-dependency bloat and executes smoothly on standard Python environments without requiring external physical cameras.

### Prerequisites
- Python 3.10 or 3.11
- Standard libraries: `streamlit`, `opencv-python-headless`, `plotly`, `pandas`, `numpy`, `pillow`

### Launch Command
Execute the following from the repository root:

```powershell
# Using default Python or Python 3.11
py -3.11 -m streamlit run app/dashboard.py
```
*(Or if `streamlit` is directly in your PATH)*:
```powershell
streamlit run app/dashboard.py
```

The Command Center will launch immediately at `http://localhost:8501`.

---

## 3. Module-by-Module Walkthrough

### Module 1: Master Command Center & Live Feed
- **Header Telemetry:** Shows live system status (`PERCEPTION ENGINE READY`), hardware engine details, and the active video stream profile (`sample_office.mp4`, 1728&times;1080 @ 30.00 FPS).
- **Industrial KPI Cards:**
  - **Total Incidents:** 6 verified safety events
  - **Critical Violations:** 6 high-severity exclusion breaches
  - **Tracked Entities:** 2 distinct personnel (`Track 0` and `Track 1`)
  - **Zone Breaches:** 2 initial entries
  - **Occupancy Dwell:** Estimated hazard exposure duration in seconds
- **Real-Time Incursion Feed:** Displays a live alert strip with severity badges (`CRITICAL`, `WARNING`), event types (`ZONE_INTRUSION_ENTRY`, `ZONE_INTRUSION_OCCUPANCY`), and normalized ground contact coordinates.

---

### Module 2: Video Replay & Synchronized Scrub Telemetry
Accessible via the **"📹 Video Replay & Monitoring"** tab:
- **Frame-Accurate Scrubber:** Scrub across all 163 frames ($0.00\text{s}$ to $5.43\text{s}$) with instantaneous visual updates.
- **Ground-Contact Crosshairs:** Displays bottom-center foot coordinates $(x_{\text{center}}, y_{\text{max}})$ with precision target crosshairs to distinguish true floor contact from upper-body leaning.
- **Zone Polygon Overlay:** Hazard zones are rendered with semi-transparent alpha fills and distinct hazard borders.
- **Quick Breach Navigation:** One-click shortcut buttons (`F3 (Entry)`, `F64`, `F125`) jump directly to recorded security breaches.
- **Full Video Playback:** An integrated HTML5 video player option allows playing the verified demonstration MP4 stream (`sentinel_demo_repaired.mp4`).

---

### Module 3: Incident Timeline & Chronology
Accessible via the **"⏱️ Incident Timeline"** tab:
- Lists all incidents chronologically with exact video timestamps ($t=0.18\text{s}, t=2.21\text{s}, t=4.24\text{s}$) and frame indexes.
- Color-coded badges indicate severity and event transition states (`ENTRY` &rarr; `OCCUPANCY`).
- Expandable incident cards provide confidence scores, zone IDs, footprint coordinates, and raw JSON payloads.
- **"Jump to F#"** buttons sync the main video scrubber directly to that incident's frame index.

---

### Module 4: Interactive Zone Calibration & Polygon Editor
Accessible via the **"📐 Zone Calibration Editor"** tab:
- **Reference Frame Alignment:** Loads genuine surveillance footage (Frame 0) as an active calibration canvas.
- **Zone Selection & Custom Creation:** Edit existing zones (`Heavy Equipment Perimeter A`, `Walkway Buffer B`) or initialize new custom safety zones.
- **Vertex Coordinate Editor:** Enter or adjust normalized $[0.0, 1.0]$ coordinates. Includes automatic geometric validation (checks for minimum 3 vertices and $[0.0, 1.0]$ boundaries).
- **Live Visual Preview:** Renders numbered vertices, boundary lines, and alpha tints in real time over the video frame.
- **Safe Persistence:**
  - Automated backup creation (`configs/restricted_zone.json.bak`) prior to any write.
  - One-click **Revert from Backup** button.
  - Direct **Download JSON** button for exporting configuration files.

---

### Module 5: Incident Explorer & Forensic Drilldown
Accessible via the **"🔍 Incident Forensics Explorer"** tab:
- **Multi-Dimensional Filters:** Filter simultaneously by Track ID, Zone Name, Event Type, and Severity.
- **Confidence Slider & Chronological Sorting:** Filter low-confidence detections or invert sort order (Oldest vs. Newest).
- **Forensic Evidence Snapshot:** Select any incident to see its metadata accompanied by an extracted visual frame highlighting the intruding personnel and ground-contact datum.
- **Export Capabilities:** Download individual incident records as JSON files for audit compliance.

---

### Module 6: Safety Copilot (Conversational EHS Assistant)
Accessible via the **"🤖 Safety Copilot"** tab:
- **Suggested Question Pills:** One-click shortcuts for standard audit queries:
  - *"📋 Summarize all incidents"*
  - *"🚨 Show critical violations"*
  - *"⏱️ Longest zone occupancy"*
  - *"⚠️ Highest-risk incident"*
  - *"📍 Summarize Zone 1 events"*
- **Backend Integration Ready:**
  - Automatically queries `src/safety_copilot.py` (`answer_question`) if implemented by Developer 1.
  - If backend is offline, gracefully shifts to deterministic local reasoning over authentic loaded incidents with clear visual status indicators.
- **Referenced Evidence:** Every response cites specific incident IDs and frame numbers for auditable verification.

---

## 4. Backend Integration Contract

The dashboard expects backend functions conforming to:

```python
# Optional contracts provided by Developer 1 in src/safety_copilot.py
def answer_question(question: str) -> str:
    """Answers safety questions over incident records."""
    pass

def get_incident_statistics() -> dict:
    """Returns aggregated incident metrics."""
    pass
```

The frontend components do **not** modify `src/` and remain 100% functional even when backend modules are offline or in-development.

---

## 5. Verification & Testing Checklist

| Test Item | Verification Target | Status |
| :--- | :--- | :--- |
| **Dashboard Startup** | Launches without syntax or import errors | **VERIFIED** |
| **Video Decoding** | OpenCV extracts all 163 frames at 30 FPS | **VERIFIED** |
| **Authentic JSONL Ingestion** | Reads all 6 verified incidents without mock injection | **VERIFIED** |
| **KPI Calculations** | Total, Critical, Tracks, and Dwell times match records | **VERIFIED** |
| **Zone Editor Validation** | Prevents invalid geometries; creates `.bak` backups | **VERIFIED** |
| **Multi-Filter Explorer** | Real-time filtering by track, severity, and event type | **VERIFIED** |
| **Copilot Offline Grace** | Handles missing `src/safety_copilot.py` without crash | **VERIFIED** |
| **Git Branch Rule** | All commits isolated strictly to `feat/safety-dashboard` | **VERIFIED** |
