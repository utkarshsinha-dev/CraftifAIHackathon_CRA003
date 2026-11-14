# Sentinel Vision AI 2.0 — Safety Operations Command Center
## Comprehensive Demonstration Walkthrough, QA Hardening & Operator Guide

> **Project:** Sentinel Vision AI 2.0
> **Track:** Track A — CraftifAI Physical AI Hackathon (PipeGen Perception)
> **Branch:** `feat/safety-dashboard`
> **Role:** Senior QA Engineer, Frontend Architect & Integration Engineer (Developer 2)
> **System Architecture:** NVIDIA DeepStream 7.1 &bull; TensorRT 10.3 YOLO11s &bull; NvDCF Multi-Object Tracking &bull; Streamlit Command Center

---

## 1. Executive Summary & Hardening Status

**Sentinel Vision AI 2.0** transforms raw edge perception into an interactive, industrial-grade **Safety Operations Command Center**. Built for EHS (Environmental Health and Safety) managers, security dispatchers, and plant supervisors, the platform delivers continuous spatial monitoring across hazardous machinery envelopes without human operator fatigue.

### Core Architecture & Capabilities
- **Incident Investigation & Evidence Management Workspace:** Dedicated multi-step forensic workspace featuring synchronized video frames, 2D trajectory path kinematics, multi-factor risk scoring, contextual Safety Copilot Q&A, and full CAPA report generation.
- **2D Trajectory Kinematics (Real DeepStream Data):** Reconstructs 160 continuous frames of tracked personnel movement directly from authentic `metadata.json`, with image-coordinate displacement and velocity mapping.
- **Modular Backend Integration Adapter (`app/components/backend_adapter.py`):** Centralizes all data ingestion and reasoning contracts. Detects `src/safety_copilot.py` dynamically; operates seamlessly in `REAL_BACKEND_AI`, `DETERMINISTIC_FALLBACK`, or `MOCK_PREVIEW` modes.
- **Dynamic Intrusion State Coloring:** Tracked personnel bounding boxes dynamically transition between **GREEN (`SAFE`)** and **RED (`INCURSION`)** in real-time as their bottom-center ground contact $(x_{\text{center}}, y_{\text{max}})$ intersects restricted polygons.
- **Zero-Latency In-Memory Video Caching:** Decorated video frame extraction with `@st.cache_data(max_entries=170)` for buttery-smooth timeline scrubbing across all 163 frames ($5.43\text{s}$).
- **Safe Zone Staging & Automated Backups:** Prevents accidental overwriting of master configuration files. All polygon edits stage to `configs/restricted_zone_working.json` by default with explicit two-step confirmation required to commit to production.
- **Categorized KPI Provenance:** Explicitly categorizes metrics into **Direct Measurements**, **Derived Metrics**, and **Unavailable/Decoupled Measurements** to comply strictly with product authenticity principles.
- **100% Relative Path Portability:** Zero hardcoded machine paths; clone and execute on any standard Windows laptop.

---

## 2. Installation & Portability Requirements

The application runs locally without requiring external physical cameras, GPU drivers, or cloud APIs.

### Environment Requirements
- **Operating System:** Windows 10/11 (or Linux/WSL2/macOS)
- **Python Version:** Python 3.10 or Python 3.11 (Recommended)

### Package Installation
```powershell
# Using Python 3.11 launcher (Windows)
py -3.11 -m pip install streamlit opencv-python-headless plotly pandas numpy pillow
```
*(Or standard pip)*:
```bash
pip install streamlit opencv-python-headless plotly pandas numpy pillow
```

### Application Launch Command
From the root repository directory (`CraftifAIHackathon_CRA003`):

```powershell
py -3.11 -m streamlit run app/dashboard.py
```
*(Or if streamlit is on PATH)*:
```powershell
streamlit run app/dashboard.py
```

The Command Center will open in your default browser at:
👉 **`http://localhost:8501`**

---

## 3. Scripted 3-Minute Demonstration Walkthrough for Judges

When presenting Sentinel Vision AI 2.0 to judges, follow this authoritative 3-minute sequence:

### Minute 1: Command Center Overview & KPI Telemetry
1. **Show the Header:** Point out the dark industrial safety theme (`#0b0f19`) and system status indicators: `PERCEPTION ENGINE READY` and `RECORDED PERCEPTION REPLAY`.
2. **Review KPI Metric Cards:**
   - **Total Incidents:** 6 verified safety events.
   - **Critical Violations:** 6 high-severity exclusion breaches.
   - **Tracked Entities:** 2 distinct workers (NvDCF Track IDs 0 and 1).
   - **Zone Breaches:** 2 initial entries.
   - **Occupancy Dwell:** Estimated hazard exposure duration (~8.5s cumulative).
3. **Open the Provenance Disclosure:** Highlight that all numbers originate strictly from genuine DeepStream metadata (`metadata.json`), with clear distinctions between direct measurements and derived hysteresis metrics.
4. **Scrub the Video (Tab 1):** Click Quick Jump button `F3 (Entry)`. Point out that the bounding boxes and ground-contact crosshairs immediately turn **RED (`INCURSION`)** as feet enter the zone.

### Minute 2: Primary Feature — Incident Investigation Workspace (Tab 2)
1. **Navigate to Tab 2 ("🕵️ Incident Investigation"):**
   - Point out the 6-step EHS workflow strip: *Select &rarr; Evidence &rarr; Risk & Trajectory &rarr; Copilot Q&A &rarr; CAPA Notes &rarr; Export*.
2. **Select Incident #01 (Frame #3, Person #0, `ZONE_INTRUSION_ENTRY`):**
   - **Visual Evidence:** High-resolution video snapshot highlighting the intruder and zone boundary with one-click synchronization back to the master video player.
   - **2D Trajectory Map:** Live Plotly chart plotting 160 authentic frames of worker path inside the exclusion boundary with inception, breach datum, and terminus points.
3. **Examine Risk Factors:**
   - Review calculated Risk Score (**85/100, CRITICAL**).
   - Show evaluated risk factors: Zone Severity, Ground-Contact Encroachment, and Dwell Hysteresis.
4. **Interrogate the Copilot:**
   - Click *"❓ Why is this critical?"* or *"🛡️ Recommended CAPA?"*.
   - Demonstrate the contextual, incident-grounded explanation citing specific timestamps ($t=0.18\text{s}$) and zone IDs (`zone_restricted_01`).

### Minute 3: CAPA Recording, Report Export & Safe Zone Calibration
1. **Record Operator Observations & CAPA:**
   - Select Contributing Conditions (e.g. *Maintenance cart obstructing walkway*, *Floor hazard demarcation faded*).
   - Update Investigation Status to `Under Review` or `Resolved` (session-recorded).
2. **Export the Complete Package:**
   - Click **"📄 Download Markdown Report (.md)"** or **"📦 Download JSON Evidence Package"**.
   - Show that the package contains all incident metadata, trajectory statistics, risk scores, operator notes, and regulatory disclaimers.
3. **Show Safe Zone Calibration (Tab 5):**
   - Show that changes stage safely to `restricted_zone_working.json` without mutating master production configurations until explicitly confirmed.

---

## 4. Backend Integration Contract Specification

The dashboard communicates through `BackendAdapter` (`app/components/backend_adapter.py`), which exposes robust dual-signature function contracts for Developer 1's backend modules (`src/safety_copilot.py`, `src/zone_monitor.py`):

```python
from app.components.backend_adapter import backend_adapter

# 1. Load incidents from JSONL or backend persistence (supports log_path or path)
incidents = backend_adapter.load_incidents(log_path="data/output/incident_records.jsonl")

# 2. Retrieve categorized statistics (supports log_path keyword or direct incident list)
stats = backend_adapter.get_incident_statistics(log_path="data/output/incident_records.jsonl")
# or: stats = backend_adapter.get_incident_statistics(incidents)

# 3. Retrieve 2D trajectory kinematics for a tracked entity across 160 frames
trajectory = backend_adapter.get_trajectory(track_id=0)

# 4. Retrieve multi-factor risk assessment (authoritative backend or transparent provisional rule)
risk = backend_adapter.get_risk_assessment(incident)

# 5. Query Safety Copilot (supports query=..., question=..., log_path=..., or direct incidents)
response = backend_adapter.answer_question(
    query="Why was this incident classified as critical?",
    log_path="data/output/incident_records.jsonl",
    incident_context=incident
)
# Modes: "BACKEND COPILOT CONNECTED" vs "DETERMINISTIC FALLBACK MODE"

# 6. Generate official Markdown investigation package
report_md = backend_adapter.generate_investigation_report_markdown(
    incident=incident,
    operator_notes="Visual verification of cart obstruction",
    capa_actions="Deploy optical light curtains",
    investigation_status="Under Review",
    contributing_conditions=["Maintenance cart obstruction"]
)
```

---

## 5. Troubleshooting & Common Inquiries

| Issue Observed | Root Cause | Solution |
| :--- | :--- | :--- |
| **`ModuleNotFoundError: cv2`** | OpenCV not installed in active environment | Run `pip install opencv-python-headless` |
| **Video slider does not seek** | Video format decoding error | The app falls back automatically to raw `sample_office.mp4` or `sentinel_demo_repaired.mp4` |
| **Zone Editor says "Invalid Geometry"** | Fewer than 3 points or values $> 1.0$ | Ensure at least 3 vertices with values between `0.0` and `1.0` |
| **Port 8501 already in use** | An existing Streamlit instance is running | Run `streamlit run app/dashboard.py --server.port 8502` |
