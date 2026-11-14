# Sentinel Vision AI 2.0 — Safety Operations Command Center
## Comprehensive Demonstration Walkthrough, QA Hardening & Operator Guide

> **Project:** Sentinel Vision AI 2.0  
> **Track:** Track A — CraftifAI Physical AI Hackathon (PipeGen Perception)  
> **Branch:** `feat/safety-dashboard`  
> **Role:** Senior QA Engineer, Frontend Architect & Integration Engineer (Developer 2)
> **System Architecture:** NVIDIA DeepStream 7.1 &bull; TensorRT 10.3 YOLO11s &bull; NvDCF Multi-Object Tracking &bull; Streamlit Command Center

---

## 1. Executive Summary & Hardening Status

**Sentinel Vision AI 2.0** transforms edge vision perception into an industrial-grade **Safety Operations Command Center**. Built for EHS (Environmental Health and Safety) managers, security dispatchers, and plant supervisors, the platform delivers continuous spatial monitoring across hazardous machinery envelopes without human operator fatigue.

### Frontend Hardening Accomplishments
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

## 3. Scripted 3-Minute Demonstration Walkthrough

When presenting Sentinel Vision AI 2.0 to judges, follow this concise 3-minute sequence:

### Minute 1: Command Center Overview & KPI Telemetry
1. **Show the Header:** Point out the dark industrial safety theme (`#0b0f19`) and system status indicators: `PERCEPTION ENGINE READY` and `RECORDED PERCEPTION REPLAY`.
2. **Review KPI Metric Cards:**
   - **Total Incidents:** 6 verified safety events.
   - **Critical Violations:** 6 high-severity exclusion breaches.
   - **Tracked Entities:** 2 distinct workers (NvDCF Track IDs 0 and 1).
   - **Zone Breaches:** 2 initial entries.
   - **Occupancy Dwell:** Estimated hazard exposure duration (~8.5s cumulative).
3. **Open the Provenance Disclosure:** Highlight that all numbers originate strictly from genuine DeepStream metadata (`metadata.json`), with clear distinctions between direct measurements and derived hysteresis metrics.

### Minute 2: Synchronized Video Scrubbing & Ground-Contact Verification
1. **Navigate to Tab 1 ("📹 Video Replay & Monitoring"):**
   - Note the dual-tab interface featuring **Frame-Accurate Scrub Telemetry** and **Full Video Playback**.
2. **Scrub to Frame 0:**
   - Observe that detected personnel are rendered with **CYAN / GREEN** bounding boxes labeled `Person #0 [SAFE]` and `Person #1 [SAFE]`.
3. **Click Quick Jump Button `F3 (Entry)`:**
   - The scrubber immediately jumps to Frame 3 ($t=0.18\text{s}$).
   - Notice the bounding boxes and ground-contact crosshairs immediately flip to **BRIGHT RED** labeled `Person #0 [INTRUSION]` and `Person #1 [INTRUSION]`.
   - Point out the bottom-center footprint datum $(x_{\text{center}}, y_{\text{max}})$: explain how using the ground-contact point prevents false alarms when workers lean forward.
4. **Click `F64` and `F125`:**
   - Show periodic `ZONE_INTRUSION_OCCUPANCY` alerts throttled by the 2.0-second cooldown timer, demonstrating the elimination of alarm fatigue.

### Minute 3: Zone Calibration, Forensics & Safety Copilot
1. **Navigate to Tab 4 ("📐 Zone Calibration Editor"):**
   - Demonstrate the interactive reference frame canvas.
   - Show how vertices are validated in normalized $[0.0, 1.0]$ space.
   - Click **"Stage to Working Copy"** to show safe staging (`restricted_zone_working.json`) without mutating production files.
   - Explain the backend reprocessing command (`python scripts/pipegen_to_sentinel.py --zones configs/restricted_zone.json`).
2. **Navigate to Tab 5 ("🔍 Incident Forensics Explorer"):**
   - Filter by Track ID 0 or 1; demonstrate instant CSV/JSON exports.
   - Select an incident to inspect the zoomed visual evidence snapshot.
3. **Navigate to Tab 6 ("🤖 Safety Copilot"):**
   - Click the prompt pill: *"🚨 Show critical violations"* or *"⏱️ Longest zone occupancy"*.
   - Watch the loading state resolve into an authoritative, evidence-grounded response citing specific incident IDs and frame timestamps.
   - Point out the active mode tag (`DETERMINISTIC_FALLBACK` or `REAL_BACKEND_AI`).

---

## 4. Backend Integration Contract Specification

The dashboard communicates through `BackendAdapter` (`app/components/backend_adapter.py`), which exposes standard function contracts for Developer 1's backend modules:

```python
from app.components.backend_adapter import backend_adapter

# 1. Load incidents from JSONL or backend persistence
incidents = backend_adapter.load_incidents(path=None)

# 2. Retrieve categorized statistics
stats = backend_adapter.get_incident_statistics(incidents)

# 3. Inspect individual incident
record = backend_adapter.get_incident_details(event_id="0bf89c6b-...")

# 4. Query Safety Copilot
response = backend_adapter.answer_question("Summarize all safety incidents", incidents)
# Returns: {"mode": "REAL_BACKEND_AI" | "DETERMINISTIC_FALLBACK", "text": "...", "evidence": [...]}

# 5. Generate official EHS audit report
report = backend_adapter.generate_incident_report(output_path="data/output/audit_report.json")
```

---

## 5. Troubleshooting & Common Inquiries

| Issue Observed | Root Cause | Solution |
| :--- | :--- | :--- |
| **`ModuleNotFoundError: cv2`** | OpenCV not installed in active environment | Run `pip install opencv-python-headless` |
| **Video slider does not seek** | Video format decoding error | The app falls back automatically to raw `sample_office.mp4` or `sentinel_demo_repaired.mp4` |
| **Zone Editor says "Invalid Geometry"** | Fewer than 3 points or values $> 1.0$ | Ensure at least 3 vertices with values between `0.0` and `1.0` |
| **Port 8501 already in use** | An existing Streamlit instance is running | Run `streamlit run app/dashboard.py --server.port 8502` |
