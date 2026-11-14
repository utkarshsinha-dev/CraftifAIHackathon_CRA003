# SENTINEL VISION — AI-Powered Industrial Safety Intelligence
## Enterprise Platform Operations Guide & Verification Architecture

> **Product:** SENTINEL VISION  
> **Platform Version:** 2.0.0 (Enterprise Release Candidate)  
> **Target Deployments:** Manufacturing Envelopes, Heavy Industrial Plants, Automated Logistics Facilities  
> **Core Pipeline:** NVIDIA DeepStream 7.1 &bull; TensorRT 10.3 YOLO11s &bull; NvDCF Multi-Object Tracking &bull; Streamlit Enterprise UI

---

## 1. Executive Summary & Product Architecture

**SENTINEL VISION** is an enterprise-grade industrial workplace safety intelligence platform designed to eliminate hazardous machinery incursions and automate Environmental Health & Safety (EHS) compliance. By converting high-framerate edge vision telemetry into actionable spatial intelligence, SENTINEL VISION delivers continuous spatial oversight without operator fatigue.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           EDGE SENSOR PERCEPTION LAYER                          │
│   RTSP Camera Stream ──▶ NVIDIA DeepStream 7.1 ──▶ TensorRT 10.3 YOLO11s       │
│                                                          │                      │
│                                            NvDCF Multi-Object Tracking          │
│                                                          │                      │
│                                      Spatial Ground-Contact Incursion Monitor   │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │ telemetry / incident records
┌────────────────────────────────────────▼────────────────────────────────────────┐
│                        BACKEND SERVICE LAYER (Deterministic)                    │
│   src/incident_service.py ──▶ Dwell Analytics & Identity Stabilization          │
│   src/risk_engine.py      ──▶ Multi-Factor Spatial Risk Scoring (0-100)         │
│   src/safety_copilot.py   ──▶ Contextual Offline EHS Copilot                    │
│   scripts/generate_ehs_report.py ──▶ Formal EHS Compliance Audit Engine         │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │ BackendAdapter abstraction
┌────────────────────────────────────────▼────────────────────────────────────────┐
│                   SENTINEL VISION ENTERPRISE WEB WORKSPACE                      │
│   [Left Dark Navy Sidebar]                                                      │
│   ├── Overview                  ──▶ Executive KPIs, Sensor Status, Recent Incursions
│   ├── Live Monitoring           ──▶ High-Definition Replay, Frame Scrubber, Alerts
│   ├── Incident Explorer         ──▶ Forensic Table, Multi-Facet Filters, Evidence Drawer
│   ├── Investigation Workspace   ──▶ 2-Column Split: Trajectory Map, Risk, Copilot, CAPA
│   ├── AI Safety Copilot         ──▶ Grounded NL Dialog for EHS Compliance
│   ├── Zone Management           ──▶ Geometric Polygon Editor with Safe Staging
│   ├── Analytics & Reports       ──▶ Exposure Dwell Distribution & EHS Report Export
│   └── System Health             ──▶ Subsystem Telemetry & Microservice Diagnostics
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Design System & User Experience

The user interface follows an enterprise design philosophy inspired by high-reliability operational software (Linear, Vercel, Stripe, Palantir):

- **Premium Dark Navy Navigation (`#090e17`):** High visual contrast for facility-wide command room visibility.
- **Clean Light Slate Workspace (`#f8fafc`):** Clean backgrounds with subtle borders (`#e2e8f0`) to prioritize operational data legibility.
- **Restrained Semantic Palette:**
  - Emerald Green (`#059669` / `#10b981`): Safe status, normal operation, and primary actions.
  - Signal Red (`#dc2626` / `#ef4444`): High-severity incursion events and critical risk thresholds.
  - Warning Amber (`#d97706` / `#f59e0b`): Hysteresis exposure and cautious zone proximity.
- **Responsive Navigation State:** Unified session state routing (`sidebar_nav`) enabling seamless 1-click transitions from global incident tables directly into forensic investigation workspaces.

---

## 3. Platform Workspace Guide

### 3.1 Overview (Executive Command Center)
- **KPI Metrics:**
  - **Total Incidents:** Real-time tally of verified zone incursions.
  - **Critical Violations:** Immediate indicator of high-severity exclusion breaches.
  - **Active Tracked Workers:** Unique personnel tracked across the active perimeter via NvDCF.
  - **Active Hazard Zones:** Configured geometric exclusion boundaries currently under surveillance.
  - **Avg Incursion Dwell:** Continuous exposure hysteresis calculated across all breaches.
- **Live Operations View:** Real-time visual feed with immediate incursion detection bounding boxes.
- **Recent Safety Incursions:** Priority event log with 1-click **Inspect** routing into deep forensic analysis.
- **Operational Telemetry:** Hourly incident density, risk distribution charts, and severity breakdowns.

### 3.2 Live Monitoring & Sensor Telemetry
- **Sensor Header:** Displays camera ID (`CAM-04 Perimeter A`), ingest status (`RECORDED REPLAY`), stream frame rate (`30.00 FPS`), tracked count, and active incursion count.
- **Interactive Replay Scrubber:** Frame-accurate scrubbing across all frames with cached low-latency seeking.
- **Ground-Contact Spatial Highlighting:** Personnel bounding boxes dynamically switch between green (`SAFE`) and red (`INCURSION`) as their ground contact coordinates $(x_{\text{center}}, y_{\text{max}})$ intersect the restricted polygon.
- **Quick Breach Navigation:** Direct jump controls to jump directly to initial entry ($F3$) and subsequent breach frames.

### 3.3 Incident Explorer
- **Enterprise Data Grid:** Multi-column view tracking Incident ID, Timestamp, Sensor, Type, Restricted Zone, Worker ID, Risk Level, and Resolution Status.
- **Multi-Dimensional Filters:** Filter by severity (Critical / High / Medium / Low), restricted zone, or keyword search across workers and event IDs.
- **Incident Evidence Detail Drawer:** Expandable quick-view displaying spatial bounding coordinates, confidence metrics, and quick navigation actions.
- **Data Export:** Export current filtered datasets to industry-standard CSV and JSON packages.

### 3.4 Forensic Investigation Workspace
A specialized split-screen investigation workspace for EHS officers:
- **Left Column (Spatial & Kinematic Evidence):**
  - **Synchronized Video Snapshot:** Evidence frame rendering the worker bounding box and zone boundary.
  - **2D Trajectory Kinematics:** Plotly Cartesian map plotting tracked entity movement over time, displaying inception points, breach entry, and exit vectors.
- **Right Column (Risk Assessment & Governance):**
  - **Authoritative Risk Score:** Quantitative risk calculation (0-100) based on zone severity, incursion depth, and dwell duration.
  - **Contextual Safety Copilot:** Grounded incident Q&A explaining the regulatory and mechanical basis of the violation.
  - **CAPA Action Recording:** Checklists for contributing conditions (e.g. lighting, physical obstacles) and resolution status tracking (Under Review, CAPA Assigned, Resolved).
  - **One-Click Evidence Package:** Generate and download comprehensive Markdown and JSON audit dossiers.

### 3.5 AI Safety Copilot
- Conversational safety assistant grounded exclusively in real-time incident logs and plant metadata.
- Pre-built query chips for immediate inquiries:
  - *"Summarize all critical incidents"*
  - *"Which workers had the longest dwell time?"*
  - *"What are the recommended corrective actions for Zone 1?"*
- Explicit provenance reporting: indicates whether answers are generated from the primary backend engine or the deterministic offline safety ruleset.

### 3.6 Zone Management & Geometric Calibration
- **Interactive Polygon Editor:** Define and calibrate 2D convex hull coordinates for restricted machinery boundaries.
- **Safe Staging Workflow:** Configuration updates are saved to a staged file (`restricted_zone_working.json`) with validation checks before committing to production.
- **Safety Hysteresis:** Configurable entry/exit debounce timers (default: 3 frames / 0.1s) to prevent false triggering.

### 3.7 Analytics & EHS Compliance Reports
- **Cumulative Exposure Tracking:** Detailed dwell-time histograms analyzing worker exposure patterns.
- **Automated Formal Audit Generation:** Invoke `scripts/generate_ehs_report.py` to generate complete, regulatory-ready EHS compliance reports ready for download.

### 3.8 System Health & Edge Diagnostics
- **Microservice Status Indicators:** Edge Perception (DeepStream 7.1), TensorRT Model Engine (YOLO11s), Multi-Object Tracker (NvDCF), Incident Analytics Engine, and Persistent Telemetry Store.
- **Edge Deployment Specifications:** Precision details including model weights, inference batch size, tracking algorithm, and coordinate normalization boundaries.

---

## 4. Local Execution & Quick Start

### Prerequisites
- Python 3.10 or 3.11 installed.
- Modern web browser (Chrome, Edge, Firefox, Safari).

### Setup & Launch
```powershell
# 1. Clone repository & checkout feature branch
git clone https://github.com/utkarshsinha-dev/CraftifAIHackathon_CRA003.git
cd CraftifAIHackathon_CRA003
git switch feat/safety-dashboard

# 2. Install dependencies
py -3.11 -m pip install streamlit opencv-python-headless plotly pandas numpy pillow pytest

# 3. Launch Sentinel Vision Enterprise Platform
py -3.11 -m streamlit run app/dashboard.py
```

Access the application in your browser at `http://localhost:8501`.

---

## 5. Verification & Test Suite Matrix

The platform is fortified with comprehensive automated testing spanning frontend UI, integration adapters, and backend algorithms:

| Test Suite | Execution Command | Scope | Result |
| :--- | :--- | :--- | :--- |
| **Frontend Tests** | `py -3.11 app/test_frontend.py` | UI components, adapter contracts, data rendering | **11 / 11 PASSED** (100%) |
| **Backend Integration Checks** | `py -3.11 scripts/verify_backend_integration.py` | Dual signature compatibility, JSONL ingestion, trajectory API | **24 / 24 PASSED** (100%) |
| **Backend Core Tests** | `py -3.11 -m pytest tests/ -v` | Spatial geometry, dwell time, risk engine, copilot offline rules | **44 / 44 PASSED** (100%) |
| **Total Automated Tests** | | **All Subsystems Verified** | **79 / 79 PASSED** |

---

## 6. Enterprise Data Authenticity Principle

SENTINEL VISION enforces a strict **Reality-First Policy**:
- **Zero Synthetic Mock Data:** Every incident record, bounding box, timestamp, and trajectory vector originates strictly from verified edge perception outputs (`data/output/incident_records.jsonl`, `metadata.json`).
- **Traceable Coordinates:** Worker ground coordinates $(x, y)$ map directly to camera sensor coordinate space with documented mathematical transformations.
- **Honest Provenance:** Every metric explicitly distinguishes between raw sensor observations, derived algorithmic metrics (e.g. dwell time), and unavailable data channels.
