# SENTINEL VISION AI — EHS INCIDENT INVESTIGATION REPORT

> **Report Generated:** `2026-11-14 06:39:08 UTC`  
> **System:** Sentinel Vision AI v2.0 (PipeGen Perception + DeepStream 7.1)  
> **Hardware Node:** NVIDIA GeForce RTX 4060 Laptop GPU (TensorRT 10.3 FP32)  
> **Audit Target:** Workplace Perimeter Safety & Restricted Zone Monitoring  

---

## 1. Executive Summary

During the surveillance monitoring session, **6 safety incident events** were automatically logged across **2 detected personnel** (Personnel IDs: 0, 1). A total of **6 events were assessed at CRITICAL severity**, requiring automated supervisor dispatch. The maximum recorded continuous dwell duration inside hazardous boundaries was **4.07 seconds**.

### Safety Metrics & KPIs

| Metric | Measured Value | Operational Threshold | Status |
| :--- | :--- | :--- | :--- |
| **Total Logged Incidents** | **6** | 0 Target | FLAGGED |
| **Critical Breaches** | **6** | 0 Target | CRITICAL |
| **Peak Dwell Duration** | **4.07 s** | < 2.0 s Cooldown | EXCEEDED |
| **Unique Personnel Involved** | **2** | Authorized Only | INVESTIGATE |
| **Restricted Zones Breached** | **Heavy Equipment Perimeter A** | 0 Zones | PERIMETER COMPROMISED |
| **Alarm Anti-Spam Suppression** | **320 raw &rarr; 6 alerts** | > 95% noise reduction | **98.1% EFFECTIVE** |

---

## 2. Chronological Incident Audit Log

| Seq | Video Time | Frame | Track ID | Zone | Event Type | Dwell | Risk Score | Assessed Severity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 01 | `0.18s` | #3 | Track 0 | Heavy Equipment Perimeter A | `ZONE_INTRUSION_ENTRY` | 0.0s | **85/100** | `CRITICAL` |
| 02 | `0.18s` | #3 | Track 1 | Heavy Equipment Perimeter A | `ZONE_INTRUSION_ENTRY` | 0.0s | **85/100** | `CRITICAL` |
| 03 | `2.21s` | #64 | Track 0 | Heavy Equipment Perimeter A | `ZONE_INTRUSION_OCCUPANCY` | 2.0s | **80/100** | `CRITICAL` |
| 04 | `2.21s` | #64 | Track 1 | Heavy Equipment Perimeter A | `ZONE_INTRUSION_OCCUPANCY` | 2.0s | **80/100** | `CRITICAL` |
| 05 | `4.24s` | #125 | Track 0 | Heavy Equipment Perimeter A | `ZONE_INTRUSION_OCCUPANCY` | 4.1s | **85/100** | `CRITICAL` |
| 06 | `4.24s` | #125 | Track 1 | Heavy Equipment Perimeter A | `ZONE_INTRUSION_OCCUPANCY` | 4.1s | **85/100** | `CRITICAL` |

---

## 3. Spatial & Kinematic Incident Analysis

### Incident #1: ZONE_INTRUSION_ENTRY (Track 0)

- **Incident ID:** `b60605a2-b05c-49d9-a5c2-91cc516d91c5`
- **Session Timestamp:** `2026-11-14T06:37:45.669389+00:00` (Video elapsed: `0.1765s`, Frame `#3`)
- **Perimeter Location:** `Heavy Equipment Perimeter A` (`zone_restricted_01`)
- **Footprint (Ground Contact Point):** `[0.5608, 0.6479]` (Bounding Box: `[0.5439, 0.5806, 0.5777, 0.6479]`)
- **Detection Confidence:** `52.33%` (TensorRT YOLO11s FP32)
- **Continuous Dwell Duration:** `0.00 seconds`
- **Kinematic Motion:** State: `stationary`, Heading: `STATIONARY`, Speed: `0.0` screen/s
- **Risk Assessment:** Score: **85/100** (`CRITICAL`)
- **Explainable Root Cause:** Assessed Risk: CRITICAL (Score 85/100). Base zone hazard is CRITICAL. Immediate perimeter breach into restricted boundary with deep penetration near hazard core. Motion dynamics: Stationary within active hazard zone.

### Incident #2: ZONE_INTRUSION_ENTRY (Track 1)

- **Incident ID:** `ef91ec24-c081-4f65-a5cd-972ae5d6d5a8`
- **Session Timestamp:** `2026-11-14T06:37:45.669389+00:00` (Video elapsed: `0.1765s`, Frame `#3`)
- **Perimeter Location:** `Heavy Equipment Perimeter A` (`zone_restricted_01`)
- **Footprint (Ground Contact Point):** `[0.4995, 0.6513]` (Bounding Box: `[0.4815, 0.5882, 0.5175, 0.6513]`)
- **Detection Confidence:** `59.48%` (TensorRT YOLO11s FP32)
- **Continuous Dwell Duration:** `0.00 seconds`
- **Kinematic Motion:** State: `stationary`, Heading: `STATIONARY`, Speed: `0.0` screen/s
- **Risk Assessment:** Score: **85/100** (`CRITICAL`)
- **Explainable Root Cause:** Assessed Risk: CRITICAL (Score 85/100). Base zone hazard is CRITICAL. Immediate perimeter breach into restricted boundary with deep penetration near hazard core. Motion dynamics: Stationary within active hazard zone.

### Incident #3: ZONE_INTRUSION_OCCUPANCY (Track 0)

- **Incident ID:** `1ff162d5-f613-4621-8a12-c2aa9fb83531`
- **Session Timestamp:** `2026-11-14T06:37:47.702356+00:00` (Video elapsed: `2.2095s`, Frame `#64`)
- **Perimeter Location:** `Heavy Equipment Perimeter A` (`zone_restricted_01`)
- **Footprint (Ground Contact Point):** `[0.561, 0.6479]` (Bounding Box: `[0.5444, 0.5805, 0.5775, 0.6479]`)
- **Detection Confidence:** `52.77%` (TensorRT YOLO11s FP32)
- **Continuous Dwell Duration:** `2.03 seconds`
- **Kinematic Motion:** State: `stationary`, Heading: `STATIONARY`, Speed: `0.0` screen/s
- **Risk Assessment:** Score: **80/100** (`CRITICAL`)
- **Explainable Root Cause:** Assessed Risk: CRITICAL (Score 80/100). Base zone hazard is CRITICAL. Continued occupancy (2.0s) past baseline safety cooldown with deep penetration near hazard core. Motion dynamics: Stationary within active hazard zone.

### Incident #4: ZONE_INTRUSION_OCCUPANCY (Track 1)

- **Incident ID:** `a2302ea5-8f27-4af4-a174-1239445d2c53`
- **Session Timestamp:** `2026-11-14T06:37:47.702356+00:00` (Video elapsed: `2.2095s`, Frame `#64`)
- **Perimeter Location:** `Heavy Equipment Perimeter A` (`zone_restricted_01`)
- **Footprint (Ground Contact Point):** `[0.5038, 0.6512]` (Bounding Box: `[0.4826, 0.5836, 0.5249, 0.6512]`)
- **Detection Confidence:** `66.78%` (TensorRT YOLO11s FP32)
- **Continuous Dwell Duration:** `2.03 seconds`
- **Kinematic Motion:** State: `stationary`, Heading: `STATIONARY`, Speed: `0.0028` screen/s
- **Risk Assessment:** Score: **80/100** (`CRITICAL`)
- **Explainable Root Cause:** Assessed Risk: CRITICAL (Score 80/100). Base zone hazard is CRITICAL. Continued occupancy (2.0s) past baseline safety cooldown with deep penetration near hazard core. Motion dynamics: Stationary within active hazard zone.

### Incident #5: ZONE_INTRUSION_OCCUPANCY (Track 0)

- **Incident ID:** `e4e9280a-8c53-4501-b943-5703429799d4`
- **Session Timestamp:** `2026-11-14T06:37:49.735422+00:00` (Video elapsed: `4.2426s`, Frame `#125`)
- **Perimeter Location:** `Heavy Equipment Perimeter A` (`zone_restricted_01`)
- **Footprint (Ground Contact Point):** `[0.561, 0.6479]` (Bounding Box: `[0.5444, 0.5806, 0.5775, 0.6479]`)
- **Detection Confidence:** `52.96%` (TensorRT YOLO11s FP32)
- **Continuous Dwell Duration:** `4.07 seconds`
- **Kinematic Motion:** State: `stationary`, Heading: `STATIONARY`, Speed: `0.0` screen/s
- **Risk Assessment:** Score: **85/100** (`CRITICAL`)
- **Explainable Root Cause:** Assessed Risk: CRITICAL (Score 85/100). Base zone hazard is CRITICAL. Substantial occupancy duration (4.1s) exceeding safety threshold with deep penetration near hazard core. Motion dynamics: Stationary within active hazard zone.

### Incident #6: ZONE_INTRUSION_OCCUPANCY (Track 1)

- **Incident ID:** `9780f040-6d9a-4d28-9c18-3df82ca50181`
- **Session Timestamp:** `2026-11-14T06:37:49.735422+00:00` (Video elapsed: `4.2426s`, Frame `#125`)
- **Perimeter Location:** `Heavy Equipment Perimeter A` (`zone_restricted_01`)
- **Footprint (Ground Contact Point):** `[0.5037, 0.6512]` (Bounding Box: `[0.4826, 0.5836, 0.5248, 0.6512]`)
- **Detection Confidence:** `66.85%` (TensorRT YOLO11s FP32)
- **Continuous Dwell Duration:** `4.07 seconds`
- **Kinematic Motion:** State: `stationary`, Heading: `STATIONARY`, Speed: `0.0009` screen/s
- **Risk Assessment:** Score: **85/100** (`CRITICAL`)
- **Explainable Root Cause:** Assessed Risk: CRITICAL (Score 85/100). Base zone hazard is CRITICAL. Substantial occupancy duration (4.1s) exceeding safety threshold with deep penetration near hazard core. Motion dynamics: Stationary within active hazard zone.

---

## 4. Root-Cause Analysis & Risk Factors

1. **Proximity to Automated Diagnostic Machinery:** Personnel entered the machine envelope at Frame 3 (t=0.18s) and remained stationary for 4.24 seconds without safety barrier de-energization.
2. **Stationary Loitering:** Prolonged dwell time indicates intentional maintenance or troubleshooting activity without lockout/tagout (LOTO) verification.
3. **Dual Personnel Exposure:** Simultaneous intrusion by Track 0 and Track 1 increases multi-worker risk.

---

## 5. Corrective & Preventive Actions (CAPA)

- [ ] **CAPA-01 (Immediate Dispatch):** Trigger automated supervisor intercom alert when dwell duration exceeds 2.0s in Heavy Equipment Perimeter A.
- [ ] **CAPA-02 (Perimeter Interlock):** Evaluate optical interlock integration with machinery emergency-stop (E-stop) circuit.
- [ ] **CAPA-03 (Worker Verification):** Verify badge credentials for Personnel Track 0 and Track 1 to confirm safety maintenance authorization.
- [ ] **CAPA-04 (Zone Re-calibration):** Audit boundary coordinates in `configs/restricted_zone.json` to ensure 0.5m buffer clearance.

---

## 6. Regulatory & Operational Disclaimer

> [!NOTE]
> This document is an automated computer-vision telemetry investigation report generated by Sentinel Vision AI.
> It provides empirical perception analytics and spatial containment evidence for workplace safety auditing.
> This report does not constitute formal certification or legal endorsement by OSHA, ISO, or regulatory safety bodies.
