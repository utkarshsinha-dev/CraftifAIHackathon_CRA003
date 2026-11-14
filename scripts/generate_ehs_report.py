"""
Sentinel Vision AI - Automated EHS Incident Investigation Report Generator

Compiles verified perception telemetry and incident records into a publication-ready
Environmental Health and Safety (EHS) Investigation & Audit Report.

Includes:
- Audit Header & Session Timestamps
- Incident KPI Executive Summary
- Severity & Event Type Breakdown Tables
- Chronological Incident Timeline
- Track & Hazard Zone Spatial Analysis (Dwell, Kinematics, Ground Contact)
- Explainable Risk Scoring & Contributing Factors
- Corrective & Preventive Actions (CAPA)
- Compliance & Operational Disclaimer
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def create_ehs_report(
    incidents: Optional[List[Dict[str, Any]]] = None,
    log_path: str = "data/output/incident_records.jsonl",
    output_path: str = "data/output/EHS_INCIDENT_REPORT.md",
    format: str = "markdown",
) -> str:
    """
    Builds the formal EHS Investigation Report from structured incident records.
    """
    records = incidents
    if records is None:
        records = []
        if os.path.isfile(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    total_events = len(records)
    unique_tracks = sorted(
        list(set(str(i.get("track_id") if i.get("track_id") is not None else i.get("entity_id", "untracked")) for i in records))
    )
    zones = sorted(list(set(str(i.get("zone_name") or i.get("zone_id") or "Unknown") for i in records)))
    max_dwell = max([float(i.get("dwell_time_s", 0.0)) for i in records], default=0.0)

    crit_count = sum(1 for i in records if (i.get("severity") or "").upper() == "CRITICAL")
    warn_count = sum(1 for i in records if (i.get("severity") or "").upper() == "WARNING")
    info_count = sum(1 for i in records if (i.get("severity") or "").upper() == "INFO")

    lines = []
    lines.append("# SENTINEL VISION AI — EHS INCIDENT INVESTIGATION REPORT")
    lines.append("")
    lines.append(f"> **Report Generated:** `{now_iso}`<br/>")
    lines.append(f"> **System:** Sentinel Vision AI v2.0 (PipeGen Perception + DeepStream 7.1)<br/>")
    lines.append(f"> **Hardware Node:** NVIDIA GeForce RTX 4060 Laptop GPU (TensorRT 10.3 FP32)<br/>")
    lines.append(f"> **Audit Target:** Workplace Perimeter Safety & Restricted Zone Monitoring")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Executive Summary")
    lines.append("")
    lines.append(
        f"During the surveillance monitoring session, **{total_events} safety incident events** were "
        f"automatically logged across **{len(unique_tracks)} detected personnel** (Personnel IDs: {', '.join(unique_tracks) if unique_tracks else 'None'}). "
        f"A total of **{crit_count} events were assessed at CRITICAL severity**, requiring automated supervisor dispatch. "
        f"The maximum recorded continuous dwell duration inside hazardous boundaries was **{max_dwell:.2f} seconds**."
    )
    lines.append("")
    lines.append("### Safety Metrics & KPIs")
    lines.append("")
    lines.append("| Metric | Measured Value | Operational Threshold | Status |")
    lines.append("| :--- | :--- | :--- | :--- |")
    lines.append(f"| **Total Logged Incidents** | **{total_events}** | 0 Target | {'FLAGGED' if total_events > 0 else 'CLEAR'} |")
    lines.append(f"| **Critical Breaches** | **{crit_count}** | 0 Target | {'CRITICAL' if crit_count > 0 else 'CLEAR'} |")
    lines.append(f"| **Peak Dwell Duration** | **{max_dwell:.2f} s** | < 2.0 s Cooldown | {'EXCEEDED' if max_dwell > 2.0 else 'NORMAL'} |")
    lines.append(f"| **Unique Personnel Involved** | **{len(unique_tracks)}** | Authorized Only | {'INVESTIGATE' if unique_tracks else 'CLEAR'} |")
    lines.append(f"| **Restricted Zones Breached** | **{', '.join(zones) if zones else 'None'}** | 0 Zones | {'PERIMETER COMPROMISED' if zones else 'CLEAR'} |")
    lines.append(f"| **Alarm Anti-Spam Suppression** | **Cooldown Active &rarr; {total_events} alerts** | Noise Suppression Active | **OPERATIONAL** |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Chronological Incident Audit Log")
    lines.append("")
    lines.append("| Seq | Video Time | Frame | Track ID | Zone | Event Type | Dwell | Risk Score | Assessed Severity |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    for idx, inc in enumerate(records, 1):
        v_time = inc.get("video_time_s", 0.0)
        f_idx = inc.get("frame_index", "N/A")
        tid = inc.get("track_id") if inc.get("track_id") is not None else inc.get("entity_id", "untracked")
        zname = inc.get("zone_name", inc.get("zone_id", "Zone"))
        evt = inc.get("event_type", "UNKNOWN")
        dwell = inc.get("dwell_time_s", 0.0)
        risk_score = inc.get("risk_assessment", {}).get("risk_score", "N/A")
        sev = inc.get("severity", "WARNING")
        lines.append(f"| {idx:02d} | `{v_time:.2f}s` | #{f_idx} | Track {tid} | {zname} | `{evt}` | {dwell:.1f}s | **{risk_score}/100** | `{sev}` |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Spatial & Kinematic Incident Analysis")
    lines.append("")

    for idx, inc in enumerate(records, 1):
        tid = inc.get("track_id") if inc.get("track_id") is not None else inc.get("entity_id", "untracked")
        lines.append(f"### Incident #{idx}: {inc.get('event_type')} (Track {tid})")
        lines.append("")
        lines.append(f"- **Incident ID:** `{inc.get('incident_id')}`")
        lines.append(f"- **Session Timestamp:** `{inc.get('timestamp_iso')}` (Video elapsed: `{inc.get('video_time_s')}s`, Frame `#{inc.get('frame_index')}`)")
        lines.append(f"- **Perimeter Location:** `{inc.get('zone_name')}` (`{inc.get('zone_id')}`)")
        lines.append(f"- **Footprint (Ground Contact Point):** `{inc.get('ground_contact_point')}` (Bounding Box: `{inc.get('bbox')}`)")
        lines.append(f"- **Detection Confidence:** `{inc.get('confidence', 0.0):.2%}` (TensorRT YOLO11s FP32)")
        lines.append(f"- **Continuous Dwell Duration:** `{inc.get('dwell_time_s', 0.0):.2f} seconds`")

        kin = inc.get("kinematics", {})
        if kin:
            lines.append(f"- **Kinematic Motion:** State: `{kin.get('motion_state')}`, Heading: `{kin.get('direction')}`, Speed: `{kin.get('speed_normalized')}` screen/s")

        risk = inc.get("risk_assessment", {})
        if risk:
            lines.append(f"- **Risk Assessment:** Score: **{risk.get('risk_score')}/100** (`{risk.get('assessed_severity')}`)")
            lines.append(f"- **Explainable Root Cause:** {risk.get('explanation')}")
        lines.append("")

    lines.append("---")
    lines.append("")
    if not records:
        lines.append("## 4. Root-Cause Analysis & Risk Factors")
        lines.append("")
        lines.append("- No perimeter breaches or safety risk factors detected. Monitored perimeters remained clear.")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## 5. Corrective & Preventive Actions (CAPA)")
        lines.append("")
        lines.append("- [ ] **CAPA-01 (Routine Audit):** Maintain scheduled sensor and optical camera inspection intervals.\n"
                     "- [ ] **CAPA-02 (Perimeter Assurance):** Verify zone coordinates in `configs/restricted_zone.json` remain aligned with facility markings.")
    else:
        lines.append("## 4. Root-Cause Analysis & Risk Factors")
        lines.append("")
        primary_zone = zones[0] if zones else "Restricted Area"
        lines.append(
            f"1. **Hazard Perimeter Incursion:** Active intrusion into `{', '.join(zones)}` "
            f"involving {len(unique_tracks)} personnel (IDs: {', '.join(unique_tracks)}).\n"
            f"2. **Continuous Dwell Exposure:** Maximum duration reached {max_dwell:.2f}s inside "
            f"restricted envelope, triggering persistent occupancy alerts under cooldown supervision.\n"
            f"3. **Multi-Worker Safety Risk:** {total_events} cumulative alert events recorded, "
            f"of which {crit_count} reached CRITICAL severity classification."
        )
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## 5. Corrective & Preventive Actions (CAPA)")
        lines.append("")
        lines.append(
            f"- [ ] **CAPA-01 (Supervisor Intervention):** Verify authorization and credentials for personnel ({', '.join(unique_tracks)}) involved in perimeter breach.\n"
            f"- [ ] **CAPA-02 (Perimeter Review):** Inspect physical barriers and signage around `{primary_zone}` to ensure optical clearance.\n"
            f"- [ ] **CAPA-03 (Interlock Verification):** Audit e-stop interlock response timing when dwell exceeds safety threshold.\n"
            f"- [ ] **CAPA-04 (Configuration Audit):** Confirm `configs/restricted_zone.json` maintains proper boundary buffers."
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 6. Regulatory & Operational Disclaimer")
    lines.append("")
    lines.append(
        "> [!NOTE]\n"
        "> This document is an automated computer-vision telemetry investigation report generated by Sentinel Vision AI.\n"
        "> It provides empirical perception analytics and spatial containment evidence for workplace safety auditing.\n"
        "> This report does not constitute formal certification or legal endorsement by OSHA, ISO, or regulatory safety bodies."
    )
    lines.append("")

    report_content = "\n".join(lines)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"[REPORT] EHS Investigation Report generated: {output_path}")

    return report_content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentinel Vision AI EHS Incident Report Generator")
    parser.add_argument("--log", default="data/output/incident_records.jsonl", help="Input incident JSONL log")
    parser.add_argument("--out", default="data/output/EHS_INCIDENT_REPORT.md", help="Output Markdown report path")
    args = parser.parse_args()

    create_ehs_report(log_path=args.log, output_path=args.out)
