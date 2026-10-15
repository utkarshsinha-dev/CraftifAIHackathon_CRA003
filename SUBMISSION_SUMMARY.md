# Sentinel Vision AI — Hackathon Submission Summary

> **CraftifAI Physical AI Hackathon**  
> **Track**: Track A — PipeGen Perception  
> **Project Name**: Sentinel Vision AI  
> **Author / Team**: Sentinel Safety AI  
> **Target Hardware**: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM) / WSL2 Ubuntu 24.04  

---

## 1. Problem Statement

In industrial manufacturing, warehousing, and construction sites, unauthorized worker access to hazardous zones (heavy robotic cells, conveyor paths, and high-voltage exclusion zones) remains a primary cause of severe occupational injuries. Current safety solutions rely either on expensive physical fences or basic presence sensors that fail to distinguish safe personnel from genuine intruders, resulting in alarm fatigue or missed violations.

## 2. Solution Overview

**Sentinel Vision AI** is an intelligent workplace safety perception system built on **CraftifAI PipeGen** and **NVIDIA DeepStream**. It:
1. Ingests video surveillance streams.
2. Detects and tracks personnel with persistent identities across frames.
3. Projects a stable ground contact point (footprint) $(x_{\text{center}}, y_{\text{max}})$.
4. Performs real-time ray-casting point-in-polygon containment against configurable restricted zones.
5. Manages boundary state transitions (`OUTSIDE` &rarr; `ZONE_INTRUSION_ENTRY` &rarr; `ZONE_INTRUSION_OCCUPANCY` &rarr; `ZONE_INTRUSION_EXIT`).
6. Applies anti-spam cooldown throttling and emits structured, machine-readable JSON Lines (`.jsonl`) incident records.

---

## 3. System Architecture

```
[ Video Feed (NVIDIA Sample Video / File) ]
                     │
                     ▼
┌────────────────────────────────────────────────────────┐
│  CraftifAI PipeGen Perception Pipeline (DeepStream)    │
│  - Hardware Accelerated Decoding (nvv4l2decoder)      │
│  - Video Preprocessing & Scaling (nvvideoconvert)      │
│  - TensorRT Engine Inference (Person Detector)         │
│  - Multi-Object Tracker (NvDCF)                        │
└────────────────────────────────────────────────────────┘
                     │  (NvDsBatchMeta / Detection Records)
                     ▼
┌────────────────────────────────────────────────────────┐
│  DeepStream Metadata Adapter (src/deepstream_adapter)  │
│  - Filters Target Person Class (COCO class_id 0)       │
│  - Normalizes Pixel Rectangles to [0.0, 1.0] Space     │
│  - Cleans Tracking Sentinels (0xFFFFFFFFFFFFFFFF -> None)│
│  - Filters Out Noise & Malformed Frame Payloads        │
└────────────────────────────────────────────────────────┘
                     │  (Normalized Detections: bbox, track_id, conf)
                     ▼
┌────────────────────────────────────────────────────────┐
│  Spatial Zone Intrusion Monitor (src/zone_monitor)    │
│  - Configurable 2D Polygon ROI Definition             │
│  - Footprint Intersection: ((x_min + x_max)/2, y_max)  │
│  - State Machine: OUTSIDE -> ENTRY -> OCCUPANCY -> EXIT│
│  - Anti-Spam Alert Cooldown Throttling                │
└────────────────────────────────────────────────────────┘
                     │
                     ▼
[ Actionable Structured Output (data/output/incident_records.jsonl) ]
```

---

## 4. Completed Components & Deliverables Inventory

| Component / Artifact | File Path | Implementation Status |
| :--- | :--- | :--- |
| **Zone Monitor** | [`src/zone_monitor.py`](file:///e:/SentinelVisionAI/src/zone_monitor.py) | **COMPLETE**: Ray-casting point-in-polygon, footprint calculation, cooldown state machine. |
| **DeepStream Adapter** | [`src/deepstream_adapter.py`](file:///e:/SentinelVisionAI/src/deepstream_adapter.py) | **COMPLETE**: DeepStream pixel coordinate normalization, class filtering, untracked sentinel sanitization. |
| **Stream Runner** | [`src/sentinel_pipeline.py`](file:///e:/SentinelVisionAI/src/sentinel_pipeline.py) | **COMPLETE**: Dual-mode runner supporting stdin piping, JSONL stream consumption, and GStreamer probe callbacks. |
| **Demo Launcher** | [`launch_demo.py`](file:///e:/SentinelVisionAI/launch_demo.py) | **COMPLETE**: Pre-flight deliverable inventory, configuration validation, and stream execution. |
| **Simulation Demo** | [`demo_simulation.py`](file:///e:/SentinelVisionAI/demo_simulation.py) | **COMPLETE**: Verified offline logic demonstration with clear disclaimers. |
| **Artifact Synchronizer**| [`scripts/sync_pipegen_artifacts.py`](file:///e:/SentinelVisionAI/scripts/sync_pipegen_artifacts.py) | **COMPLETE**: Scans WSL candidate paths to ingest authentic PipeGen IR, engines, and logs without placeholders. |
| **Zone Configuration** | [`configs/restricted_zone.json`](file:///e:/SentinelVisionAI/configs/restricted_zone.json) | **COMPLETE**: Normalized polygonal danger zones with severity and cooldown definitions. |
| **Incident Schema** | [`configs/incident_schema.json`](file:///e:/SentinelVisionAI/configs/incident_schema.json) | **COMPLETE**: Formal JSON Schema (Draft 2020-12) defining emitted incident records. |
| **Automated Tests** | [`tests/`](file:///e:/SentinelVisionAI/tests/) | **COMPLETE**: 8/8 automated pytest integration and unit tests passing. |
| **Test Output Log** | [`artifacts/benchmarks/test_results.txt`](file:///e:/SentinelVisionAI/artifacts/benchmarks/test_results.txt) | **COMPLETE**: Genuine pytest execution transcript. |
| **Simulated Output** | [`data/output/simulation_incident_records.jsonl`](file:///e:/SentinelVisionAI/data/output/simulation_incident_records.jsonl) | **COMPLETE**: Verified sample incident records matching schema. |

---

## 5. Verification & Test Evidence

Command:
```powershell
python -m pytest tests/ -v
```

Verified Results (`artifacts/benchmarks/test_results.txt`):
- `tests/test_deepstream_integration.py::test_multi_person_trajectory_stream` &rarr; **PASSED**
- `tests/test_deepstream_integration.py::test_malformed_metadata_handling` &rarr; **PASSED**
- `tests/test_deepstream_integration.py::test_occupancy_cooldown_and_jsonl_persistence` &rarr; **PASSED**
- `tests/test_deepstream_integration.py::test_multiple_untracked_entities_distinct_locations` &rarr; **PASSED**
- `tests/test_deepstream_integration.py::test_stream_consumer_cli_integration` &rarr; **PASSED**
- `tests/test_zone_monitor.py::test_point_in_polygon` &rarr; **PASSED**
- `tests/test_zone_monitor.py::test_zone_monitor` &rarr; **PASSED**
- `tests/test_zone_monitor.py::test_deepstream_adapter` &rarr; **PASSED**
- **Summary: 8 passed in 0.03 seconds (100% assertions passing).**

---

## 6. Known Limitations & Disclosure

1. **Active Image Pull**: The official NVIDIA DeepStream container (`craftifaidocker/deepstream:7.1-amd64-v3`, ~12 GB compressed) was actively downloading and extracting in WSL during project preparation.
2. **Zero Fabrication**: No simulated benchmark numbers, placeholder `.engine` binaries, or mock IR files were generated. The authentic PipeGen Pipeline IR and compiled TensorRT engine will be generated directly through the platform upon container completion.
