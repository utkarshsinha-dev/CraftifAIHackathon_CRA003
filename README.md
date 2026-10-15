# Sentinel Vision AI

**Automated Workplace Safety Monitoring & Restricted Zone Intrusion Detection**  
Built with **CraftifAI PipeGen v0.5.1** and **NVIDIA DeepStream** for the **CraftifAI Physical AI Hackathon (Track A: PipeGen Perception)**.

---

## 1. Problem & Executive Summary

In high-risk industrial environments—such as automated manufacturing floors, heavy construction sites, and robotic assembly cells—unauthorized personnel entering hazardous exclusion zones represents a leading cause of severe workplace injuries.

**Sentinel Vision AI** provides continuous, automated computer-vision perimeter monitoring. It:
1. Ingests video surveillance streams.
2. Detects and tracks human personnel with persistent IDs.
3. Calculates the person's ground-contact footprint $(x_{\text{center}}, y_{\text{max}})$.
4. Evaluates spatial containment against configurable 2D polygonal restricted zones in real time.
5. Emits structured, machine-readable JSON Lines (`.jsonl`) incident records with timestamps, tracking IDs, zone identifiers, severity levels, and detection confidences for instant security dispatch and safety audits.

---

## 2. System Architecture

```
[ Video Feed (NVIDIA Sample Video / File) ]
                     │
                     ▼
┌────────────────────────────────────────────────────────┐
│  CraftifAI PipeGen Perception Pipeline (DeepStream)    │
│                                                        │
│  [ Hardware Video Decoder (nvv4l2decoder) ]            │
│               │                                        │
│  [ Video Preprocessing & Scaling (nvvideoconvert) ]   │
│               │                                        │
│  [ TensorRT Engine Inference (Person Detector) ]       │
│               │                                        │
│  [ Multi-Object Tracker (NvDCF) ]                      │
└────────────────────────────────────────────────────────┘
                     │
                     ▼  (NvDsBatchMeta / Detection Records)
┌────────────────────────────────────────────────────────┐
│  DeepStream Metadata Adapter (src/deepstream_adapter)  │
│  - Filters Target Person Class (COCO class_id 0)       │
│  - Normalizes Pixel Rectangles to [0.0, 1.0] Space     │
│  - Cleans Tracking Sentinels (0xFFFFFFFFFFFFFFFF -> None)│
│  - Filters Out Noise & Malformed Frame Payloads        │
└────────────────────────────────────────────────────────┘
                     │
                     ▼  (Normalized Detections: bbox, track_id, conf)
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

## 3. Environment & Hardware Specifications

- **Host Operating System**: Windows 11 (64-bit)
- **Host GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB GDDR6 VRAM, Compute Capability 8.9)
- **Linux Subsystem**: WSL2 (Ubuntu 24.04 LTS)
- **Containerization**: Docker Engine + NVIDIA Container Toolkit (`nvidia-ctk`)
- **Perception Platform**: CraftifAI PipeGen v0.5.1
- **Underlying Engine**: NVIDIA DeepStream SDK 7.1 in Docker (`craftifaidocker/deepstream:7.1-amd64-v3`)

---

## 4. Repository Structure

```
SentinelVisionAI/
├── README.md                      # Comprehensive project documentation & guide
├── PROJECT_PLAN.md                # Hackathon development milestones
├── REQUIREMENTS.md                # Functional and technical requirements
├── launch_demo.py                 # Reproducible demo launcher & inventory validator
├── demo_simulation.py             # Software simulation & logic verification demo
├── configs/
│   ├── restricted_zone.json       # Configurable 2D polygon restricted zones
│   └── incident_schema.json       # Formal JSON Schema (Draft 2020-12) for incident records
├── data/
│   ├── samples/                   # Reference video clips
│   └── output/                    # Generated JSON incident records and logs
├── artifacts/
│   ├── ir/                        # PipeGen Pipeline IR specifications
│   └── benchmarks/                # TensorRT latency and throughput reports
├── docs/
│   └── METADATA_INTEGRATION.md    # DeepStream metadata integration specification
├── scripts/
│   └── sync_pipegen_artifacts.py  # Safe artifact scanner & repository sync tool
├── src/
│   ├── zone_monitor.py            # Geometric polygon containment and state machine
│   ├── deepstream_adapter.py      # DeepStream pyds / JSON metadata normalizer
│   └── sentinel_pipeline.py       # End-to-end pipeline runner and stream consumer
└── tests/
    ├── test_zone_monitor.py       # Unit tests for containment, states, and cooldown
    └── test_deepstream_integration.py # Multi-person stream, malformed data & CLI integration tests
```

---

## 5. Verification & Automated Test Suite

Sentinel Vision AI features an automated pytest test suite covering unit geometry, state transitions, metadata normalization, and end-to-end streaming.

```powershell
python -m pytest tests/ -v
```

### Verified Test Results (8 of 8 Passing):
| Test Name | Verification Focus | Status |
| :--- | :--- | :--- |
| `test_multi_person_trajectory_stream` | 60-frame continuous multi-person crossing (`OUTSIDE` &rarr; `ENTRY` &rarr; `OCCUPANCY` &rarr; `EXIT`), non-person filtering, and untracked detection. | **PASSED** |
| `test_malformed_metadata_handling` | Resilience against truncated JSON, NaN coordinates, and invalid values without unhandled exceptions. | **PASSED** |
| `test_occupancy_cooldown_and_jsonl_persistence` | Exact interval throttling and JSON Lines formatting. | **PASSED** |
| `test_multiple_untracked_entities_distinct_locations` | Simultaneous untracked entities tracked separately without collision. | **PASSED** |
| `test_stream_consumer_cli_integration` | CLI stream consumer reading and serializing JSONL records. | **PASSED** |
| `test_point_in_polygon` | Ray-casting geometric containment inside and outside arbitrary polygons. | **PASSED** |
| `test_zone_monitor` | State machine transitions and incident persistence. | **PASSED** |
| `test_deepstream_adapter` | DeepStream pixel-space normalization, non-person filtering, and schema validation. | **PASSED** |

---

## 6. Execution & Demo Instructions

### A. Run the Software Simulation Demo
To verify spatial zone containment, state transitions, and JSON Lines alerting logic independently of GPU container initialization:
```powershell
python demo_simulation.py
```
*(Clearly labelled as a software simulation test fixture; outputs to `data/output/simulation_incident_records.jsonl`).*

### B. Pre-Flight Deliverables Inventory Check
```powershell
python launch_demo.py
```
Validates required zone configurations, output paths, and scans for compiled PipeGen pipeline deliverables.

### C. Replay or Stream Ingestion Mode
```powershell
python launch_demo.py --stream path/to/detections.jsonl
```

### D. Synchronize Genuine PipeGen Artifacts
Once PipeGen compiles the pipeline in WSL, harvest the authentic deliverables:
```powershell
python scripts/sync_pipegen_artifacts.py
```

### E. Run the Live Compiled DeepStream Pipeline
```bash
# Inside WSL generated/ directory once compiled:
cd generated && ./run.sh | python ../src/sentinel_pipeline.py --stream -
```

---

## 7. Deliverables & Current Hackathon Inventory

| Category | Deliverable | Status | Evidence / Location |
| :--- | :--- | :--- | :--- |
| **Safety Logic** | Zone Monitor (`src/zone_monitor.py`) | **COMPLETE** | Ray-casting point-in-polygon, footprint calculation, cooldown state machine. |
| **Pipeline Bridge**| DeepStream Adapter (`src/deepstream_adapter.py`) | **COMPLETE** | Normalizes DeepStream pixel bounding boxes, cleans tracker IDs. |
| **Configuration** | Zone Config & Schema (`configs/`) | **COMPLETE** | [`restricted_zone.json`](configs/restricted_zone.json), [`incident_schema.json`](configs/incident_schema.json). |
| **Test Suite** | Comprehensive Tests (`tests/`) | **COMPLETE** | 8/8 automated pytest integration tests passing. |
| **Sync Tooling** | Multi-path Ingestion (`scripts/`) | **COMPLETE** | Safe artifact synchronizer scanning WSL paths without mock files. |
| **Pipeline IR** | `artifacts/ir/pipeline_ir.yaml` | **PENDING** | To be generated directly by PipeGen requirement prompt upon container completion. |
| **TensorRT Engine**| `artifacts/benchmarks/model.engine` | **PENDING** | To be compiled on RTX 4060 through PipeGen model compiler. |
| **Benchmark Log** | `artifacts/benchmarks/speed_report.md` | **PENDING** | To record measured FPS, latency, and VRAM utilization from real execution. |

---

## 8. Honest Limitations & Disclosure

1. **Active Container Download**:
   The required NVIDIA DeepStream container (`craftifaidocker/deepstream:7.1-amd64-v3`, ~12 GB compressed) was actively pulling and extracting in the background during implementation.
2. **Zero Fabrication Policy**:
   In strict accordance with hackathon rules, no mock TensorRT engines, fabricated benchmark figures, or simulated detections were presented as real pipeline outputs. As soon as the container finishes extraction, the verified requirement prompt is submitted to produce the authentic IR and TensorRT engine.
