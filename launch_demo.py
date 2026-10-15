"""
Sentinel Vision AI - Reproducible Demo Launcher

Manages the execution of the workplace safety perception system:
1. Validates environment, configurations, and required deliverables.
2. Checks for compiled PipeGen DeepStream pipeline files in generated/.
3. Launches real pipeline execution or offline metadata stream consumption.
4. Exits cleanly with actionable error messages without simulating inference.
"""

import argparse
import glob
import os
import sys
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def check_deliverable(label: str, pattern: str) -> bool:
    matches = glob.glob(os.path.join(PROJECT_ROOT, pattern))
    found = len(matches) > 0 and any(os.path.getsize(m) > 0 for m in matches)
    status_str = "[READY]" if found else "[PENDING]"
    details = f"({matches[0]})" if found else f"(pattern: {pattern})"
    print(f"  {status_str:<10} {label:<28} {details}")
    return found


def main():
    parser = argparse.ArgumentParser(description="Sentinel Vision AI Demo Launcher")
    parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "restricted_zone.json"), help="Zone config file")
    parser.add_argument("--output", default=os.path.join(PROJECT_ROOT, "data", "output", "incident_records.jsonl"), help="Incident output log")
    parser.add_argument("--stream", default=None, help="Input detection stream JSONL file (optional)")
    parser.add_argument("--pipeline-dir", default=os.path.join(PROJECT_ROOT, "generated"), help="Directory containing PipeGen generated pipeline")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("        SENTINEL VISION AI — WORKPLACE SAFETY MONITORING SYSTEM")
    print("        Track A: CraftifAI PipeGen Perception Submission")
    print("=" * 80)

    # 1. Verification of configurations
    print("\n[PRE-FLIGHT] Checking Configurations & Schemas:")
    if not os.path.isfile(args.config):
        print(f"  [ERROR] Restricted zone configuration missing: {args.config}")
        sys.exit(1)
    print(f"  [OK] Restricted Zone Config: {args.config}")

    schema_file = os.path.join(PROJECT_ROOT, "configs", "incident_schema.json")
    if not os.path.isfile(schema_file):
        print(f"  [ERROR] Incident JSON Schema missing: {schema_file}")
        sys.exit(1)
    print(f"  [OK] Incident JSON Schema:   {schema_file}")

    # 2. Output directory verification
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"  [OK] Incident Records Sink:  {args.output}")

    # 3. Deliverables Status Check
    print("\n[INVENTORY] Hackathon Deliverables Status:")
    ir_ok = check_deliverable("PipeGen Pipeline IR", "artifacts/ir/*ir*.*")
    eng_ok = check_deliverable("TensorRT Engine (.engine)", "artifacts/benchmarks/*.engine")
    bench_ok = check_deliverable("Benchmark Speed Report", "artifacts/benchmarks/speed_report.md")
    pipe_ok = check_deliverable("Generated Pipeline Code", "generated/*")

    print("\n" + "-" * 80)

    # 4. Determine Execution Mode
    pipe_sh = os.path.join(args.pipeline_dir, "run.sh")
    pipe_py = os.path.join(args.pipeline_dir, "pipeline.py")

    # Mode A: Real Compiled Pipeline Execution
    if os.path.isfile(pipe_sh) or os.path.isfile(pipe_py):
        print(f"[ACTIVE] Found PipeGen Generated Pipeline in: {args.pipeline_dir}")
        print("         Launching DeepStream perception pipeline with Sentinel Safety Adapter...")
        cmd = [
            sys.executable,
            os.path.join(PROJECT_ROOT, "src", "sentinel_pipeline.py"),
            "--config", args.config,
            "--output", args.output,
        ]
        if args.stream:
            cmd.extend(["--stream", args.stream])
        subprocess.run(cmd)

    # Mode B: Explicit Stream Ingestion Mode (e.g. piped or recorded metadata)
    elif args.stream:
        if args.stream != "-" and not os.path.isfile(args.stream):
            print(f"[ERROR] Specified stream file does not exist: {args.stream}")
            sys.exit(1)
        print(f"[ACTIVE] Ingestion Mode: Processing detection stream from '{args.stream}'...")
        cmd = [
            sys.executable,
            os.path.join(PROJECT_ROOT, "src", "sentinel_pipeline.py"),
            "--config", args.config,
            "--output", args.output,
            "--stream", args.stream,
        ]
        subprocess.run(cmd)

    # Mode C: Standby / Pre-Flight Mode
    else:
        print("[STANDBY] No active perception stream provided and generated/ is not yet populated.")
        print("          Sentinel Vision AI is in verification standby.")
        print("\n[ACTIONABLE NEXT STEPS]:")
        print("  1. Complete PipeGen pipeline compilation to populate 'generated/' and 'artifacts/'.")
        print("  2. To run with a pre-recorded or live detection stream, pass:")
        print(f"       python launch_demo.py --stream <path_to_detections.jsonl>")
        print("  3. To run all automated integration tests, run:")
        print("       python -m pytest tests/ -v")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
