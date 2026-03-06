#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pipeline_core import ROOT, now_utc, run_stage

VALID_STAGES = ["preflight", "extract", "transform", "build", "build_v2", "qa"]


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _input_paths_for_stage(stage: str) -> list[Path]:
    mapping = {
        "preflight": [
            ROOT / "data/raw/Copy of BX-HH-Vista Ranch_10-30-2025.xlsm",
            ROOT / "data/raw/BX-FM_10-30-2025-Model.pdf",
            ROOT / "data/raw/BX-FM_10-30-2025-Model2.pdf",
            ROOT / "data/raw/Hydraulic Model Planning.docx",
        ],
        "extract": [
            ROOT / "data/raw/Copy of BX-HH-Vista Ranch_10-30-2025.xlsm",
            ROOT / "data/raw/BX-FM_10-30-2025-Model.pdf",
            ROOT / "data/raw/BX-FM_10-30-2025-Model2.pdf",
        ],
        "transform": [
            ROOT / "data/interim/nodes_raw.csv",
            ROOT / "data/interim/links_raw.csv",
            ROOT / "data/interim/rational_raw.csv",
        ],
        "build": [
            ROOT / "data/processed/nodes.csv",
            ROOT / "data/processed/links.csv",
            ROOT / "data/processed/rational_data.csv",
            ROOT / "data/processed/subcatchment_defaults.csv",
        ],
        "build_v2": [
            ROOT / "data/raw/Branch_Logic_GPT.xlsx",
            ROOT / "data/raw/Copy of BX-HH-Vista Ranch_10-30-2025.xlsm",
        ],
        "qa": [
            ROOT / "models/model.inp",
            ROOT / "data/processed/nodes.csv",
            ROOT / "data/processed/links.csv",
        ],
    }
    return mapping[stage]


def _count_rows(path: Path) -> int | None:
    if not path.exists() or path.suffix != ".csv":
        return None
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return 0
    return max(0, txt.count("\n"))


def _key_counts(stage: str) -> dict:
    counts: dict[str, int] = {}
    for rel in [
        "data/interim/nodes_raw.csv",
        "data/interim/links_raw.csv",
        "data/interim/rational_raw.csv",
        "data/processed/nodes.csv",
        "data/processed/links.csv",
        "data/processed/rational_data.csv",
        "data/processed/subcatchment_defaults.csv",
        "outputs/qa/qa_findings.csv",
    ]:
        c = _count_rows(ROOT / rel)
        if c is not None:
            counts[rel] = c
    if (ROOT / "models/model.inp").exists():
        counts["models/model.inp_lines"] = len((ROOT / "models/model.inp").read_text(encoding="utf-8").splitlines())
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SWMM automation pipeline by stage")
    parser.add_argument("--stage", choices=VALID_STAGES, required=True)
    args = parser.parse_args()

    result = run_stage(args.stage)

    inputs_used = []
    for p in _input_paths_for_stage(args.stage):
        inputs_used.append(
            {
                "path": str(p.relative_to(ROOT)),
                "exists": p.exists(),
                "sha256": _sha256(p),
            }
        )

    assumptions = {
        "synthetic_fallback_allowed": args.stage in ["build", "qa"],
        "default_config": "configs/defaults.yaml",
    }
    if (ROOT / "outputs/logs/build_summary.json").exists():
        try:
            build_summary = json.loads((ROOT / "outputs/logs/build_summary.json").read_text(encoding="utf-8"))
            assumptions["build_assumptions"] = build_summary.get("assumptions", {})
        except Exception:
            pass

    pipeline_status = result.status
    next_action = "run next pipeline stage" if result.status == "ready" else "resolve blocking issues"
    blockers: list[str] = []

    if args.stage == "qa":
        qa_summary_path = ROOT / "outputs/qa/qa_summary.json"
        qa_findings_path = ROOT / "outputs/qa/qa_findings.csv"
        qa_summary = {}
        if qa_summary_path.exists():
            try:
                qa_summary = json.loads(qa_summary_path.read_text(encoding="utf-8"))
            except Exception:
                qa_summary = {}
        qa_status = str(qa_summary.get("status", result.status))
        high_count = int(qa_summary.get("high_count", 0) or 0)

        if qa_status in {"ready", "ready_for_swmm_gui"} and high_count == 0:
            pipeline_status = "ready"
            next_action = "swmm_gui_manual_qaqc"
        else:
            pipeline_status = "blocked"
            next_action = "resolve blocking issues"
            if qa_findings_path.exists():
                lines = [ln.strip() for ln in qa_findings_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
                if len(lines) > 1:
                    headers = lines[0].split(",")
                    for raw in lines[1:]:
                        vals = raw.split(",")
                        row = {headers[i]: vals[i] if i < len(vals) else "" for i in range(len(headers))}
                        if row.get("severity", "").lower() == "high":
                            blockers.append(f"{row.get('check', 'unknown_check')}: {row.get('message', '')}")
            if not blockers and high_count > 0:
                blockers.append(f"qa_summary reported {high_count} high-severity finding(s)")

    payload = {
        "generated_at_utc": now_utc(),
        "current_stage": result.stage,
        "status": pipeline_status,
        "message": result.message,
        "inputs_used": inputs_used,
        "outputs_produced": result.produced_files,
        "key_counts": _key_counts(args.stage),
        "assumptions_used": assumptions,
        "next_action": next_action,
    }
    if blockers:
        payload["blockers"] = blockers

    state_path = ROOT / "outputs/logs/pipeline_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"State written: {state_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
