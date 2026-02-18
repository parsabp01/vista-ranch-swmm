#!/usr/bin/env python3
"""Compatibility wrapper for staged pipeline."""
from pathlib import Path
from pipeline_core import run_stage

STAGE_MAP = {
    "00_inventory_inputs.py": "preflight",
    "01_extract_excel.py": "extract",
    "02_extract_pdf_annotations.py": "extract",
    "03_normalize_entities.py": "transform",
    "04_build_network_graph.py": "transform",
    "05_generate_subcatchments.py": "transform",
    "06_build_swmm_inp.py": "build",
    "07_run_qa_checks.py": "qa",
    "08_export_review_pack.py": "qa",
}

if __name__ == "__main__":
    this = Path(__file__).name
    result = run_stage(STAGE_MAP[this])
    print(result)
