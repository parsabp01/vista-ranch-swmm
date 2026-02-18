# Product Requirements Document (PRD)
## SWMM Hydraulic Model Automation (Codex Browser Workflow)

## 1. Source of Truth and Purpose
This PRD formalizes the governing architecture in `data/raw/Hydraulic Model Planning.docx` into an executable, repo-based workflow.

Goal: automate SWMM model generation from:
- Rational-method workbook (`.xlsm`)
- Two marked-up plan PDFs
- Planning framework document

The workflow must run in the Codex web environment without requiring a local IDE/terminal.

## 2. Objectives
1. Extract node/link/rational inputs from workbook/PDF metadata into reproducible tables.
2. Normalize canonical tables for model build.
3. Auto-create homogeneous SWMM subcatchments using heuristics.
4. Generate runnable `.inp` model artifact and QA outputs.
5. Execute RALPH loops (Reflect-Act-Learn-Plan-Hone) for iterative quality improvement.

## 3. Scope
### In Scope
- Scripted pipeline stages: preflight, extract, transform, build, qa
- Intake manifest and data lineage logs
- Canonical CSV outputs
- SWMM `.inp` generation with hydraulics + simplified hydrology scaffolding
- Deterministic QA checks and exception logs

### Out of Scope (for current iteration)
- GIS polygon catchments
- Full hydrologic calibration workflow
- Detailed CAD-based coordinates

## 4. Assumptions
- Workbook has enough information to discover candidate node/link/rational tables.
- PDF parsing is metadata-confirming in this iteration (full OCR parsing deferred).
- Homogeneous subcatchments are acceptable for first-pass modeling.
- Width heuristic:
  - `Width = 2 * Area / TcLength`
  - if `TcLength` missing, use configurable fallback.

## 5. Repository Architecture
```text
.
├─ prd.md
├─ docs/
│  ├─ runbook.md
│  └─ notes.md
├─ configs/
│  ├─ defaults.yaml
│  ├─ extraction.yaml
│  └─ ralph.yaml
├─ scripts/
│  ├─ run_pipeline.py
│  └─ pipeline_core.py
├─ data/
│  ├─ raw/
│  ├─ interim/
│  └─ processed/
├─ outputs/
│  ├─ logs/
│  └─ qa/
└─ models/
```

## 6. Agent Roles (Logical)
- Coordinator: orchestration, stage routing, status summary
- Extractor: workbook/PDF discovery + extraction outputs
- Validator: preflight and QA checks
- Model Builder: SWMM section assembly
- Reporter: assumptions and QA artifacts

(Implemented as functions/modules in scripts for deterministic execution.)

## 7. Data Contracts / Schemas
## 7.1 Intake Manifest (`outputs/logs/intake_manifest.json`)
- `generated_at_utc`
- `required_inputs` (workbook, pdfs, planning_doc)
- `files[]` with:
  - `path`, `exists`, `size_bytes`, `sha256`, `modified_time_utc`
- `status` (`ready` or `blocked`)
- `notes[]`

## 7.2 Interim Extraction Tables (`data/interim/`)
- `workbook_sheet_inventory.csv`
- `nodes_raw.csv`
- `links_raw.csv`
- `rational_raw.csv`
- `pdf_metadata.csv`

## 7.3 Processed Canonical Tables (`data/processed/`)
- `nodes.csv`
- `links.csv`
- `rational_data.csv`
- `subcatchment_defaults.csv`

Canonical columns are normalized to lowercase snake_case.

## 8. Functional Requirements
1. Preflight confirms workbook + two PDFs + planning doc exist.
2. Manifest writes deterministic metadata and hashes.
3. Extract stage:
   - discovers workbook sheets and headers programmatically
   - classifies candidate rows for nodes/links/rational using keyword maps
   - writes extracted tables even if partially populated
   - records discovered sheet details in inventory
4. Transform stage:
   - normalizes columns and IDs
   - generates baseline subcatchment defaults using heuristic width formula
5. Build stage:
   - writes a baseline `.inp` model in `models/`
6. QA stage:
   - validates required fields, missing IDs, connectivity references, negative lengths
   - writes report in `outputs/qa/`

## 9. RALPH Verification Loops
Each loop produces logs and explicit next actions.

1. **Reflect**: summarize anomalies from preflight/extract/qa outputs.
2. **Act**: apply deterministic corrections (column mapping, ID cleanup, default fills).
3. **Learn**: capture what rules improved extraction/quality.
4. **Plan**: decide next rerun stage and unresolved questions.
5. **Hone**: tighten config defaults/mappings; rerun from preflight or extract.

Artifacts:
- `outputs/qa/qa_summary.json`
- `outputs/qa/qa_findings.csv`
- `outputs/logs/pipeline_state.json`

## 10. Acceptance Tests
A run is acceptable when:
1. `python scripts/run_pipeline.py --stage preflight` returns `status: ready`.
2. `python scripts/run_pipeline.py --stage extract` writes:
   - intake manifest
   - workbook inventory
   - nodes_raw/links_raw/rational_raw (non-empty preferred; empty allowed only with explicit warning)
3. `python scripts/run_pipeline.py --stage transform` writes canonical processed CSVs.
4. `python scripts/run_pipeline.py --stage build` writes a `.inp` file in `models/`.
5. `python scripts/run_pipeline.py --stage qa` writes QA summary + findings.

## 11. Non-Developer Operating Model
- User only needs to request stage runs in Codex browser.
- Scripts generate all outputs in-repo.
- No local machine commands required.

## 12. Risks and Mitigations
- Workbook ambiguity: mitigate with sheet/header discovery inventory.
- Missing TcLength: fallback defaults from config, logged in assumptions.
- Partial extraction quality: RALPH loop reruns and mapping updates.
