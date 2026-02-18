# SWMM Auto-Build Blueprint (No GIS Geometry)

## 1) Proposed Project Architecture

Use a **pipeline architecture** with explicit stages and checkpoint files so a non-developer can run one step at a time and inspect outputs before proceeding.

### Architecture layers
1. **Raw Inputs Layer**
   - Excel workbook (rational method source)
   - 2 annotated plan PDFs (inlets/manholes/pipes)
   - Planning document

2. **Extraction Layer**
   - Pull tabular source data from Excel
   - Pull pipe/node annotations from PDFs via OCR + parsing
   - Produce standardized CSV/JSON tables

3. **Normalization & Modeling Layer**
   - Harmonize IDs (e.g., `MH-101` vs `MH101`)
   - Build a connected network graph (nodes + links)
   - Create homogeneous subcatchments algorithmically (no GIS polygons)

4. **SWMM Assembly Layer**
   - Convert normalized tables into SWMM sections
   - Emit `.inp` model plus traceability reports

5. **QA / Verification Layer**
   - RALPH-style loops (rapid extract → check → fix → rerun)
   - Connectivity, slope, continuity, and sanity checks
   - Flag exceptions in machine-readable and human-readable reports

6. **Delivery Layer**
   - Versioned output model and QA package
   - Final checklist and handoff notes for manual engineering review

---

## 2) Recommended Folder Structure

```text
vista-ranch-swmm/
├─ data/
│  ├─ raw/                     # source files (Excel, PDFs, planning doc)
│  ├─ interim/                 # extracted raw tables and OCR text
│  └─ processed/               # cleaned, canonical tables used by model builder
├─ config/
│  ├─ project_config.yaml      # global settings (units, defaults, assumptions)
│  ├─ field_mappings.yaml      # Excel/PDF field mapping rules
│  └─ id_rules.yaml            # naming/normalization rules
├─ scripts/
│  ├─ 00_inventory_inputs.py
│  ├─ 01_extract_excel.py
│  ├─ 02_extract_pdf_annotations.py
│  ├─ 03_normalize_entities.py
│  ├─ 04_build_network_graph.py
│  ├─ 05_generate_subcatchments.py
│  ├─ 06_build_swmm_inp.py
│  ├─ 07_run_qa_checks.py
│  └─ 08_export_review_pack.py
├─ qa/
│  ├─ checklists/
│  ├─ reports/
│  └─ exceptions/
├─ logs/
├─ outputs/
│  ├─ model/
│  └─ review/
├─ templates/
│  ├─ swmm_section_defaults.yaml
│  └─ qa_report_template.md
└─ docs/
   ├─ SWMM_AUTOBUILD_BLUEPRINT.md
   └─ RUNBOOK.md
```

---

## 3) Required Scripts and Responsibilities

### `00_inventory_inputs.py`
- Confirms required files exist.
- Generates an input manifest (`data/interim/input_inventory.json`) with file hashes.
- Prevents accidental use of stale files.

### `01_extract_excel.py`
- Reads workbook sheets and cell ranges.
- Exports structured tables (CSV/JSON): runoff coefficients, design intensities, known drainage areas, pipe schedules (if present), notes.
- Captures sheet-level provenance (`sheet`, `range`, timestamp).

### `02_extract_pdf_annotations.py`
- OCRs both PDFs and extracts text near callouts.
- Detects entities and attributes:
  - Nodes: inlets/manholes, rim/invert (if present)
  - Links: pipe IDs, diameters, materials, lengths, slopes, direction cues
- Stores raw + parsed outputs to allow manual correction.

### `03_normalize_entities.py`
- Standardizes IDs and units.
- Resolves duplicates/conflicts between Excel and PDF sources.
- Produces canonical tables:
  - `nodes.csv`
  - `links.csv`
  - `catchment_inputs.csv`
  - `assumptions_log.csv`

### `04_build_network_graph.py`
- Builds directed graph from links/nodes.
- Detects disconnected components, cycles (where invalid), dangling links.
- Writes connectivity report and fix suggestions.

### `05_generate_subcatchments.py`
- Creates **homogeneous subcatchments** without GIS polygons.
- Strategy:
  - one subcatchment per inlet/manhole catchment assignment
  - distribute known area by rational-method metadata and plan intent
  - assign representative width/slope/imperviousness via config defaults
- Outputs subcatchments + subareas + infiltration tables.

### `06_build_swmm_inp.py`
- Assembles SWMM `.inp` sections:
  - `[JUNCTIONS]`, `[OUTFALLS]`, `[CONDUITS]`, `[XSECTIONS]`
  - `[SUBCATCHMENTS]`, `[SUBAREAS]`, `[INFILTRATION]`
  - `[RAINGAGES]`, `[TIMESERIES]`/`[OPTIONS]` placeholders as configured
- Emits traceability map from each `.inp` row to source records.

### `07_run_qa_checks.py`
- Performs deterministic checks (see section 5).
- Creates pass/fail report with severity tiers.
- Blocks final export when high-severity errors exist.

### `08_export_review_pack.py`
- Bundles `.inp`, QA reports, exceptions, and assumptions.
- Produces concise reviewer packet for engineer signoff.

---

## 4) Data Extraction Approach (Excel + PDFs)

## A. Excel workbook extraction
1. Enumerate workbook sheets and named ranges.
2. Apply a config-driven mapping (`config/field_mappings.yaml`) so parsing is stable even if columns move.
3. Preserve raw values and typed values (text/float/date).
4. Record data lineage:
   - source file
   - sheet name
   - row/column
   - parsing rule version

**Tip for browser-only users:** keep all extraction behavior in YAML mappings, not hardcoded edits.

## B. PDF plan extraction
1. Convert each PDF page to image (high DPI for annotation readability).
2. OCR full page text.
3. Use regex and spatial clustering to parse likely callouts (examples):
   - `MH[- ]?\d+`, `INLET[- ]?\d+`, `CB[- ]?\d+`
   - pipe size patterns (`12"`, `18 IN`, etc.)
   - invert/rim markers (`INV`, `RIM`, `IE`)
4. Build candidate node/link records with confidence score.
5. Save uncertain extractions to `qa/exceptions/pdf_parse_exceptions.csv` for manual review.

## C. Cross-source reconciliation
- Precedence rules:
  1. planning document explicit decisions
  2. Excel tabular values
  3. PDF annotation inference
  4. configured defaults
- Any fallback/default must be logged in `assumptions_log.csv`.

---

## 5) Verification Steps (RALPH-Style Loops)

Use short loops where each pass improves data quality before full model build.

### RALPH Loop Definition
- **R**ecord assumptions and extraction outputs
- **A**udit with scripted checks
- **L**ocalize issues to source records
- **P**atch mappings/rules (not ad-hoc data edits)
- **H**arden by rerunning from raw inputs

### Loop checks
1. **Schema checks**
   - required fields present
   - unit consistency
2. **ID integrity**
   - unique node/link IDs
   - valid upstream/downstream references
3. **Hydraulic plausibility**
   - non-negative lengths
   - slopes within expected ranges
   - invert logic (upstream >= downstream when applicable)
4. **Topologic integrity**
   - no orphan nodes (except outfalls)
   - no impossible loops unless intentionally modeled
5. **Catchment balance checks**
   - total assigned area vs known project area
   - runoff coefficient ranges by land use assumptions
6. **SWMM compile checks**
   - `.inp` section completeness
   - optional SWMM dry-run parse if engine is available

Each loop should generate:
- `qa/reports/loop_<n>_summary.md`
- `qa/exceptions/loop_<n>_exceptions.csv`
- updated `assumptions_log.csv`

---

## 6) Clear Next Actions for Build Mode (Browser-Only User)

1. **Create configuration first**
   - Populate `config/project_config.yaml` with units, default Manning’s n, infiltration method, and homogeneous subcatchment assumptions.

2. **Build extraction mappings**
   - Fill `config/field_mappings.yaml` for workbook sheets and expected PDF annotation patterns.

3. **Run scripts sequentially**
   - `00` → `08` in order; do not skip QA.

4. **Resolve exceptions after each RALPH loop**
   - Fix mapping rules or ID normalization rules, then rerun from raw inputs.

5. **Freeze a candidate model**
   - Export one versioned `.inp` only after all high-severity QA checks pass.

6. **Engineer review checkpoint**
   - Review assumptions/defaults for subcatchment generation before simulation scenarios are finalized.

---

## 7) Minimal Definition of Done

- Canonical `nodes.csv`, `links.csv`, `subcatchments.csv` generated from raw files.
- SWMM `.inp` created reproducibly from scripts/config only.
- QA report shows no unresolved high-severity issues.
- Assumptions and source lineage fully documented.
