# SWMM Automation Runbook (Codex Browser Only)

This workflow runs entirely in this repository via Codex web execution.

## What happens when I run stages
- **preflight**: verifies source files exist and writes `outputs/logs/intake_manifest.json`.
- **extract**: discovers workbook sheets/headers, classifies records into nodes/links/rational tables, records PDF metadata.
- **transform**: normalizes raw extracts to canonical processed tables and derives subcatchment defaults.
- **build**: writes a baseline SWMM `.inp` file in `models/`.
- **qa**: runs basic QA checks and writes findings/reports under `outputs/qa/`.

## Commands used in this web environment
1. `python scripts/run_pipeline.py --stage preflight`
2. `python scripts/run_pipeline.py --stage extract`

Optional full flow:
- `python scripts/run_pipeline.py --stage transform`
- `python scripts/run_pipeline.py --stage build`
- `python scripts/run_pipeline.py --stage qa`

## Interpretation
- `status: ready` means required inputs are present.
- `status: blocked` means one or more required inputs are missing or unreadable.

All outputs are written to repo folders so you can inspect artifacts directly from GitHub/Codex.
