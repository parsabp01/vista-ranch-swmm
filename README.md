# vista-ranch-swmm

SWMM hydraulic model automation workspace driven by the source-of-truth planning document and executable pipeline scripts.

## Primary docs
- `prd.md`
- `docs/runbook.md`
- `data/raw/Hydraulic Model Planning.docx`

## Pipeline entrypoint
- `python scripts/run_pipeline.py --stage preflight`
- `python scripts/run_pipeline.py --stage extract`

Additional stages are available: `transform`, `build`, `qa`.
