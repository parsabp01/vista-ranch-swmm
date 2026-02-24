# Phase 2 Runtime QA/QC Summary

- QA Status: **needs_review**
- Findings: HIGH=0, MEDIUM=3, LOW=2
- Required sections present: 12/12
- Model counts: junctions=174, conduits=249, inflows=137

## Key Runtime Notes
- Phase 2A inflow preflight completed; see outputs/qa/phase2a_inflow_preflight.md.
- Static checks only (no SWMM engine execution in Codex environment).

## Top Findings
- [MEDIUM] build_summary_count_mismatch | build_summary inflow_count | build_summary inflow_count=134 but model section count=137.
- [MEDIUM] default_heavy_junction_maxdepth | junction JUNCTIONS | MaxDepth appears default-heavy: 100.00% at value 8.0.
- [MEDIUM] duplicate_ids_in_section | model_section [INFLOWS] | Duplicate IDs present (3): J086, J101, J114.
- [LOW] dead_end_junctions | junction dead_end_summary | 50 degree-1 junctions found.
- [LOW] inflow_zero_baseline_concentration | inflow INFLOWS | 94.16% of inflow baseline values are zero.
