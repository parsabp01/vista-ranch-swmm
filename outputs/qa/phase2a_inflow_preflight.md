# Phase 2A Inflow Preflight

## inflow_duplicate_source_row
- Affected count: 302
- Classification: **runtime_nonblocking_bookkeeping**
- Runtime impact: duplicate INFLOWS nodes in model: none
- Example entities: HYDROLOGY|330; HYDROLOGY|265; HYDROLOGY|499; HYDROLOGY|431; HYDROLOGY|458; HYDROLOGY|456; HYDROLOGY|455; HYDROLOGY|452; HYDROLOGY|450; HYDROLOGY|446
- Notes: bookkeeping duplicates exist in id_map_inflows; model-level dedupe mostly achieved but 3 duplicate target nodes remain.
- Runtime impact: Creates duplicated [INFLOWS] entries for nodes: J086, J101, J114
- Example entities: Formluas|33; HYDROLOGY|101; HYDROLOGY|104; HYDROLOGY|108; HYDROLOGY|109; HYDROLOGY|113; HYDROLOGY|115; HYDROLOGY|116; HYDROLOGY|12; HYDROLOGY|120
- Notes: Most duplicates map to same node and many are zero-baseline rows; however duplicate [INFLOWS] node lines exist and should be consolidated.

## inflow_source_row_multi_node
- Affected count: 302
- Classification: **runtime_nonblocking_bookkeeping**
- Runtime impact: appears to be expansion artifact in mapping table; model consumes node-level inflow records
- Example entities: Formluas|33; HYDROLOGY|101; HYDROLOGY|104; HYDROLOGY|108; HYDROLOGY|109; HYDROLOGY|113; HYDROLOGY|115; HYDROLOGY|116; HYDROLOGY|12; HYDROLOGY|120
- Notes: source_key to node mapping should be normalized for traceability, but not a runtime parser blocker.
- Runtime impact: No direct 1:1 source-row provenance in model.inp; appears to be mapping-table expansion artifact.
- Example entities: Formluas|33->0.96|J001; HYDROLOGY|101->0.95|J035; HYDROLOGY|104->0.95|J036; HYDROLOGY|108->0.95|J038; HYDROLOGY|109->0.95|J038; HYDROLOGY|113->0.95|J040; HYDROLOGY|115->0.95|J041; HYDROLOGY|116->0.95|J041; HYDROLOGY|12->0.95|J001; HYDROLOGY|120->0.95|J043
- Notes: Model [INFLOWS] is node-based; ambiguity exists in bookkeeping table, not in INP parser execution path.

## inflow_unmapped
- Affected count: 2
- Classification: **accepted_baseline_assumption**
- Runtime impact: omitted from model inflows; omitted |q|=0.0000 cfs
- Example entities: HYDROLOGY|687; HYDROLOGY|688
- Notes: rows align with trailing non-input records after final BASIN in previous parser notes.
- Runtime impact: Omitted from [INFLOWS]; total omitted |q|=0.0000 cfs.
- Example entities: HYDROLOGY|687|raw_inlet=nan|raw_junction=nan|q=nan; HYDROLOGY|688|raw_inlet=nan|raw_junction=nan|q=nan
- Notes: Both unmapped records have null q_cfs and occur after final BASIN/trailing non-input rows in prior parser notes.
