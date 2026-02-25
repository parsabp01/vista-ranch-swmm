# Phase 2A Inflow Preflight

## inflow_duplicate_source_row
- Affected count: 302
- Classification: **runtime_nonblocking_bookkeeping**
- Runtime impact: duplicate INFLOWS nodes in model: none
- Example entities: HYDROLOGY|330; HYDROLOGY|265; HYDROLOGY|499; HYDROLOGY|431; HYDROLOGY|458; HYDROLOGY|456; HYDROLOGY|455; HYDROLOGY|452; HYDROLOGY|450; HYDROLOGY|446
- Notes: bookkeeping duplicates exist in id_map_inflows; model-level dedupe mostly achieved but 3 duplicate target nodes remain.

## inflow_source_row_multi_node
- Affected count: 302
- Classification: **runtime_nonblocking_bookkeeping**
- Runtime impact: appears to be expansion artifact in mapping table; model consumes node-level inflow records
- Example entities: Formluas|33; HYDROLOGY|101; HYDROLOGY|104; HYDROLOGY|108; HYDROLOGY|109; HYDROLOGY|113; HYDROLOGY|115; HYDROLOGY|116; HYDROLOGY|12; HYDROLOGY|120
- Notes: source_key to node mapping should be normalized for traceability, but not a runtime parser blocker.

## inflow_unmapped
- Affected count: 2
- Classification: **accepted_baseline_assumption**
- Runtime impact: omitted from model inflows; omitted |q|=0.0000 cfs
- Example entities: HYDROLOGY|687; HYDROLOGY|688
- Notes: rows align with trailing non-input records after final BASIN in previous parser notes.
