# Phase 1 Data QA/QC Summary (Pre-SWMM Runtime)

- QA Status: **needs_review**
- Junctions: 248
- Links: 252
- Inflow records: 1215
- Systems: 325
- Findings: HIGH=0, MEDIUM=3, LOW=1
- Findings: HIGH=0, MEDIUM=6, LOW=1

## Key Integrity Metrics
- Orphan junctions: 0
- Missing link endpoints: 0
- Excluded non-data link rows: 3
- Self-loop links: 0
- Cross-system edges: 0
- Unmapped inflows: 2
- Duplicate inflow source rows: 302
- HYDROLOGY Column D used as inlet ID count: 0

## Top Risks
- [MEDIUM] link_row_exclusion_candidate | link row_250 | Processed link row appears to be non-link worksheet noise and should be excluded from link QA denominator. (dataset: links, tab: Reference, row: excel_row=8;raw_extracted_row_index=250)
- [MEDIUM] link_row_exclusion_candidate | link row_251 | Processed link row appears to be non-link worksheet noise and should be excluded from link QA denominator. (dataset: links, tab: Formluas, row: excel_row=23;raw_extracted_row_index=251)
- [MEDIUM] link_row_exclusion_candidate | link row_252 | Processed link row appears to be non-link worksheet noise and should be excluded from link QA denominator. (dataset: links, tab: VBA reference, row: excel_row=19;raw_extracted_row_index=252)
- [MEDIUM] inflow_duplicate_source_row | inflow source_row_duplicates | 302 duplicate inflow source rows. (dataset: inflow, tab: HYDROLOGY, row: grouped_by_source_tab_and_source_row)
- [MEDIUM] inflow_source_row_multi_node | inflow source_row_multi_node | 302 source rows map to multiple SWMM nodes. (dataset: inflow, tab: HYDROLOGY, row: grouped_by_source_tab_and_source_row)
- [MEDIUM] inflow_unmapped | inflow unmapped | 2 inflow records missing node mapping. (dataset: inflow, tab: HYDROLOGY, row: id_map_inflows_unmapped_rows)
- [LOW] topology_dead_end_nodes | junction dead_end_summary | 83 degree-1 nodes detected. (dataset: topology, tab: HYDRAULICS, row: derived_from_links_graph)

## Phase 2 Handoff (Runtime/Simulation Focus)
- Proceed to Phase 2 once MEDIUM findings are reviewed and accepted.
