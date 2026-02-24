# Phase 1 Triage Log (MEDIUM+LOW set and auto-resolutions)

## Auto-resolved this pass (excluded non-input)
1. MEDIUM | link_row_exclusion_candidate | link row_250 | system_id=n/a | source_dataset=links | source_tab=Reference | source_row=excel_row=8;raw_extracted_row_index=250 | explanation: non-link worksheet noise row in processed links denominator | proposed_disposition=exclude_non_input | status=resolved
2. MEDIUM | link_row_exclusion_candidate | link row_251 | system_id=n/a | source_dataset=links | source_tab=Formluas | source_row=excel_row=23;raw_extracted_row_index=251 | explanation: non-link worksheet noise row in processed links denominator | proposed_disposition=exclude_non_input | status=resolved
3. MEDIUM | link_row_exclusion_candidate | link row_252 | system_id=n/a | source_dataset=links | source_tab=VBA reference | source_row=excel_row=19;raw_extracted_row_index=252 | explanation: non-link worksheet noise row in processed links denominator | proposed_disposition=exclude_non_input | status=resolved

## Remaining findings
4. MEDIUM | inflow_duplicate_source_row | inflow source_row_duplicates | system_id=n/a | source_dataset=inflow | source_tab=HYDROLOGY | source_row=grouped_by_source_tab_and_source_row | explanation: duplicate tab+row keys exist in inflow map table and likely reflect mapping expansion behavior | proposed_disposition=fix_mapping | status=open
5. MEDIUM | inflow_source_row_multi_node | inflow source_row_multi_node | system_id=n/a | source_dataset=inflow | source_tab=HYDROLOGY | source_row=grouped_by_source_tab_and_source_row | explanation: same source key maps to multiple SWMM nodes in current inflow map table | proposed_disposition=fix_mapping | status=open
6. MEDIUM | inflow_unmapped | inflow unmapped | system_id=n/a | source_dataset=inflow | source_tab=HYDROLOGY | source_row=id_map_inflows_unmapped_rows | explanation: two inflow records remain unmapped in processed inflow table | proposed_disposition=manual_engineering_review | status=open
7. LOW | topology_dead_end_nodes | junction dead_end_summary | system_id=n/a | source_dataset=topology | source_tab=HYDRAULICS | source_row=derived_from_links_graph | explanation: degree-1 nodes exist and are often expected at starts/termini | proposed_disposition=accept_as_assumption | status=open
