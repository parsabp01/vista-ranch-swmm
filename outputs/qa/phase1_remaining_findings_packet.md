# Phase 1 Remaining Findings Packet (MEDIUM + LOW)

- [MEDIUM] inflow_duplicate_source_row | inflow source_row_duplicates | disposition=fix_mapping | 302 duplicate inflow source rows. | source_dataset=inflow tab=HYDROLOGY row=grouped_by_source_tab_and_source_row
- [MEDIUM] inflow_source_row_multi_node | inflow source_row_multi_node | disposition=fix_mapping | 302 source rows map to multiple SWMM nodes. | source_dataset=inflow tab=HYDROLOGY row=grouped_by_source_tab_and_source_row
- [MEDIUM] inflow_unmapped | inflow unmapped | disposition=manual_engineering_review | 2 inflow records missing node mapping. | source_dataset=inflow tab=HYDROLOGY row=id_map_inflows_unmapped_rows
- [LOW] topology_dead_end_nodes | junction dead_end_summary | disposition=accept_as_assumption | 83 degree-1 nodes detected. | source_dataset=topology tab=HYDRAULICS row=derived_from_links_graph
- [MEDIUM] link_row_exclusion_candidate | link row_250 | Processed link row appears to be non-link worksheet noise and should be excluded from link QA denominator. | source_dataset=links tab=Reference row=excel_row=8;raw_extracted_row_index=250
- [MEDIUM] link_row_exclusion_candidate | link row_251 | Processed link row appears to be non-link worksheet noise and should be excluded from link QA denominator. | source_dataset=links tab=Formluas row=excel_row=23;raw_extracted_row_index=251
- [MEDIUM] link_row_exclusion_candidate | link row_252 | Processed link row appears to be non-link worksheet noise and should be excluded from link QA denominator. | source_dataset=links tab=VBA reference row=excel_row=19;raw_extracted_row_index=252
- [MEDIUM] inflow_duplicate_source_row | inflow source_row_duplicates | 302 duplicate inflow source rows. | source_dataset=inflow tab=HYDROLOGY row=grouped_by_source_tab_and_source_row
- [MEDIUM] inflow_source_row_multi_node | inflow source_row_multi_node | 302 source rows map to multiple SWMM nodes. | source_dataset=inflow tab=HYDROLOGY row=grouped_by_source_tab_and_source_row
- [MEDIUM] inflow_unmapped | inflow unmapped | 2 inflow records missing node mapping. | source_dataset=inflow tab=HYDROLOGY row=id_map_inflows_unmapped_rows
- [LOW] topology_dead_end_nodes | junction dead_end_summary | 83 degree-1 nodes detected. | source_dataset=topology tab=HYDRAULICS row=derived_from_links_graph
