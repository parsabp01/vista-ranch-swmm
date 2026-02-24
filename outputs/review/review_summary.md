# Review Summary

- Inflow records mapped: 304
- Inflow matched %: 99.34%
- Strict match %: 97.7%
- Normalized/suffix match %: 1.64%
- Low-confidence match count: 2
- Unresolved count: 2
- Inlet->junction coverage: 99.34%
- Inflow->SWMM node coverage: 99.34%
- BASIN boundary consistency: PASS

## HYDROLOGY schema lock validation
- Column B parsed as junction IDs
- Column C parsed as inlet IDs
- Column D parsed as inlet area (never inlet ID)
- Regression guard count (D looked numeric while C empty): 0

## Target row checks
- Row 265: inlet_ids=['I_99'], inlet_id_source_cols=['C']
- Row 330: inlet_ids=['I_117_1'], inlet_id_source_cols=['C']
- Row 431: inlet_ids=['I_137_1'], inlet_id_source_cols=['C']
- Row 499: inlet_ids=['I_165'], inlet_id_source_cols=['C']
- Inflow records mapped: 306
- Inflow->inlet coverage: 98.37%
- Inlet->junction coverage: 98.69%
- Inflow->SWMM node coverage: 98.69%
- BASIN boundary consistency: PASS

See outputs/logs/inflow_mapping_summary.json for detailed diagnostics.
