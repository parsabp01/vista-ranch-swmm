# Topology Gap Analysis

## Symptom
- Prior runs showed `data/interim/links_raw.csv` with only 3 rows while build produced multiple conduits via synthetic chaining.

## Root cause
1. Generic keyword classification captured only sparse formula/reference rows from non-hydraulic sheets.
2. The true hydraulic link table in `HYDRAULICS` sheet was not parsed structurally.
3. Header detection logic looked for `STREAM JCT` and unit tokens in the same row, but workbook separates labels/units across adjacent rows.
4. Column indexing for upstream/downstream/diameter/length/slope was offset from actual workbook positions.

## Fix applied
- Added structured parser `_extract_hydraulic_links_from_sheet()`.
- Parser now detects hydraulic table by repeated `STREAM JCT` label row and extracts rows using known column positions:
  - downstream col2, upstream col3, diameter col7, Q col8, length col9, slope col12.
- Appends extracted links with explicit `extraction_method=hydraulics_structured_table` and source lineage.
- BUILD now uses extracted links topology by default (`topology_mode=extracted_links`) and only falls back to synthetic chain if required columns are absent.

## Expected impact
- Link extraction count increases from 3 to full hydraulic table rows.
- SWMM conduits are now derived from extracted topology rather than synthetic chaining.
