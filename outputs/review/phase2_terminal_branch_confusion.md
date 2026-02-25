# Terminal Branch Confusion + Excel Coverage Reconciliation

- HYDROLOGY BASIN markers: 9
- Modeled outfalls: 4
- Missing basin-terminal outfalls: 5

## Basin terminal reconciliation
- Basin 1: terminal J_77 -> expected O_77 -> modeled (BASIN row 232)
- Basin 2: terminal J_134 -> expected O_134 -> missing_outfall (BASIN row 486)
- Basin 3: terminal J_134_1 -> expected O_134_1 -> modeled (BASIN row 491)
- Basin 4: terminal J_135 -> expected O_135 -> modeled (BASIN row 496)
- Basin 5: terminal J_156 -> expected O_156 -> modeled (BASIN row 582)
- Basin 6: terminal J_184 -> expected O_184 -> missing_outfall (BASIN row 669)
- Basin 7: terminal J_185 -> expected O_185 -> missing_outfall (BASIN row 674)
- Basin 8: terminal J_186 -> expected O_186 -> missing_outfall (BASIN row 679)
- Basin 9: terminal J_187 -> expected O_187 -> missing_outfall (BASIN row 684)

## Excel structures not yet accounted in model

- Missing junction IDs: 95
- Missing inlet IDs: 113

### Missing junction IDs (full list)
36, 55, 65.2, 65.25, 65.3, 68, 68.05, 71.2, 71.3, 71.4, 84, 84.1, 89, 91, 95.1, 96.1, 97, 97.1, 99, 101, 101.1, 102, 102.1, 102.2, 102.3, 102.4, 103, 103.1, 103.2, 103.3, 103.4, 103.5, 104, 105, 105.1, 105.2, 105.3, 105.4, 106, 107, 108, 108.1, 108.2, 108.3, 108.4, 109, 109.1, 109.2, 110, 110.1, 110.2, 110.3, 113.2, 114, 114.1, 114.2, 116, 134, 136, 136.1, 136.2, 136.3, 136.4, 136.5, 136.6, 136.7, 136.8, 136.9, 137, 137.1, 137.2, 157, 158, 159, 159.1, 159.2, 163, 170, 171, 172, 173, 174, 175, 176, 177, 180, 180.2, 182, 182.1, 182.2, 183, 184, 185, 186, 187

### Missing inlet IDs (full list)
1, 39, 56, 56.1, 75.1, 75.2, 75.25, 75.3, 75.4, 77, 77.1, 77.2, 95, 96, 102, 106, 106.1, 112, 115.1, 116, 116.01, 116.02, 116.2, 116.3, 116.4, 117, 117.1, 117.2, 118, 118.1, 119, 119.1, 119.2, 120, 121, 122, 122.1, 122.2, 123, 123.1, 123.2, 123.3, 123.4, 124, 124.1, 124.2, 124.3, 124.4, 124.5, 124.6, 125, 126, 127, 127.1, 127.2, 128, 128.1, 128.2, 129, 129.1, 129.2, 130, 130.1, 130.2, 130.3, 130.4, 130.5, 137, 137.1, 138, 139, 140, 141, 149, 156, 162, 164, 165, 165.1, 165.2, 165.3, 165.4, 165.5, 165.6, 165.7, 165.8, 165.9, 166, 166.1, 166.2, 194, 195, 196, 197, 204, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 231, 232, 233, 234, 235

## Notes
- This report compares raw HYDROLOGY presence against current strict hydraulic baseline build inclusion.
- Structures are omitted when required geometry fields are missing under current no-fabrication rules.
