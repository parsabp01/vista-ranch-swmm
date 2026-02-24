# Phase 1 Data QA/QC Summary (Pre-SWMM Runtime)

- QA Status: **blocked**
- Junctions: 174
- Links: 252
- Inflow records: 302
- Systems: 9
- Findings: HIGH=121, MEDIUM=6, LOW=1

## Key Integrity Metrics
- Orphan junctions: 2
- Missing link endpoints: 3
- Self-loop links: 0
- Cross-system edges: 0
- Unmapped inflows: 0
- Duplicate inflow source rows: 0
- HYDROLOGY Column D used as inlet ID count: 0

## Top Risks
- [HIGH] link_endpoint_node_missing | link 101-101.1 | Endpoint node reference not found in canonical node set. (source: data/processed/links.csv)
- [HIGH] link_endpoint_node_missing | link 101.1-107 | Endpoint node reference not found in canonical node set. (source: data/processed/links.csv)
- [HIGH] link_endpoint_node_missing | link 102-102.1 | Endpoint node reference not found in canonical node set. (source: data/processed/links.csv)
- [HIGH] link_endpoint_node_missing | link 102.1-102.3 | Endpoint node reference not found in canonical node set. (source: data/processed/links.csv)
- [HIGH] link_endpoint_node_missing | link 102.3-102.4 | Endpoint node reference not found in canonical node set. (source: data/processed/links.csv)
- [HIGH] link_endpoint_node_missing | link 102.4-103.3 | Endpoint node reference not found in canonical node set. (source: data/processed/links.csv)
- [HIGH] link_endpoint_node_missing | link 103-103.2 | Endpoint node reference not found in canonical node set. (source: data/processed/links.csv)
- [HIGH] link_endpoint_node_missing | link 103.1-103.2 | Endpoint node reference not found in canonical node set. (source: data/processed/links.csv)
- [HIGH] link_endpoint_node_missing | link 103.2-103.3 | Endpoint node reference not found in canonical node set. (source: data/processed/links.csv)
- [HIGH] link_endpoint_node_missing | link 103.3-103.4 | Endpoint node reference not found in canonical node set. (source: data/processed/links.csv)

## Phase 2 Handoff (Runtime/Simulation Focus)
- Verify the highest-risk MEDIUM findings in SWMM context (defaults concentration, disconnected components, duplicate inflow source-row semantics) before interpreting hydraulic performance results.
