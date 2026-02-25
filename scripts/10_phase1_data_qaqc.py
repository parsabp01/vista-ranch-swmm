#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from id_utils import typed_canonical_id

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data/processed'
INTERIM = ROOT / 'data/interim'
OUT_QA = ROOT / 'outputs/qa'
OUT_REV = ROOT / 'outputs/review'
OUT_LOG = ROOT / 'outputs/logs'
for d in (OUT_QA, OUT_REV, OUT_LOG):
    d.mkdir(parents=True, exist_ok=True)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()


nodes = read_csv(PROC / 'nodes.csv')
links = read_csv(PROC / 'links.csv')
inflows = read_csv(PROC / 'id_map_inflows.csv')
topo = read_csv(PROC / 'junction_topology_map.csv')
def safe_json(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text(encoding='utf-8')
    try:
        return json.loads(text)
    except Exception:
        return {}

inf_summary = safe_json(OUT_LOG / 'inflow_mapping_summary.json')

findings: list[dict] = []


def add_finding(
    check_name: str,
    severity: str,
    entity_type: str,
    entity_id: str,
    system_id,
    status: str,
    message: str,
    source_dataset: str,
    source_tab,
    source_row,
    processed_row,
    source_columns_used: str,
    raw_upstream_value,
    raw_downstream_value,
    raw_link_id,
    exclusion_candidate: bool,
    exclusion_reason,
    recommended_action: str,
):
    findings.append({
        'check_name': check_name,
        'severity': severity,
        'entity_type': entity_type,
        'entity_id': entity_id,
        'system_id': system_id,
        'status': status,
        'message': message,
        'source_dataset': source_dataset,
        'source_tab': source_tab,
        'source_row': source_row,
        'processed_row': processed_row,
        'source_columns_used': source_columns_used,
        'raw_upstream_value': raw_upstream_value,
        'raw_downstream_value': raw_downstream_value,
        'raw_link_id': raw_link_id,
        'exclusion_candidate': bool(exclusion_candidate),
        'exclusion_reason': exclusion_reason,
        'recommended_action': recommended_action,
    })


# Junction and link-derived node universe
node_set = set()
if 'jct' in nodes.columns:
    node_set |= {typed_canonical_id(v, 'junction') for v in nodes['jct']}
for col in ('upstream_node', 'downstream_node'):
    if col in links.columns:
        node_set |= {typed_canonical_id(v, 'junction') for v in links[col]}
node_set = {n for n in node_set if n}

# Build adjacency from complete links only
adj: dict[str, set[str]] = defaultdict(set)
missing_endpoint_count = 0
self_loop_count = 0
excluded_noise_rows = 0
link_qaqc = []

for idx, r in links.reset_index(drop=True).iterrows():
    processed_row = idx + 1
    us_raw = r.get('upstream_node')
    ds_raw = r.get('downstream_node')
    pipe_id = r.get('pipe_id')
    src_row = r.get('source_row')
    src_tab = r.get('source_sheet')

    us = typed_canonical_id(us_raw, 'junction')
    ds = typed_canonical_id(ds_raw, 'junction')

    is_noise_candidate = (
        (pd.isna(pipe_id) or str(pipe_id).strip() == '')
        and str(src_tab) != 'HYDRAULICS'
        and (pd.isna(us_raw) or str(us_raw).strip() == '')
        and (pd.isna(ds_raw) or str(ds_raw).strip() == '')
    )

    if is_noise_candidate:
        excluded_noise_rows += 1
        add_finding(
            'link_row_exclusion_candidate',
            'MEDIUM',
            'link',
            f'row_{processed_row}',
            None,
            'warn',
            'Processed link row appears to be non-link worksheet noise and should be excluded from link QA denominator.',
            'links',
            src_tab,
            f"excel_row={src_row};raw_extracted_row_index={processed_row}",
            processed_row,
            'pipe_id,upstream_node,downstream_node,source_sheet,source_row',
            us_raw,
            ds_raw,
            pipe_id,
            True,
            'non_hydraulics_blank_link_row',
            'Exclude this row from operational link QA denominator or hard-filter non-HYDRAULICS link rows upstream.',
        )
        continue

    if not us or not ds:
        missing_endpoint_count += 1
        add_finding(
            'link_missing_endpoint',
            'HIGH',
            'link',
            f'row_{processed_row}',
            None,
            'fail',
            'Missing upstream/downstream endpoint.',
            'links',
            src_tab,
            f"excel_row={src_row};raw_extracted_row_index={processed_row}",
            processed_row,
            'upstream_node,downstream_node,pipe_id,source_sheet,source_row',
            us_raw,
            ds_raw,
            pipe_id,
            False,
            None,
            'Populate both upstream and downstream node IDs or formally exclude this row as non-data.',
        )
        continue

    if us == ds:
        self_loop_count += 1
        add_finding(
            'link_self_loop', 'HIGH', 'link', str(pipe_id) if pd.notna(pipe_id) else f'row_{processed_row}', None, 'fail',
            'Upstream and downstream nodes are identical.', 'links', src_tab,
            f"excel_row={src_row};raw_extracted_row_index={processed_row}", processed_row,
            'upstream_node,downstream_node,pipe_id', us_raw, ds_raw, pipe_id, False, None,
            'Correct upstream/downstream assignment.'
        )
    adj[us].add(ds)
    adj[ds].add(us)

# Junction checks
orphan_nodes = sorted([n for n in sorted(node_set) if len(adj.get(n, set())) == 0])
junction_qaqc = [
    {'check_name': 'junction_orphan_count', 'status': 'warn' if orphan_nodes else 'pass', 'metric': 'count', 'value': len(orphan_nodes), 'message': 'Junctions with no connected links.'},
]
for node in orphan_nodes:
    add_finding(
        'junction_orphan', 'MEDIUM', 'junction', node, None, 'warn', 'Junction has no connected links.',
        'links', 'HYDRAULICS', 'derived_from_links_graph', None, 'upstream_node,downstream_node', None, None, None, False, None,
        'Confirm intentional isolation or add link connectivity.'
    )

# Topology checks
components = []
seen = set()
for n in sorted(node_set):
    if n in seen:
        continue
    q = deque([n])
    seen.add(n)
    comp = []
    while q:
        cur = q.popleft()
        comp.append(cur)
        for nxt in adj.get(cur, set()):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    components.append(comp)

if len(components) > 1:
    add_finding(
        'topology_isolated_subnetworks', 'MEDIUM', 'system', 'graph_components', None, 'warn',
        f'Network has {len(components)} connected components.', 'topology', 'HYDRAULICS', 'derived_from_links_graph', None,
        'upstream_node,downstream_node', None, None, None, False, None,
        'Validate disconnected components against intended basin segmentation.'
    )

dead_end_count = sum(1 for n in node_set if len(adj.get(n, set())) == 1)
if dead_end_count > 0:
    add_finding(
        'topology_dead_end_nodes', 'LOW', 'junction', 'dead_end_summary', None, 'warn',
        f'{dead_end_count} degree-1 nodes detected.', 'topology', 'HYDRAULICS', 'derived_from_links_graph', None,
        'upstream_node,downstream_node', None, None, None, False, None,
        'Review whether degree-1 nodes are expected starts/termini.'
    )

cross_system_edge_count = int(topo['crosses_basin_boundary'].fillna(False).sum()) if 'crosses_basin_boundary' in topo.columns else 0
if cross_system_edge_count > 0:
    add_finding(
        'topology_cross_system_leakage', 'HIGH', 'system', 'basin_boundary', None, 'fail',
        f'{cross_system_edge_count} edges cross BASIN boundaries.', 'topology', 'HYDRAULICS', 'derived_from_junction_topology_map', None,
        'upstream_system_id,downstream_system_id,crosses_basin_boundary', None, None, None, False, None,
        'Repair system segmentation/link direction mapping.'
    )

topology_qaqc = [
    {'check_name': 'connected_component_count', 'status': 'warn' if len(components) > 1 else 'pass', 'metric': 'count', 'value': len(components), 'message': 'Connected components in graph.'},
    {'check_name': 'dead_end_count', 'status': 'warn' if dead_end_count else 'pass', 'metric': 'count', 'value': dead_end_count, 'message': 'Degree-1 nodes.'},
    {'check_name': 'cross_system_edge_count', 'status': 'fail' if cross_system_edge_count else 'pass', 'metric': 'count', 'value': cross_system_edge_count, 'message': 'Edges crossing BASIN boundaries.'},
]

# Inflow checks
unmapped_inflow_count = int(inflows['canonical_swmm_node_id'].isna().sum()) if 'canonical_swmm_node_id' in inflows.columns else 0
dup_source_row = 0
source_multi_node = 0
if {'source_tab', 'source_row'}.issubset(inflows.columns):
    key = inflows[['source_tab', 'source_row']].astype(str).agg('|'.join, axis=1)
    dup_source_row = int((key.value_counts() > 1).sum())
    source_multi_node = int((inflows.groupby(['source_tab', 'source_row'])['canonical_swmm_node_id'].nunique(dropna=True) > 1).sum())

column_d_guard = int(inf_summary.get('regression_validation', {}).get('column_d_used_as_inlet_id_count', 0))

if dup_source_row > 0:
    add_finding(
        'inflow_duplicate_source_row', 'MEDIUM', 'inflow', 'source_row_duplicates', None, 'warn',
        f'{dup_source_row} duplicate inflow source rows.', 'inflow', 'HYDROLOGY', 'grouped_by_source_tab_and_source_row', None,
        'source_tab,source_row,canonical_swmm_node_id', None, None, None, False, None,
        'Confirm duplicate source rows are expected (multi-record inflows) or deduplicate upstream.'
    )
if source_multi_node > 0:
    add_finding(
        'inflow_source_row_multi_node', 'MEDIUM', 'inflow', 'source_row_multi_node', None, 'fail',
        f'{source_multi_node} source rows map to multiple SWMM nodes.', 'inflow', 'HYDROLOGY', 'grouped_by_source_tab_and_source_row', None,
        'source_tab,source_row,canonical_swmm_node_id', None, None, None, False, None,
        'Enforce 1:1 mapping per source row.'
    )
if unmapped_inflow_count > 0:
    add_finding(
        'inflow_unmapped', 'MEDIUM', 'inflow', 'unmapped', None, 'fail',
        f'{unmapped_inflow_count} inflow records missing node mapping.', 'inflow', 'HYDROLOGY', 'id_map_inflows_unmapped_rows', None,
        'canonical_swmm_node_id', None, None, None, False, None,
        'Resolve receiving node mapping.'
    )
if column_d_guard != 0:
    add_finding(
        'inflow_column_d_regression_guard', 'HIGH', 'inflow', 'hydrology_column_d', None, 'fail',
        f'Column D used as inlet ID count={column_d_guard}.', 'inflow', 'HYDROLOGY', 'inflow_mapping_summary.regression_validation', None,
        'column_d_used_as_inlet_id_count', None, None, None, False, None,
        'Fix HYDROLOGY parser to keep inlet IDs in Column C only.'
    )

inflow_qaqc = [
    {'check_name': 'unmapped_inflow_count', 'status': 'pass' if unmapped_inflow_count == 0 else 'warn', 'metric': 'count', 'value': unmapped_inflow_count, 'message': 'Inflows missing receiving node.'},
    {'check_name': 'duplicate_inflow_source_row_count', 'status': 'warn' if dup_source_row > 0 else 'pass', 'metric': 'count', 'value': dup_source_row, 'message': 'Duplicate inflow source rows.'},
    {'check_name': 'inflow_source_row_multi_node_count', 'status': 'pass' if source_multi_node == 0 else 'warn', 'metric': 'count', 'value': source_multi_node, 'message': 'Source rows mapped to >1 node.'},
    {'check_name': 'column_d_used_as_inlet_id_count', 'status': 'pass' if column_d_guard == 0 else 'fail', 'metric': 'count', 'value': column_d_guard, 'message': 'HYDROLOGY Column D regression guard.'},
]

link_qaqc.extend([
    {'check_name': 'missing_link_endpoint_count', 'status': 'fail' if missing_endpoint_count else 'pass', 'metric': 'count', 'value': missing_endpoint_count, 'message': 'Links missing endpoint references (post-exclusion).'},
    {'check_name': 'excluded_link_noise_row_count', 'status': 'warn' if excluded_noise_rows else 'pass', 'metric': 'count', 'value': excluded_noise_rows, 'message': 'Rows excluded as non-link worksheet noise.'},
    {'check_name': 'self_loop_link_count', 'status': 'fail' if self_loop_count else 'pass', 'metric': 'count', 'value': self_loop_count, 'message': 'Self-loop links.'},
])

# Save checks and findings
pd.DataFrame(junction_qaqc).to_csv(OUT_QA / 'junction_qaqc_checks.csv', index=False)
pd.DataFrame(link_qaqc).to_csv(OUT_QA / 'link_qaqc_checks.csv', index=False)
pd.DataFrame(topology_qaqc).to_csv(OUT_QA / 'topology_qaqc_checks.csv', index=False)
pd.DataFrame(inflow_qaqc).to_csv(OUT_QA / 'inflow_qaqc_checks.csv', index=False)

findings_df = pd.DataFrame(findings)
if findings_df.empty:
    findings_df = pd.DataFrame(columns=[
        'check_name', 'severity', 'entity_type', 'entity_id', 'system_id', 'status', 'message', 'source_dataset', 'source_tab', 'source_row',
        'processed_row', 'source_columns_used', 'raw_upstream_value', 'raw_downstream_value', 'raw_link_id', 'exclusion_candidate', 'exclusion_reason', 'recommended_action'
    ])

sev_rank = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
findings_df['sev_rank'] = findings_df['severity'].map(sev_rank).fillna(9)
findings_df = findings_df.sort_values(['sev_rank', 'system_id', 'entity_id', 'check_name'], na_position='last').drop(columns=['sev_rank'])
findings_df.to_csv(OUT_QA / 'phase1_data_qaqc_findings.csv', index=False)

priority = findings_df.copy()
priority.to_csv(OUT_REV / 'phase1_priority_review_list.csv', index=False)

# Remaining MEDIUM/LOW review packet
remaining = findings_df[findings_df['severity'].isin(['MEDIUM', 'LOW'])].copy()
remaining['plain_english_interpretation'] = remaining['message'].fillna('').astype(str)

def propose_disposition(check_name: str) -> str:
    if check_name in {'inflow_duplicate_source_row', 'inflow_source_row_multi_node'}:
        return 'fix_mapping'
    if check_name in {'inflow_unmapped'}:
        return 'manual_engineering_review'
    if check_name in {'topology_dead_end_nodes'}:
        return 'accept_as_assumption'
    return 'manual_engineering_review'

remaining['recommended_disposition'] = remaining['check_name'].map(propose_disposition)
remaining = remaining.sort_values(['severity', 'system_id', 'entity_id'], key=lambda s: s.map({'MEDIUM': 0, 'LOW': 1}) if s.name == 'severity' else s, na_position='last')
remaining.to_csv(OUT_QA / 'phase1_remaining_findings_packet.csv', index=False)

packet_md = ['# Phase 1 Remaining Findings Packet (MEDIUM + LOW)', '']
if remaining.empty:
    packet_md.append('- No MEDIUM or LOW findings remain.')
else:
    for _, r in remaining.iterrows():
        packet_md.append(f"- [{r['severity']}] {r['check_name']} | {r['entity_type']} {r['entity_id']} | disposition={r['recommended_disposition']} | {r['plain_english_interpretation']} | source_dataset={r['source_dataset']} tab={r['source_tab']} row={r['source_row']}")
        packet_md.append(f"- [{r['severity']}] {r['check_name']} | {r['entity_type']} {r['entity_id']} | {r['plain_english_interpretation']} | source_dataset={r['source_dataset']} tab={r['source_tab']} row={r['source_row']}")
(OUT_QA / 'phase1_remaining_findings_packet.md').write_text('\n'.join(packet_md) + '\n', encoding='utf-8')

high_count = int((findings_df['severity'] == 'HIGH').sum())
medium_count = int((findings_df['severity'] == 'MEDIUM').sum())
low_count = int((findings_df['severity'] == 'LOW').sum())

metrics = {
    'junction_count': int(len(node_set)),
    'link_count': int(len(links)),
    'inflow_record_count': int(len(inflows)),
    'system_count': int(inflows['system_id'].nunique(dropna=True)) if 'system_id' in inflows.columns else 0,
    'high_count': high_count,
    'medium_count': medium_count,
    'low_count': low_count,
    'orphan_junction_count': int(len(orphan_nodes)),
    'missing_link_endpoint_count': int(missing_endpoint_count),
    'self_loop_link_count': int(self_loop_count),
    'excluded_link_noise_row_count': int(excluded_noise_rows),
    'cross_system_edge_count': int(cross_system_edge_count),
    'unmapped_inflow_count': int(unmapped_inflow_count),
    'duplicate_inflow_source_row_count': int(dup_source_row),
    'column_d_used_as_inlet_id_count': int(column_d_guard),
    'qa_status': 'blocked' if high_count > 0 else ('needs_review' if medium_count > 0 else 'ready_for_phase2'),
}
(OUT_QA / 'phase1_data_qaqc_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')

summary = [
    '# Phase 1 Data QA/QC Summary (Pre-SWMM Runtime)', '',
    f"- QA Status: **{metrics['qa_status']}**",
    f"- Junctions: {metrics['junction_count']}",
    f"- Links: {metrics['link_count']}",
    f"- Inflow records: {metrics['inflow_record_count']}",
    f"- Systems: {metrics['system_count']}",
    f"- Findings: HIGH={high_count}, MEDIUM={medium_count}, LOW={low_count}", '',
    '## Key Integrity Metrics',
    f"- Orphan junctions: {metrics['orphan_junction_count']}",
    f"- Missing link endpoints: {metrics['missing_link_endpoint_count']}",
    f"- Excluded non-data link rows: {metrics['excluded_link_noise_row_count']}",
    f"- Self-loop links: {metrics['self_loop_link_count']}",
    f"- Cross-system edges: {metrics['cross_system_edge_count']}",
    f"- Unmapped inflows: {metrics['unmapped_inflow_count']}",
    f"- Duplicate inflow source rows: {metrics['duplicate_inflow_source_row_count']}",
    f"- HYDROLOGY Column D used as inlet ID count: {metrics['column_d_used_as_inlet_id_count']}", '',
    '## Top Risks',
]
for _, r in findings_df.head(10).iterrows():
    summary.append(f"- [{r['severity']}] {r['check_name']} | {r['entity_type']} {r['entity_id']} | {r['message']} (dataset: {r['source_dataset']}, tab: {r['source_tab']}, row: {r['source_row']})")
summary.append('')
summary.append('## Phase 2 Handoff (Runtime/Simulation Focus)')
summary.append('- Proceed to Phase 2 once MEDIUM findings are reviewed and accepted.')
(OUT_QA / 'phase1_data_qaqc_summary.md').write_text('\n'.join(summary) + '\n', encoding='utf-8')

# pipeline state update
pipeline_state = {
    'generated_at_utc': now_utc(),
    'current_stage': 'phase1_data_qaqc',
    'status': 'ready' if metrics['qa_status'] != 'blocked' else 'blocked',
    'key_metrics': metrics,
    'blockers': [] if high_count == 0 else [f'{high_count} HIGH findings remain'],
    'next_action': 'phase2_runtime_qa' if high_count == 0 else 'resolve remaining HIGH blockers',
}
(OUT_LOG / 'pipeline_state.json').write_text(json.dumps(pipeline_state, indent=2), encoding='utf-8')

print(json.dumps(metrics, indent=2))
