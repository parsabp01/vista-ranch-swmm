#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, defaultdict, deque
from pathlib import Path

import pandas as pd

from id_utils import norm_numeric_id, swmm_junction_id, typed_canonical_id

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data/processed'
OUT_QA = ROOT / 'outputs/qa'
OUT_REV = ROOT / 'outputs/review'
OUT_LOG = ROOT / 'outputs/logs'
for d in (OUT_QA, OUT_REV, OUT_LOG):
    d.mkdir(parents=True, exist_ok=True)


nodes = pd.read_csv(PROC / 'nodes.csv') if (PROC / 'nodes.csv').exists() else pd.DataFrame()
links = pd.read_csv(PROC / 'links.csv') if (PROC / 'links.csv').exists() else pd.DataFrame()
inflows = pd.read_csv(PROC / 'id_map_inflows.csv') if (PROC / 'id_map_inflows.csv').exists() else pd.DataFrame()
bridge = pd.read_csv(PROC / 'inlet_to_junction_map.csv') if (PROC / 'inlet_to_junction_map.csv').exists() else pd.DataFrame()
topo = pd.read_csv(PROC / 'junction_topology_map.csv') if (PROC / 'junction_topology_map.csv').exists() else pd.DataFrame()
jmap = pd.read_csv(PROC / 'id_map_junctions.csv') if (PROC / 'id_map_junctions.csv').exists() else pd.DataFrame()
inf_summary = json.loads((OUT_LOG / 'inflow_mapping_summary.json').read_text()) if (OUT_LOG / 'inflow_mapping_summary.json').exists() else {}

findings: list[dict] = []

def add_finding(check_name, severity, entity_type, entity_id, system_id, status, message, source_file, source_row, action):
    findings.append({'check_name': check_name, 'severity': severity, 'entity_type': entity_type, 'entity_id': entity_id, 'system_id': system_id, 'status': status, 'message': message, 'source_file': source_file, 'source_row': source_row, 'recommended_action': action})
    findings.append({
        'check_name': check_name,
        'severity': severity,
        'entity_type': entity_type,
        'entity_id': entity_id,
        'system_id': system_id,
        'status': status,
        'message': message,
        'source_file': source_file,
        'source_row': source_row,
        'recommended_action': action,
    })


def as_num(v):
    if pd.isna(v):
        return None
    try:
        x = float(str(v).strip())
        return None if pd.isna(x) else x
    except Exception:
        return None

# Parse model IDs for reference
# Model parsing for cross-checks
model_path = ROOT / 'models/model.inp'
model_nodes, model_links = set(), set()
if model_path.exists():
    sec = None
    for line in model_path.read_text(encoding='utf-8').splitlines():
        t = line.strip()
        if not t or t.startswith(';;'):
            continue
        if t.startswith('[') and t.endswith(']'):
            sec = t
            continue
        parts = t.split()
        if sec == '[JUNCTIONS]' and parts:
            model_nodes.add(parts[0])
        elif sec == '[CONDUITS]' and parts:
            model_links.add(parts[0])

# canonical node sets
link_typed_nodes = set()
for _, r in links.iterrows():
    us = typed_canonical_id(r.get('upstream_node'), 'junction')
    ds = typed_canonical_id(r.get('downstream_node'), 'junction')
    if us:
        link_typed_nodes.add(us)
    if ds:
        link_typed_nodes.add(ds)

jmap_typed_nodes = set(jmap['canonical_junction_id'].dropna().astype(str).tolist()) if not jmap.empty and 'canonical_junction_id' in jmap.columns else set()
canonical_junction_set = jmap_typed_nodes | link_typed_nodes

# A) Junction QA
junction_qaqc = []
junction_ids = sorted(canonical_junction_set)
# Junction canonical ids from model naming pattern
node_ids_from_links = set()
if not links.empty and {'upstream_node', 'downstream_node'}.issubset(links.columns):
    for _, r in links.iterrows():
        for c in ('upstream_node', 'downstream_node'):
            n = as_num(r.get(c))
            if n is not None:
                node_ids_from_links.add(f"J{int(n):03d}" if float(n).is_integer() else f"J{str(n).replace('.', '_')}")

# A) Junction QA
junction_qaqc = []

node_ids_from_nodes_table = set()
if not nodes.empty and 'jct' in nodes.columns:
    for v in nodes['jct']:
        n = as_num(v)
        if n is not None:
            node_ids_from_nodes_table.add(f"J{int(n):03d}" if float(n).is_integer() else f"J{str(n).replace('.', '_')}")
junction_ids = sorted((node_ids_from_nodes_table | node_ids_from_links) if node_ids_from_nodes_table else (model_nodes if model_nodes else node_ids_from_links))
j_count = len(junction_ids)

dup_j = [k for k, v in Counter(junction_ids).items() if v > 1]
for j in dup_j:
    add_finding('junction_duplicate_typed_id', 'HIGH', 'junction', j, None, 'fail', 'Duplicate typed canonical junction ID.', 'data/processed/id_map_junctions.csv', None, 'Ensure canonical junction IDs are unique.')

if j_count:
    add_finding('junction_defaults_concentration', 'MEDIUM', 'junction', 'ALL', None, 'warn', 'Synthetic/default geometry concentration appears high under current assumptions.', 'models/model.inp', None, 'Validate key junction geometry against engineering sheets.')
    junction_qaqc.append({'check_name': 'junction_defaults_concentration', 'status': 'warn', 'metric': 'default_ratio', 'value': 1.0, 'message': 'Default-heavy junction geometry in current build.'})

adj = defaultdict(set)
for _, r in links.iterrows():
    us = typed_canonical_id(r.get('upstream_node'), 'junction')
    ds = typed_canonical_id(r.get('downstream_node'), 'junction')
    if not us or not ds:
        continue
    adj[us].add(ds)
    adj[ds].add(us)
orphans = sorted([n for n in junction_ids if len(adj.get(n, set())) == 0])
for o in orphans:
    add_finding('junction_orphan', 'MEDIUM', 'junction', o, None, 'warn', 'Junction has no connected links.', 'data/processed/links.csv', None, 'Confirm intentional isolation or add connectivity.')
junction_qaqc.append({'check_name': 'junction_orphan_count', 'status': 'pass' if not orphans else 'warn', 'metric': 'count', 'value': len(orphans), 'message': 'Orphan typed junction IDs.'})

# B) Link QA
link_qaqc = []
missing_endpoint = self_loop = invalid_length = invalid_dia = node_ref_missing = 0

for i, r in links.iterrows():
    us = typed_canonical_id(r.get('upstream_node'), 'junction')
    ds = typed_canonical_id(r.get('downstream_node'), 'junction')
    add_finding('junction_duplicate_swmm_id', 'HIGH', 'junction', j, None, 'fail', 'Duplicate SWMM junction ID generated from link endpoints', 'data/processed/links.csv', None, 'Fix source endpoint IDs to unique canonical junction mapping.')

if j_count:
    derived_defaults_ratio = 1.0  # by current build assumptions all junction elevations/depth are synthetic defaults
    junction_qaqc.append({'check_name': 'junction_defaults_concentration', 'status': 'warn', 'metric': 'default_ratio', 'value': derived_defaults_ratio, 'message': 'Current build uses synthetic default junction geometry for all modeled nodes.'})
    add_finding('junction_defaults_concentration', 'MEDIUM', 'junction', 'ALL', None, 'warn', 'Synthetic/default geometry concentration appears high under current assumptions.', 'models/model.inp', None, 'Validate key junction rim/invert/depth values against design tables before Phase 2 interpretation.')

# Orphans by model/links graph
adj = defaultdict(set)
for _, r in links.iterrows():
    us, ds = as_num(r.get('upstream_node')), as_num(r.get('downstream_node'))
    if us is None or ds is None:
        continue
    us_id = f"J{int(us):03d}" if float(us).is_integer() else f"J{str(us).replace('.', '_')}"
    ds_id = f"J{int(ds):03d}" if float(ds).is_integer() else f"J{str(ds).replace('.', '_')}"
    adj[us_id].add(ds_id)
    adj[ds_id].add(us_id)

orphans = sorted([n for n in junction_ids if len(adj.get(n, set())) == 0])
for o in orphans:
    add_finding('junction_orphan', 'MEDIUM', 'junction', o, None, 'warn', 'Junction has no connected links.', 'data/processed/links.csv', None, 'Confirm this node should be connected or intentionally isolated.')
junction_qaqc.append({'check_name': 'junction_orphan_count', 'status': 'pass' if not orphans else 'warn', 'metric': 'count', 'value': len(orphans), 'message': 'Orphan junctions not connected to any link.'})

# B) Link QA
link_qaqc = []
missing_endpoint = 0
self_loop = 0
invalid_length = 0
invalid_dia = 0
node_ref_missing = 0

for i, r in links.iterrows():
    us, ds = as_num(r.get('upstream_node')), as_num(r.get('downstream_node'))
    pid_val = r.get('pipe_id', f'row_{i+1}')
    pid = str(pid_val) if pd.notna(pid_val) and str(pid_val).strip() else f'row_{i+1}'
    src_row = int(r['source_row']) if pd.notna(r.get('source_row')) and str(r.get('source_row')).strip() else None

    if not us or not ds:
        missing_endpoint += 1
        add_finding('link_missing_endpoint', 'HIGH', 'link', pid, None, 'fail', 'Missing upstream/downstream endpoint.', 'data/processed/links.csv', src_row, 'Populate both endpoints.')
        continue
    if us == ds:
        self_loop += 1
        add_finding('link_self_loop', 'HIGH', 'link', pid, None, 'fail', 'Self-loop link detected.', 'data/processed/links.csv', src_row, 'Correct upstream/downstream assignment.')
    if us not in canonical_junction_set or ds not in canonical_junction_set:
        node_ref_missing += 1
        add_finding('link_endpoint_node_missing', 'HIGH', 'link', pid, None, 'fail', 'Endpoint node reference not found in typed canonical junction set.', 'data/processed/links.csv', src_row, 'Align endpoint canonicalization with typed node namespace.')
    if us is None or ds is None:
        missing_endpoint += 1
        add_finding('link_missing_endpoint', 'HIGH', 'link', pid, None, 'fail', 'Missing upstream/downstream endpoint.', 'data/processed/links.csv', src_row, 'Populate both upstream and downstream node IDs.')
        continue

    us_id = f"J{int(us):03d}" if float(us).is_integer() else f"J{str(us).replace('.', '_')}"
    ds_id = f"J{int(ds):03d}" if float(ds).is_integer() else f"J{str(ds).replace('.', '_')}"

    if us_id == ds_id:
        self_loop += 1
        add_finding('link_self_loop', 'HIGH', 'link', pid, None, 'fail', 'Upstream and downstream node are identical (self-loop).', 'data/processed/links.csv', src_row, 'Verify upstream/downstream assignment in HYDRAULICS source.')

    if us_id not in junction_ids or ds_id not in junction_ids:
        node_ref_missing += 1
        add_finding('link_endpoint_node_missing', 'HIGH', 'link', pid, None, 'fail', 'Endpoint node reference not found in canonical node set.', 'data/processed/links.csv', src_row, 'Fix endpoint mapping to existing canonical node IDs.')

    length = as_num(r.get('length'))
    dia = as_num(r.get('dia'))
    if length is not None and length <= 0:
        invalid_length += 1
        add_finding('link_non_positive_length', 'HIGH', 'link', pid, None, 'fail', 'Non-positive conduit length.', 'data/processed/links.csv', src_row, 'Fix length values.')
    if dia is not None and dia <= 0:
        invalid_dia += 1
        add_finding('link_invalid_diameter', 'HIGH', 'link', pid, None, 'fail', 'Non-positive conduit diameter.', 'data/processed/links.csv', src_row, 'Fix diameter values.')

# explicit validation of previously flagged links
target_links = ['101-101.1','101.1-107','102-102.1','102.1-102.3','102.3-102.4','102.4-103.3','103-103.2','103.1-103.2','103.2-103.3','103.3-103.4']
for t in target_links:
    row = links[links['pipe_id'].astype(str) == t]
    if row.empty:
        add_finding('target_link_presence', 'MEDIUM', 'link', t, None, 'warn', 'Target link not found in processed links.', 'data/processed/links.csv', None, 'Verify source extraction preserved this link ID.')
        continue
    rr = row.iloc[0]
    us = typed_canonical_id(rr.get('upstream_node'), 'junction')
    ds = typed_canonical_id(rr.get('downstream_node'), 'junction')
    if us in canonical_junction_set and ds in canonical_junction_set:
        link_qaqc.append({'check_name': 'target_link_endpoint_resolution', 'status': 'pass', 'metric': 'bool', 'value': True, 'message': f'{t} endpoints resolve: {us}->{ds}'})
    else:
        add_finding('target_link_endpoint_resolution', 'HIGH', 'link', t, None, 'fail', f'Target link unresolved endpoint(s): {us}->{ds}.', 'data/processed/links.csv', int(rr['source_row']) if pd.notna(rr.get('source_row')) else None, 'Normalize endpoint IDs to typed canonical junction namespace.')

vc = links['pipe_id'].astype(str).value_counts() if 'pipe_id' in links.columns else pd.Series(dtype=int)
for p in vc[vc > 1].index.tolist():
    add_finding('link_duplicate_id', 'MEDIUM', 'link', p, None, 'warn', 'Duplicate pipe_id appears in processed links.', 'data/processed/links.csv', None, 'Confirm if parallel pipes need unique IDs.')

proc_conduits = {f"C{i+1:03d}" for i in range(min(len(links), 250))}
missing_in_model = sorted(proc_conduits - model_links)
if missing_in_model:
    add_finding('model_conduit_missing_from_model', 'MEDIUM', 'model', 'CONDUITS', None, 'warn', f'{len(missing_in_model)} processed conduits not present in model [CONDUITS].', 'models/model.inp', None, 'Review model build truncation/index assumptions.')

link_qaqc.extend([
    {'check_name': 'missing_link_endpoint_count', 'status': 'pass' if missing_endpoint == 0 else 'fail', 'metric': 'count', 'value': missing_endpoint, 'message': 'Links missing endpoint references.'},
    {'check_name': 'self_loop_link_count', 'status': 'pass' if self_loop == 0 else 'fail', 'metric': 'count', 'value': self_loop, 'message': 'Self-loop links.'},
    {'check_name': 'endpoint_not_in_canonical_set_count', 'status': 'pass' if node_ref_missing == 0 else 'fail', 'metric': 'count', 'value': node_ref_missing, 'message': 'Link endpoints outside typed canonical set.'},
        add_finding('link_non_positive_length', 'HIGH', 'link', pid, None, 'fail', 'Non-positive conduit length.', 'data/processed/links.csv', src_row, 'Correct conduit length values to positive feet.')
    if dia is not None and dia <= 0:
        invalid_dia += 1
        add_finding('link_invalid_diameter', 'HIGH', 'link', pid, None, 'fail', 'Non-positive conduit diameter.', 'data/processed/links.csv', src_row, 'Correct conduit diameter values to positive inches.')

# duplicate link IDs
dup_pipe = []
if 'pipe_id' in links.columns:
    vc = links['pipe_id'].astype(str).value_counts()
    dup_pipe = vc[vc > 1].index.tolist()
for p in dup_pipe:
    add_finding('link_duplicate_id', 'MEDIUM', 'link', p, None, 'warn', 'Duplicate pipe_id appears in processed links.', 'data/processed/links.csv', None, 'Confirm whether parallel pipes should have distinct IDs.')

# model conduit presence cross-check
proc_conduits = {f"C{i+1:03d}" for i in range(min(len(links), 250))}
missing_in_model = sorted(proc_conduits - model_links)
extra_in_model = sorted(model_links - proc_conduits)
if missing_in_model:
    add_finding('model_conduit_missing_from_model', 'MEDIUM', 'model', 'CONDUITS', None, 'warn', f'{len(missing_in_model)} processed conduits not present in model [CONDUITS].', 'models/model.inp', None, 'Check build truncation/head limits and ensure all intended links are carried to model.')
if extra_in_model:
    add_finding('model_conduit_extra_in_model', 'LOW', 'model', 'CONDUITS', None, 'warn', f'{len(extra_in_model)} model conduits not found in expected processed-index set.', 'models/model.inp', None, 'Review build indexing assumptions if this is unexpected.')

link_qaqc.extend([
    {'check_name': 'missing_link_endpoint_count', 'status': 'pass' if missing_endpoint == 0 else 'fail', 'metric': 'count', 'value': missing_endpoint, 'message': 'Links missing endpoint references.'},
    {'check_name': 'self_loop_link_count', 'status': 'pass' if self_loop == 0 else 'fail', 'metric': 'count', 'value': self_loop, 'message': 'Self-loop links where upstream=downstream.'},
    {'check_name': 'non_positive_length_count', 'status': 'pass' if invalid_length == 0 else 'fail', 'metric': 'count', 'value': invalid_length, 'message': 'Links with non-positive length.'},
    {'check_name': 'invalid_diameter_count', 'status': 'pass' if invalid_dia == 0 else 'fail', 'metric': 'count', 'value': invalid_dia, 'message': 'Links with non-positive diameter.'},
])

# C) Topology QA
all_nodes = set(junction_ids)
visited = set(); components = []
topology_qaqc = []
# connected components
all_nodes = set(junction_ids)
visited = set()
components = []
for n in all_nodes:
    if n in visited:
        continue
    q = deque([n]); visited.add(n); comp = []
    while q:
        cur = q.popleft(); comp.append(cur)
        for nxt in adj.get(cur, set()):
            if nxt not in visited:
                visited.add(nxt); q.append(nxt)
    components.append(sorted(comp))

if len(components) > 1:
    add_finding('topology_isolated_subnetworks', 'MEDIUM', 'system', 'graph_components', None, 'warn', f'Network has {len(components)} connected components.', 'data/processed/links.csv', None, 'Validate disconnected systems against BASIN segmentation.')

dead_ends = [n for n in all_nodes if len(adj.get(n, set())) == 1]
if dead_ends:
    add_finding('topology_dead_end_nodes', 'LOW', 'junction', 'dead_end_summary', None, 'warn', f'{len(dead_ends)} degree-1 nodes detected.', 'data/processed/links.csv', None, 'Review whether dead ends are expected starts/termini.')

# directed cycle
dir_adj = defaultdict(list)
for _, r in links.iterrows():
    us = typed_canonical_id(r.get('upstream_node'), 'junction')
    ds = typed_canonical_id(r.get('downstream_node'), 'junction')
    if us and ds:
        dir_adj[us].append(ds)
WHITE, GRAY, BLACK = 0, 1, 2
color = {n: WHITE for n in all_nodes}

def dfs(u):
    add_finding('topology_isolated_subnetworks', 'MEDIUM', 'system', 'graph_components', None, 'warn', f'Network has {len(components)} connected components.', 'data/processed/links.csv', None, 'Verify intended disconnected systems and basin segmentation.')

# dead ends
deg = {n: len(adj.get(n, set())) for n in all_nodes}
dead_ends = [n for n, d in deg.items() if d == 1]
if dead_ends:
    add_finding('topology_dead_end_nodes', 'LOW', 'junction', 'dead_end_summary', None, 'warn', f'{len(dead_ends)} degree-1 nodes detected (could include valid system starts/ends).', 'data/processed/links.csv', None, 'Review degree-1 nodes for expected starts/outfalls.')

# directed cycle check using links direction
dir_adj = defaultdict(list)
for _, r in links.iterrows():
    us, ds = as_num(r.get('upstream_node')), as_num(r.get('downstream_node'))
    if us is None or ds is None:
        continue
    us_id = f"J{int(us):03d}" if float(us).is_integer() else f"J{str(us).replace('.', '_')}"
    ds_id = f"J{int(ds):03d}" if float(ds).is_integer() else f"J{str(ds).replace('.', '_')}"
    dir_adj[us_id].append(ds_id)

WHITE, GRAY, BLACK = 0, 1, 2
color = {n: WHITE for n in all_nodes}
cycle_found = False

def dfs(u):
    global cycle_found
    color[u] = GRAY
    for v in dir_adj.get(u, []):
        if color.get(v, WHITE) == GRAY:
            return True
        if color.get(v, WHITE) == WHITE and dfs(v):
            return True
    color[u] = BLACK
    return False
cycle_found = any(color[n] == WHITE and dfs(n) for n in list(all_nodes))
if cycle_found:
    add_finding('topology_cycle_detected', 'MEDIUM', 'system', 'directed_graph', None, 'warn', 'Directed cycle detected in link graph.', 'data/processed/links.csv', None, 'Confirm loop is intentional.')

cross_system_edge_count = int(topo['crosses_basin_boundary'].fillna(False).sum()) if not topo.empty and 'crosses_basin_boundary' in topo.columns else 0
if cross_system_edge_count > 0:
    add_finding('topology_cross_system_leakage', 'HIGH', 'system', 'basin_boundary', None, 'fail', f'{cross_system_edge_count} edges cross BASIN boundary.', 'data/processed/junction_topology_map.csv', None, 'Repair system segmentation/link direction mapping.')

topology_qaqc = [
    {'check_name': 'connected_component_count', 'status': 'pass' if len(components) == 1 else 'warn', 'metric': 'count', 'value': len(components), 'message': 'Connected components in network graph.'},
    {'check_name': 'dead_end_count', 'status': 'warn' if dead_ends else 'pass', 'metric': 'count', 'value': len(dead_ends), 'message': 'Degree-1 nodes.'},
    {'check_name': 'cycle_detected', 'status': 'warn' if cycle_found else 'pass', 'metric': 'bool', 'value': bool(cycle_found), 'message': 'Directed cycle presence.'},
    {'check_name': 'cross_system_edge_count', 'status': 'pass' if cross_system_edge_count == 0 else 'fail', 'metric': 'count', 'value': cross_system_edge_count, 'message': 'Edges crossing BASIN boundaries.'},
]

# D) Inflow QA
unmapped_inflow_count = int(inflows['canonical_swmm_node_id'].isna().sum()) if not inflows.empty else 0
dup_source_row = source_multi_node = 0
if not inflows.empty and {'source_tab', 'source_row'}.issubset(inflows.columns):
    key = inflows[['source_tab', 'source_row']].astype(str).agg('|'.join, axis=1)
    dup_source_row = int((key.value_counts() > 1).sum())
    source_multi_node = int((inflows.groupby(['source_tab', 'source_row'])['canonical_swmm_node_id'].nunique(dropna=True) > 1).sum())
if source_multi_node > 0:
    add_finding('inflow_source_row_multi_node', 'HIGH', 'inflow', 'source_row_multi_node', None, 'fail', f'{source_multi_node} source rows map to multiple SWMM nodes.', 'data/processed/id_map_inflows.csv', None, 'Enforce 1:1 mapping per source row.')
if unmapped_inflow_count > 0:
    add_finding('inflow_unmapped', 'HIGH', 'inflow', 'unmapped', None, 'fail', f'{unmapped_inflow_count} inflow records missing node mapping.', 'data/processed/id_map_inflows.csv', None, 'Resolve inflow receiving node mapping.')
if dup_source_row > 0:
    add_finding('inflow_duplicate_source_row', 'MEDIUM', 'inflow', 'source_row_duplicates', None, 'warn', f'{dup_source_row} duplicate inflow source rows.', 'data/processed/id_map_inflows.csv', None, 'Confirm whether duplicates are expected.')

column_d_guard = int(inf_summary.get('regression_validation', {}).get('column_d_used_as_inlet_id_count', -1))
if column_d_guard != 0:
    add_finding('inflow_column_d_regression_guard', 'HIGH', 'inflow', 'hydrology_column_d', None, 'fail', f'Column D used as inlet ID count={column_d_guard}.', 'outputs/logs/inflow_mapping_summary.json', None, 'Fix hydrology column provenance parser.')

inflow_qaqc = [
    {'check_name': 'unmapped_inflow_count', 'status': 'pass' if unmapped_inflow_count == 0 else 'fail', 'metric': 'count', 'value': unmapped_inflow_count, 'message': 'Inflows missing receiving node.'},
    {'check_name': 'duplicate_inflow_source_row_count', 'status': 'warn' if dup_source_row > 0 else 'pass', 'metric': 'count', 'value': dup_source_row, 'message': 'Duplicate inflow source rows.'},
    {'check_name': 'inflow_source_row_multi_node_count', 'status': 'pass' if source_multi_node == 0 else 'fail', 'metric': 'count', 'value': source_multi_node, 'message': 'Source rows mapped to >1 node.'},
    {'check_name': 'column_d_used_as_inlet_id_count', 'status': 'pass' if column_d_guard == 0 else 'fail', 'metric': 'count', 'value': column_d_guard, 'message': 'HYDROLOGY Column D regression guard.'},
]

# system rollup
if not inflows.empty:
    inflows.groupby('system_id', dropna=False).agg(
        inflow_record_count=('record_index', 'count'),
        mapped_node_count=('canonical_swmm_node_id', lambda s: s.notna().sum()),
        total_q_cfs=('q_cfs', lambda s: pd.to_numeric(s, errors='coerce').fillna(0).sum()),
    ).reset_index().to_csv(OUT_QA / 'system_qaqc_rollup.csv', index=False)


for n in all_nodes:
    if color[n] == WHITE and dfs(n):
        cycle_found = True
        break
if cycle_found:
    add_finding('topology_cycle_detected', 'MEDIUM', 'system', 'directed_graph', None, 'warn', 'Directed cycle detected in upstream->downstream links.', 'data/processed/links.csv', None, 'Confirm if looped network is intentional.')

cross_system_edge_count = int(topo['crosses_basin_boundary'].fillna(False).sum()) if not topo.empty and 'crosses_basin_boundary' in topo.columns else 0
if cross_system_edge_count > 0:
    add_finding('topology_cross_system_leakage', 'HIGH', 'system', 'basin_boundary', None, 'fail', f'{cross_system_edge_count} topology edges cross BASIN system boundaries.', 'data/processed/junction_topology_map.csv', None, 'Correct system segmentation or topology mapping.')

topology_qaqc.extend([
    {'check_name': 'connected_component_count', 'status': 'pass' if len(components) == 1 else 'warn', 'metric': 'count', 'value': len(components), 'message': 'Connected components in undirected network graph.'},
    {'check_name': 'dead_end_count', 'status': 'warn' if dead_ends else 'pass', 'metric': 'count', 'value': len(dead_ends), 'message': 'Degree-1 nodes.'},
    {'check_name': 'cycle_detected', 'status': 'warn' if cycle_found else 'pass', 'metric': 'bool', 'value': bool(cycle_found), 'message': 'Directed cycle presence in network.'},
    {'check_name': 'cross_system_edge_count', 'status': 'pass' if cross_system_edge_count == 0 else 'fail', 'metric': 'count', 'value': cross_system_edge_count, 'message': 'Edges crossing BASIN system boundaries.'},
])

# D) Inflow QA
inflow_qaqc = []
unmapped_inflow_count = int(inflows['canonical_swmm_node_id'].isna().sum()) if not inflows.empty else 0
dup_source_row = 0
source_multi_node = 0
if not inflows.empty and {'source_tab', 'source_row'}.issubset(inflows.columns):
    key = inflows[['source_tab', 'source_row']].astype(str).agg('|'.join, axis=1)
    vc = key.value_counts()
    dup_source_row = int((vc > 1).sum())
    bykey = inflows.groupby(['source_tab', 'source_row'])['canonical_swmm_node_id'].nunique(dropna=True)
    source_multi_node = int((bykey > 1).sum())

if dup_source_row > 0:
    add_finding('inflow_duplicate_source_row', 'MEDIUM', 'inflow', 'source_row_duplicates', None, 'warn', f'{dup_source_row} source rows appear multiple times (review if expected from bridge augmentation).', 'data/processed/id_map_inflows.csv', None, 'Confirm duplicate source-row behavior is intentional.')
if source_multi_node > 0:
    add_finding('inflow_source_row_multi_node', 'HIGH', 'inflow', 'source_row_multi_node', None, 'fail', f'{source_multi_node} source rows map to multiple SWMM nodes.', 'data/processed/id_map_inflows.csv', None, 'Enforce one source row to one receiving node mapping unless explicitly justified.')
if unmapped_inflow_count > 0:
    add_finding('inflow_unmapped', 'HIGH', 'inflow', 'unmapped', None, 'fail', f'{unmapped_inflow_count} inflow records missing SWMM node mapping.', 'data/processed/id_map_inflows.csv', None, 'Resolve missing receiving junction/SWMM node assignments.')

# system inflow rollup
system_roll = pd.DataFrame()
if not inflows.empty:
    g = inflows.groupby('system_id', dropna=False)
    system_roll = g.agg(inflow_record_count=('record_index', 'count'), mapped_node_count=('canonical_swmm_node_id', lambda s: s.notna().sum()), total_q_cfs=('q_cfs', lambda s: pd.to_numeric(s, errors='coerce').fillna(0).sum())).reset_index()
    system_roll.to_csv(OUT_QA / 'system_qaqc_rollup.csv', index=False)

column_d_guard = int(inf_summary.get('regression_validation', {}).get('column_d_used_as_inlet_id_count', -1))
if column_d_guard != 0:
    add_finding('inflow_column_d_regression_guard', 'HIGH', 'inflow', 'hydrology_column_d', None, 'fail', f'Column D used as inlet ID count is {column_d_guard}.', 'outputs/logs/inflow_mapping_summary.json', None, 'Fix parser provenance logic: only HYDROLOGY Column C can supply inlet IDs.')

inflow_qaqc.extend([
    {'check_name': 'unmapped_inflow_count', 'status': 'pass' if unmapped_inflow_count == 0 else 'fail', 'metric': 'count', 'value': unmapped_inflow_count, 'message': 'Inflows missing receiving SWMM node.'},
    {'check_name': 'duplicate_inflow_source_row_count', 'status': 'warn' if dup_source_row > 0 else 'pass', 'metric': 'count', 'value': dup_source_row, 'message': 'Duplicate inflow source rows.'},
    {'check_name': 'inflow_source_row_multi_node_count', 'status': 'pass' if source_multi_node == 0 else 'fail', 'metric': 'count', 'value': source_multi_node, 'message': 'Source rows mapped to >1 SWMM node.'},
    {'check_name': 'column_d_used_as_inlet_id_count', 'status': 'pass' if column_d_guard == 0 else 'fail', 'metric': 'count', 'value': column_d_guard, 'message': 'Regression guard for HYDROLOGY Column D provenance.'},
])

# Save detailed checks
pd.DataFrame(junction_qaqc).to_csv(OUT_QA / 'junction_qaqc_checks.csv', index=False)
pd.DataFrame(link_qaqc).to_csv(OUT_QA / 'link_qaqc_checks.csv', index=False)
pd.DataFrame(topology_qaqc).to_csv(OUT_QA / 'topology_qaqc_checks.csv', index=False)
pd.DataFrame(inflow_qaqc).to_csv(OUT_QA / 'inflow_qaqc_checks.csv', index=False)

findings_df = pd.DataFrame(findings)
if findings_df.empty:
    findings_df = pd.DataFrame(columns=['check_name', 'severity', 'entity_type', 'entity_id', 'system_id', 'status', 'message', 'source_file', 'source_row', 'recommended_action'])
findings_df.to_csv(OUT_QA / 'phase1_data_qaqc_findings.csv', index=False)

priority = findings_df.copy()
if not priority.empty:
    priority['sev_rank'] = priority['severity'].map({'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}).fillna(9)
# Priority review list
priority = findings_df.copy()
if not priority.empty:
    sev_rank = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    priority['sev_rank'] = priority['severity'].map(sev_rank).fillna(9)
    priority = priority.sort_values(['sev_rank', 'check_name', 'entity_id']).drop(columns=['sev_rank'])
priority.to_csv(OUT_REV / 'phase1_priority_review_list.csv', index=False)

high_count = int((findings_df['severity'] == 'HIGH').sum()) if not findings_df.empty else 0
medium_count = int((findings_df['severity'] == 'MEDIUM').sum()) if not findings_df.empty else 0
low_count = int((findings_df['severity'] == 'LOW').sum()) if not findings_df.empty else 0

metrics = {
    'junction_count': int(j_count),
    'link_count': int(len(links)),
    'inflow_record_count': int(len(inflows)),
    'system_count': int(inflows['system_id'].nunique(dropna=True)) if not inflows.empty and 'system_id' in inflows.columns else 0,
    'high_count': high_count,
    'medium_count': medium_count,
    'low_count': low_count,
    'orphan_junction_count': int(len(orphans)),
    'missing_link_endpoint_count': int(missing_endpoint),
    'self_loop_link_count': int(self_loop),
    'cross_system_edge_count': int(cross_system_edge_count),
    'unmapped_inflow_count': int(unmapped_inflow_count),
    'duplicate_inflow_source_row_count': int(dup_source_row),
    'column_d_used_as_inlet_id_count': int(column_d_guard),
    'qa_status': 'blocked' if high_count > 0 else ('needs_review' if medium_count > 0 else 'ready_for_phase2'),
}
(OUT_QA / 'phase1_data_qaqc_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')

summary = [
    '# Phase 1 Data QA/QC Summary (Pre-SWMM Runtime)', '',
# Summary markdown
summary = [
    '# Phase 1 Data QA/QC Summary (Pre-SWMM Runtime)',
    '',
    f"- QA Status: **{metrics['qa_status']}**",
    f"- Junctions: {metrics['junction_count']}",
    f"- Links: {metrics['link_count']}",
    f"- Inflow records: {metrics['inflow_record_count']}",
    f"- Systems: {metrics['system_count']}",
    f"- Findings: HIGH={high_count}, MEDIUM={medium_count}, LOW={low_count}", '',
    f"- Findings: HIGH={high_count}, MEDIUM={medium_count}, LOW={low_count}",
    '',
    '## Key Integrity Metrics',
    f"- Orphan junctions: {metrics['orphan_junction_count']}",
    f"- Missing link endpoints: {metrics['missing_link_endpoint_count']}",
    f"- Self-loop links: {metrics['self_loop_link_count']}",
    f"- Cross-system edges: {metrics['cross_system_edge_count']}",
    f"- Unmapped inflows: {metrics['unmapped_inflow_count']}",
    f"- Duplicate inflow source rows: {metrics['duplicate_inflow_source_row_count']}",
    f"- HYDROLOGY Column D used as inlet ID count: {metrics['column_d_used_as_inlet_id_count']}", '',
    f"- HYDROLOGY Column D used as inlet ID count: {metrics['column_d_used_as_inlet_id_count']}",
    '',
    '## Top Risks',
]
if not priority.empty:
    for _, r in priority.head(10).iterrows():
        summary.append(f"- [{r['severity']}] {r['check_name']} | {r['entity_type']} {r['entity_id']} | {r['message']} (source: {r['source_file']})")
else:
    summary.append('- No findings generated.')
summary.append('')
summary.append('## Phase 2 Handoff (Runtime/Simulation Focus)')
summary.append('- Validate residual HIGH findings first; then assess MEDIUM findings (defaults concentration/disconnected components/cycles) before interpreting runtime hydraulic results.')

summary.append('')
summary.append('## Phase 2 Handoff (Runtime/Simulation Focus)')
summary.append('- Verify the highest-risk MEDIUM findings in SWMM context (defaults concentration, disconnected components, duplicate inflow source-row semantics) before interpreting hydraulic performance results.')
(OUT_QA / 'phase1_data_qaqc_summary.md').write_text('\n'.join(summary) + '\n', encoding='utf-8')

print(json.dumps(metrics, indent=2))
