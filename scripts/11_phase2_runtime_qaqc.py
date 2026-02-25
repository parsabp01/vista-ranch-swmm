#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data/processed'
OUT_QA = ROOT / 'outputs/qa'
OUT_REV = ROOT / 'outputs/review'
OUT_LOG = ROOT / 'outputs/logs'
MODEL = ROOT / 'models/model.inp'
for d in (OUT_QA, OUT_REV, OUT_LOG):
    d.mkdir(parents=True, exist_ok=True)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_inp(path: Path) -> dict[str, list[str]]:
    sec = None
    out: dict[str, list[str]] = defaultdict(list)
def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()


def parse_inp(path: Path) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = defaultdict(list)
    sec = None
    for line in path.read_text(encoding='utf-8').splitlines():
        t = line.strip()
        if not t or t.startswith(';;'):
            continue
        if t.startswith('[') and t.endswith(']'):
            sec = t
            continue
        if sec:
            out[sec].append(t)
    return out


sections = parse_inp(MODEL)
findings: list[dict] = []


def add(sev: str, check: str, entity_type: str, entity_id: str, msg: str, src: str, prov: str, action: str, runtime_classification: str):
    findings.append({
        'severity': sev,
        'check_name': check,
        'entity_type': entity_type,
        'entity_id': entity_id,
        'message': msg,
        'source_dataset': src,
        'provenance': prov,
            sections[sec].append(t)
    return sections


def add_finding(rows: list[dict], severity: str, check_name: str, entity_type: str, entity_id: str, message: str, source_dataset: str, provenance: str, action: str, runtime_classification: str):
    rows.append({
        'severity': severity,
        'check_name': check_name,
        'entity_type': entity_type,
        'entity_id': entity_id,
        'message': message,
        'source_dataset': source_dataset,
        'provenance': provenance,
        'recommended_action': action,
        'runtime_classification': runtime_classification,
    })


# ---------- Phase 2A preflight for prior MEDIUM inflow findings ----------
inflows_df = pd.read_csv(ROOT / 'data/processed/id_map_inflows.csv')
inflows_df['source_key'] = inflows_df[['source_tab', 'source_row']].astype(str).agg('|'.join, axis=1)
model_inflows = [ln.split() for ln in sections.get('[INFLOWS]', [])]
model_inflow_nodes = [r[0] for r in model_inflows if r]

# duplicate source rows
key_counts = inflows_df['source_key'].value_counts()
dup_keys = key_counts[key_counts > 1]
dup_runtime_nodes = [n for n, c in Counter(model_inflow_nodes).items() if c > 1]

# multi-node source rows
multi_map = inflows_df.groupby('source_key')['canonical_swmm_node_id'].nunique(dropna=True)
multi_keys = multi_map[multi_map > 1]

# unmapped rows
unmapped = inflows_df[inflows_df['canonical_swmm_node_id'].isna()].copy()
unmapped['q_num'] = pd.to_numeric(unmapped['q_cfs'], errors='coerce').fillna(0.0)

preflight_rows = [
    {
        'finding': 'inflow_duplicate_source_row',
        'affected_count': int(len(dup_keys)),
        'example_entities': '; '.join(dup_keys.index.tolist()[:10]),
        'runtime_impact': f"duplicate INFLOWS nodes in model: {', '.join(dup_runtime_nodes) if dup_runtime_nodes else 'none'}",
        'classification': 'runtime_nonblocking_bookkeeping',
        'notes': 'bookkeeping duplicates exist in id_map_inflows; model-level dedupe mostly achieved but 3 duplicate target nodes remain.'
    },
    {
        'finding': 'inflow_source_row_multi_node',
        'affected_count': int(len(multi_keys)),
        'example_entities': '; '.join(multi_keys.index.tolist()[:10]),
        'runtime_impact': 'appears to be expansion artifact in mapping table; model consumes node-level inflow records',
        'classification': 'runtime_nonblocking_bookkeeping',
        'notes': 'source_key to node mapping should be normalized for traceability, but not a runtime parser blocker.'
    },
    {
        'finding': 'inflow_unmapped',
        'affected_count': int(len(unmapped)),
        'example_entities': '; '.join((unmapped['source_tab'].astype(str) + '|' + unmapped['source_row'].astype(str)).tolist()),
        'runtime_impact': f"omitted from model inflows; omitted |q|={unmapped['q_num'].abs().sum():.4f} cfs",
        'classification': 'accepted_baseline_assumption' if unmapped['q_num'].abs().sum() == 0 else 'needs_user_engineering_decision',
        'notes': 'rows align with trailing non-input records after final BASIN in previous parser notes.'
    },
]

pre_df = pd.DataFrame(preflight_rows)
pre_df.to_csv(OUT_QA / 'phase2a_inflow_preflight.csv', index=False)

pre_md = ['# Phase 2A Inflow Preflight', '']
for _, r in pre_df.iterrows():
    pre_md += [
        f"## {r['finding']}",
        f"- Affected count: {r['affected_count']}",
        f"- Classification: **{r['classification']}**",
        f"- Runtime impact: {r['runtime_impact']}",
        f"- Example entities: {r['example_entities']}",
        f"- Notes: {r['notes']}",
        ''
    ]
(OUT_QA / 'phase2a_inflow_preflight.md').write_text('\n'.join(pre_md), encoding='utf-8')


# ---------- Runtime structural/hydraulic static checks ----------
required_sections = ['[OPTIONS]', '[JUNCTIONS]', '[OUTFALLS]', '[CONDUITS]', '[XSECTIONS]', '[INFLOWS]', '[COORDINATES]']
for sec in required_sections:
    if sec not in sections or not sections[sec]:
        add('HIGH', 'missing_required_section', 'section', sec, f'missing required section {sec}', 'models/model.inp', sec, 'regenerate model with required section', 'runtime_blocking')

# Subcatchments must be absent or empty for this phase
if sections.get('[SUBCATCHMENTS]'):
    add('HIGH', 'subcatchments_present_in_hydraulic_only_phase', 'section', '[SUBCATCHMENTS]', 'subcatchments should be removed in hydraulic-only baseline', 'models/model.inp', '[SUBCATCHMENTS]', 'remove subcatchments from build', 'runtime_blocking')

junctions = [r.split()[0] for r in sections.get('[JUNCTIONS]', [])]
outfalls = [r.split()[0] for r in sections.get('[OUTFALLS]', [])]
conduits = [r.split() for r in sections.get('[CONDUITS]', [])]
xsections = [r.split() for r in sections.get('[XSECTIONS]', [])]
inflows = [r.split() for r in sections.get('[INFLOWS]', [])]
node_set = set(junctions + outfalls)

# ID conventions
if any(n.startswith('J0') and n[1:].isdigit() for n in junctions):
    add('HIGH', 'generic_node_ids_present', 'junction', 'J*', 'generic renumbered IDs detected; expected J_<source_id>', 'models/model.inp', '[JUNCTIONS]', 'preserve source-traceable IDs', 'runtime_blocking')
if any(not n.startswith(('J_', 'IN_')) for n in junctions):
    add('MEDIUM', 'unexpected_junction_naming', 'junction', 'JUNCTIONS', 'some junction IDs do not follow J_/IN_ convention', 'models/model.inp', '[JUNCTIONS]', 'normalize node naming', 'runtime_nonblocking_bookkeeping')
if any(not o.startswith('O_') for o in outfalls):
    add('HIGH', 'unexpected_outfall_naming', 'outfall', 'OUTFALLS', 'outfall IDs must follow O_<terminal_id>', 'models/model.inp', '[OUTFALLS]', 'normalize outfall naming', 'runtime_blocking')

# duplicates
for sec, ids in {
    '[JUNCTIONS]': junctions,
    '[OUTFALLS]': outfalls,
    '[CONDUITS]': [r[0] for r in conduits if len(r) > 2],
    '[INFLOWS]': [r[0] for r in inflows if len(r) > 2],
}.items():
    dups = [k for k, v in Counter(ids).items() if v > 1]
    if dups:
        sev = 'MEDIUM' if sec == '[INFLOWS]' else 'HIGH'
        add(sev, 'duplicate_ids_in_section', 'section', sec, f'duplicate IDs: {", ".join(dups[:10])}', 'models/model.inp', sec, 'deduplicate IDs', 'runtime_nonblocking_bookkeeping' if sec == '[INFLOWS]' else 'runtime_blocking')

# conduit integrity
self_loops = []
missing_end = []
nonpos_len = []
for r in conduits:
    if len(r) < 5:
        continue
    cid, us, ds, ln = r[0], r[1], r[2], r[3]
    if us == ds:
        self_loops.append(cid)
    if us not in node_set or ds not in node_set:
        missing_end.append(cid)
    try:
        if float(ln) <= 0:
            nonpos_len.append(cid)
    except Exception:
        nonpos_len.append(cid)
if self_loops:
    add('HIGH', 'self_loop_conduits', 'conduit', 'CONDUITS', f'{len(self_loops)} self-loop conduits found', 'models/model.inp', '[CONDUITS]', 'remove/fix self loops', 'runtime_blocking')
if missing_end:
    add('HIGH', 'conduit_endpoint_missing_node', 'conduit', 'CONDUITS', f'{len(missing_end)} conduits with missing endpoint nodes', 'models/model.inp', '[CONDUITS]', 'fix endpoint node references', 'runtime_blocking')
if nonpos_len:
    add('HIGH', 'non_positive_conduit_length', 'conduit', 'CONDUITS', f'{len(nonpos_len)} conduits with non-positive length', 'models/model.inp', '[CONDUITS]', 'fix conduit lengths', 'runtime_blocking')

# xsection diameters
bad_geom = []
for r in xsections:
    if len(r) < 3:
        continue
    try:
        if float(r[2]) <= 0:
            bad_geom.append(r[0])
    except Exception:
        bad_geom.append(r[0])
if bad_geom:
    add('HIGH', 'invalid_conduit_diameter', 'conduit', 'XSECTIONS', f'{len(bad_geom)} conduits with invalid diameters', 'models/model.inp', '[XSECTIONS]', 'fix diameters', 'runtime_blocking')

# inflows syntax/targets
bad_targets = [r[0] for r in inflows if len(r) >= 3 and r[0] not in node_set]
if bad_targets:
    add('HIGH', 'inflow_target_missing_node', 'inflow', 'INFLOWS', f'{len(set(bad_targets))} inflow target nodes missing from model nodes', 'models/model.inp', '[INFLOWS]', 'repair inflow target nodes', 'runtime_blocking')
non_inlet_targets = [r[0] for r in inflows if len(r) >= 3 and not r[0].startswith('IN_')]
if non_inlet_targets:
    add('HIGH', 'inflow_targets_not_inlets', 'inflow', 'INFLOWS', f'{len(set(non_inlet_targets))} inflow targets are not inlet nodes', 'models/model.inp', '[INFLOWS]', 'apply inflows only to IN_* nodes', 'runtime_blocking')
# SWMM Error 209 guard: third token must be quoted empty or a name, but we expect static syntax with FLOW in col4
for r in inflows:
    if len(r) < 8:
        add('HIGH', 'malformed_inflows_line', 'inflow', r[0] if r else 'unknown', 'malformed INFLOWS row; expected 8 tokens', 'models/model.inp', '[INFLOWS]', 'emit full static inflow syntax', 'runtime_blocking')
        break

# outfall segmentation / cross-system sanity from topology map
if (ROOT / 'data/processed/junction_topology_map.csv').exists():
    topo = pd.read_csv(ROOT / 'data/processed/junction_topology_map.csv')
    cross = int(topo.get('crosses_basin_boundary', pd.Series(dtype=bool)).fillna(False).sum()) if not topo.empty else 0
    if cross > 0:
        add('HIGH', 'cross_system_links_present', 'topology', 'junction_topology_map', f'{cross} cross-system links detected', 'data/processed/junction_topology_map.csv', 'crosses_basin_boundary', 'remove cross-basin links', 'runtime_blocking')

# default-heavy maxdepth
depths = []
rims = []
for row in sections.get('[JUNCTIONS]', []):
    p = row.split()
    if len(p) >= 3:
        try:
            depths.append(float(p[2]))
            rims.append(float(p[1]) + float(p[2]))
        except Exception:
            pass
if depths:
    mode_val, mode_cnt = Counter(depths).most_common(1)[0]
    ratio = mode_cnt / len(depths)
    if ratio > 0.9:
        add('MEDIUM', 'default_heavy_maxdepth', 'junction', 'JUNCTIONS', f'maxdepth default-heavy ({ratio:.2%} at {mode_val})', 'models/model.inp', '[JUNCTIONS]', 'engineering review of depths', 'accepted_baseline_assumption')

# write findings
fdf = pd.DataFrame(findings)
if fdf.empty:
    fdf = pd.DataFrame(columns=['severity', 'check_name', 'entity_type', 'entity_id', 'message', 'source_dataset', 'provenance', 'recommended_action', 'runtime_classification'])
sev_rank = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
fdf['sev_rank'] = fdf['severity'].map(sev_rank).fillna(9)
fdf = fdf.sort_values(['sev_rank', 'check_name', 'entity_id']).drop(columns=['sev_rank'])
fdf.to_csv(OUT_QA / 'phase2_runtime_qaqc_findings.csv', index=False)

high = int((fdf['severity'] == 'HIGH').sum())
med = int((fdf['severity'] == 'MEDIUM').sum())
low = int((fdf['severity'] == 'LOW').sum())
status = 'blocked' if high > 0 else ('needs_review' if med > 0 else 'ready_for_swmm_gui')
id_map_inflows = read_csv(PROC / 'id_map_inflows.csv')
inlet_bridge = read_csv(PROC / 'inlet_to_junction_map.csv')
links = read_csv(PROC / 'links.csv')
nodes = read_csv(PROC / 'nodes.csv')
sections = parse_inp(MODEL)
build_summary = json.loads((OUT_LOG / 'build_summary.json').read_text(encoding='utf-8')) if (OUT_LOG / 'build_summary.json').exists() else {}

# ========== Phase 2A inflow preflight ==========
pre_rows = []

inp_inflows = [ln.split() for ln in sections.get('[INFLOWS]', [])]
inp_inflow_df = pd.DataFrame([{'node': r[0] if len(r)>0 else '', 'constituent': r[1] if len(r)>1 else '', 'baseline': r[2] if len(r)>2 else ''} for r in inp_inflows])
if inp_inflow_df.empty:
    inp_inflow_df = pd.DataFrame(columns=['node','constituent','baseline'])
inp_inflow_df['baseline_num'] = pd.to_numeric(inp_inflow_df['baseline'], errors='coerce').fillna(0.0)

if not id_map_inflows.empty:
    id_map_inflows['source_key'] = id_map_inflows[['source_tab', 'source_row']].astype(str).agg('|'.join, axis=1)

    dup_keys = id_map_inflows['source_key'].value_counts()
    dup_keys = dup_keys[dup_keys > 1]
    dup_df = id_map_inflows[id_map_inflows['source_key'].isin(dup_keys.index)].copy()
    dup_group = dup_df.groupby('source_key').agg(
        row_count=('source_key', 'count'),
        swmm_node_count=('canonical_swmm_node_id', lambda s: s.nunique(dropna=True)),
        swmm_nodes=('canonical_swmm_node_id', lambda s: '|'.join(sorted(set(s.dropna().astype(str))))),
        source_tab=('source_tab', 'first'),
        source_row=('source_row', 'first'),
    ).reset_index()

    duplicated_runtime_nodes = sorted([n for n, c in Counter(inp_inflow_df['node']).items() if c > 1])
    dup_runtime_impact = 'Creates duplicated [INFLOWS] entries for nodes: ' + (', '.join(duplicated_runtime_nodes) if duplicated_runtime_nodes else 'none')
    pre_rows.append({
        'finding': 'inflow_duplicate_source_row',
        'affected_count': int(len(dup_group)),
        'example_entities': '; '.join(dup_group.head(10)['source_key'].astype(str).tolist()),
        'runtime_impact': dup_runtime_impact,
        'classification': 'runtime_nonblocking_bookkeeping',
        'notes': 'Most duplicates map to same node and many are zero-baseline rows; however duplicate [INFLOWS] node lines exist and should be consolidated.'
    })

    multi = id_map_inflows.groupby('source_key').agg(
        swmm_node_count=('canonical_swmm_node_id', lambda s: s.nunique(dropna=True)),
        swmm_nodes=('canonical_swmm_node_id', lambda s: '|'.join(sorted(set(s.dropna().astype(str))))),
        source_tab=('source_tab', 'first'),
        source_row=('source_row', 'first')
    ).reset_index()
    multi = multi[multi['swmm_node_count'] > 1]
    pre_rows.append({
        'finding': 'inflow_source_row_multi_node',
        'affected_count': int(len(multi)),
        'example_entities': '; '.join((multi['source_tab'].astype(str) + '|' + multi['source_row'].astype(str) + '->' + multi['swmm_nodes']).head(10).tolist()),
        'runtime_impact': 'No direct 1:1 source-row provenance in model.inp; appears to be mapping-table expansion artifact.',
        'classification': 'runtime_nonblocking_bookkeeping',
        'notes': 'Model [INFLOWS] is node-based; ambiguity exists in bookkeeping table, not in INP parser execution path.'
    })

    unmapped = id_map_inflows[id_map_inflows['canonical_swmm_node_id'].isna()].copy()
    unmapped_examples = []
    for _, r in unmapped.iterrows():
        unmapped_examples.append(f"{r.get('source_tab')}|{r.get('source_row')}|raw_inlet={r.get('raw_inlet')}|raw_junction={r.get('raw_junction')}|q={r.get('q_cfs')}")
    material_omission = pd.to_numeric(unmapped.get('q_cfs', pd.Series(dtype=float)), errors='coerce').fillna(0).abs().sum()
    pre_rows.append({
        'finding': 'inflow_unmapped',
        'affected_count': int(len(unmapped)),
        'example_entities': '; '.join(unmapped_examples),
        'runtime_impact': f'Omitted from [INFLOWS]; total omitted |q|={material_omission:.4f} cfs.',
        'classification': 'accepted_baseline_assumption' if material_omission == 0 else 'needs_user_engineering_decision',
        'notes': 'Both unmapped records have null q_cfs and occur after final BASIN/trailing non-input rows in prior parser notes.'
    })

preflight_df = pd.DataFrame(pre_rows)
preflight_df.to_csv(OUT_QA / 'phase2a_inflow_preflight.csv', index=False)

md = ['# Phase 2A Inflow Preflight', '']
for _, r in preflight_df.iterrows():
    md.append(f"## {r['finding']}")
    md.append(f"- Affected count: {r['affected_count']}")
    md.append(f"- Classification: **{r['classification']}**")
    md.append(f"- Runtime impact: {r['runtime_impact']}")
    md.append(f"- Example entities: {r['example_entities']}")
    md.append(f"- Notes: {r['notes']}")
    md.append('')
(OUT_QA / 'phase2a_inflow_preflight.md').write_text('\n'.join(md), encoding='utf-8')

# ========== Phase 2 runtime QA ==========
findings: list[dict] = []

required_sections = ['[OPTIONS]', '[RAINGAGES]', '[TIMESERIES]', '[JUNCTIONS]', '[OUTFALLS]', '[CONDUITS]', '[XSECTIONS]', '[SUBCATCHMENTS]', '[SUBAREAS]', '[INFILTRATION]', '[INFLOWS]', '[COORDINATES]']
for sec in required_sections:
    if sec not in sections or len(sections.get(sec, [])) == 0:
        add_finding(findings, 'HIGH', 'missing_required_section', 'model_section', sec, f'Missing or empty required section {sec}.', 'models/model.inp', sec, 'Add/repair section before runtime.', 'runtime_blocking')

junctions = [ln.split()[0] for ln in sections.get('[JUNCTIONS]', [])]
outfalls = [ln.split()[0] for ln in sections.get('[OUTFALLS]', [])]
conduits = [ln.split() for ln in sections.get('[CONDUITS]', [])]
xsections = [ln.split() for ln in sections.get('[XSECTIONS]', [])]
inflows_lines = [ln.split() for ln in sections.get('[INFLOWS]', [])]
coords = [ln.split()[0] for ln in sections.get('[COORDINATES]', [])]

node_set = set(junctions + outfalls)

# duplicate IDs per section
for sec, ids in {
    '[JUNCTIONS]': junctions,
    '[OUTFALLS]': outfalls,
    '[CONDUITS]': [r[0] for r in conduits if len(r) >= 3],
    '[XSECTIONS]': [r[0] for r in xsections if len(r) >= 2],
    '[INFLOWS]': [r[0] for r in inflows_lines if len(r) >= 3],
}.items():
    counts = Counter(ids)
    dups = [k for k, v in counts.items() if v > 1]
    if dups:
        sev = 'MEDIUM' if sec == '[INFLOWS]' else 'HIGH'
        add_finding(findings, sev, 'duplicate_ids_in_section', 'model_section', sec, f'Duplicate IDs present ({len(dups)}): {", ".join(dups[:10])}.', 'models/model.inp', sec, 'Ensure IDs are unique per section; consolidate duplicate inflow node lines.', 'runtime_nonblocking_bookkeeping' if sec == '[INFLOWS]' else 'runtime_blocking')

# conduit endpoint/node checks
bad_conduit_ep = []
non_pos_len = []
for row in conduits:
    if len(row) < 5:
        continue
    cid, us, ds, length = row[0], row[1], row[2], row[3]
    if us not in node_set or ds not in node_set:
        bad_conduit_ep.append(cid)
    try:
        if float(length) <= 0:
            non_pos_len.append(cid)
    except Exception:
        non_pos_len.append(cid)
if bad_conduit_ep:
    add_finding(findings, 'HIGH', 'conduit_endpoint_missing_node', 'conduit', 'CONDUITS', f'{len(bad_conduit_ep)} conduits reference missing nodes.', 'models/model.inp', '[CONDUITS]', 'Fix conduit endpoints to valid nodes.', 'runtime_blocking')
if non_pos_len:
    add_finding(findings, 'HIGH', 'non_positive_conduit_length', 'conduit', 'CONDUITS', f'{len(non_pos_len)} conduits have non-positive/invalid length.', 'models/model.inp', '[CONDUITS]', 'Correct conduit lengths.', 'runtime_blocking')

# xsection diameter sanity
bad_geom = []
for row in xsections:
    if len(row) < 3:
        continue
    try:
        if float(row[2]) <= 0:
            bad_geom.append(row[0])
    except Exception:
        bad_geom.append(row[0])
if bad_geom:
    add_finding(findings, 'HIGH', 'invalid_xsection_geom1', 'conduit', 'XSECTIONS', f'{len(bad_geom)} links have invalid Geom1.', 'models/model.inp', '[XSECTIONS]', 'Fix xsection geometry values.', 'runtime_blocking')

# inflow target nodes exist
bad_inflow_nodes = []
for row in inflows_lines:
    if len(row) >= 1 and row[0] not in node_set:
        bad_inflow_nodes.append(row[0])
if bad_inflow_nodes:
    add_finding(findings, 'HIGH', 'inflow_target_node_missing', 'inflow', 'INFLOWS', f'{len(set(bad_inflow_nodes))} inflow nodes are missing in JUNCTIONS/OUTFALLS.', 'models/model.inp', '[INFLOWS]', 'Fix inflow target node references.', 'runtime_blocking')

# build summary consistency
if build_summary:
    expected = {
        'junction_count': len(junctions),
        'conduit_count': len(conduits),
        'subcatchment_count': len(sections.get('[SUBCATCHMENTS]', [])),
        'inflow_count': len(inflows_lines),
    }
    for k, actual in expected.items():
        logged = build_summary.get(k)
        if logged is not None and int(logged) != int(actual):
            add_finding(findings, 'MEDIUM', 'build_summary_count_mismatch', 'build_summary', k, f'build_summary {k}={logged} but model section count={actual}.', 'outputs/logs/build_summary.json', k, 'Regenerate build summary from current model.', 'runtime_nonblocking_bookkeeping')

# disconnected / dead ends
adj = defaultdict(set)
for row in conduits:
    if len(row) >= 3:
        adj[row[1]].add(row[2])
        adj[row[2]].add(row[1])
isolated_junc = [j for j in junctions if len(adj.get(j, set())) == 0]
if isolated_junc:
    add_finding(findings, 'MEDIUM', 'isolated_junctions_in_model', 'junction', 'JUNCTIONS', f'{len(isolated_junc)} isolated junctions in model graph.', 'models/model.inp', '[JUNCTIONS]/[CONDUITS]', 'Review connectivity in GUI and source mapping.', 'needs_user_engineering_decision')

dead_ends = [j for j in junctions if len(adj.get(j, set())) == 1]
if dead_ends:
    add_finding(findings, 'LOW', 'dead_end_junctions', 'junction', 'dead_end_summary', f'{len(dead_ends)} degree-1 junctions found.', 'models/model.inp', '[CONDUITS]', 'Treat as acceptable where they represent starts/termini/outfalls.', 'accepted_baseline_assumption')

# default-heavy values
if sections.get('[JUNCTIONS]'):
    max_depths = []
    for ln in sections['[JUNCTIONS]']:
        parts = ln.split()
        if len(parts) >= 3:
            try:
                max_depths.append(float(parts[2]))
            except Exception:
                pass
    if max_depths:
        c = Counter(max_depths)
        mode_val, mode_count = c.most_common(1)[0]
        ratio = mode_count / len(max_depths)
        if ratio > 0.9:
            add_finding(findings, 'MEDIUM', 'default_heavy_junction_maxdepth', 'junction', 'JUNCTIONS', f'MaxDepth appears default-heavy: {ratio:.2%} at value {mode_val}.', 'models/model.inp', '[JUNCTIONS]', 'Validate representative junction depths in engineering review.', 'accepted_baseline_assumption')

if inflows_lines:
    baselines = []
    for r in inflows_lines:
        if len(r) >= 3:
            try:
                baselines.append(float(r[2]))
            except Exception:
                pass
    if baselines:
        zero_ratio = sum(1 for v in baselines if abs(v) < 1e-12) / len(baselines)
        if zero_ratio > 0.9:
            add_finding(findings, 'LOW', 'inflow_zero_baseline_concentration', 'inflow', 'INFLOWS', f'{zero_ratio:.2%} of inflow baseline values are zero.', 'models/model.inp', '[INFLOWS]', 'Expected in baseline build; confirm non-zero inflows match intended design points.', 'accepted_baseline_assumption')

findings_df = pd.DataFrame(findings)
if findings_df.empty:
    findings_df = pd.DataFrame(columns=['severity', 'check_name', 'entity_type', 'entity_id', 'message', 'source_dataset', 'provenance', 'recommended_action', 'runtime_classification'])

sev_rank = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
findings_df['sev_rank'] = findings_df['severity'].map(sev_rank).fillna(9)
findings_df = findings_df.sort_values(['sev_rank', 'check_name', 'entity_id']).drop(columns=['sev_rank'])
findings_df.to_csv(OUT_QA / 'phase2_runtime_qaqc_findings.csv', index=False)

high = int((findings_df['severity'] == 'HIGH').sum())
med = int((findings_df['severity'] == 'MEDIUM').sum())
low = int((findings_df['severity'] == 'LOW').sum())
status = 'blocked' if high > 0 else ('needs_review' if med > 0 else 'ready_for_gui_runtime')

metrics = {
    'generated_at_utc': now_utc(),
    'required_sections_present_count': sum(1 for s in required_sections if s in sections and len(sections[s]) > 0),
    'required_sections_total': len(required_sections),
    'junction_count': len(junctions),
    'outfall_count': len(outfalls),
    'conduit_count': len(conduits),
    'inflow_entry_count': len(inflows),
    'unique_inflow_nodes': len(set([r[0] for r in inflows if r])),
    'self_loop_conduit_count': len(self_loops),
    'xsection_count': len(xsections),
    'inflow_entry_count': len(inflows_lines),
    'unique_inflow_nodes': int(len(set([r[0] for r in inflows_lines if r]))),
    'high_count': high,
    'medium_count': med,
    'low_count': low,
    'qa_status': status,
}
(OUT_QA / 'phase2_runtime_qaqc_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')

summary = [
    '# Phase 2 Runtime QA/QC Summary',
    '',
    f"- QA Status: **{status}**",
    f"- Findings: HIGH={high}, MEDIUM={med}, LOW={low}",
    f"- Required sections present: {metrics['required_sections_present_count']}/{metrics['required_sections_total']}",
    f"- Node counts: junctions={len(junctions)}, outfalls={len(outfalls)}, inflow nodes={metrics['unique_inflow_nodes']}",
    f"- Conduit count: {len(conduits)} | self-loops={len(self_loops)}",
    '',
    '## Top Findings',
]
for _, r in fdf.head(20).iterrows():
    summary.append(f"- [{r['severity']}] {r['check_name']} | {r['entity_type']} {r['entity_id']} | {r['message']}")
(OUT_QA / 'phase2_runtime_qaqc_summary.md').write_text('\n'.join(summary) + '\n', encoding='utf-8')

# checklist / spotchecks
checklist = [
    '# SWMM GUI QA/QC Checklist (Hydraulic Baseline)',
    '',
    '- Confirm each BASIN system discharges to its own O_* outfall.',
    '- Confirm inlet nodes (IN_*) exist and receive static [INFLOWS].',
    '- Verify no self-loop conduits and no cross-system links.',
    '- Run SWMM and review continuity and stability reports.',
    '- Spot-check top risk links/nodes against HYDROLOGY/DI TABLE source rows in crosswalk.',
    '- Verify inlet rim elevation assumption (rim = receiving junction rim) is acceptable.',
]
(OUT_REV / 'swmm_gui_qaqc_checklist.md').write_text('\n'.join(checklist) + '\n', encoding='utf-8')

priority = []
for i, n in enumerate(sorted([n for n, c in Counter(model_inflow_nodes).items() if c > 1]), start=1):
    priority.append({'priority_rank': i, 'entity_type': 'node', 'entity_id': n, 'risk_reason': 'duplicate_inflow_target', 'source': 'model [INFLOWS]'})
for r in model_inflows:
    if len(r) >= 7:
        try:
            q = float(r[6])
        except Exception:
            q = 0.0
        if q > 0:
            priority.append({'priority_rank': len(priority)+1, 'entity_type': 'node', 'entity_id': r[0], 'risk_reason': f'nonzero_inflow_{q:.4f}_cfs', 'source': 'model [INFLOWS]'})
priority_df = pd.DataFrame(priority).drop_duplicates(subset=['entity_type', 'entity_id', 'risk_reason']).head(20)
if not priority_df.empty:
    priority_df['priority_rank'] = range(1, len(priority_df) + 1)
priority_df.to_csv(OUT_REV / 'swmm_priority_spotcheck_list.csv', index=False)

# pipeline state
state = {
    f"- Model counts: junctions={len(junctions)}, conduits={len(conduits)}, inflows={len(inflows_lines)}",
    '',
    '## Key Runtime Notes',
    '- Phase 2A inflow preflight completed; see outputs/qa/phase2a_inflow_preflight.md.',
    '- Static checks only (no SWMM engine execution in Codex environment).',
    '',
    '## Top Findings',
]
for _, r in findings_df.head(15).iterrows():
    summary.append(f"- [{r['severity']}] {r['check_name']} | {r['entity_type']} {r['entity_id']} | {r['message']}")
(OUT_QA / 'phase2_runtime_qaqc_summary.md').write_text('\n'.join(summary) + '\n', encoding='utf-8')

# GUI checklist
checklist = [
    '# SWMM GUI QA/QC Checklist (Phase 2 Runtime)',
    '',
    '## 1) Simulation run controls',
    '- Open models/model.inp in SWMM GUI and run with dynamic wave settings from [OPTIONS].',
    '- Confirm no fatal continuity/stability errors in Status Report.',
    '',
    '## 2) Continuity / stability review',
    '- Check flow routing continuity error (%).',
    '- Check highest node surcharge/flooding summary tables.',
    '- Check conduit instability index / Courant-related warnings.',
    '',
    '## 3) Inflow-focused checks',
    '- Verify duplicate inflow nodes J086, J101, J114 do not double-count intended hydrograph/steady inflow.',
    '- Confirm unmapped HYDROLOGY rows 687 and 688 are non-input trailing records and not intended inflows.',
    '',
    '## 4) Spot-check vs source workbook/PDF',
    '- Compare top priority links (length/diameter/slope) against HYDRAULICS tab values.',
    '- Compare non-zero inflow nodes against HYDROLOGY sheet rows for magnitude sanity.',
    '',
    '## 5) Acceptance criteria',
    '- No runtime-fatal errors.',
    '- Continuity/stability warnings understood and documented.',
    '- Priority spot-check discrepancies either fixed or accepted with engineering note.',
]
(OUT_REV / 'swmm_gui_qaqc_checklist.md').write_text('\n'.join(checklist) + '\n', encoding='utf-8')

# priority spot-check list (top 20)
priority_rows = []
# duplicate inflow nodes first
for n in sorted([n for n, c in Counter([r[0] for r in inflows_lines]).items() if c > 1]):
    priority_rows.append({'priority_rank': len(priority_rows)+1, 'entity_type': 'node', 'entity_id': n, 'risk_reason': 'duplicate_inflow_entries', 'source': 'model [INFLOWS]'})
# non-zero inflow nodes
for r in inflows_lines:
    if len(r) >= 3:
        try:
            q = float(r[2])
        except Exception:
            q = 0.0
        if q > 0:
            priority_rows.append({'priority_rank': len(priority_rows)+1, 'entity_type': 'node', 'entity_id': r[0], 'risk_reason': f'non_zero_inflow_{q:.4f}_cfs', 'source': 'model [INFLOWS]'})
# longest conduits
if not links.empty and {'pipe_id', 'length', 'source_sheet', 'source_row'}.issubset(links.columns):
    l2 = links.copy()
    l2['length_num'] = pd.to_numeric(l2['length'], errors='coerce')
    l2 = l2.sort_values('length_num', ascending=False).head(30)
    for _, row in l2.iterrows():
        priority_rows.append({'priority_rank': len(priority_rows)+1, 'entity_type': 'link', 'entity_id': str(row.get('pipe_id')), 'risk_reason': f"long_conduit_{row.get('length_num')}", 'source': f"{row.get('source_sheet')} row {row.get('source_row')}"})

priority_df = pd.DataFrame(priority_rows).drop_duplicates(subset=['entity_type', 'entity_id', 'risk_reason']).head(20).copy()
priority_df['priority_rank'] = range(1, len(priority_df) + 1)
priority_df.to_csv(OUT_REV / 'swmm_priority_spotcheck_list.csv', index=False)

# update pipeline state
pipeline_state = {
    'generated_at_utc': now_utc(),
    'current_stage': 'phase2_runtime_qaqc',
    'status': 'blocked' if high > 0 else 'ready',
    'key_metrics': {
        'phase2_required_sections_present': metrics['required_sections_present_count'],
        'phase2_required_sections_total': metrics['required_sections_total'],
        'phase2_high_count': high,
        'phase2_medium_count': med,
        'phase2_low_count': low,
        'phase2_qa_status': status,
        'phase2_inflow_preflight_rows': int(len(pre_df)),
    },
    'blockers': fdf[fdf['severity'] == 'HIGH']['check_name'].tolist(),
    'next_action': 'fix_runtime_blockers' if high > 0 else 'swmm_gui_manual_qaqc',
}
(OUT_LOG / 'pipeline_state.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
        'phase2_inflow_preflight_rows': int(len(preflight_df)),
    },
    'blockers': findings_df[findings_df['severity'] == 'HIGH']['check_name'].tolist(),
    'next_action': 'fix_runtime_blockers' if high > 0 else 'swmm_gui_manual_qaqc',
}
(OUT_LOG / 'pipeline_state.json').write_text(json.dumps(pipeline_state, indent=2), encoding='utf-8')

print(json.dumps({'phase2_status': status, 'high': high, 'medium': med, 'low': low}, indent=2))
