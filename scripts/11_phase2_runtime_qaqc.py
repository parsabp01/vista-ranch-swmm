#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
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
    'generated_at_utc': now_utc(),
    'current_stage': 'phase2_runtime_qaqc',
    'status': 'blocked' if high > 0 else 'ready',
    'key_metrics': {
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

print(json.dumps({'phase2_status': status, 'high': high, 'medium': med, 'low': low}, indent=2))
