#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
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


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()


def parse_inp(path: Path) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    sec = None
    for line in path.read_text(encoding='utf-8').splitlines():
        t = line.strip()
        if not t or t.startswith(';;'):
            continue
        if t.startswith('[') and t.endswith(']'):
            sec = t
            sections.setdefault(sec, [])
            continue
        if sec:
            sections[sec].append(t)
    return sections


def add(rows: list[dict], sev: str, check: str, entity_type: str, entity_id: str, msg: str, src: str, prov: str, action: str, cls: str) -> None:
    rows.append({
        'severity': sev,
        'check_name': check,
        'entity_type': entity_type,
        'entity_id': entity_id,
        'message': msg,
        'source_dataset': src,
        'provenance': prov,
        'recommended_action': action,
        'runtime_classification': cls,
    })


sections = parse_inp(MODEL)
findings: list[dict] = []

required_sections = ['[OPTIONS]', '[JUNCTIONS]', '[OUTFALLS]', '[CONDUITS]', '[XSECTIONS]', '[LOSSES]', '[INFLOWS]', '[COORDINATES]']
for sec in required_sections:
    if sec not in sections:
        add(findings, 'HIGH', 'required_section_missing', 'model', sec, f'Missing section {sec}', 'models/model.inp', sec, 'emit required section in build stage', 'runtime_blocking')

junctions = [r.split()[0] for r in sections.get('[JUNCTIONS]', []) if r.split()]
outfalls = [r.split()[0] for r in sections.get('[OUTFALLS]', []) if r.split()]
conduits = [r.split() for r in sections.get('[CONDUITS]', [])]
inflows = [r.split() for r in sections.get('[INFLOWS]', [])]

# no self loops
self_loops = [c[0] for c in conduits if len(c) >= 3 and c[1] == c[2]]
if self_loops:
    add(findings, 'HIGH', 'self_loop_conduits', 'conduit', ','.join(self_loops[:10]), f'{len(self_loops)} self-loop conduits detected', 'models/model.inp', '[CONDUITS]', 'remove self-loops', 'runtime_blocking')

# inflow target validation
inflow_nodes = [r[0] for r in inflows if len(r) >= 1]
bad_targets = sorted({n for n in inflow_nodes if not n.startswith('IN_')})
if bad_targets:
    add(findings, 'HIGH', 'inflow_targets_not_inlets', 'inflow', ','.join(bad_targets[:10]), f'{len(bad_targets)} inflow targets are not IN_* nodes', 'models/model.inp', '[INFLOWS]', 'target only inlet nodes', 'runtime_blocking')

# duplicate inflow targets should be aggregated
dup_nodes = sorted([n for n, c in Counter(inflow_nodes).items() if c > 1])
if dup_nodes:
    add(findings, 'MEDIUM', 'duplicate_inflow_targets', 'inflow', ','.join(dup_nodes[:10]), f'{len(dup_nodes)} inlet nodes have duplicate inflow lines', 'models/model.inp', '[INFLOWS]', 'aggregate into one inflow per node', 'runtime_nonblocking_bookkeeping')

# no subcatchment objects in this phase
if sections.get('[SUBCATCHMENTS]'):
    add(findings, 'HIGH', 'subcatchments_present', 'model', 'SUBCATCHMENTS', 'Subcatchments present in hydraulic-only phase', 'models/model.inp', '[SUBCATCHMENTS]', 'remove runoff objects', 'runtime_blocking')

# optional cross-system check from topology map
jtop = read_csv(PROC / 'junction_topology_map.csv')
if not jtop.empty and 'crosses_basin_boundary' in jtop.columns:
    cross = int(jtop['crosses_basin_boundary'].fillna(False).sum())
    if cross > 0:
        add(findings, 'HIGH', 'cross_system_links_present', 'topology', 'junction_topology_map', f'{cross} cross-system links detected', 'data/processed/junction_topology_map.csv', 'crosses_basin_boundary', 'remove cross-basin links', 'runtime_blocking')

fdf = pd.DataFrame(findings)
if fdf.empty:
    fdf = pd.DataFrame(columns=['severity', 'check_name', 'entity_type', 'entity_id', 'message', 'source_dataset', 'provenance', 'recommended_action', 'runtime_classification'])
fdf.to_csv(OUT_QA / 'phase2_runtime_qaqc_findings.csv', index=False)

high = int((fdf['severity'] == 'HIGH').sum()) if not fdf.empty else 0
med = int((fdf['severity'] == 'MEDIUM').sum()) if not fdf.empty else 0
low = int((fdf['severity'] == 'LOW').sum()) if not fdf.empty else 0
status = 'blocked' if high > 0 else ('needs_review' if med > 0 else 'ready_for_swmm_gui')

metrics = {
    'generated_at_utc': now_utc(),
    'required_sections_total': len(required_sections),
    'required_sections_present_count': sum(1 for s in required_sections if s in sections),
    'junction_count': len(junctions),
    'outfall_count': len(outfalls),
    'conduit_count': len(conduits),
    'inflow_line_count': len(inflows),
    'duplicate_inflow_target_count': len(dup_nodes),
    'high_count': high,
    'medium_count': med,
    'low_count': low,
    'phase2_status': status,
}
(OUT_QA / 'phase2_runtime_qaqc_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')

summary = [
    '# Phase 2 Runtime QA/QC Summary',
    '',
    f"- Status: **{status}**",
    f"- Model counts: junctions={len(junctions)}, outfalls={len(outfalls)}, conduits={len(conduits)}, inflows={len(inflows)}",
    f"- Findings: HIGH={high}, MEDIUM={med}, LOW={low}",
    '',
    '## Top Findings',
]
for _, r in fdf.head(20).iterrows():
    summary.append(f"- [{r['severity']}] {r['check_name']} | {r['entity_type']} {r['entity_id']} | {r['message']}")
(OUT_QA / 'phase2_runtime_qaqc_summary.md').write_text('\n'.join(summary) + '\n', encoding='utf-8')

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

priority_rows = []
for i, n in enumerate(dup_nodes, start=1):
    priority_rows.append({'priority_rank': i, 'entity_type': 'node', 'entity_id': n, 'risk_reason': 'duplicate_inflow_target', 'source': 'model [INFLOWS]'})
for r in inflows:
    if len(r) >= 7:
        try:
            q = float(r[6])
        except Exception:
            q = 0.0
        if q > 0:
            priority_rows.append({'priority_rank': len(priority_rows) + 1, 'entity_type': 'node', 'entity_id': r[0], 'risk_reason': f'nonzero_inflow_{q:.4f}_cfs', 'source': 'model [INFLOWS]'})
priority_df = pd.DataFrame(priority_rows).drop_duplicates(subset=['entity_type', 'entity_id', 'risk_reason']).head(20)
if not priority_df.empty:
    priority_df['priority_rank'] = range(1, len(priority_df) + 1)
priority_df.to_csv(OUT_REV / 'swmm_priority_spotcheck_list.csv', index=False)

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
    },
    'blockers': fdf[fdf['severity'] == 'HIGH']['check_name'].tolist() if not fdf.empty else [],
    'next_action': 'fix_runtime_blockers' if high > 0 else 'swmm_gui_manual_qaqc',
}
(OUT_LOG / 'pipeline_state.json').write_text(json.dumps(pipeline_state, indent=2), encoding='utf-8')

print(json.dumps({'phase2_status': status, 'high': high, 'medium': med, 'low': low}, indent=2))
