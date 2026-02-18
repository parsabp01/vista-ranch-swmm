#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/review"
OUT.mkdir(parents=True, exist_ok=True)


def _safe_float(v):
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    try:
        return float(str(v).strip())
    except Exception:
        return None


def main() -> None:
    processed = ROOT / "data/processed"
    model_path = ROOT / "models/model.inp"
    nodes_df = pd.read_csv(processed / "nodes.csv") if (processed / "nodes.csv").exists() and (processed / "nodes.csv").stat().st_size > 0 else pd.DataFrame()
    links_df = pd.read_csv(processed / "links.csv") if (processed / "links.csv").exists() and (processed / "links.csv").stat().st_size > 0 else pd.DataFrame()
    rational_df = pd.read_csv(processed / "rational_data.csv") if (processed / "rational_data.csv").exists() and (processed / "rational_data.csv").stat().st_size > 0 else pd.DataFrame()
    subs_df = pd.read_csv(processed / "subcatchment_defaults.csv") if (processed / "subcatchment_defaults.csv").exists() and (processed / "subcatchment_defaults.csv").stat().st_size > 0 else pd.DataFrame()

    swmm_nodes = []
    swmm_links = []
    if model_path.exists():
        sec = None
        for line in model_path.read_text(encoding="utf-8").splitlines():
            t = line.strip()
            if not t:
                continue
            if t.startswith('[') and t.endswith(']'):
                sec = t
                continue
            if t.startswith(';;'):
                continue
            parts = t.split()
            if sec == '[JUNCTIONS]' and parts:
                swmm_nodes.append(parts[0])
            elif sec == '[CONDUITS]' and len(parts) >= 3:
                swmm_links.append((parts[0], parts[1], parts[2]))

    q_vals = rational_df['q'].apply(_safe_float) if 'q' in rational_df.columns else pd.Series(dtype=float)
    q_rows = rational_df[q_vals.fillna(0) > 0].reset_index(drop=True) if not q_vals.empty else pd.DataFrame()
    if q_rows.empty:
        q_rows = rational_df.head(len(swmm_nodes)).reset_index(drop=True)

    node_rows = []
    for i, nid in enumerate(swmm_nodes):
        src = q_rows.iloc[i] if i < len(q_rows) else pd.Series(dtype=object)
        q = _safe_float(src.get('q')) if not src.empty else None
        area = _safe_float(subs_df.iloc[i].get('area_ac')) if i < len(subs_df) else None
        width = _safe_float(subs_df.iloc[i].get('width_ft')) if i < len(subs_df) else None
        slope = _safe_float(subs_df.iloc[i].get('slope_percent')) if i < len(subs_df) else None
        imperv = _safe_float(subs_df.iloc[i].get('percent_impervious')) if i < len(subs_df) else None
        node_rows.append({
            'swmm_node_id': nid,
            'excel_node_id': src.get('jct', '') if not src.empty else '',
            'pdf_label': '',
            'rim_elev': src.get('rim', '') if not src.empty else '',
            'invert_elev': src.get('elev', '') if not src.empty else '',
            'node_type': 'outfall' if nid.startswith('OUT') else 'junction',
            'assigned_inflow_q_cfs': q if q is not None else '',
            'inflow_source': f"{src.get('source_sheet','')}/{src.get('source_row','')}" if not src.empty else '',
            'assigned_subcatchment_area_ac': area if area is not None else '',
            'assigned_subcatchment_width_ft': width if width is not None else '',
            'assigned_subcatchment_slope_pct': slope if slope is not None else '',
            'assigned_subcatchment_pct_imperv': imperv if imperv is not None else '',
            'missing_flag': int(q is None),
            'derived_flag': int(src.empty or not str(src.get('jct','')).strip()),
            'notes': 'Derived mapping from rational row order; verify against plan labels',
        })

    link_rows = []
    for i, (cid, us, ds) in enumerate(swmm_links):
        src = links_df.iloc[i] if i < len(links_df) else pd.Series(dtype=object)
        length = _safe_float(src.get('length'))
        dia = _safe_float(src.get('dia'))
        slope = _safe_float(src.get('slope'))
        method = str(src.get("extraction_method", "")).strip() if not src.empty else ""
        inferred = src.empty
        if method == "hydraulics_structured_table":
            conf = 0.9
        elif not src.empty:
            conf = 0.6
        else:
            conf = 0.35
        link_rows.append({
            'swmm_conduit_id': cid,
            'excel_pipe_id': src.get('description', '') if not src.empty else '',
            'upstream_node': us,
            'downstream_node': ds,
            'diameter_in': dia if dia is not None else 18.0,
            'shape': 'CIRCULAR',
            'length_ft': length if length is not None else 50.0,
            'slope_ftft': slope if slope is not None else '',
            'invert_in_ft': src.get('inv_in', '') if not src.empty else '',
            'invert_out_ft': src.get('inv_out', '') if not src.empty else '',
            'losses': '0/0/0',
            'confidence': conf,
            'inferred_flag': int(inferred),
            'notes': 'Row-order mapped from extracted links when available',
        })

    node_x = pd.DataFrame(node_rows)
    link_x = pd.DataFrame(link_rows)
    node_x.to_csv(OUT / 'node_crosswalk.csv', index=False)
    link_x.to_csv(OUT / 'link_crosswalk.csv', index=False)

    coverage = {
        'nodes_matched_pct': round(100.0 * (1 - node_x['missing_flag'].mean()) if len(node_x) else 0.0, 2),
        'links_matched_pct': round(100.0 * ((link_x['confidence'] >= 0.7).mean()) if len(link_x) else 0.0, 2),
        'inflows_matched_pct': round(100.0 * (node_x['assigned_inflow_q_cfs'].astype(str).str.len() > 0).mean() if len(node_x) else 0.0, 2),
        'subcatchments_matched_pct': round(100.0 * (node_x['assigned_subcatchment_area_ac'].astype(str).str.len() > 0).mean() if len(node_x) else 0.0, 2),
        'counts': {
            'swmm_nodes': int(len(node_x)),
            'swmm_links': int(len(link_x)),
            'excel_nodes_rows': int(len(nodes_df)),
            'excel_links_rows': int(len(links_df)),
            'excel_rational_rows': int(len(rational_df)),
        },
    }
    (OUT / 'source_coverage.json').write_text(json.dumps(coverage, indent=2), encoding='utf-8')

    critical = []
    for _, r in node_x[node_x['derived_flag'] == 1].head(20).iterrows():
        critical.append(f"NODE {r['swmm_node_id']}: derived mapping; verify Excel/PDF ID alignment")
    for _, r in link_x[link_x['confidence'] < 0.7].head(20 - len(critical)).iterrows():
        critical.append(f"LINK {r['swmm_conduit_id']}: low confidence ({r['confidence']}) from limited link extraction")

    summary = [
        '# Review Summary',
        '',
        f"- Node crosswalk rows: {len(node_x)}",
        f"- Link crosswalk rows: {len(link_x)}",
        f"- Nodes with derived values: {int(node_x['derived_flag'].sum()) if len(node_x) else 0}",
        f"- Links inferred/low-confidence: {int((link_x['confidence'] < 0.7).sum()) if len(link_x) else 0}",
        f"- Synthetic/defaulted values present: {'YES' if (len(node_x) and node_x['derived_flag'].sum()>0) or (len(link_x) and (link_x['confidence']<0.7).sum()>0) else 'NO'}",
        '',
        '## Top 20 Critical Items',
    ]
    summary.extend([f"{i+1}. {item}" for i, item in enumerate(critical[:20])])
    (OUT / 'review_summary.md').write_text('\n'.join(summary) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
