#!/usr/bin/env python3
"""Core SWMM automation pipeline stages."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import openpyxl
import yaml

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class StageResult:
    stage: str
    status: str
    message: str
    produced_files: List[str]


class PipelineContext:
    def __init__(self) -> None:
        self.extraction_cfg = self._load_yaml(ROOT / "configs" / "extraction.yaml")
        self.defaults_cfg = self._load_yaml(ROOT / "configs" / "defaults.yaml")

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dirs() -> None:
    for rel in ["data/interim", "data/processed", "outputs/logs", "outputs/qa", "models"]:
        (ROOT / rel).mkdir(parents=True, exist_ok=True)


def required_paths(ctx: PipelineContext) -> Dict[str, List[Path] | Path]:
    req = ctx.extraction_cfg["required_inputs"]
    return {
        "workbook": ROOT / req["workbook"],
        "pdfs": [ROOT / p for p in req["pdfs"]],
        "planning_doc": ROOT / req["planning_doc"],
    }


def run_preflight(ctx: PipelineContext) -> StageResult:
    ensure_dirs()
    paths = required_paths(ctx)
    files_meta: List[dict] = []

    all_required: List[Tuple[str, Path]] = [
        ("workbook", paths["workbook"]),
        ("planning_doc", paths["planning_doc"]),
    ] + [("pdf", p) for p in paths["pdfs"]]

    blocked = False
    notes: List[str] = []
    for kind, p in all_required:
        exists = p.exists()
        if not exists:
            blocked = True
            notes.append(f"Missing required {kind}: {p.relative_to(ROOT)}")
        files_meta.append(
            {
                "kind": kind,
                "path": str(p.relative_to(ROOT)),
                "exists": exists,
                "size_bytes": p.stat().st_size if exists else None,
                "sha256": sha256_file(p) if exists else None,
                "modified_time_utc": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()
                if exists
                else None,
            }
        )

    status = "blocked" if blocked else "ready"
    manifest = {
        "generated_at_utc": now_utc(),
        "required_inputs": {
            "workbook": str(paths["workbook"].relative_to(ROOT)),
            "pdfs": [str(p.relative_to(ROOT)) for p in paths["pdfs"]],
            "planning_doc": str(paths["planning_doc"].relative_to(ROOT)),
        },
        "files": files_meta,
        "status": status,
        "notes": notes,
    }

    manifest_path = ROOT / "outputs/logs/intake_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return StageResult("preflight", status, "Required input check complete", [str(manifest_path.relative_to(ROOT))])


def normalize_header(value: object) -> str:
    return str(value or "").strip().lower().replace("\n", " ")


def row_to_dict(headers: List[str], row: Iterable[object]) -> dict:
    return {headers[i]: row[i] for i in range(min(len(headers), len(row))) if headers[i]}


def classify_record(record: dict, keywords: Dict[str, List[str]]) -> str | None:
    blob = " ".join(str(v).lower() for v in record.values() if v is not None)
    scores = {group: sum(1 for kw in kws if kw in blob) for group, kws in keywords.items()}
    best_group, best_score = max(scores.items(), key=lambda item: item[1])
    return best_group if best_score > 0 else None


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        val = float(value)
        return val if math.isfinite(val) else None
    text = str(value).strip()
    if text == "":
        return None
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None




def _extract_hydraulic_links_from_sheet(all_rows: List[tuple], sheet_name: str) -> List[dict]:
    links: List[dict] = []
    header_idx = None
    for i, row in enumerate(all_rows):
        row_txt = " ".join(str(v).lower() for v in row if v is not None)
        if row_txt.count("stream jct") >= 2:
            header_idx = i
            break
    if header_idx is None:
        return links

    # table starts after units/header rows; robustly scan following rows
    for r in range(header_idx + 1, len(all_rows)):
        row = all_rows[r]
        vals = [v for v in row]
        if len(vals) < 8:
            continue
        ds = vals[1] if len(vals) > 1 else None
        us = vals[2] if len(vals) > 2 else None
        dia = _to_float(vals[6] if len(vals) > 6 else None)
        q = _to_float(vals[7] if len(vals) > 7 else None)
        length = _to_float(vals[8] if len(vals) > 8 else None)
        slope = _to_float(vals[11] if len(vals) > 11 else None)
        hl = _to_float(vals[12] if len(vals) > 12 else None)
        jtype = str(vals[14]).strip() if len(vals) > 14 and vals[14] is not None else ""

        if us is None or ds is None:
            continue
        us_str = str(us).strip()
        ds_str = str(ds).strip()
        if not us_str or not ds_str:
            continue
        if _to_float(us_str) is None or _to_float(ds_str) is None:
            continue

        links.append(
            {
                "pipe_id": f"{us_str}-{ds_str}",
                "upstream_node": us_str,
                "downstream_node": ds_str,
                "dia": dia,
                "q": q,
                "length": length,
                "slope": slope,
                "headloss": hl,
                "junction_type": jtype,
                "source_sheet": sheet_name,
                "source_row": r + 1,
                "extraction_method": "hydraulics_structured_table",
            }
        )
    return links

def extract_workbook(ctx: PipelineContext) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    workbook_path = required_paths(ctx)["workbook"]
    wb = openpyxl.load_workbook(workbook_path, data_only=True, read_only=True)

    inventory_rows: List[dict] = []
    nodes_rows: List[dict] = []
    links_rows: List[dict] = []
    rational_rows: List[dict] = []

    keywords = ctx.extraction_cfg["classification_keywords"]
    scan_rows = int(ctx.extraction_cfg.get("candidate_header_scan_rows", 20))

    for sheet in wb.worksheets:
        all_rows = list(sheet.iter_rows(values_only=True))
        if not all_rows:
            continue

        max_row = len(all_rows)
        max_col = max(len(r) for r in all_rows)

        header_row_idx = 1
        best_count = -1
        best_headers: List[str] = []
        for i in range(1, min(max_row, scan_rows) + 1):
            vals = [normalize_header(v) for v in all_rows[i - 1]]
            count = sum(1 for v in vals if v)
            if count > best_count:
                best_count = count
                header_row_idx = i
                best_headers = vals

        inventory_rows.append(
            {
                "sheet_name": sheet.title,
                "rows": max_row,
                "cols": max_col,
                "detected_header_row": header_row_idx,
                "non_empty_headers": best_count,
                "headers_preview": " | ".join([h for h in best_headers if h][:12]),
            }
        )

        headers = [h if h else f"col_{idx+1}" for idx, h in enumerate(best_headers)]

        for offset, row in enumerate(all_rows[header_row_idx:], start=header_row_idx + 1):
            values = list(row)
            if not any(v is not None and str(v).strip() != "" for v in values):
                continue
            rec = row_to_dict(headers, values)
            rec["source_sheet"] = sheet.title
            rec["source_row"] = offset
            group = classify_record(rec, keywords)
            if group == "nodes":
                nodes_rows.append(rec)
            elif group == "links":
                links_rows.append(rec)
            elif group == "rational":
                rational_rows.append(rec)

        if sheet.title.strip().upper() == "HYDRAULICS":
            links_rows.extend(_extract_hydraulic_links_from_sheet(all_rows, sheet.title))

    return inventory_rows, nodes_rows, links_rows, rational_rows


def extract_pdf_metadata(ctx: PipelineContext) -> List[dict]:
    rows: List[dict] = []
    for p in required_paths(ctx)["pdfs"]:
        rows.append(
            {
                "path": str(p.relative_to(ROOT)),
                "exists": p.exists(),
                "size_bytes": p.stat().st_size if p.exists() else None,
                "sha256": sha256_file(p) if p.exists() else None,
                "note": "PDF parsing pass-through in current iteration",
            }
        )
    return rows


def run_extract(ctx: PipelineContext) -> StageResult:
    preflight = run_preflight(ctx)
    if preflight.status == "blocked":
        return StageResult("extract", "blocked", "Preflight blocked extraction due to missing inputs", preflight.produced_files)

    inventory, nodes, links, rational = extract_workbook(ctx)
    pdf_meta = extract_pdf_metadata(ctx)

    outputs = {
        "inventory": ROOT / "data/interim/workbook_sheet_inventory.csv",
        "nodes": ROOT / "data/interim/nodes_raw.csv",
        "links": ROOT / "data/interim/links_raw.csv",
        "rational": ROOT / "data/interim/rational_raw.csv",
        "pdf": ROOT / "data/interim/pdf_metadata.csv",
    }
    for k, path in outputs.items():
        if k == "inventory":
            write_csv(path, inventory)
        elif k == "nodes":
            write_csv(path, nodes)
        elif k == "links":
            write_csv(path, links)
        elif k == "rational":
            write_csv(path, rational)
        elif k == "pdf":
            write_csv(path, pdf_meta)

    summary = {
        "generated_at_utc": now_utc(),
        "counts": {
            "inventory_rows": len(inventory),
            "nodes_raw": len(nodes),
            "links_raw": len(links),
            "rational_raw": len(rational),
            "pdf_files": len(pdf_meta),
        },
    }
    summary_path = ROOT / "outputs/logs/extract_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return StageResult(
        "extract",
        "ready",
        "Extraction completed (workbook discovered programmatically; PDFs metadata captured)",
        [str(p.relative_to(ROOT)) for p in list(outputs.values()) + [summary_path]],
    )


def canonicalize_rows(rows: List[dict]) -> List[dict]:
    canon: List[dict] = []
    for r in rows:
        nr = {}
        for k, v in r.items():
            nk = k.strip().lower().replace(" ", "_").replace("/", "_")
            nr[nk] = v
        canon.append(nr)
    return canon


def run_transform(ctx: PipelineContext) -> StageResult:
    import pandas as pd

    interim = ROOT / "data/interim"
    processed = ROOT / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)

    sources = {
        "nodes": interim / "nodes_raw.csv",
        "links": interim / "links_raw.csv",
        "rational": interim / "rational_raw.csv",
    }
    missing = [k for k, p in sources.items() if not p.exists()]
    if missing:
        return StageResult("transform", "blocked", f"Missing interim files: {missing}", [])

    nodes_df = pd.read_csv(sources["nodes"]) if sources["nodes"].stat().st_size > 0 else pd.DataFrame()
    links_df = pd.read_csv(sources["links"]) if sources["links"].stat().st_size > 0 else pd.DataFrame()
    rational_df = pd.read_csv(sources["rational"]) if sources["rational"].stat().st_size > 0 else pd.DataFrame()

    nodes = canonicalize_rows(nodes_df.to_dict(orient="records"))
    links = canonicalize_rows(links_df.to_dict(orient="records"))
    rational = canonicalize_rows(rational_df.to_dict(orient="records"))

    tc_default = float(ctx.defaults_cfg["subcatchment"]["tc_length_default_ft"])
    min_width = float(ctx.defaults_cfg["subcatchment"]["min_width_ft"])
    imperv = float(ctx.defaults_cfg["subcatchment"]["percent_impervious_default"])
    slope = float(ctx.defaults_cfg["subcatchment"]["slope_percent_default"])

    subcatchment_defaults: List[dict] = []
    for rec in rational:
        area = _to_float(rec.get("area")) or _to_float(rec.get("area_ac"))
        q = _to_float(rec.get("q"))
        if area is None and q is not None and q > 0:
            area = max(0.05, min(5.0, q / 5.0))
        area = area or 0.0
        tc_len = _to_float(rec.get("tclength")) or _to_float(rec.get("tc_length")) or tc_default
        width = max((2.0 * area * 43560.0) / tc_len if tc_len > 0 else min_width, min_width)
        subcatchment_defaults.append(
            {
                "source_sheet": rec.get("source_sheet"),
                "source_row": rec.get("source_row"),
                "area_ac": round(area, 4),
                "tc_length_ft": round(tc_len, 3),
                "width_ft": round(width, 3),
                "slope_percent": slope,
                "percent_impervious": imperv,
            }
        )

    outputs = {
        "nodes": processed / "nodes.csv",
        "links": processed / "links.csv",
        "rational": processed / "rational_data.csv",
        "subs": processed / "subcatchment_defaults.csv",
    }
    write_csv(outputs["nodes"], nodes)
    write_csv(outputs["links"], links)
    write_csv(outputs["rational"], rational)
    write_csv(outputs["subs"], subcatchment_defaults)

    return StageResult("transform", "ready", "Transform completed", [str(p.relative_to(ROOT)) for p in outputs.values()])


def _build_runnable_model(ctx: PipelineContext) -> Tuple[str, dict]:
    import pandas as pd

    processed = ROOT / "data/processed"
    rational_path = processed / "rational_data.csv"
    subs_path = processed / "subcatchment_defaults.csv"
    links_path = processed / "links.csv"

    if not rational_path.exists() or not subs_path.exists() or not links_path.exists():
        raise FileNotFoundError("Missing processed inputs; run transform first")

    rational = pd.read_csv(rational_path) if rational_path.stat().st_size > 0 else pd.DataFrame()
    subs = pd.read_csv(subs_path) if subs_path.stat().st_size > 0 else pd.DataFrame()
    links = pd.read_csv(links_path) if links_path.stat().st_size > 0 else pd.DataFrame()

    roughness = float(ctx.defaults_cfg["hydraulics"]["manning_n_pipe_default"])
    def_len = float(ctx.defaults_cfg["hydraulics"]["conduit_length_default_ft"])

    required_cols = {"upstream_node", "downstream_node"}
    topology_mode = "extracted_links"
    if links.empty or not required_cols.issubset(set(links.columns)):
        topology_mode = "synthetic_chain"

    conduits = []
    xsections = []
    node_ids: List[str] = []
    outfall_id = "OUT1"

    if topology_mode == "extracted_links":
        ldf = links.copy()
        ldf = ldf[ldf["upstream_node"].astype(str).str.len() > 0]
        ldf = ldf[ldf["downstream_node"].astype(str).str.len() > 0]
        ldf = ldf.dropna(subset=["upstream_node", "downstream_node"])
        ldf = ldf.head(250).reset_index(drop=True)

        for i, row in ldf.iterrows():
            us = f"J{int(float(row['upstream_node'])):03d}" if _to_float(row['upstream_node']) is not None else f"J{i+1:03d}"
            ds = f"J{int(float(row['downstream_node'])):03d}" if _to_float(row['downstream_node']) is not None else f"J{i+2:03d}"
            length = _to_float(row.get("length"))
            diameter_in = _to_float(row.get("dia"))
            if length is None or not math.isfinite(length):
                length = def_len
            if diameter_in is None or not math.isfinite(diameter_in):
                diameter_in = 18.0
            length = max(length, 10.0)
            diameter_in = max(diameter_in, 6.0)
            geom_ft = diameter_in / 12.0
            cid = f"C{i+1:03d}"
            conduits.append((cid, us, ds, length, roughness, 0.0, 0.0, 0.0, 0.0))
            xsections.append((cid, "CIRCULAR", round(geom_ft, 3), 0.0, 0.0, 0.0, 1))
            node_ids.extend([us, ds])

        node_ids = sorted(set(node_ids))
        out_candidates = sorted({c[2] for c in conduits} - {c[1] for c in conduits})
        if out_candidates:
            outfall_id = out_candidates[0]
            conduits = [c if c[2] != outfall_id else (c[0], c[1], "OUT1", c[3], c[4], c[5], c[6], c[7], c[8]) for c in conduits]
            node_ids = [n for n in node_ids if n != outfall_id]
    else:
        q_vals = rational["q"].apply(_to_float) if "q" in rational.columns else pd.Series(dtype=float)
        q_rows = rational[q_vals.fillna(0) > 0].copy() if not q_vals.empty else rational.head(0).copy()
        if q_rows.empty:
            q_rows = rational.head(min(len(rational), 25)).copy()
            if q_rows.empty:
                q_rows = pd.DataFrame([{"source_row": 1, "source_sheet": "synthetic"}])
        q_rows = q_rows.head(min(len(q_rows), 120)).reset_index(drop=True)
        node_ids = [f"J{i+1:03d}" for i in range(len(q_rows))]
        for i, nid in enumerate(node_ids):
            to_node = "OUT1" if i == len(node_ids) - 1 else node_ids[i + 1]
            cid = f"C{i+1:03d}"
            conduits.append((cid, nid, to_node, def_len, roughness, 0.0, 0.0, 0.0, 0.0))
            xsections.append((cid, "CIRCULAR", 1.5, 0.0, 0.0, 0.0, 1))

    node_ids = sorted(set(node_ids))
    junctions = []
    coordinates = []
    for i, nid in enumerate(node_ids):
        base_elev = 100.0 - (0.2 * i)
        junctions.append((nid, base_elev, 8.0, 0.0, 0.0, 0.0))
        coordinates.append((nid, float(i) * 100.0, 1000.0 - float(i) * 8.0))
    coordinates.append(("OUT1", float(len(node_ids)) * 100.0 + 50.0, 900.0 - float(len(node_ids)) * 8.0))

    # map inflows/subcatchments to nodes using rational Q-positive rows
    q_vals = rational["q"].apply(_to_float) if "q" in rational.columns else pd.Series(dtype=float)
    q_rows = rational[q_vals.fillna(0) > 0].reset_index(drop=True) if not q_vals.empty else pd.DataFrame()
    if q_rows.empty:
        q_rows = rational.head(len(node_ids)).reset_index(drop=True)

    subcatchments = []
    subareas = []
    infiltration = []
    inflows = []

    for i, nid in enumerate(node_ids):
        src = q_rows.iloc[i] if i < len(q_rows) else pd.Series(dtype=object)
        area_ac = _to_float(subs.iloc[i]["area_ac"]) if i < len(subs) and "area_ac" in subs.columns else 0.25
        width = _to_float(subs.iloc[i]["width_ft"]) if i < len(subs) and "width_ft" in subs.columns else 150.0
        slope_pct = _to_float(subs.iloc[i]["slope_percent"]) if i < len(subs) and "slope_percent" in subs.columns else 1.0
        imperv = _to_float(subs.iloc[i]["percent_impervious"]) if i < len(subs) and "percent_impervious" in subs.columns else 55.0
        q = _to_float(src.get("q")) if not src.empty else None

        area_ac = max(area_ac or 0.25, 0.01)
        width = max(width or 10.0, 10.0)
        slope_pct = max(slope_pct or 1.0, 0.1)
        imperv = max(min(imperv or 55.0, 100.0), 0.0)

        sc_id = f"S{i+1:03d}"
        subcatchments.append((sc_id, "RG1", nid, area_ac, imperv, width, slope_pct, 0.0, ""))
        subareas.append((sc_id, 0.05, 0.10, 0.01, 0.1, 25.0, "OUTLET", ""))
        infiltration.append((sc_id, 3.0, 0.5, 4.0, 7.0, 0.0))
        inflows.append((nid, "FLOW", "", "", "", round(q or 0.0, 4), "", ""))

    model_text = []
    model_text.append("[TITLE]\n;; Auto-generated SWMM model (pipeline build stage)")
    model_text.append("\n[OPTIONS]\nFLOW_UNITS           CFS\nINFILTRATION         HORTON\nFLOW_ROUTING         DYNWAVE\nSTART_DATE           01/01/2025\nSTART_TIME           00:00:00\nREPORT_START_DATE    01/01/2025\nREPORT_START_TIME    00:00:00\nEND_DATE             01/01/2025\nEND_TIME             06:00:00\nWET_STEP             00:05:00\nDRY_STEP             01:00:00\nROUTING_STEP         00:00:30\nALLOW_PONDING        NO")
    model_text.append("\n[RAINGAGES]\n;;Name  Format  Interval  SCF  Source\nRG1    INTENSITY  0:05    1.0  TIMESERIES TS1")
    model_text.append("\n[TIMESERIES]\n;;Name Date Time Value\nTS1 01/01/2025 00:00 0.00\nTS1 01/01/2025 00:30 0.25\nTS1 01/01/2025 01:00 0.50\nTS1 01/01/2025 01:30 0.25\nTS1 01/01/2025 02:00 0.00")

    model_text.append("\n[JUNCTIONS]\n;;Name  Elevation  MaxDepth  InitDepth  SurDepth  Aponded")
    for r in junctions:
        model_text.append(f"{r[0]}  {r[1]:.3f}  {r[2]:.3f}  {r[3]:.3f}  {r[4]:.3f}  {r[5]:.3f}")

    model_text.append("\n[OUTFALLS]\n;;Name  Elevation  Type  Stage Data  Gated  Route To\nOUT1  95.000  FREE  ")

    model_text.append("\n[CONDUITS]\n;;Name  FromNode  ToNode  Length  Roughness  InOffset  OutOffset  InitFlow  MaxFlow")
    for r in conduits:
        model_text.append(f"{r[0]}  {r[1]}  {r[2]}  {r[3]:.2f}  {r[4]:.3f}  {r[5]:.2f}  {r[6]:.2f}  {r[7]:.2f}  {r[8]:.2f}")

    model_text.append("\n[XSECTIONS]\n;;Link  Shape  Geom1  Geom2  Geom3  Geom4  Barrels")
    for r in xsections:
        model_text.append(f"{r[0]}  {r[1]}  {r[2]:.3f}  {r[3]:.3f}  {r[4]:.3f}  {r[5]:.3f}  {r[6]}")

    model_text.append("\n[LOSSES]\n;;Link  Kentry  Kexit  Kavg  FlapGate  Seepage")
    for r in conduits:
        model_text.append(f"{r[0]}  0.0  0.0  0.0  NO  0.0")

    model_text.append("\n[SUBCATCHMENTS]\n;;Name  RainGage  Outlet  Area  %Imperv  Width  Slope  CurbLen  SnowPack")
    for r in subcatchments:
        model_text.append(f"{r[0]}  {r[1]}  {r[2]}  {r[3]:.4f}  {r[4]:.2f}  {r[5]:.2f}  {r[6]:.2f}  {r[7]:.2f}  {r[8]}")

    model_text.append("\n[SUBAREAS]\n;;Subcatchment  N-Imperv  N-Perv  S-Imperv  S-Perv  PctZero  RouteTo  PctRouted")
    for r in subareas:
        model_text.append(f"{r[0]}  {r[1]:.3f}  {r[2]:.3f}  {r[3]:.3f}  {r[4]:.3f}  {r[5]:.1f}  {r[6]}  {r[7]}")

    model_text.append("\n[INFILTRATION]\n;;Subcatchment  MaxRate  MinRate  Decay  DryTime  MaxInfil")
    for r in infiltration:
        model_text.append(f"{r[0]}  {r[1]:.3f}  {r[2]:.3f}  {r[3]:.3f}  {r[4]:.3f}  {r[5]:.3f}")

    model_text.append("\n[INFLOWS]\n;;Node  Constituent  TimeSeries  Type  Mfactor  Sfactor  Baseline  Pattern")
    for r in inflows:
        model_text.append(f"{r[0]}  {r[1]}  {r[2]}  {r[3]}  {r[4]}  {r[5]}  {r[6]}  {r[7]}")

    model_text.append("\n[COORDINATES]\n;;Node  X-Coord  Y-Coord")
    for r in coordinates:
        model_text.append(f"{r[0]}  {r[1]:.2f}  {r[2]:.2f}")

    metadata = {
        "junction_count": len(junctions),
        "conduit_count": len(conduits),
        "subcatchment_count": len(subcatchments),
        "inflow_count": len(inflows),
        "assumptions": {
            "topology_mode": topology_mode,
            "single_outfall": "OUT1",
            "rainfall_timeseries": "TS1 design storm placeholder",
        },
    }
    return "\n".join(model_text) + "\n", metadata


def run_build(ctx: PipelineContext) -> StageResult:
    processed = ROOT / "data/processed"
    if not (processed / "nodes.csv").exists() or not (processed / "links.csv").exists():
        return StageResult("build", "blocked", "Missing processed nodes/links", [])

    model_text, metadata = _build_runnable_model(ctx)
    model_path = ROOT / "models/model.inp"
    model_path.write_text(model_text, encoding="utf-8")

    meta_path = ROOT / "outputs/logs/build_summary.json"
    meta_path.write_text(json.dumps({"generated_at_utc": now_utc(), **metadata}, indent=2), encoding="utf-8")

    return StageResult("build", "ready", "Runnable SWMM model written", [str(model_path.relative_to(ROOT)), str(meta_path.relative_to(ROOT))])


def run_qa(ctx: PipelineContext) -> StageResult:
    import pandas as pd

    findings: List[dict] = []
    nodes_path = ROOT / "data/processed/nodes.csv"
    links_path = ROOT / "data/processed/links.csv"
    model_path = ROOT / "models/model.inp"

    if not nodes_path.exists() or not links_path.exists():
        findings.append({"severity": "high", "check": "processed_files", "message": "Missing processed nodes or links"})
    else:
        nodes_df = pd.read_csv(nodes_path) if nodes_path.stat().st_size > 0 else pd.DataFrame()
        links_df = pd.read_csv(links_path) if links_path.stat().st_size > 0 else pd.DataFrame()
        if nodes_df.empty:
            findings.append({"severity": "high", "check": "nodes_non_empty", "message": "nodes.csv is empty"})
        if links_df.empty:
            findings.append({"severity": "medium", "check": "links_non_empty", "message": "links.csv is empty; build uses synthetic chaining"})

    required_sections = [
        "[OPTIONS]",
        "[RAINGAGES]",
        "[TIMESERIES]",
        "[JUNCTIONS]",
        "[OUTFALLS]",
        "[CONDUITS]",
        "[XSECTIONS]",
        "[LOSSES]",
        "[SUBCATCHMENTS]",
        "[SUBAREAS]",
        "[INFILTRATION]",
        "[INFLOWS]",
        "[COORDINATES]",
    ]

    if not model_path.exists():
        findings.append({"severity": "high", "check": "model_exists", "message": "models/model.inp is missing"})
    else:
        model_text = model_path.read_text(encoding="utf-8")
        for section in required_sections:
            if section not in model_text:
                findings.append({"severity": "high", "check": "required_section", "message": f"Missing {section} in model.inp"})

    status = "ready" if not any(f["severity"] == "high" for f in findings) else "blocked"

    qa_dir = ROOT / "outputs/qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    findings_path = qa_dir / "qa_findings.csv"
    write_csv(findings_path, findings)

    summary = {
        "generated_at_utc": now_utc(),
        "status": status,
        "finding_count": len(findings),
        "high_count": sum(1 for f in findings if f.get("severity") == "high"),
    }
    summary_path = qa_dir / "qa_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return StageResult("qa", status, "QA complete", [str(findings_path.relative_to(ROOT)), str(summary_path.relative_to(ROOT))])


def run_stage(stage: str) -> StageResult:
    ctx = PipelineContext()
    if stage == "preflight":
        return run_preflight(ctx)
    if stage == "extract":
        return run_extract(ctx)
    if stage == "transform":
        return run_transform(ctx)
    if stage == "build":
        return run_build(ctx)
    if stage == "qa":
        return run_qa(ctx)
    raise ValueError(f"Unknown stage: {stage}")
