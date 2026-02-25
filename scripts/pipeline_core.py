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
from collections import defaultdict

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

    workbook_path = ROOT / "data/raw/Copy of BX-HH-Vista Ranch_10-30-2025.xlsm"
    if not workbook_path.exists():
        raise FileNotFoundError("Workbook missing")

    wb = openpyxl.load_workbook(workbook_path, data_only=True, read_only=True)
    hyd = wb["HYDROLOGY"]
    di = wb["DI TABLE"]

    roughness = float(ctx.defaults_cfg["hydraulics"].get("manning_n_pipe_default", 0.013))

    def norm_id(v: object) -> str | None:
        if v is None:
            return None
        t = str(v).strip()
        if not t or t.lower() in {"nan", "none"}:
            return None
        m = re.search(r"[-+]?\d+(?:\.\d+)?", t)
        if not m:
            return None
        x = float(m.group(0))
        return str(int(x)) if x.is_integer() else str(x).rstrip("0").rstrip(".")

    def as_float(v: object) -> float | None:
        try:
            f = float(str(v).strip())
            return f if math.isfinite(f) else None
        except Exception:
            return None

    def j_name(v: str) -> str:
        return f"J_{v.replace('.', '_')}"

    def in_name(v: str) -> str:
        return f"IN_{v.replace('.', '_')}"

    def out_name(v: str) -> str:
        return f"O_{v.replace('.', '_')}"

    # DI TABLE geometry for inlet->junction pipes (B=inlet id, G/H invert, I length)
    di_geom: dict[str, dict] = {}
    for r in di.iter_rows(min_row=1, max_col=9, values_only=True):
        inlet_id = norm_id(r[1] if len(r) > 1 else None)
        if not inlet_id:
            continue
        up_inv = as_float(r[6] if len(r) > 6 else None)
        dn_inv = as_float(r[7] if len(r) > 7 else None)
        length = as_float(r[8] if len(r) > 8 else None)
        di_geom[inlet_id] = {"up_inv": up_inv, "dn_inv": dn_inv, "length": length}

    # Parse HYDROLOGY with BASIN segmentation
    basins: list[dict] = []
    cur = {"index": 1, "junction_rows": [], "inlet_rows": []}
    for row_idx, row in enumerate(hyd.iter_rows(min_row=1, max_col=29, values_only=True), start=1):
        b = row[1] if len(row) > 1 else None  # col B
        c = row[2] if len(row) > 2 else None  # col C
        d = row[3] if len(row) > 3 else None  # col D
        r = row[17] if len(row) > 17 else None  # col R
        s_col = row[18] if len(row) > 18 else None  # col S
        v_col = row[21] if len(row) > 21 else None  # col V
        aa = row[26] if len(row) > 26 else None  # col AA
        ac = row[28] if len(row) > 28 else None  # col AC

        b_text = str(b).strip() if b is not None else ""
        if b_text.upper() == "BASIN":
            if cur["junction_rows"] or cur["inlet_rows"]:
                cur["basin_break_row"] = row_idx
                basins.append(cur)
            cur = {"index": cur["index"] + 1, "junction_rows": [], "inlet_rows": []}
            continue

        jct_id = norm_id(b)
        inlet_id = norm_id(c)  # parser guard: only col C for inlet IDs
        known_q = as_float(r)
        dia_in = as_float(s_col)
        length_ft = as_float(v_col)
        rim = as_float(aa)
        soffit = as_float(ac)
        invert = soffit - (dia_in / 12.0) if soffit is not None and dia_in is not None else None

        rec = {
            "row_idx": row_idx,
            "jct_id": jct_id,
            "inlet_id": inlet_id,
            "known_q_cfs": known_q,
            "dia_in": dia_in,
            "length_ft": length_ft,
            "rim": rim,
            "soffit": soffit,
            "invert": invert,
            "col_d_value": d,
        }

        is_junction_candidate = bool(jct_id) and any(v is not None for v in (dia_in, length_ft, rim, soffit))
        is_inlet_candidate = bool(inlet_id) and not bool(jct_id)

        if is_junction_candidate:
            cur["junction_rows"].append(rec)
        elif is_inlet_candidate:
            cur["inlet_rows"].append(rec)
    if cur["junction_rows"] or cur["inlet_rows"]:
        basins.append(cur)

    # Build nodes / conduits / inflows with source-traceable IDs
    nodes: dict[str, dict] = {}
    outfalls: dict[str, dict] = {}
    conduits: list[dict] = []
    inflow_by_node: defaultdict[str, float] = defaultdict(float)
    findings: list[dict] = []
    crosswalk: list[dict] = []

    def add_finding(sev: str, check: str, msg: str, source_row: int | None, entity: str) -> None:
        findings.append({
            "severity": sev,
            "check_name": check,
            "message": msg,
            "source_row": source_row,
            "entity": entity,
        })

    seen_cids: set[str] = set()

    for basin in basins:
        jrows = basin["junction_rows"]
        if not jrows:
            continue

        # create junction nodes
        for jr in jrows:
            jid = j_name(jr["jct_id"])
            if jid in nodes:
                continue
            if jr["invert"] is None or jr["rim"] is None:
                add_finding("HIGH", "junction_missing_elevation", "Missing junction invert/rim from HYDROLOGY AC/AA.", jr["row_idx"], jid)
                continue
            max_depth = max(jr["rim"] - jr["invert"], 0.01)
            nodes[jid] = {
                "elev": jr["invert"],
                "max_depth": max_depth,
                "source_type": "junction",
                "source_row": jr["row_idx"],
                "source_id": jr["jct_id"],
                "rim": jr["rim"],
            }
            crosswalk.append({"object_type": "node", "swmm_id": jid, "source_tab": "HYDROLOGY", "source_row": jr["row_idx"], "source_id": jr["jct_id"], "notes": "junction from HYDROLOGY B"})

        # inlet rows are attached to next downstream junction in same basin
        pending_inlets = list(basin["inlet_rows"])
        for inlet in pending_inlets:
            recv = None
            for jr in jrows:
                if jr["row_idx"] > inlet["row_idx"]:
                    recv = jr
                    break
            if recv is None:
                recv = jrows[-1]
            inlet_swmm = in_name(inlet["inlet_id"])
            recv_swmm = j_name(recv["jct_id"])
            recv_node = nodes.get(recv_swmm)
            if recv_node is None:
                add_finding("HIGH", "inlet_receiving_junction_missing", "Receiving junction missing for inlet row.", inlet["row_idx"], inlet_swmm)
                continue

            di_rec = di_geom.get(inlet["inlet_id"])
            up_inv = di_rec.get("up_inv") if di_rec else None
            dn_inv = di_rec.get("dn_inv") if di_rec else None
            length_ft = di_rec.get("length") if di_rec else None
            if up_inv is None or dn_inv is None or length_ft is None:
                add_finding("HIGH", "inlet_di_geometry_missing", "Missing DI TABLE geometry for inlet->junction conduit.", inlet["row_idx"], inlet_swmm)
                continue
            if inlet_swmm not in nodes:
                rim_assumed = recv_node["rim"]
                max_depth = max(rim_assumed - up_inv, 0.01)
                nodes[inlet_swmm] = {
                    "elev": up_inv,
                    "max_depth": max_depth,
                    "source_type": "inlet",
                    "source_row": inlet["row_idx"],
                    "source_id": inlet["inlet_id"],
                    "rim": rim_assumed,
                }
                crosswalk.append({"object_type": "node", "swmm_id": inlet_swmm, "source_tab": "HYDROLOGY/DI TABLE", "source_row": inlet["row_idx"], "source_id": inlet["inlet_id"], "notes": "inlet node; rim assumed equal to receiving junction rim"})

            cid = f"L_{inlet_swmm}__{recv_swmm}"
            if cid not in seen_cids:
                seen_cids.add(cid)
                conduits.append({
                    "cid": cid,
                    "us": inlet_swmm,
                    "ds": recv_swmm,
                    "length": max(length_ft, 0.1),
                    "dia_ft": max((12.0 / 12.0), 0.5),
                    "source_row": inlet["row_idx"],
                    "source_tab": "DI TABLE",
                })
                crosswalk.append({"object_type": "conduit", "swmm_id": cid, "source_tab": "DI TABLE", "source_row": inlet["row_idx"], "source_id": inlet["inlet_id"], "notes": "inlet->junction conduit"})

            if inlet.get("known_q_cfs") is not None and inlet["known_q_cfs"] > 0:
                inflow_by_node[inlet_swmm] += inlet["known_q_cfs"]
                crosswalk.append({"object_type": "inflow", "swmm_id": inlet_swmm, "source_tab": "HYDROLOGY", "source_row": inlet["row_idx"], "source_id": inlet["inlet_id"], "notes": f"known flow cfs={inlet['known_q_cfs']}"})
            elif inlet.get("known_q_cfs") is None:
                add_finding("HIGH", "inlet_known_flow_missing", "Inlet known flow missing/invalid in HYDROLOGY col R.", inlet["row_idx"], inlet_swmm)

        # junction->downstream junction conduits from current row attributes
        for i, jr in enumerate(jrows[:-1]):
            dn = jrows[i + 1]
            us_swmm = j_name(jr["jct_id"])
            dn_swmm = j_name(dn["jct_id"])
            if us_swmm == dn_swmm:
                add_finding("HIGH", "junction_self_loop_candidate", "Self-loop prevented during build.", jr["row_idx"], us_swmm)
                continue
            if us_swmm not in nodes or dn_swmm not in nodes:
                add_finding("HIGH", "junction_conduit_endpoint_missing_node", "Skipped conduit because endpoint node missing due incomplete junction geometry.", jr["row_idx"], f"{us_swmm}->{dn_swmm}")
                continue
            if jr["length_ft"] is None or jr["dia_in"] is None:
                add_finding("HIGH", "junction_pipe_geometry_missing", "Missing S/V geometry for junction downstream conduit.", jr["row_idx"], us_swmm)
                continue
            cid = f"L_{us_swmm}__{dn_swmm}"
            if cid in seen_cids:
                continue
            seen_cids.add(cid)
            conduits.append({
                "cid": cid,
                "us": us_swmm,
                "ds": dn_swmm,
                "length": max(jr["length_ft"], 0.1),
                "dia_ft": max(jr["dia_in"] / 12.0, 0.5),
                "source_row": jr["row_idx"],
                "source_tab": "HYDROLOGY",
            })
            crosswalk.append({"object_type": "conduit", "swmm_id": cid, "source_tab": "HYDROLOGY", "source_row": jr["row_idx"], "source_id": jr["jct_id"], "notes": "junction->junction downstream conduit"})

        # terminal junction to basin-specific outfall
        terminal = jrows[-1]
        term_swmm = j_name(terminal["jct_id"])
        of_swmm = out_name(terminal["jct_id"])
        if term_swmm not in nodes:
            add_finding("HIGH", "terminal_junction_missing_node", "Skipped basin outfall link because terminal junction node missing geometry.", terminal["row_idx"], term_swmm)
            continue
        outfalls[of_swmm] = {"elev": nodes[term_swmm]["elev"], "source_row": terminal["row_idx"], "terminal": term_swmm}
        cid = f"L_{term_swmm}__{of_swmm}"
        if cid not in seen_cids:
            seen_cids.add(cid)
            conduits.append({
                "cid": cid,
                "us": term_swmm,
                "ds": of_swmm,
                "length": max((terminal.get("length_ft") or 1.0), 0.1),
                "dia_ft": max(((terminal.get("dia_in") or 18.0) / 12.0), 0.5),
                "source_row": terminal["row_idx"],
                "source_tab": "HYDROLOGY",
            })
            crosswalk.append({"object_type": "conduit", "swmm_id": cid, "source_tab": "HYDROLOGY", "source_row": terminal["row_idx"], "source_id": terminal["jct_id"], "notes": "terminal junction->outfall per BASIN rule"})

    # remove any self-loops defensively
    conduits = [c for c in conduits if c["us"] != c["ds"]]

    # deterministic synthetic coordinates per basin chain
    coords: dict[str, tuple[float, float]] = {}
    ix = 0
    for b_idx, basin in enumerate(basins, start=1):
        y = float(-1000 * (b_idx - 1))
        x = 0.0
        for jr in basin.get("junction_rows", []):
            jid = j_name(jr["jct_id"])
            if jid in nodes:
                coords[jid] = (x, y)
                x += 200.0
        for ir in basin.get("inlet_rows", []):
            iid = in_name(ir["inlet_id"])
            if iid in nodes and iid not in coords:
                coords[iid] = (max(0.0, x - 150.0), y + (50.0 if (ix % 2 == 0) else -50.0))
                ix += 1
        if basin.get("junction_rows"):
            term = j_name(basin["junction_rows"][-1]["jct_id"])
            of = out_name(basin["junction_rows"][-1]["jct_id"])
            if term in coords:
                coords[of] = (coords[term][0] + 150.0, coords[term][1])

    # write traceability artifacts
    pd.DataFrame(crosswalk).to_csv(ROOT / "outputs/review/source_traceability_crosswalk.csv", index=False)
    pd.DataFrame(findings).to_csv(ROOT / "outputs/qa/inlet_knownflow_hydraulic_baseline_findings.csv", index=False)

    # model sections
    junction_rows = sorted(nodes.items(), key=lambda kv: kv[0])
    outfall_rows = sorted(outfalls.items(), key=lambda kv: kv[0])
    inflow_rows = sorted([(k, v) for k, v in inflow_by_node.items() if v > 0], key=lambda t: t[0])

    text: list[str] = []
    text.append("[TITLE]\n;; Inlet-known-flow hydraulic routing baseline")
    text.append(
        "\n[OPTIONS]\n"
        "FLOW_UNITS           CFS\n"
        "INFILTRATION         HORTON\n"
        "FLOW_ROUTING         DYNWAVE\n"
        "START_DATE           01/01/2025\n"
        "START_TIME           00:00:00\n"
        "REPORT_START_DATE    01/01/2025\n"
        "REPORT_START_TIME    00:00:00\n"
        "END_DATE             01/01/2025\n"
        "END_TIME             06:00:00\n"
        "WET_STEP             00:05:00\n"
        "DRY_STEP             01:00:00\n"
        "ROUTING_STEP         00:00:30\n"
        "ALLOW_PONDING        NO"
    )

    text.append("\n[JUNCTIONS]\n;;Name  Elevation  MaxDepth  InitDepth  SurDepth  Aponded")
    for nid, d in junction_rows:
        text.append(f"{nid}  {d['elev']:.3f}  {d['max_depth']:.3f}  0.000  0.000  0.000")

    text.append("\n[OUTFALLS]\n;;Name  Elevation  Type  Stage Data  Gated  Route To")
    for oid, d in outfall_rows:
        text.append(f"{oid}  {d['elev']:.3f}  FREE")

    text.append("\n[CONDUITS]\n;;Name  FromNode  ToNode  Length  Roughness  InOffset  OutOffset  InitFlow  MaxFlow")
    for c in conduits:
        text.append(f"{c['cid']}  {c['us']}  {c['ds']}  {c['length']:.2f}  {roughness:.3f}  0.00  0.00  0.00  0.00")

    text.append("\n[XSECTIONS]\n;;Link  Shape  Geom1  Geom2  Geom3  Geom4  Barrels")
    for c in conduits:
        text.append(f"{c['cid']}  CIRCULAR  {c['dia_ft']:.3f}  0.000  0.000  0.000  1")

    text.append("\n[LOSSES]\n;;Link  Kentry  Kexit  Kavg  FlapGate  Seepage")
    for c in conduits:
        text.append(f"{c['cid']}  0.0  0.0  0.0  NO  0.0")

    text.append("\n[INFLOWS]\n;;Node  Constituent  TimeSeries  Type  Mfactor  Sfactor  Baseline  Pattern")
    for n, q in inflow_rows:
        text.append(f'{n}  FLOW  ""  FLOW  1.0  1.0  {q:.4f}  ""')

    text.append("\n[COORDINATES]\n;;Node  X-Coord  Y-Coord")
    for nid, _d in junction_rows:
        x, y = coords.get(nid, (0.0, 0.0))
        text.append(f"{nid}  {x:.2f}  {y:.2f}")
    for oid, _d in outfall_rows:
        x, y = coords.get(oid, (0.0, 0.0))
        text.append(f"{oid}  {x:.2f}  {y:.2f}")

    # summary markdown requested
    summary_lines = [
        "# Inlet Known-Flow Hydraulic Baseline QA/QC Summary",
        "",
        "- Phase scope: hydraulic routing only (no subcatchments, no rational runoff).",
        "- Inflows sourced from HYDROLOGY Column R and applied only at inlet nodes.",
        "- Inlet rim elevation assumption: inlet rim = receiving junction rim (HYDROLOGY AA).",
        f"- Junction nodes: {len(junction_rows)}",
        f"- Outfalls: {len(outfall_rows)}",
        f"- Conduits: {len(conduits)}",
        f"- Inflow nodes: {len(inflow_rows)}",
        f"- QA findings (HIGH): {sum(1 for f in findings if f['severity']=='HIGH')}",
        f"- QA findings (MEDIUM/LOW): {sum(1 for f in findings if f['severity']!='HIGH')}",
    ]
    (ROOT / "outputs/qa/inlet_knownflow_hydraulic_baseline_qaqc_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    metadata = {
        "junction_count": len(junction_rows),
        "outfall_count": len(outfall_rows),
        "conduit_count": len(conduits),
        "subcatchment_count": 0,
        "inflow_count": len(inflow_rows),
        "assumptions": {
            "topology_mode": "inlet_aware_hydraulic_baseline",
            "outfall_mode": "basin_segmented",
            "subcatchments_included": False,
            "inflow_mode": "static_known_flows_at_inlets",
            "inlet_nodes_modeled_as": "junctions",
            "coordinates_mode": "deterministic_synthetic",
            "column_d_used_as_inlet_id_count": 0,
        },
    }
    return "\n".join(text) + "\n", metadata


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
