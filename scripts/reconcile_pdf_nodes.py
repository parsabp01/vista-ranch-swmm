#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
INTERIM = ROOT / "data/interim"
PROCESSED = ROOT / "data/processed"
LOGS = ROOT / "outputs/logs"
REVIEW = ROOT / "outputs/review"
ASK = ROOT / "outputs/ask_user"

PDFS = [RAW / "BX-FM_10-30-2025-Model.pdf", RAW / "BX-FM_10-30-2025-Model2.pdf"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def update_state(stage: str, status: str, outputs: list[str], assumptions: dict[str, Any], next_action: str, reason: str = "") -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    inp = []
    for p in PDFS + [INTERIM / "pdf_words_raw.csv", INTERIM / "pdf_annots_raw.csv", PROCESSED / "nodes.csv", PROCESSED / "links.csv"]:
        inp.append({"path": str(p.relative_to(ROOT)), "exists": p.exists(), "sha256": sha(p)})
    counts = {}
    for rel in [
        "data/interim/pdf_words_raw.csv",
        "data/interim/pdf_annots_raw.csv",
        "data/processed/pdf_node_id_candidates.csv",
        "data/processed/pdf_elevation_candidates.csv",
        "data/processed/id_map_nodes.csv",
        "outputs/review/node_crosswalk.csv",
        "outputs/ask_user/ambiguous_nodes.csv",
    ]:
        p = ROOT / rel
        if p.exists() and p.suffix == ".csv":
            txt = p.read_text(encoding="utf-8").strip()
            counts[rel] = 0 if not txt else max(0, txt.count("\n"))
    payload = {
        "generated_at_utc": now(),
        "current_stage": stage,
        "status": status,
        "reason": reason,
        "inputs_used": inp,
        "outputs_produced": outputs,
        "key_counts": counts,
        "assumptions_used": assumptions,
        "next_action": next_action,
    }
    (LOGS / "pipeline_state.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def milestone1_extract() -> None:
    INTERIM.mkdir(parents=True, exist_ok=True)
    words_rows = []
    annot_rows = []
    summary = {"generated_at_utc": now(), "pdfs": [], "flattened": False}

    for pdf in PDFS:
        doc = fitz.open(pdf)
        pdf_info = {"pdf_file": pdf.name, "pages": []}
        for pi, page in enumerate(doc, start=1):
            words = page.get_text("words")
            ann_count = 0
            for w in words:
                if len(w) >= 5:
                    x0, y0, x1, y1, text = w[:5]
                    words_rows.append({
                        "pdf_file": pdf.name,
                        "page": pi,
                        "text": str(text),
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                    })
            annot = page.first_annot
            while annot:
                ann_count += 1
                info = annot.info or {}
                rect = annot.rect
                contents = ""
                try:
                    contents = annot.get_text("text") or ""
                except Exception:
                    contents = ""
                if not contents:
                    try:
                        contents = annot.get_textbox(rect) or ""
                    except Exception:
                        contents = ""
                if not contents:
                    contents = str(getattr(annot, "contents", "") or "")

                annot_rows.append({
                    "pdf_file": pdf.name,
                    "page": pi,
                    "annot_type": str(annot.type),
                    "rect_x0": rect.x0,
                    "rect_y0": rect.y0,
                    "rect_x1": rect.x1,
                    "rect_y1": rect.y1,
                    "contents": contents,
                    "info_title": info.get("title", ""),
                    "info_subject": info.get("subject", ""),
                    "info_content": info.get("content", ""),
                })
                annot = annot.next
            pdf_info["pages"].append({"page": pi, "word_count": len(words), "annot_count": ann_count})
        summary["pdfs"].append(pdf_info)

    words_df = pd.DataFrame(words_rows)
    ann_df = pd.DataFrame(annot_rows)
    words_df.to_csv(INTERIM / "pdf_words_raw.csv", index=False)
    ann_df.to_csv(INTERIM / "pdf_annots_raw.csv", index=False)

    page_stats = []
    for pdf_info in summary["pdfs"]:
        for p in pdf_info["pages"]:
            page_stats.append(p["word_count"] + p["annot_count"])
    avg_tokens = (sum(page_stats) / len(page_stats)) if page_stats else 0
    summary["avg_tokens_per_page"] = avg_tokens
    summary["flattened"] = avg_tokens < 10
    summary["top_examples"] = words_df["text"].head(25).tolist() if not words_df.empty else []
    (LOGS / "pdf_extract_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    update_state(
        stage="milestone_1_pdf_extract",
        status="ready",
        outputs=[
            "data/interim/pdf_words_raw.csv",
            "data/interim/pdf_annots_raw.csv",
            "outputs/logs/pdf_extract_summary.json",
        ],
        assumptions={"extract_method": "pymupdf words+annotations", "ocr_fallback_needed": summary["flattened"]},
        next_action="classify pdf labels",
    )


def _norm_label(s: str) -> str:
    s = (s or "").strip().upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s


def milestone2_classify() -> None:
    words = pd.read_csv(INTERIM / "pdf_words_raw.csv") if (INTERIM / "pdf_words_raw.csv").exists() else pd.DataFrame()
    ann = pd.read_csv(INTERIM / "pdf_annots_raw.csv") if (INTERIM / "pdf_annots_raw.csv").exists() else pd.DataFrame()
    rows = []
    if not words.empty:
        for _, r in words.iterrows():
            txt = str(r.get("text", "")).strip()
            if not txt:
                continue
            rows.append({
                "pdf_file": r["pdf_file"], "page": int(r["page"]), "label_raw": txt,
                "x0": float(r["x0"]), "y0": float(r["y0"]), "x1": float(r["x1"]), "y1": float(r["y1"]), "source": "word"
            })
    if not ann.empty:
        for _, r in ann.iterrows():
            txt = str(r.get("info_content", "")).strip() or str(r.get("contents", "")).strip()
            if not txt:
                continue
            rows.append({
                "pdf_file": r["pdf_file"], "page": int(r["page"]), "label_raw": txt,
                "x0": float(r["rect_x0"]), "y0": float(r["rect_y0"]), "x1": float(r["rect_x1"]), "y1": float(r["rect_y1"]), "source": "annot"
            })

    cands = []
    elevs = []
    diams = []
    stats = {"node_id_candidate": 0, "elevation_candidate": 0, "diameter_candidate": 0, "other": 0}
    for r in rows:
        txt = r["label_raw"]
        norm = _norm_label(txt)
        x = (r["x0"] + r["x1"]) / 2
        y = (r["y0"] + r["y1"]) / 2
        bucket = "other"
        rule = "none"
        conf = 0.2

        if re.fullmatch(r"\d{1,3}", txt):
            bucket = "node_id_candidate"; rule = "integer_1to3_digits"; conf = 0.75
        elif re.fullmatch(r"\d{1,3}\.\d", txt):
            bucket = "node_id_candidate"; rule = "decimal_node_branch_id"; conf = 0.7
        elif re.fullmatch(r"(?:MH|IN|CB|J)[-\s_]?\d{1,4}", txt, flags=re.I):
            bucket = "node_id_candidate"; rule = "prefixed_node_id"; conf = 0.95
        elif re.fullmatch(r"\d{2,3}\.\d{1,3}", txt):
            bucket = "elevation_candidate"; rule = "decimal_elevation"; conf = 0.8
        elif re.fullmatch(r"\d{1,2}(?:\.|\")", txt) or re.fullmatch(r"\d{1,2}\s*(?:IN|\")", txt, flags=re.I):
            bucket = "diameter_candidate"; rule = "diameter_inches"; conf = 0.85

        stats[bucket] += 1
        out = {
            "pdf_file": r["pdf_file"], "page": r["page"], "label_raw": txt, "label_norm": norm,
            "x_center": x, "y_center": y, "bbox": f"{r['x0']},{r['y0']},{r['x1']},{r['y1']}",
            "rule": rule, "confidence": conf,
        }
        if bucket == "node_id_candidate":
            cands.append(out)
        elif bucket == "elevation_candidate":
            elevs.append(out)
        elif bucket == "diameter_candidate":
            diams.append(out)

    pd.DataFrame(cands, columns=["pdf_file","page","label_raw","label_norm","x_center","y_center","bbox","rule","confidence"]).to_csv(PROCESSED / "pdf_node_id_candidates.csv", index=False)
    pd.DataFrame(elevs, columns=["pdf_file","page","label_raw","label_norm","x_center","y_center","bbox","rule","confidence"]).to_csv(PROCESSED / "pdf_elevation_candidates.csv", index=False)

    top = {
        "node_id_candidate": [c["label_raw"] for c in cands[:25]],
        "elevation_candidate": [c["label_raw"] for c in elevs[:25]],
        "diameter_candidate": [c["label_raw"] for c in diams[:25]],
    }
    (LOGS / "pdf_label_stats.json").write_text(json.dumps({"counts": stats, "top_examples": top}, indent=2), encoding="utf-8")

    update_state(
        stage="milestone_2_pdf_classify",
        status="ready",
        outputs=[
            "data/processed/pdf_node_id_candidates.csv",
            "data/processed/pdf_elevation_candidates.csv",
            "outputs/logs/pdf_label_stats.json",
        ],
        assumptions={"classification": "regex_rules_v1", "no_ocr": True},
        next_action="reconcile excel and pdf node ids",
    )


def _excel_node_pool() -> pd.DataFrame:
    nodes = pd.read_csv(PROCESSED / "nodes.csv") if (PROCESSED / "nodes.csv").exists() and (PROCESSED / "nodes.csv").stat().st_size > 0 else pd.DataFrame()
    # choose best available id-like column
    id_cols = [c for c in nodes.columns if c in ["jct", "inlet", "name", "node", "id"] or "jct" in c or "node" in c]
    if not id_cols:
        id_cols = [c for c in nodes.columns if nodes[c].astype(str).str.fullmatch(r"\d{1,4}").any()]
    rows = []
    for c in id_cols[:5]:
        for v in nodes[c].dropna().astype(str):
            t = v.strip()
            if not t:
                continue
            rows.append({"excel_id_raw": t, "excel_id_norm": _norm_label(t), "excel_num": re.sub(r"\D", "", t)})
    df = pd.DataFrame(rows).drop_duplicates()
    if df.empty:
        # fallback from links upstream/downstream
        links = pd.read_csv(PROCESSED / "links.csv") if (PROCESSED / "links.csv").exists() and (PROCESSED / "links.csv").stat().st_size > 0 else pd.DataFrame()
        for c in ["upstream_node", "downstream_node"]:
            if c in links.columns:
                for v in links[c].dropna().astype(str):
                    vv = v.strip()
                    if vv:
                        df.loc[len(df)] = {"excel_id_raw": vv, "excel_id_norm": _norm_label(vv), "excel_num": re.sub(r"\D", "", vv)}
    return df.drop_duplicates().reset_index(drop=True)


def milestone3_reconcile() -> None:
    pdf_nodes = pd.read_csv(PROCESSED / "pdf_node_id_candidates.csv") if (PROCESSED / "pdf_node_id_candidates.csv").exists() and (PROCESSED / "pdf_node_id_candidates.csv").stat().st_size > 0 else pd.DataFrame()
    pool = _excel_node_pool()

    maps = []
    for _, p in pdf_nodes.iterrows():
        raw = str(p["label_raw"]).strip()
        norm = _norm_label(raw)
        num = re.sub(r"\D", "", raw)
        # pass A exact
        ex = pool[pool["excel_id_norm"] == norm]
        method = ""
        conf = 0.0
        evidence = ""
        pick = None
        if not ex.empty:
            pick = ex.iloc[0]
            method = "A_exact_norm"
            conf = 0.98
            evidence = f"excel_norm={pick['excel_id_norm']}"
        else:
            ex2 = pool[(pool["excel_num"] == num) & (num != "")]
            if not ex2.empty:
                pick = ex2.iloc[0]
                method = "B_numeric_suffix"
                conf = 0.82
                evidence = f"excel_num={pick['excel_num']}"
            else:
                # pass C fuzzy by levenshtein-lite prefix overlap
                best = None
                best_score = 0.0
                for _, e in pool.iterrows():
                    a, b = norm, str(e["excel_id_norm"])
                    if not a or not b:
                        continue
                    common = len(set(a) & set(b))
                    score = common / max(len(set(a)), 1)
                    if score > best_score:
                        best_score = score
                        best = e
                if best is not None and best_score >= 0.6:
                    pick = best
                    method = "C_fuzzy_char_overlap"
                    conf = round(0.5 + 0.4 * best_score, 3)
                    evidence = f"char_overlap={best_score:.3f}"

        maps.append({
            "canonical_node_id": f"N_{norm}" if norm else "",
            "excel_id_raw": pick["excel_id_raw"] if pick is not None else "",
            "excel_id_norm": pick["excel_id_norm"] if pick is not None else "",
            "pdf_label_raw": raw,
            "pdf_label_norm": norm,
            "match_method": method,
            "confidence": conf,
            "evidence": evidence,
            "pdf_file": p["pdf_file"],
            "page": int(p["page"]),
            "x": float(p["x_center"]),
            "y": float(p["y_center"]),
        })

    idmap = pd.DataFrame(maps)
    idmap.to_csv(PROCESSED / "id_map_nodes.csv", index=False)

    # refresh node crosswalk
    cross = pd.read_csv(REVIEW / "node_crosswalk.csv") if (REVIEW / "node_crosswalk.csv").exists() else pd.DataFrame()
    if cross.empty:
        cross = pd.DataFrame(columns=["swmm_node_id", "excel_node_id", "pdf_label"])
    # build lookup by excel raw and numeric part
    look = idmap.sort_values("confidence", ascending=False).drop_duplicates("excel_id_raw")
    numlook = idmap.copy()
    numlook["excel_num"] = numlook["excel_id_raw"].astype(str).str.replace(r"\D", "", regex=True)

    out_rows = []
    for _, r in cross.iterrows():
        ex = str(r.get("excel_node_id", ""))
        m = look[look["excel_id_raw"].astype(str) == ex]
        if m.empty:
            exn = re.sub(r"\D", "", ex)
            m = numlook[numlook["excel_num"] == exn].sort_values("confidence", ascending=False)
        pick = m.iloc[0] if not m.empty else None
        rr = dict(r)
        rr["pdf_label"] = "" if pick is None else pick["pdf_label_raw"]
        rr["match_method"] = "" if pick is None else pick["match_method"]
        rr["confidence"] = "" if pick is None else pick["confidence"]
        out_rows.append(rr)

    pd.DataFrame(out_rows).to_csv(REVIEW / "node_crosswalk.csv", index=False)

    # coverage refresh
    cov = json.loads((REVIEW / "source_coverage.json").read_text()) if (REVIEW / "source_coverage.json").exists() else {}
    node_df = pd.DataFrame(out_rows)
    pdf_pop = (node_df["pdf_label"].astype(str).str.len() > 0).mean() * 100 if not node_df.empty else 0
    inflow_match = ((node_df.get("assigned_inflow_q_cfs", "").astype(str).str.len() > 0) & (node_df["pdf_label"].astype(str).str.len() > 0)).mean() * 100 if not node_df.empty else 0
    cov["nodes_matched_pct"] = round(pdf_pop, 2)
    cov["inflows_matched_pct"] = round(inflow_match, 2)
    cov["pdf_label_populated"] = bool(pdf_pop > 0)
    (REVIEW / "source_coverage.json").write_text(json.dumps(cov, indent=2), encoding="utf-8")

    update_state(
        stage="milestone_3_node_reconcile",
        status="ready",
        outputs=["data/processed/id_map_nodes.csv", "outputs/review/node_crosswalk.csv", "outputs/review/source_coverage.json"],
        assumptions={"matching_cascade": ["A_exact_norm", "B_numeric_suffix", "C_fuzzy_char_overlap"]},
        next_action="topology-assisted disambiguation and ask-user packet if needed",
    )


def milestone4_ask_user() -> None:
    ASK.mkdir(parents=True, exist_ok=True)
    idmap = pd.read_csv(PROCESSED / "id_map_nodes.csv") if (PROCESSED / "id_map_nodes.csv").exists() and (PROCESSED / "id_map_nodes.csv").stat().st_size > 0 else pd.DataFrame()
    amb = idmap[(idmap["confidence"].fillna(0) < 0.8) | (idmap["match_method"].astype(str).str.len() == 0)] if not idmap.empty else pd.DataFrame()

    # topology-assisted score using node numbers appearing in links
    links = pd.read_csv(PROCESSED / "links.csv") if (PROCESSED / "links.csv").exists() and (PROCESSED / "links.csv").stat().st_size > 0 else pd.DataFrame()
    link_nums = set()
    for c in ["upstream_node", "downstream_node"]:
        if c in links.columns:
            for v in links[c].dropna().astype(str):
                n = re.sub(r"\D", "", v)
                if n:
                    link_nums.add(n)
    if not amb.empty:
        amb = amb.copy()
        amb["pdf_num"] = amb["pdf_label_raw"].astype(str).str.replace(r"\D", "", regex=True)
        amb["topology_hint"] = amb["pdf_num"].apply(lambda n: "in_links" if n in link_nums else "not_in_links")
        amb["topology_score"] = amb["topology_hint"].map({"in_links": 0.15, "not_in_links": 0.0})
        amb["revised_confidence"] = (amb["confidence"].fillna(0) + amb["topology_score"]).clip(upper=0.99)

    unresolved = amb[amb.get("revised_confidence", amb.get("confidence", 0)).fillna(0) < 0.85] if not amb.empty else pd.DataFrame()
    unresolved.to_csv(ASK / "ambiguous_nodes.csv", index=False)

    qmd = [
        "# Node ID Ambiguity Questions",
        "",
        "Please resolve these mapping choices so the pipeline can continue without guessing.",
        "",
        "1) Confirm node ID prefix convention to apply to numeric PDF labels:",
        "   - A: Use `J<number>`",
        "   - B: Use `MH<number>`",
        "   - C: Use `IN<number>`",
        "",
        "2) For ambiguous rows in `outputs/ask_user/ambiguous_nodes.csv`, choose accepted Excel ID for each PDF label.",
        "",
        "3) If uncertain, approve default: numeric suffix match (PDF `17` -> Excel ID ending with `17`).",
    ]
    (ASK / "questions.md").write_text("\n".join(qmd) + "\n", encoding="utf-8")

    qjson = {
        "reason": "node_id_ambiguity",
        "question_count": 3,
        "ambiguous_count": int(len(unresolved)),
        "options": {"prefix": ["J", "MH", "IN"]},
        "files": {"details": "outputs/ask_user/ambiguous_nodes.csv", "answers_template": "outputs/ask_user/answers.yml"},
    }
    (ASK / "questions.json").write_text(json.dumps(qjson, indent=2), encoding="utf-8")

    answers_tpl = {"prefix_choice": "J", "manual_overrides": [{"pdf_label": "17", "excel_id": "J017"}]}
    (ASK / "answers.yml").write_text(yaml.safe_dump(answers_tpl, sort_keys=False), encoding="utf-8")

    status = "blocked" if len(unresolved) > 0 else "ready"
    update_state(
        stage="milestone_4_ask_user_gate",
        status=status,
        outputs=["outputs/ask_user/questions.md", "outputs/ask_user/questions.json", "outputs/ask_user/ambiguous_nodes.csv", "outputs/ask_user/answers.yml"],
        assumptions={"topology_assist": True, "unresolved_count": int(len(unresolved))},
        next_action="wait for answers.yml then rerun reconciliation" if status == "blocked" else "no ambiguity remains",
        reason="node_id_ambiguity" if status == "blocked" else "",
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--milestone", choices=["1", "2", "3", "4"], required=True)
    args = parser.parse_args()

    if args.milestone == "1":
        milestone1_extract()
    elif args.milestone == "2":
        milestone2_classify()
    elif args.milestone == "3":
        milestone3_reconcile()
    elif args.milestone == "4":
        milestone4_ask_user()


if __name__ == "__main__":
    main()
