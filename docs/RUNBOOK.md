# Build Mode Runbook

Run these scripts in order after creating config files:

1. `python scripts/00_inventory_inputs.py`
2. `python scripts/01_extract_excel.py`
3. `python scripts/02_extract_pdf_annotations.py`
4. `python scripts/03_normalize_entities.py`
5. `python scripts/04_build_network_graph.py`
6. `python scripts/05_generate_subcatchments.py`
7. `python scripts/06_build_swmm_inp.py`
8. `python scripts/07_run_qa_checks.py`
9. `python scripts/08_export_review_pack.py`

If QA fails, fix config and rerun from script 01.
