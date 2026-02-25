# SWMM GUI QA/QC Checklist (Hydraulic Baseline)

- Confirm each BASIN system discharges to its own O_* outfall.
- Confirm inlet nodes (IN_*) exist and receive static [INFLOWS].
- Verify no self-loop conduits and no cross-system links.
- Run SWMM and review continuity and stability reports.
- Spot-check top risk links/nodes against HYDROLOGY/DI TABLE source rows in crosswalk.
- Verify inlet rim elevation assumption (rim = receiving junction rim) is acceptable.
# SWMM GUI QA/QC Checklist (Phase 2 Runtime)

## 1) Simulation run controls
- Open models/model.inp in SWMM GUI and run with dynamic wave settings from [OPTIONS].
- Confirm no fatal continuity/stability errors in Status Report.

## 2) Continuity / stability review
- Check flow routing continuity error (%).
- Check highest node surcharge/flooding summary tables.
- Check conduit instability index / Courant-related warnings.

## 3) Inflow-focused checks
- Verify duplicate inflow nodes J086, J101, J114 do not double-count intended hydrograph/steady inflow.
- Confirm unmapped HYDROLOGY rows 687 and 688 are non-input trailing records and not intended inflows.

## 4) Spot-check vs source workbook/PDF
- Compare top priority links (length/diameter/slope) against HYDRAULICS tab values.
- Compare non-zero inflow nodes against HYDROLOGY sheet rows for magnitude sanity.

## 5) Acceptance criteria
- No runtime-fatal errors.
- Continuity/stability warnings understood and documented.
- Priority spot-check discrepancies either fixed or accepted with engineering note.
