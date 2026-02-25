# SWMM GUI QA/QC Checklist (Hydraulic Baseline)

- Confirm each BASIN system discharges to its own O_* outfall.
- Confirm inlet nodes (IN_*) exist and receive static [INFLOWS].
- Verify no self-loop conduits and no cross-system links.
- Run SWMM and review continuity and stability reports.
- Spot-check top risk links/nodes against HYDROLOGY/DI TABLE source rows in crosswalk.
- Verify inlet rim elevation assumption (rim = receiving junction rim) is acceptable.
