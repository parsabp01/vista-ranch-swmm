# Duplicate/Conflict Evidence

This packet verifies whether remaining modeled=False inlet records are caused by real object collisions or by classification artifacts.

## IN_233
- source: HYDROLOGY row 671 (inlet_id=233, basin=7).
- expected receiving junction: J_185.
- duplicate check (node id): IN_233 -> already exists.
- duplicate check (conduit id): L_IN_233__J_185 -> already exists.
- duplicate check (inflow target): IN_233 -> already targeted.
- conflicting object IDs: IN_233, L_IN_233__J_185
- conflict provenance:
  - node IN_233 from HYDROLOGY/DI TABLE row 671 (source_id=233)
  - inflow IN_233 from HYDROLOGY row 671 (source_id=233)
  - conduit L_IN_233__J_185 from DI TABLE row 671 (source_id=233)
- collision verdict: ARTIFACT_OF_CONFLICT_DETECTION_LOGIC.
- current modeled status: modeled=True reason=MODELED_WITH_ASSUMPTION.

## IN_234
- source: HYDROLOGY row 676 (inlet_id=234, basin=8).
- expected receiving junction: J_186.
- duplicate check (node id): IN_234 -> already exists.
- duplicate check (conduit id): L_IN_234__J_186 -> already exists.
- duplicate check (inflow target): IN_234 -> already targeted.
- conflicting object IDs: IN_234, L_IN_234__J_186
- conflict provenance:
  - node IN_234 from HYDROLOGY/DI TABLE row 676 (source_id=234)
  - inflow IN_234 from HYDROLOGY row 676 (source_id=234)
  - conduit L_IN_234__J_186 from DI TABLE row 676 (source_id=234)
- collision verdict: ARTIFACT_OF_CONFLICT_DETECTION_LOGIC.
- current modeled status: modeled=True reason=MODELED_WITH_ASSUMPTION.

## IN_235
- source: HYDROLOGY row 681 (inlet_id=235, basin=9).
- expected receiving junction: J_187.
- duplicate check (node id): IN_235 -> already exists.
- duplicate check (conduit id): L_IN_235__J_187 -> already exists.
- duplicate check (inflow target): IN_235 -> already targeted.
- conflicting object IDs: IN_235, L_IN_235__J_187
- conflict provenance:
  - node IN_235 from HYDROLOGY/DI TABLE row 681 (source_id=235)
  - inflow IN_235 from HYDROLOGY row 681 (source_id=235)
  - conduit L_IN_235__J_187 from DI TABLE row 681 (source_id=235)
- collision verdict: ARTIFACT_OF_CONFLICT_DETECTION_LOGIC.
- current modeled status: modeled=True reason=MODELED_WITH_ASSUMPTION.

