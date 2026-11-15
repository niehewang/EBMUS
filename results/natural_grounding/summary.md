# EBMUS A2.13 Natural-Evidence Settlement Grounding — Mechanical Summary

This file reports mechanical validation results only; scientific interpretation is documented separately.

## Frozen source
- week: `2026-08-30_2026-09-05`
- all five frozen SHA256 match: `True`
- manifest/tables/parquet integrity chain: `True`

## Natural cache-reporting evidence
- total requests: 190544
- non-cache-reporting requests: 3325 (0.017450037787)
- child-session requests: 3700
- child non-cache-reporting requests: 210 (0.056756756757)
- Q_partition: 1.000000000000

## Week interval
- L_week: 57483.149830
- U_week: 61337.053906
- W_week: 3853.904076
- Wrel_week: 0.064869507959
- exact finality: False
- 1% tolerance finality: False

## Complete-case analytical baseline
- S_complete_case: 56974.727116
- lower_bound_gap: 0.008844726072

## Real workflow trees
- top-level workflows: 1806
- affected top-level workflows: 929
- nested/delegated workflows: 66
- affected nested/delegated workflows: 48
- affected nested median Wrel: 0.046965615133
- nested workflows failing 1% tolerance finality: 41

## Numeric validity
- invalid request intervals: 0
- negative P used: 0
- negative O used: 0
- all workflow intervals ordered: True

## Claim boundary
All values are under the frozen FreeInference reference modeled cost contract.
Do not interpret interval width as actual financial loss, billing error, over/undercharge, or model-owner royalty.
