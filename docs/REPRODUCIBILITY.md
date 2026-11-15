# Reproducibility notes

## Frozen sources

### FreeInference
- Dataset: `harvardMadsys/freeinference_agentic_trace`
- Schema/code identity used in the study: `2b795dbf5eac490a7c081906fdce6cc7a184b7ef`
- Release: 16 weekly partitions from `2026-05-17_2026-05-23` through `2026-08-30_2026-09-05`
- Exact file hashes: `experiments/paper_scale/freeinference_frozen_hashes.json`

### TraceLab
- Dataset: `UW-SyFI/TraceLab`
- Revision: `7256bbf77e76d5cb83f4ddbd5ad31d8faee9dc7a`
- File: `data/v0.0.2/rounds/train.parquet`
- SHA256: `cb8c2b1d23307090c8057c747f955663bacde16a49f84a3c5071a1e2e96a7ab8`
- Frozen schema reference: `experiments/paper_scale/configs/syfi_schema_preflight_reference.json`

## Deterministic protocol

- Session/workflow split seed: `20261109`
- Train/validation/test target: `60/20/20`
- Mask seeds: `23, 47, 71, 101, 131`
- Masking fractions: `0.10, 0.20, 0.30`
- Masking mechanisms: MCAR, contiguous session block, context-size-biased
- Multiple-imputation seeds: `1001` through `1020`
- Bootstrap seed: `20261109`

The exact split-ID lists and mask arrays are deliberately not committed because they reproduce deterministically from frozen public inputs and contain pseudonymous trace identifiers. The scripts regenerate them locally before any method evaluation.

## Output separation

Runtime outputs are written under `experiments/paper_scale/` and ignored by Git:

- `data/` — downloaded public data
- `cache/` — prepared local tables containing trace identifiers
- `data_contracts/splits_*.json` — exact local split membership
- `masks/` — exact saved mask arrays
- `raw_metrics/` — session-level metrics
- `paper_facing/` — regenerated aggregate outputs
- `mechanical_checks/` — regenerated integrity checks

The committed `results/` directory contains only aggregate or fully synthetic reference outputs.
