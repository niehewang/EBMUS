# EBMUS

**Evidence-Bounded Model-Usage Settlement (EBMUS)** is research code for computing evidence-consistent settlement bounds and finality under incomplete AI/agent usage-accounting evidence.

The repository contains the frozen scientific implementation used for the paper experiments: the LaminarFlow structured exact solver, the scoped discrete-ambiguity solver, controlled exactness/scalability studies, real-trace masking comparisons, evidence-refinement and sensitivity experiments, natural FreeInference analysis, and quantitative figure regeneration.

## What is included

- `src/ebmus/core.py` — frozen EBMUS core implementation used by the method-validation study.
- `experiments/a0/` — exactness, truth-containment, non-degeneracy, and scalability validation.
- `experiments/paper_scale/` — controlled real-trace comparison, generic-vs-structured solver benchmark, evidence refinement, sensitivity, and natural-trace analysis.
- `results/` — **aggregate or synthetic** reference results used by the manuscript.
- `figures/` — exact-data quantitative figure generation code and the vendored IEEE/science Matplotlib style.

## What is intentionally not included

To minimize unnecessary disclosure, this public release does **not** contain raw third-party trace files, session-level raw metrics, exact session-ID split lists, saved mask arrays, machine hostnames, absolute server paths, API credentials, internal project-management documents, or manuscript source. These artifacts are either third-party data or operational records and are not needed in the repository.

The controlled splits and masks are deterministic and are regenerated from the frozen public sources using the committed seeds, so their omission does not change the scientific protocol.

## Environment

Tested scientific environment:

- Python 3.10
- NumPy 1.26.4
- pandas 2.2.3
- PyArrow 17.0.0
- SciPy 1.14.1
- scikit-learn 1.5.2
- psutil 6.1.0

Create an environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick validation

```bash
PYTHONPATH=src python experiments/a0/self_test_frozen.py
cd experiments/paper_scale
python scripts/self_test.py
```

## Reproduce the synthetic method validation

```bash
mkdir -p experiments/a0/run
PYTHONPATH=src python experiments/a0/run.py --run-dir experiments/a0/run
```

This runs the frozen normative tests, LaminarFlow/brute-force exactness checks, scoped discrete ambiguity exactness checks, controlled non-degeneracy study, and CPU scaling study.

## Reproduce the paper-scale real-trace experiments

The experiments use two public datasets:

1. **FreeInference Agentic Trace** — `harvardMadsys/freeinference_agentic_trace`. The repository records exact file SHA256 values for all 16 weekly partitions.
2. **TraceLab** — `UW-SyFI/TraceLab`, frozen file `data/v0.0.2/rounds/train.parquet` at revision `7256bbf77e76d5cb83f4ddbd5ad31d8faee9dc7a`, SHA256 `cb8c2b1d23307090c8057c747f955663bacde16a49f84a3c5071a1e2e96a7ab8`.

Run:

```bash
cd experiments/paper_scale
python run_all.py
```

By default acquisition uses `https://huggingface.co`. In network environments that require a compatible mirror, set for example:

```bash
export EBMUS_HF_ENDPOINT=https://hf-mirror.com
python run_all.py
```

If the FreeInference files already exist elsewhere, point the acquisition script at the directory containing the weekly folders:

```bash
export EBMUS_FI_DATA_ROOT=/path/to/freeinference/release
python run_all.py
```

Every downloaded FreeInference file is checked against the committed frozen SHA256 values before analysis. TraceLab is checked against the frozen revision/file SHA256. After acquisition, the formal scientific-analysis scripts run offline.

> The public datasets are not redistributed here. Please follow each upstream dataset's license and terms.

## Regenerate quantitative figures

The repository includes the exact aggregate CSV/JSON assets used by the quantitative figures.

```bash
python figures/generate_quantitative_figures.py
```

Generated PDFs are written to `figures/generated/`.

## Scientific boundary

The `1:10:50` cached-input : fresh-input : output vector used in the controlled real-trace experiments is a transparent **modeled reference settlement contract**, not a provider invoice or royalty tariff. The release does not claim actual billing loss, overcharge, undercharge, or model-owner royalty loss.

The scoped discrete solver is validated only for the explicit duplicate/distinct execution-attempt and mutually-exclusive correction ambiguity classes implemented in the code; it should not be read as a claim of arbitrary MILP tractability.

## Reproducibility and privacy

See `docs/REPRODUCIBILITY.md` and `docs/PRIVACY_AND_RELEASE.md`.
