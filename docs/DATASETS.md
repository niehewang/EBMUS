# Dataset usage and licensing

## FreeInference Agentic Trace

Source: `harvardMadsys/freeinference_agentic_trace`.

The frozen source dataset is published under the Creative Commons Attribution 4.0 International License (CC BY 4.0). The code downloads the four frozen files needed from each weekly release partition and validates every file against the committed SHA256 map. Raw files are kept locally under `experiments/paper_scale/data/` and are not committed.

## UW-SyFI TraceLab

Source: `UW-SyFI/TraceLab`.

The public TraceLab dataset is published under CC BY 4.0. The upstream TraceLab software is licensed separately under Apache License 2.0. This repository uses only the frozen rounds parquet required for the controlled masking benchmark. Schema preflight verifies the grouping/provider/model fields and the token partition identity before enabling the cross-dataset analysis. No natural-missingness claim is made for TraceLab.

## Redistribution boundary

This repository does not redistribute either raw dataset. Users should obtain the frozen inputs from the upstream sources and comply with the corresponding attribution and license terms. Committed EBMUS outputs are aggregate or synthetic reference artifacts rather than copies of the raw third-party traces.
