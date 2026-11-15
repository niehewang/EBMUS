#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


errors = []

a0 = load("results/a0/summary.json")
if a0["subtest_A"]["mismatch_count"] != 0:
    errors.append("A0 subtest_A mismatch_count != 0")
if a0["subtest_B"]["mismatch_count"] != 0:
    errors.append("A0 subtest_B mismatch_count != 0")
if a0["subtest_C"]["mismatch_count"] != 0:
    errors.append("A0 subtest_C mismatch_count != 0")
if a0["subtest_D"]["truth_containment_violation_count"] != 0:
    errors.append("A0 truth-containment violations != 0")

agreement = load("results/paper_scale/solver_agreement.json")
for key in (
    "bruteforce_vs_evidence_mismatches",
    "lp_vs_laminar_mismatches",
    "milp_vs_evidence_mismatches",
):
    if agreement.get(key) != 0:
        errors.append(f"{key} != 0")

for rel in (
    "results/paper_scale/truth_preservation_freeinference.json",
    "results/paper_scale/truth_preservation_syfi.json",
):
    obj = load(rel)
    if obj.get("match") is not True or obj.get("masked_copy_only") is not True:
        errors.append(f"truth-preservation failure: {rel}")

temporal = load("results/temporal_replication/heldout_aggregate.json")
if temporal.get("heldout_weeks") != 15:
    errors.append("heldout_weeks != 15")
if temporal.get("numeric_validity_all_weeks") is not True:
    errors.append("numeric_validity_all_weeks != true")
if temporal.get("known_primary_week_excluded") is not True:
    errors.append("known_primary_week_excluded != true")

if errors:
    print("FROZEN_EVIDENCE_VERIFY_FAIL")
    for e in errors:
        print("-", e)
    raise SystemExit(2)

print("FROZEN_EVIDENCE_VERIFY_PASS")
