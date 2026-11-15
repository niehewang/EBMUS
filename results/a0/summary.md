# EBMUS A0 Method Validation — Mechanical Summary

No scientific GO / MODIFY / NO-GO decision is made by the experiment execution session.

## Subtest A — normative sanity
- cases: 6
- mismatches: 0

## Subtest B — LaminarFlow exactness
- instances: 4000
- exactness/feasibility/witness mismatches: 0

## Subtest C — EvidenceMILP exactness
- instances: 1000
- exactness/feasibility/witness mismatches: 0
- ambiguity type counts: `{"correction_branch": 744, "duplicate_distinct": 745}`

## Subtest D — controlled non-degeneracy
- instances: 5400
- truth-containment violations: 0
- primary cell nonfinal fraction: 1.0
- primary cell fraction with L<U and W_rel>=0.05: 0.935
- primary cell fraction with L<U and E_recv>=0.05: 0.995
- primary cell finite-U count: 200/200
- primary cell median W_rel among non-final: 0.13460287120257025

## Subtest E — scalability
- n=10,000 successful instances: 50/50
- n=10,000 median lower+upper solve seconds: 0.027471196837723255
- n=10,000 max process peak RSS MB: 20.29296875

## Boundary
This A0 validates exactness, controlled non-degeneracy, and CPU scalability only.
It does not establish real-world anomaly prevalence, monetary loss prevalence, or production superiority.
