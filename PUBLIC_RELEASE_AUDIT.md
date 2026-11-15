# Public Release Audit

Updated: 2026-11-15

This document records the public-code release audit for the EBMUS repository. It distinguishes the current default-branch tree from Git-history residuals.

## Current-tree scope

The public tree contains the scientific implementation, deterministic protocol parameters, public-source integrity contracts, aggregate or synthetic reference results, and figure-regeneration code needed for the reported experiments.

Deliberately excluded are raw third-party traces, exact pseudonymous split membership, saved row-level masks, session-level raw metrics, credentials, private infrastructure details, private operational records, and unpublished document source.

## G0–G12 status

| Gate | Status | Evidence |
| --- | --- | --- |
| G0 Scope | PASS | Public release is limited to scientific code, public-source provenance, disclosure-minimized outputs, and documentation. |
| G1 Secret | PASS (current tree) | Automated scan covers token/key patterns, bearer credentials, private-key headers, direct email addresses, and forbidden local environment files. |
| G2 Infrastructure | PASS (current tree) | Automated scan rejects absolute local paths and private Drive links; no current-tree infrastructure identifier was found. |
| G3 Venue anonymity | PASS (current tree) | Target-venue strings are rejected by the audit; public documentation is venue-neutral. |
| G4 Production firewall | PASS (current tree) | Internal workflow/state identifiers are rejected by the audit and were removed from public summaries. |
| G5 Data/license/privacy | PASS | Raw third-party datasets are not redistributed; upstream dataset licensing and the privacy boundary are documented. |
| G6 Repository quality | PASS_WITH_RIGHTS_NOTICE | README, CI, security guidance, reproducibility documentation, and rights status are present and factual. No repository-wide open-source license is claimed. |
| G7 Portability | PASS | Public-source acquisition supports the default endpoint, a compatible mirror endpoint, and pre-existing verified local data; frozen hashes are checked. |
| G8 Automated audit | PASS | CI compiles Python, runs both self-tests, verifies committed frozen evidence, and runs the public-release audit. |
| G9 Current tree | PASS | The latest current-tree CI run after hardening completed successfully. |
| G10 History | RESIDUAL_RISK | Historical commits retain an old server-path/hostname pattern (P1) and earlier internal workflow wording (P2). No real credential/private key was found in the audited historical snapshot and subsequent diffs. |
| G11 External view | PASS_WITH_INDEXING_LIMITATION | The repository is publicly readable, has one public branch and no GitHub releases; repository files and CI are accessible. Search-engine indexing is not relied upon as a release condition. |
| G12 Final decision | BLOCKED_PENDING_OWNER_HISTORY_DECISION | Current tree is release-clean, but historical P1/P2 residuals remain. Destructive history rewrite/force-push is not performed without explicit owner authorization. |

## Current-tree checks

The GitHub Actions workflow performs:

- Python compilation of `src`, `experiments`, `figures`, and `tools`;
- frozen A0 self-test;
- paper-scale self-test;
- compact verification of committed exactness/truth-preservation/held-out evidence;
- public-release scan for secrets, private infrastructure, internal workflow metadata, target-venue strings, row-level binary trace artifacts, and large files.

The stale static release manifest was removed because its stored hashes no longer represented the evolving default branch. CI is the live current-tree integrity signal.

## Historical residual

The repository was already public before this G0–G12 execution. Historical inspection found:

- P1: an earlier audit-script revision contains a literal local server path prefix and hostname used as forbidden-pattern examples;
- P2: earlier public summaries/audit patterns contain internal workflow identifiers.

These strings are absent from the current default-branch tree. No destructive history rewrite has been performed.

## Rights boundary

No repository-wide open-source license has been selected for EBMUS-authored code. See `RIGHTS_AND_LICENSES.md`. Third-party raw datasets are not redistributed and retain upstream terms.

## Scientific scope boundary

The common `1:10:50` cached/fresh/output rate vector is a transparent modeled comparison contract, not an actual provider invoice or royalty schedule. TraceLab is used only for controlled masking/generalization. The public release does not make a natural-missingness claim for TraceLab.
