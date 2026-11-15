# Public Release Audit

Updated: 2026-11-15

This document records the completed G0–G12 public-code release audit for the EBMUS repository.

## Final decision

**PUBLIC_RELEASE_STATUS = PASS**

The current public tree passes the strengthened release checks, and the reachable `main` history was destructively rewritten with owner authorization to remove the previously identified P1/P2 historical residuals.

## G0–G12 status

| Gate | Status | Evidence |
| --- | --- | --- |
| G0 Scope | PASS | Public release is limited to scientific code, public-source provenance, disclosure-minimized outputs, and documentation. |
| G1 Secret | PASS | Automated scan covers token/key patterns, bearer credentials, private-key headers, direct email addresses, and forbidden local environment files. |
| G2 Infrastructure | PASS | Automated scan rejects absolute local paths and private Drive links; the previously identified local server path/hostname are absent from the reachable rewritten history. |
| G3 Venue anonymity | PASS | Target-venue strings are rejected by the audit; public documentation is venue-neutral. |
| G4 Production firewall | PASS | Internal workflow/state identifiers are rejected by the audit and the previously identified historical workflow wording was removed from reachable history. |
| G5 Data/license/privacy | PASS | Raw third-party datasets are not redistributed; upstream dataset licensing and the privacy boundary are documented. |
| G6 Repository quality | PASS_WITH_RIGHTS_NOTICE | README, CI, security guidance, reproducibility documentation, and rights status are present and factual. No repository-wide open-source license is claimed. |
| G7 Portability | PASS | Public-source acquisition supports the default endpoint, a compatible mirror endpoint, and pre-existing verified local data; frozen hashes are checked. |
| G8 Automated audit | PASS | CI compiles Python, runs both self-tests, verifies committed frozen evidence, and runs the public-release audit. |
| G9 Current tree | PASS | The rewritten public tree completed the full CI successfully. |
| G10 History | PASS | Owner-authorized destructive rewrite replaced the old reachable chain. The public `main` history now contains only the initialization commit and the sanitized release-state commit before this audit-finalization commit; the identified P1/P2 commits are no longer reachable from any repository branch. |
| G11 External view | PASS | Repository is publicly readable, has one public branch and no GitHub releases; public files and CI are accessible. |
| G12 Final decision | PASS | G0–G11 pass; release gate closed. |

## Current-tree checks

The GitHub Actions workflow performs:

- Python compilation of `src`, `experiments`, `figures`, and `tools`;
- frozen A0 self-test;
- paper-scale self-test;
- compact verification of committed exactness/truth-preservation/held-out evidence;
- public-release scan for secrets, private infrastructure, internal workflow metadata, target-venue strings, row-level binary trace artifacts, and large files.

The stale static release manifest was removed because its stored hashes no longer represented the evolving default branch. CI is the live current-tree integrity signal.

## History rewrite record

Before rewrite, the reachable history contained:

- P1: a literal local server path prefix and hostname used in an old audit-script revision;
- P2: earlier internal workflow identifiers in public summaries/audit patterns.

With explicit owner authorization, `main` was force-updated using a sanitized tree. The prior affected commits are no longer reachable from the repository's branch history.

As with any hosted Git service, this operation does not assert instantaneous erasure of provider-side caches, old external clones, or previously fetched copies outside the repository's reachable Git history.

## Rights boundary

No repository-wide open-source license has been selected for EBMUS-authored code. See `RIGHTS_AND_LICENSES.md`. Third-party raw datasets are not redistributed and retain upstream terms.

## Scientific scope boundary

The common `1:10:50` cached/fresh/output rate vector is a transparent modeled comparison contract, not an actual provider invoice or royalty schedule. TraceLab is used only for controlled masking/generalization. The public release does not make a natural-missingness claim for TraceLab.
