# Privacy and release policy

This repository is a disclosure-minimized scientific code release.

## Excluded from the public repository

- API keys, access tokens, credentials, cookies, or private URLs
- local server usernames, hostnames, IP addresses, SSH configuration, or absolute filesystem paths
- private operational, coordination, and project-management records
- unpublished document source and private correspondence
- raw third-party datasets
- exact pseudonymous session-ID split lists
- saved row-level masking arrays
- raw session-level evaluation metrics
- row-level FreeInference workflow tables that could facilitate unnecessary linkage

## Included

- scientific source code
- frozen random seeds and protocol parameters
- public-source identities and cryptographic hashes
- aggregate reported results
- synthetic validation outputs
- non-identifying representative workflow summaries (no session ID or message text)

## Credentials

The reproduction code does not require model APIs or paid services. Dataset acquisition uses public HTTP(S) endpoints. Do not add `.env` files, private keys, access tokens, or credentials to the repository.
