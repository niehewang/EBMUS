# Security policy

This repository is intended to contain only public scientific code, public-source provenance information, and disclosure-minimized reference outputs.

## Reporting a potential exposure

If you find a credential, private key, access token, private infrastructure identifier, sensitive raw record, or other information that should not be public:

- do **not** paste the sensitive value into a public issue;
- contact the repository maintainer through a private GitHub contact channel, or use GitHub's private security-reporting mechanism if it is available for the repository;
- include only the minimum information needed to identify the affected file or commit.

## Repository boundary

No secret is required to run the committed self-tests or compact evidence checks. Public dataset acquisition uses unauthenticated HTTP(S) endpoints. Local environment files such as `.env`, private keys, raw trace data, and row-level linkage artifacts must not be committed.
