#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
TEXT_EXT = {'.py','.md','.json','.yml','.yaml','.txt','.csv','.gitignore','.cff','.toml','.sh','.mplstyle'}

FORBIDDEN = [
    r'(?<![A-Za-z0-9_$])/(?:data|home)/[^\s"\']+',
    r'\bC:\\Users\\[^\s"\']+',
    r'OPENROUTER_API_KEY',
    r'github_pat_',
    r'ghp_[A-Za-z0-9]+',
    r'\bsk-[A-Za-z0-9_-]{16,}',
    r'\bhf_[A-Za-z0-9]{16,}',
    r'Authorization:\s*Bearer',
    r'-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----',
    r'https?://(?:drive|docs)\.google\.com/[^\s)"\']+',
    r'\b(?:LATEST_HANDOFF|CURRENT_DECISION|NEXT_ACTION|EXPERIMENT_EXECUTION|PAPER_WRITING|IDEA_DECISION)\b',
    r'\bResearch[- ]OS\b',
    r'\bTSC_AI_SERVICE_IP\b',
    r'\bIEEE\s+TSC\b',
    r'\bIEEE\s+Transactions\s+on\s+Services\s+Computing\b',
]

FORBIDDEN_FILES = {'.env', 'source_manifest.json', 'environment.json'}
REQUIRED_FILES = {
    'README.md',
    '.gitignore',
    'SECURITY.md',
    'RIGHTS_AND_LICENSES.md',
    'docs/DATASETS.md',
    'docs/PRIVACY_AND_RELEASE.md',
    'docs/REPRODUCIBILITY.md',
    'tools/verify_frozen_evidence.py',
}
errors = []

for rel in sorted(REQUIRED_FILES):
    if not (ROOT / rel).is_file():
        errors.append(f'missing required public-release file: {rel}')

for p in ROOT.rglob('*'):
    if not p.is_file() or '.git' in p.parts:
        continue
    rel = p.relative_to(ROOT).as_posix()
    if rel == 'tools/release_audit.py':
        continue
    if p.name in FORBIDDEN_FILES:
        errors.append(f'forbidden committed file: {rel}')
    if p.suffix in {'.parquet', '.npz'}:
        errors.append(f'row-level/binary trace artifact committed: {rel}')
    if p.stat().st_size > 5 * 1024 * 1024:
        errors.append(f'large artifact >5 MiB committed: {rel}')
    if p.suffix.lower() in TEXT_EXT or p.name in {'.gitignore', 'Makefile'}:
        try:
            txt = p.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            continue
        for pat in FORBIDDEN:
            if re.search(pat, txt, re.I):
                errors.append(f'sensitive/internal metadata: {rel}')
        if re.search(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', txt, re.I):
            errors.append(f'email address committed: {rel}')

for pat in ('**/splits_*.json', '**/masks_*.npz', '**/raw_metrics/**'):
    for p in ROOT.glob(pat):
        if p.is_file():
            errors.append(f'privacy-minimized artifact unexpectedly committed: {p.relative_to(ROOT)}')

if errors:
    print('RELEASE_AUDIT_FAIL')
    for e in sorted(set(errors)):
        print('-', e)
    sys.exit(2)

print('RELEASE_AUDIT_PASS')
