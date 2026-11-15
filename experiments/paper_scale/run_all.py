#!/usr/bin/env python3
from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

PKG=Path(__file__).resolve().parent
PY=sys.executable

def run(script, env=None):
    cmd=[PY, str(PKG/'scripts'/script)]
    print('+', ' '.join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(PKG), env=env)

def main():
    if sys.version_info < (3,9):
        raise SystemExit('Python >=3.9 is required')
    run('self_test.py')
    run('acquire_sources.py')
    run('prepare_datasets.py')
    env=dict(os.environ)
    env.update({'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','NO_PROXY':'*','no_proxy':'*'})
    for script in ['run_main_comparison.py','run_solver_benchmark.py','run_refinement_sensitivity.py','run_natural_freeinference.py','mechanical_checks.py']:
        run(script, env=env)
    print('EBMUS paper-scale reproduction complete.')
    print('Generated paper-facing outputs:', PKG/'paper_facing')

if __name__=='__main__':
    main()
