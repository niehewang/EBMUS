#!/usr/bin/env python3
from __future__ import annotations
import json,math
from pathlib import Path
import pandas as pd
from common import PKG,RUNS,CONTRACTS,MASKS,PAPER,CHECKS,atomic_json,read_json,sha256_file
REQ_PAPER=[
'paper_table_main_comparison.csv','paper_solver_scaling.csv','paper_solver_exactness.csv','paper_evidence_refinement.csv','paper_sensitivity.csv',
'paper_freeinference_group_summary.csv','paper_freeinference_tolerance_curve.csv','paper_freeinference_workflow_examples.json',
'fig_main_comparison.csv','fig_solver_scaling.csv','fig_evidence_refinement.csv','fig_sensitivity.csv','fig_real_trace_generalization.csv']

def parseable(p):
    try:
      if p.suffix=='.csv': pd.read_csv(p,nrows=3)
      elif p.suffix=='.json': json.load(open(p))
      else:return False
      return True
    except Exception:return False

def main():
    acq=read_json(RUNS/'source_acquisition.json'); prep=read_json(RUNS/'prepared_datasets.json')
    solver=read_json(CHECKS/'solver_agreement.json'); ref=read_json(CHECKS/'evidence_refinement_nesting.json'); firewall=read_json(CHECKS/'natural_vs_masked_firewall.json')
    # Split leakage checks from exact saved session IDs.
    split_ok=True; split_detail={}
    for ds in ['freeinference']+(['syfi'] if prep['syfi'].get('schema_pass') else []):
        x=read_json(CONTRACTS/('splits_%s.json'%ds)); sets={k:set(x['session_ids'][k]) for k in ('train','validation','test')}; overlap=(sets['train']&sets['validation'])|(sets['train']&sets['test'])|(sets['validation']&sets['test']); split_detail[ds]={'overlap_count':len(overlap),'counts':x['counts']}; split_ok=split_ok and not overlap
    # Mask identity: a single frozen NPZ per dataset, no method-specific masks.
    mask_files=list(MASKS.glob('masks_*.npz')); mask_ok=len(mask_files)>=1 and all((MASKS/('mask_manifest_'+p.stem.replace('masks_','')+'.json')).exists() for p in mask_files)
    truth_files=list(CHECKS.glob('truth_preservation_*.json')); truth_ok=all(read_json(p).get('match') for p in truth_files) and bool(truth_files)
    feature_ok=True; feature_detail={}
    for p in CONTRACTS.glob('feature_contract_*.json'):
        x=read_json(p); forbidden=set(x['forbidden']); actual=set(x['numeric_predictors'])|set(x['categorical_predictors']); bad=actual&{'cache_read_tokens','fresh_input_tokens','uncached_input_tokens'}; feature_detail[p.stem]=sorted(bad); feature_ok=feature_ok and not bad
    syfi=read_json(PKG/'syfi_schema_preflight.json'); syfi_firewall=(not syfi.get('natural_missing_claim_authorized',False))
    required={name:{'present':(PAPER/name).exists(),'parseable':parseable(PAPER/name) if (PAPER/name).exists() else False} for name in REQ_PAPER}; complete=all(v['present'] and v['parseable'] for v in required.values())
    fi_records=acq.get('freeinference',{})
    fi_integrity=(len(fi_records)==16 and all(all(f.get('match') is True for f in rec.get('files',{}).values()) for rec in fi_records.values()))
    syfi_rec=acq.get('syfi',{}); syfi_integrity=bool(acq.get('syfi_path')) and syfi_rec.get('sha256')=='cb8c2b1d23307090c8057c747f955663bacde16a49f84a3c5071a1e2e96a7ab8'
    source_ok=('freeinference_locations' in acq and len(acq['freeinference_locations'])==16 and fi_integrity and syfi_integrity)
    exact_ok=(solver['lp_vs_laminar_mismatches']==0 and solver['milp_vs_evidence_mismatches']==0 and solver['bruteforce_vs_evidence_mismatches']==0)
    nesting_ok=ref['violation_count']==0
    gates={
      'G1_source_integrity':{'pass':source_ok,'detail':'16 FreeInference historical-hash-anchored weeks + SyFI exact file SHA recorded'},
      'G2_split_leakage':{'pass':split_ok,'detail':split_detail},
      'G3_mask_identity':{'pass':mask_ok,'mask_files':[p.name for p in mask_files]},
      'G4_target_leakage':{'pass':feature_ok,'detail':feature_detail},
      'G5_truth_preservation':{'pass':truth_ok,'files':[p.name for p in truth_files]},
      'G6_exact_solver_agreement':{'pass':exact_ok,'detail':solver},
      'G7_evidence_refinement_nesting':{'pass':nesting_ok,'detail':{'violation_count':ref['violation_count']}},
      'G8_natural_vs_masked_firewall':{'pass':bool(firewall.get('synthetic_masking') is False and firewall.get('uses_mask_files') is False),'detail':firewall},
      'G9_syfi_claim_firewall':{'pass':syfi_firewall,'schema_gate':syfi.get('schema_gate'),'natural_missing_claim_authorized':syfi.get('natural_missing_claim_authorized',False)},
      'G10_paper_facing_completeness':{'pass':complete,'files':required},
      'scientific_outcome':'NOT_ADJUDICATED_BY_EXPERIMENT_EXECUTION'
    }
    atomic_json(CHECKS/'mechanical_gates.json',gates)
    failed=[k for k,v in gates.items() if k.startswith('G') and not v['pass']]
    print('MECHANICAL_GATES', 'PASS' if not failed else 'FAIL', failed)
    return 0 if not failed else 6
if __name__=='__main__': raise SystemExit(main())
