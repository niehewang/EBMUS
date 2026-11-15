#!/usr/bin/env python3
from __future__ import annotations
import csv,json,random,time
import numpy as np
import pandas as pd
from common import PKG,PAPER,CHECKS,proc_peak_rss_mb,atomic_json,midpoint_wrel
from paper_core import generate_instance,laminar_flow_solve,independent_bounds,generic_lp,generic_milp,evidence_bounds,brute_force
CFG=json.load(open(PKG/'frozen_config.json')); sizes=CFG['solver']['sizes']; reps=CFG['solver']['repetitions']; timeout=CFG['solver']['timeout_seconds']; tol=CFG['solver']['numerical_tolerance']

def pair_structured(inst):
    t=time.perf_counter(); lo=laminar_flow_solve(inst,'min'); hi=laminar_flow_solve(inst,'max'); rt=time.perf_counter()-t
    return {'success':lo.get('feasible') and hi.get('feasible'),'L':lo.get('cost'),'U':hi.get('cost'),'runtime':rt,'timeout':False,'rss':proc_peak_rss_mb()}

def pair_lp(inst):
    lo=generic_lp(inst,'min',timeout); hi=generic_lp(inst,'max',timeout); return {'success':lo.get('success') and hi.get('success'),'L':lo.get('cost'),'U':hi.get('cost'),'runtime':lo.get('runtime',0)+hi.get('runtime',0),'timeout':lo.get('timeout') or hi.get('timeout'),'rss':proc_peak_rss_mb()}

def pair_evidence(inst):
    t=time.perf_counter(); z=evidence_bounds(inst); rt=time.perf_counter()-t; return {'success':z.get('feasible'), 'L':z.get('L'),'U':z.get('U'),'runtime':rt,'timeout':False,'rss':proc_peak_rss_mb()}

def pair_milp(inst):
    lo=generic_milp(inst,'min',timeout); hi=generic_milp(inst,'max',timeout); return {'success':lo.get('success') and hi.get('success'),'L':lo.get('cost'),'U':hi.get('cost'),'runtime':lo.get('runtime',0)+hi.get('runtime',0),'timeout':lo.get('timeout') or hi.get('timeout'),'rss':proc_peak_rss_mb()}

def close(a,b): return a is not None and b is not None and abs(float(a)-float(b))<=tol*max(1.0,abs(float(a)),abs(float(b)))

def main():
    rows=[]; exact=[]; rng=random.Random(20261109)
    for n in sizes:
      for rep in range(reps):
        inst=generate_instance(rng,n,.10,.10,.50,'medium','none'); ind=independent_bounds(inst); lf=pair_structured(inst); lp=pair_lp(inst)
        for method,z in [('independent_bounds',{'success':True,'L':ind['L'],'U':ind['U'],'runtime':0.0,'timeout':False,'rss':proc_peak_rss_mb()}),('generic_lp',lp),('laminarflow',lf)]:
            rows.append({'problem':'continuous_laminar','n':n,'rep':rep,'method':method,'success':z['success'],'timeout':z['timeout'],'runtime_seconds':z['runtime'],'peak_rss_mb_process':z['rss'],'L':z['L'],'U':z['U'],'relative_width':midpoint_wrel(z['L'],z['U']) if z['success'] else np.nan,'mismatch_to_laminar':False if method=='laminarflow' else (not (z['success'] and lf['success'] and close(z['L'],lf['L']) and close(z['U'],lf['U'])))})
        exact.append({'regime':'lp_vs_laminar','n':n,'rep':rep,'common_success':lp['success'] and lf['success'],'L_mismatch':not close(lp['L'],lf['L']) if lp['success'] and lf['success'] else None,'U_mismatch':not close(lp['U'],lf['U']) if lp['success'] and lf['success'] else None})
        # scoped discrete ambiguity: two types are generated before outcomes.
        dinst=generate_instance(rng,n,.10,.10,.50,'medium','medium'); em=pair_evidence(dinst); gm=pair_milp(dinst)
        for method,z in [('generic_milp',gm),('evidence_milp',em)]:
            rows.append({'problem':'scoped_discrete','n':n,'rep':rep,'method':method,'success':z['success'],'timeout':z['timeout'],'runtime_seconds':z['runtime'],'peak_rss_mb_process':z['rss'],'L':z['L'],'U':z['U'],'relative_width':midpoint_wrel(z['L'],z['U']) if z['success'] else np.nan,'mismatch_to_laminar':False})
        exact.append({'regime':'generic_milp_vs_evidence_milp','n':n,'rep':rep,'common_success':gm['success'] and em['success'],'L_mismatch':not close(gm['L'],em['L']) if gm['success'] and em['success'] else None,'U_mismatch':not close(gm['U'],em['U']) if gm['success'] and em['success'] else None})
        if (rep+1)%10==0: print('solver n=%d rep=%d/%d'%(n,rep+1,reps),flush=True)
    # small brute-force exactness independent check
    for n in CFG['solver']['small_exact_sizes']:
      for rep in range(CFG['solver']['small_exact_repetitions']):
        inst=generate_instance(rng,n,.20,.15,.50,'medium','low'); brute=brute_force(inst); em=evidence_bounds(inst)
        exact.append({'regime':'bruteforce_vs_evidence','n':n,'rep':rep,'common_success':brute['feasible'] and em['feasible'],'L_mismatch':not close(brute['L'],em['L']) if brute['feasible'] and em['feasible'] else None,'U_mismatch':not close(brute['U'],em['U']) if brute['feasible'] and em['feasible'] else None})
    pd.DataFrame(rows).to_csv(PAPER/'paper_solver_scaling.csv',index=False); pd.DataFrame(exact).to_csv(PAPER/'paper_solver_exactness.csv',index=False); pd.DataFrame(rows).to_csv(PAPER/'fig_solver_scaling.csv',index=False)
    ex=pd.DataFrame(exact); checks={'lp_vs_laminar_mismatches':int(((ex.regime=='lp_vs_laminar')&(ex.L_mismatch.fillna(False)|ex.U_mismatch.fillna(False))).sum()),'milp_vs_evidence_mismatches':int(((ex.regime=='generic_milp_vs_evidence_milp')&(ex.L_mismatch.fillna(False)|ex.U_mismatch.fillna(False))).sum()),'bruteforce_vs_evidence_mismatches':int(((ex.regime=='bruteforce_vs_evidence')&(ex.L_mismatch.fillna(False)|ex.U_mismatch.fillna(False))).sum()),'numerical_tolerance':tol}
    atomic_json(CHECKS/'solver_agreement.json',checks); print('SOLVER_BENCHMARK_COMPLETE',checks)
if __name__=='__main__': main()
