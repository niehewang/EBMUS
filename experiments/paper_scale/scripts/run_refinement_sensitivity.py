#!/usr/bin/env python3
from __future__ import annotations
import json,random
import numpy as np
import pandas as pd
from common import PKG,PAPER,CHECKS,atomic_json,midpoint_wrel
from paper_core import generate_instance,evidence_bounds,refinement_conditions,true_cost
CFG=json.load(open(PKG/'frozen_config.json'))

def qstats(vals):
    a=np.asarray(vals,float)
    return {'median':float(np.median(a)),'p25':float(np.quantile(a,.25)),'p75':float(np.quantile(a,.75))}

def refinement():
    rows=[]; violations=[]
    for n in CFG['evidence_refinement']['sizes']:
      for rate in CFG['evidence_refinement']['rate_heterogeneity']:
       for seed in CFG['evidence_refinement']['seeds']:
        rng=random.Random(seed*100000+n*97+sum(map(ord,rate)))
        # fixed 50 instances per (n,rate,seed), preregistered in package before outcomes
        for iid in range(50):
            inst=generate_instance(rng,n,.10,.10,.50,rate,'medium'); prevL=None; prevU=None
            for cond,x,resolved in refinement_conditions(inst):
                z=evidence_bounds(x,resolved); L=z.get('L'); U=z.get('U'); wr=midpoint_wrel(L,U) if z.get('feasible') else np.nan
                rows.append({'n':n,'rate_heterogeneity':rate,'seed':seed,'instance_id':iid,'condition':cond,'feasible':z.get('feasible',False),'L':L,'U':U,'Wrel':wr,'exact_final':bool(z.get('feasible') and abs(U-L)<=1e-9),'final_1pct':bool(z.get('feasible') and wr<=.01)})
                if prevL is not None and z.get('feasible') and (L<prevL-1e-7 or U>prevU+1e-7): violations.append({'n':n,'rate':rate,'seed':seed,'instance':iid,'condition':cond,'prevL':prevL,'prevU':prevU,'L':L,'U':U})
                if z.get('feasible'): prevL,prevU=L,U
    raw=pd.DataFrame(rows)
    out=[]
    for (n,rate,cond),g in raw.groupby(['n','rate_heterogeneity','condition']):
        s=qstats(g.Wrel.dropna()); out.append({'n':n,'rate_heterogeneity':rate,'condition':cond,'instances':len(g),'median_Wrel':s['median'],'p25_Wrel':s['p25'],'p75_Wrel':s['p75'],'exact_final_fraction':float(g.exact_final.mean()),'final_1pct_fraction':float(g.final_1pct.mean())})
    agg=pd.DataFrame(out)
    # marginal contraction against immediately previous evidence condition
    order=['E0','E1','E2','E3','E4']
    for (n,rate),idx in agg.groupby(['n','rate_heterogeneity']).groups.items():
        sub=agg.loc[idx].set_index('condition')
        for i,c in enumerate(order):
            if i==0: agg.loc[(agg.n==n)&(agg.rate_heterogeneity==rate)&(agg.condition==c),'marginal_median_contraction']=np.nan
            else:
                v=float(sub.loc[order[i-1],'median_Wrel']-sub.loc[c,'median_Wrel']); agg.loc[(agg.n==n)&(agg.rate_heterogeneity==rate)&(agg.condition==c),'marginal_median_contraction']=v
    agg.to_csv(PAPER/'paper_evidence_refinement.csv',index=False); agg.to_csv(PAPER/'fig_evidence_refinement.csv',index=False)
    raw.to_csv(PKG/'raw_metrics'/'evidence_refinement_instances.csv.gz',index=False,compression='gzip')
    atomic_json(CHECKS/'evidence_refinement_nesting.json',{'violations':violations,'violation_count':len(violations),'rule':'L cannot decrease and U cannot increase from E0 through E4 beyond 1e-7'})

def sensitivity():
    rows=[]; rng=random.Random(20261109)
    cfg=CFG['sensitivity']
    for m in cfg['missing_local']:
      for a in cfg['hidden_attribution']:
       for s in cfg['internal_summary_coverage']:
        for amb in cfg['ambiguity_density']:
         for rate in cfg['rate_heterogeneity']:
          vals=[]
          for iid in range(cfg['instances_per_cell']):
            inst=generate_instance(rng,cfg['n'],m,a,s,rate,amb); z=evidence_bounds(inst); truth=true_cost(inst)
            if not z.get('feasible'): vals.append({'feasible':False}); continue
            wr=midpoint_wrel(z['L'],z['U']); vals.append({'feasible':True,'Wrel':wr,'truth_contained':z['L']-1e-7<=truth<=z['U']+1e-7,'exact':abs(z['U']-z['L'])<=1e-9,'final1':wr<=.01})
          good=[x for x in vals if x.get('feasible')]
          arr=[x['Wrel'] for x in good]
          rows.append({'missing_local':m,'hidden_attribution':a,'internal_summary_coverage':s,'ambiguity_density':amb,'rate_heterogeneity':rate,'n':cfg['n'],'instances':len(vals),'feasible_fraction':len(good)/len(vals),'median_Wrel':float(np.median(arr)) if arr else np.nan,'p25_Wrel':float(np.quantile(arr,.25)) if arr else np.nan,'p75_Wrel':float(np.quantile(arr,.75)) if arr else np.nan,'p95_Wrel':float(np.quantile(arr,.95)) if arr else np.nan,'exact_final_fraction':sum(x.get('exact',False) for x in good)/max(len(good),1),'final_1pct_fraction':sum(x.get('final1',False) for x in good)/max(len(good),1),'truth_containment_fraction':sum(x.get('truth_contained',False) for x in good)/max(len(good),1)})
    df=pd.DataFrame(rows); df.to_csv(PAPER/'paper_sensitivity.csv',index=False); df.to_csv(PAPER/'fig_sensitivity.csv',index=False)
    atomic_json(CHECKS/'sensitivity_protocol.json',{'base_axes':['missing_local','hidden_attribution','internal_summary_coverage'],'targeted_additions':['ambiguity_density','rate_heterogeneity'],'cells':len(df),'instances_per_cell':cfg['instances_per_cell'],'post_result_tuning':False})

def main():
    refinement(); sensitivity(); print('REFINEMENT_AND_SENSITIVITY_COMPLETE')
if __name__=='__main__': main()
