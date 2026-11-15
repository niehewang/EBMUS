#!/usr/bin/env python3
from __future__ import annotations
import json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from common import PKG,RUNS,PAPER,CHECKS,atomic_json,read_json,exact_cost,bound_cost,midpoint_wrel
CFG=read_json(PKG/'frozen_config.json'); SUPPORT=CFG['natural_group_support']; TOLS=[0.0,.005,.01,.02,.05,.10]

def root_depth_map(sess):
    parent=dict(zip(sess.session_id.astype(str),sess.parent_session_id.where(sess.parent_session_id.notna(),None)))
    root={}; depth={}
    def find(s):
        if s in root:return root[s],depth[s]
        chain=[];cur=s;seen=set()
        while True:
            if cur in seen:raise RuntimeError('cycle')
            seen.add(cur);chain.append(cur);p=parent.get(cur,None)
            if p is None or (isinstance(p,float) and math.isnan(p)):r=cur;d0=0;break
            p=str(p)
            if p in root:r=root[p];d0=depth[p]+1;break
            if p not in parent:raise RuntimeError('orphan')
            cur=p
        # assign in reverse with root-relative depth
        if r==chain[-1]:
            dep=0
            for x in reversed(chain): root[x]=r; depth[x]=dep; dep+=1
        else:
            dep=d0
            for x in reversed(chain): root[x]=r; depth[x]=dep; dep+=1
        return root[s],depth[s]
    for s in parent:find(s)
    return root,depth

def rate_rows(req):
    P=pd.to_numeric(req.prompt_tokens,errors='coerce').to_numpy(float); O=pd.to_numeric(req.completion_tokens,errors='coerce').to_numpy(float); C=pd.to_numeric(req.cache_read_tokens,errors='coerce').to_numpy(float)
    good=np.isfinite(P)&np.isfinite(O)&(P>=0)&(O>=0); L=np.full(len(req),np.nan);U=np.full(len(req),np.nan); null=np.isnan(C)&good; rep=np.isfinite(C)&(C>=0)&good
    # Source exact semantics for reported cache values, including rare C>P via max(P-C,0).
    ex=(C[rep]+10*np.maximum(P[rep]-C[rep],0)+50*O[rep])/1e6;L[rep]=ex;U[rep]=ex
    L[null]=(P[null]+50*O[null])/1e6;U[null]=(10*P[null]+50*O[null])/1e6
    return P,O,C,L,U,null,good&(rep|null)

def group_record(df,group_type,group_value,week=None):
    n=len(df); nulls=int(df['is_null'].sum()); L=float(df.L.sum());U=float(df.U.sum()); wr=midpoint_wrel(L,U); workflows=int(df.root_id.nunique())
    return {'week':week or 'ALL','group_type':group_type,'group_value':str(group_value),'requests':n,'workflows':workflows,'natural_null_count':nulls,'natural_null_fraction':nulls/max(n,1),'L':L,'U':U,'Wrel':wr,'support_adequate':bool(n>=SUPPORT['min_requests'] and workflows>=SUPPORT['min_workflows'])}

def main():
    acq=read_json(RUNS/'source_acquisition.json'); group_rows=[]; workflow_rows=[]
    for week,base in sorted(acq['freeinference_locations'].items()):
        base=Path(base); sess=pq.read_table(base/'processed/sessions.parquet',columns=['session_id','harness','parent_session_id']).to_pandas();sess['session_id']=sess.session_id.astype(str);roots,depth=root_depth_map(sess);hmap=dict(zip(sess.session_id,sess.harness.fillna('UNKNOWN').astype(str)))
        pf=pq.ParquetFile(base/'processed/requests.parquet'); names=set(pf.schema_arrow.names); cols=[c for c in ['session_id','model','provider','prompt_tokens','completion_tokens','cache_read_tokens'] if c in names]; req=pq.read_table(base/'processed/requests.parquet',columns=cols).to_pandas();req['session_id']=req.session_id.astype(str)
        for c in ('model','provider'):
            if c not in req:req[c]='UNKNOWN'
            req[c]=req[c].fillna('UNKNOWN').astype(str)
        P,O,C,L,U,null,valid=rate_rows(req);req=req.loc[valid].copy();req['L']=L[valid];req['U']=U[valid];req['is_null']=null[valid];req['root_id']=req.session_id.map(roots);req['harness']=req.session_id.map(hmap).fillna('UNKNOWN')
        group_rows.append(group_record(req,'week',week,week))
        for h,g in req.groupby('harness'):group_rows.append(group_record(g,'harness',h,week))
        for col in ('provider','model'):
            for v,g in req.groupby(col):group_rows.append(group_record(g,col,v,week))
        # workflow tree aggregation and decomposition
        sess_info=defaultdict(lambda:{'session_count':0,'max_depth':0})
        for sid in sess.session_id:
            r=roots[sid];sess_info[r]['session_count']+=1;sess_info[r]['max_depth']=max(sess_info[r]['max_depth'],depth[sid])
        for r,g in req.groupby('root_id'):
            Lw=float(g.L.sum());Uw=float(g.U.sum());wr=midpoint_wrel(Lw,Uw);nnull=int(g['is_null'].sum()); exact_known=float(g.loc[~g['is_null'],'L'].sum()); missing_lower=float(g.loc[g['is_null'],'L'].sum());missing_upper=float(g.loc[g['is_null'],'U'].sum())
            workflow_rows.append({'week':week,'root_private':r,'root_harness':hmap.get(r,'UNKNOWN'),'session_count':sess_info[r]['session_count'],'max_depth':sess_info[r]['max_depth'],'requests':len(g),'natural_null_count':nnull,'L':Lw,'U':Uw,'Wrel':wr,'affected':Uw>Lw+1e-12,'exact_known_cost':exact_known,'missing_lower_cost':missing_lower,'missing_upper_cost':missing_upper})
        print('natural',week,'requests',len(req),flush=True)
    groups=pd.DataFrame(group_rows); workflows=pd.DataFrame(workflow_rows)
    # support-only association summaries are count-filtered, never outcome-filtered.
    assoc={}
    for gt in ('week','harness'):
        x=groups[(groups.group_type==gt)&groups.support_adequate]
        if len(x)>=3:
            rho,p=spearmanr(x.natural_null_fraction,x.Wrel);assoc[gt]={'groups':len(x),'spearman_rho':float(rho),'pvalue':float(p)}
    atomic_json(CHECKS/'freeinference_natural_association.json',assoc)
    # workflow size/depth bins
    workflows['size_bin']=pd.cut(workflows.session_count,bins=[0,1,3,7,np.inf],labels=['1','2-3','4-7','8+'])
    workflows['depth_bin']=pd.cut(workflows.max_depth,bins=[-1,0,1,2,np.inf],labels=['0','1','2','3+'])
    extra=[]
    for col in ('size_bin','depth_bin'):
      for v,g in workflows.groupby(col,observed=True):
        extra.append({'week':'ALL','group_type':'workflow_'+col,'group_value':str(v),'requests':int(g.requests.sum()),'workflows':len(g),'natural_null_count':int(g.natural_null_count.sum()),'natural_null_fraction':float(g.natural_null_count.sum()/max(g.requests.sum(),1)),'L':float(g.L.sum()),'U':float(g.U.sum()),'Wrel':midpoint_wrel(g.L.sum(),g.U.sum()),'support_adequate':True,'affected_workflow_fraction':float(g.affected.mean()),'median_workflow_Wrel':float(g.Wrel.median())})
    groups=pd.concat([groups,pd.DataFrame(extra)],ignore_index=True,sort=False);groups.to_csv(PAPER/'paper_freeinference_group_summary.csv',index=False)
    # tolerance curves by week and overall, on top-level workflows.
    curves=[]
    for label,g in list(workflows.groupby('week'))+[('ALL',workflows)]:
        for t in TOLS:curves.append({'week':label,'tolerance':t,'workflow_count':len(g),'finality_fraction':float((g.Wrel<=t+1e-15).mean()),'nested_only_finality_fraction':float((g.loc[g.session_count>1,'Wrel']<=t+1e-15).mean()) if (g.session_count>1).any() else np.nan})
    pd.DataFrame(curves).to_csv(PAPER/'paper_freeinference_tolerance_curve.csv',index=False)
    # Representative affected workflows chosen deterministically by Wrel order: low, median, high.
    aff=workflows[workflows.affected].sort_values(['Wrel','week','root_private']).reset_index(drop=True);examples=[]
    if len(aff):
        idxs=[0,(len(aff)-1)//2,len(aff)-1];labels=['low','median','high']
        for lab,i in zip(labels,idxs):
            r=aff.iloc[i];examples.append({'representative_level':lab,'rank_index':int(i),'affected_workflow_count':len(aff),'week':r.week,'root_harness':r.root_harness,'session_count':int(r.session_count),'max_depth':int(r.max_depth),'requests':int(r.requests),'natural_null_count':int(r.natural_null_count),'L':float(r.L),'U':float(r.U),'Wrel':float(r.Wrel),'lower_witness_decomposition':{'exact_known_cost':float(r.exact_known_cost),'missing_cache_partition_lower_cost':float(r.missing_lower_cost)},'upper_witness_decomposition':{'exact_known_cost':float(r.exact_known_cost),'missing_cache_partition_upper_cost':float(r.missing_upper_cost)},'session_id_returned':False})
    atomic_json(PAPER/'paper_freeinference_workflow_examples.json',{'selection':'low/median/high Wrel among all naturally affected top-level workflows; outcome examples requested by frozen protocol','message_text_returned':False,'examples':examples})
    # Figure generalization asset may already exist from masked comparison; append natural week rows in a separate canonical asset then merge.
    natfig=groups[groups.group_type=='week'][['week','requests','natural_null_fraction','Wrel']].copy();natfig.insert(0,'analysis','FreeInference_natural');natfig.to_csv(PAPER/'fig_freeinference_natural.csv',index=False)
    # Do not return private root IDs; raw workflow records stay local cache only.
    workflows.drop(columns=['root_private']).to_parquet(CHECKS/'freeinference_workflow_mechanical.parquet',index=False)
    atomic_json(CHECKS/'natural_vs_masked_firewall.json',{'natural_analysis_script':'run_natural_freeinference.py','uses_mask_files':False,'synthetic_masking':False,'natural_source':'unaltered public FreeInference rows','masked_benchmark_separate_script':'run_main_comparison.py'})
    print('NATURAL_FREEINFERENCE_COMPLETE workflows=%d affected=%d'%(len(workflows),int(workflows.affected.sum())))
if __name__=='__main__':main()
