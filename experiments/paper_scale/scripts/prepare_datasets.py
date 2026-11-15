#!/usr/bin/env python3
from __future__ import annotations
import json, math, os, sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from common import PKG,RUNS,CACHE,CONTRACTS,CHECKS,atomic_json,read_json,sha256_file,stable_u64

SPLIT_SEED=20261109
MIN_ROWS=10

def root_map(sessions:pd.DataFrame):
    parent=dict(zip(sessions.session_id.astype(str), sessions.parent_session_id.where(sessions.parent_session_id.notna(),None)))
    roots={}
    def find(s):
        if s in roots:return roots[s]
        seen=[]; cur=s; mark=set()
        while True:
            if cur in mark: raise RuntimeError('cycle in session graph')
            mark.add(cur); seen.append(cur)
            p=parent.get(cur,None)
            if p is None or (isinstance(p,float) and math.isnan(p)):
                r=cur; break
            p=str(p)
            if p in roots: r=roots[p]; break
            if p not in parent: raise RuntimeError('orphan parent %s'%p)
            cur=p
        for x in seen: roots[x]=r
        return r
    for s in parent: find(s)
    return roots

def split_groups(group_to_stratum:dict):
    rng=np.random.default_rng(SPLIT_SEED)
    out={'train':[],'validation':[],'test':[]}
    by=defaultdict(list)
    for g,s in group_to_stratum.items(): by[str(s)].append(str(g))
    for stratum,ids in sorted(by.items()):
        ids=sorted(ids); perm=np.array(ids,dtype=object); rng.shuffle(perm); ids=list(perm)
        n=len(ids)
        if n>=5:
            nt=int(round(.60*n)); nv=int(round(.20*n)); nt=max(1,min(nt,n-2)); nv=max(1,min(nv,n-nt-1))
            out['train']+=ids[:nt]; out['validation']+=ids[nt:nt+nv]; out['test']+=ids[nt+nv:]
        else:
            for g in ids:
                u=stable_u64('%s|%s'%(SPLIT_SEED,g))/float(2**64)
                out['train' if u<.60 else ('validation' if u<.80 else 'test')].append(g)
    # guarantee no overlap
    sets={k:set(v) for k,v in out.items()}
    assert not (sets['train']&sets['validation'] or sets['train']&sets['test'] or sets['validation']&sets['test'])
    return {k:sorted(v) for k,v in out.items()}

def majority_stratum(df,group_col='split_group_id'):
    ans={}
    for g,x in df.groupby(group_col,sort=False):
        vals=x['provider'].fillna('UNKNOWN').astype(str)
        ans[str(g)]=vals.value_counts().index[0] if len(vals) else 'UNKNOWN'
    return ans

def make_feature_contract(dataset,cols):
    obj={
      'dataset':dataset,
      'target':'cache_read_tokens',
      'numeric_predictors':[c for c in ['input_tokens','output_tokens','step_index','n_messages','session_relative_position'] if c in cols],
      'categorical_predictors':[c for c in ['model','provider','harness','step_trigger'] if c in cols],
      'categorical_encoding':'deterministic signed SHA256 feature hashing, 12 dimensions; target-independent',
      'numeric_transform':'log1p for token/message counts; raw step_index/session_relative_position; train-only standardization',
      'forbidden':['cache_read_tokens','fresh_input_tokens','uncached_input_tokens','future messages/results','any target-derived field'],
      'target_clamp':'all predictive cache estimates are clipped to [0,input_tokens] before rating',
      'split':'session/workflow-group split seed 20261109; no request-level split'
    }
    atomic_json(CONTRACTS/('feature_contract_%s.json'%dataset.lower()),obj)

def load_fi_week(week,base):
    base=Path(base)
    sessions=pq.read_table(base/'processed/sessions.parquet',columns=['session_id','harness','parent_session_id']).to_pandas()
    sessions['session_id']=sessions['session_id'].astype(str)
    roots=root_map(sessions)
    hmap=dict(zip(sessions.session_id,sessions.harness.fillna('UNKNOWN').astype(str)))
    pf=pq.ParquetFile(base/'processed/requests.parquet')
    names=set(pf.schema_arrow.names)
    wanted=['session_id','step','model','provider','prompt_tokens','completion_tokens','cache_read_tokens','n_messages','step_trigger']
    cols=[c for c in wanted if c in names]
    req=pq.read_table(base/'processed/requests.parquet',columns=cols).to_pandas()
    for c in wanted:
        if c not in req.columns: req[c]=np.nan if c in ('n_messages',) else 'UNKNOWN'
    req['session_id']=req['session_id'].astype(str)
    req['week']=week
    req['eval_session_id']=week+'::'+req['session_id']
    req['split_group_id']=week+'::'+req['session_id'].map(roots)
    req['harness']=req['session_id'].map(hmap).fillna('UNKNOWN')
    req['input_tokens']=pd.to_numeric(req['prompt_tokens'],errors='coerce')
    req['output_tokens']=pd.to_numeric(req['completion_tokens'],errors='coerce')
    req['cache_read_tokens']=pd.to_numeric(req['cache_read_tokens'],errors='coerce')
    req['step_index']=pd.to_numeric(req['step'],errors='coerce').fillna(0)
    req['n_messages']=pd.to_numeric(req['n_messages'],errors='coerce')
    req['model']=req['model'].fillna('UNKNOWN').astype(str); req['provider']=req['provider'].fillna('UNKNOWN').astype(str)
    req['step_trigger']=req['step_trigger'].fillna('UNKNOWN').astype(str)
    req['row_local_id']=req['step_index'].astype('int64').astype(str)
    maxstep=req.groupby('eval_session_id')['step_index'].transform('max').clip(lower=1)
    req['session_relative_position']=req['step_index']/maxstep
    return req[['week','eval_session_id','split_group_id','row_local_id','model','provider','harness','step_trigger','input_tokens','output_tokens','cache_read_tokens','step_index','n_messages','session_relative_position']]

def prepare_freeinference(acq):
    frames=[]; audit={'weeks':{},'source_filter':'numeric nonnegative P/O/C with 0<=C<=P; session >=10 complete rows; filtering independent of outcomes'}
    for week,base in sorted(acq['freeinference_locations'].items()):
        x=load_fi_week(week,base)
        before=len(x)
        valid=(x.input_tokens.notna()&x.output_tokens.notna()&x.cache_read_tokens.notna()&(x.input_tokens>=0)&(x.output_tokens>=0)&(x.cache_read_tokens>=0)&(x.cache_read_tokens<=x.input_tokens))
        y=x.loc[valid].copy()
        counts=y.groupby('eval_session_id').size(); eligible=set(counts[counts>=MIN_ROWS].index)
        y=y[y.eval_session_id.isin(eligible)].copy()
        audit['weeks'][week]={'all_rows':before,'complete_partition_valid_rows':int(valid.sum()),'eligible_rows':len(y),'eligible_sessions':len(eligible)}
        frames.append(y)
    df=pd.concat(frames,ignore_index=True)
    group_to_stratum=majority_stratum(df,'split_group_id'); splits=split_groups(group_to_stratum)
    g2s={g:s for s,ids in splits.items() for g in ids}
    df['split']=df.split_group_id.map(g2s)
    if df['split'].isna().any(): raise RuntimeError('unassigned FreeInference split group')
    # exact session IDs per split, inherited from root/workflow group
    split_sessions={s:sorted(df.loc[df.split==s,'eval_session_id'].unique().tolist()) for s in ('train','validation','test')}
    split_obj={'dataset':'FreeInference','seed':SPLIT_SEED,'split_groups':splits,'session_ids':split_sessions,
               'counts':{s:{'groups':len(splits[s]),'sessions':len(split_sessions[s]),'rows':int((df.split==s).sum())} for s in splits}}
    atomic_json(CONTRACTS/'splits_freeinference.json',split_obj)
    make_feature_contract('FreeInference',df.columns)
    out=CACHE/'freeinference_complete.parquet'; df.to_parquet(out,index=False,compression='zstd')
    audit['total_rows']=len(df); audit['total_sessions']=df.eval_session_id.nunique(); audit['compact_sha256']=sha256_file(out)
    atomic_json(CHECKS/'freeinference_controlled_source_filter.json',audit)
    return {'path':str(out),'splits':split_obj,'audit':audit}

def prepare_syfi(acq):
    p=Path(acq['syfi_path']); pf=pq.ParquetFile(p); fields=pf.schema_arrow.names; rows=pf.metadata.num_rows
    required=['session_id','provider','model','round_index','input_tokens_total','prefix_tokens','newly_append_tokens','output_tokens']
    missing=[c for c in required if c not in fields]
    pre={'dataset':'UW-SyFI/TraceLab','path':'data/v0.0.2/rounds/train.parquet','sha256':sha256_file(p),'row_count':rows,'field_names':fields,'required_fields':required,'missing_required_fields':missing,'natural_missing_claim_authorized':False}
    if missing:
        pre['schema_gate']='FAIL'; pre['reason']='missing_required_fields'; atomic_json(PKG/'syfi_schema_preflight.json',pre); return {'schema_pass':False}
    extra=[c for c in ['current_user_message_count','current_input_event_count'] if c in fields]
    df=pq.read_table(p,columns=required+extra).to_pandas()
    null_rates={c:float(df[c].isna().mean()) for c in required}
    pre['null_rates']=null_rates
    for c in ['input_tokens_total','prefix_tokens','newly_append_tokens','output_tokens','round_index']:
        df[c]=pd.to_numeric(df[c],errors='coerce')
    core=df[['input_tokens_total','prefix_tokens','newly_append_tokens','output_tokens']]
    nonneg=(core.notna().all(axis=1)&(core>=0).all(axis=1))
    part=(df['input_tokens_total']==df['prefix_tokens']+df['newly_append_tokens'])
    pre['nonnegative_complete_fraction']=float(nonneg.mean())
    pre['partition_identity_fraction']=float(part[nonneg].mean()) if nonneg.any() else 0.0
    pre['providers']=sorted(df.provider.dropna().astype(str).unique().tolist())
    pre['models_count']=int(df.model.nunique(dropna=True)); pre['sessions_count']=int(df.session_id.nunique(dropna=True))
    # strict schema gate: all rows have complete billable fields and exact total=cached+fresh identity
    schema_pass=(rows==665453 and not missing and all(null_rates[c]==0.0 for c in ['session_id','provider','model','input_tokens_total','prefix_tokens','newly_append_tokens','output_tokens']) and bool(nonneg.all()) and bool(part.all()))
    pre['schema_gate']='PASS' if schema_pass else 'FAIL'
    pre['schema_gate_rule']='expected 665453 rows; required grouping/provider/model fields; all core billable fields numeric/nonnegative/non-null; input_tokens_total == prefix_tokens + newly_append_tokens for every row'
    pre['natural_missing_cache_state_documented']=False
    atomic_json(PKG/'syfi_schema_preflight.json',pre)
    if not schema_pass: return {'schema_pass':False,'preflight':pre}
    outdf=pd.DataFrame({
      'week':'SYFI_v0.0.2','eval_session_id':df.session_id.astype(str),'split_group_id':df.session_id.astype(str),
      'row_local_id':df.round_index.astype('int64').astype(str),'model':df.model.astype(str),'provider':df.provider.astype(str),
      'harness':'TraceLab','step_trigger':'UNKNOWN','input_tokens':df.input_tokens_total.astype(float),
      'output_tokens':df.output_tokens.astype(float),'cache_read_tokens':df.prefix_tokens.astype(float),
      'fresh_truth_tokens':df.newly_append_tokens.astype(float),'step_index':df.round_index.astype(float),
      'n_messages':pd.to_numeric(df[extra[0]],errors='coerce') if extra else np.nan,
    })
    mx=outdf.groupby('eval_session_id')['step_index'].transform('max').clip(lower=1); outdf['session_relative_position']=outdf.step_index/mx
    counts=outdf.groupby('eval_session_id').size(); eligible=set(counts[counts>=MIN_ROWS].index); outdf=outdf[outdf.eval_session_id.isin(eligible)].copy()
    group_to_stratum={str(s):str(g.provider.mode().iloc[0]) for s,g in outdf.groupby('split_group_id')}
    splits=split_groups(group_to_stratum); g2s={g:s for s,ids in splits.items() for g in ids}; outdf['split']=outdf.split_group_id.map(g2s)
    split_sessions={s:sorted(outdf.loc[outdf.split==s,'eval_session_id'].unique().tolist()) for s in ('train','validation','test')}
    split_obj={'dataset':'SyFI','seed':SPLIT_SEED,'split_groups':splits,'session_ids':split_sessions,
               'counts':{s:{'groups':len(splits[s]),'sessions':len(split_sessions[s]),'rows':int((outdf.split==s).sum())} for s in splits}}
    atomic_json(CONTRACTS/'splits_syfi.json',split_obj); make_feature_contract('SyFI',outdf.columns)
    out=CACHE/'syfi_complete.parquet'; outdf.to_parquet(out,index=False,compression='zstd')
    pre['eligible_controlled_rows']=len(outdf); pre['eligible_sessions']=outdf.eval_session_id.nunique(); pre['compact_sha256']=sha256_file(out); atomic_json(PKG/'syfi_schema_preflight.json',pre)
    return {'schema_pass':True,'path':str(out),'splits':split_obj,'preflight':pre}

def main():
    acq=read_json(RUNS/'source_acquisition.json')
    fi=prepare_freeinference(acq); sy=prepare_syfi(acq)
    atomic_json(RUNS/'prepared_datasets.json',{'freeinference':fi,'syfi':sy})
    print('DATASETS_PREPARED FreeInference rows=%d SyFI=%s'%(fi['audit']['total_rows'],sy.get('schema_pass')))
if __name__=='__main__': main()
