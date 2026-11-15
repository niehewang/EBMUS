#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,math,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error
from common import PKG,RUNS,CACHE,CONTRACTS,MASKS,RAW,PAPER,CHECKS,atomic_json,read_json,sha256_file,stable_u64,exact_cost,bound_cost,midpoint_wrel,percentile

CFG=read_json(PKG/'frozen_config.json')
MASK_SEEDS=CFG['masking']['seeds']; QS=CFG['masking']['q']; MECHS=CFG['masking']['mechanisms']
MI_SEEDS=CFG['predictors']['multiple_imputation_seeds']; HASH_DIMS=CFG['predictors']['categorical_hash_dims']
KGRID=CFG['predictors']['knn_k_grid']; RFGRID=CFG['predictors']['rf_grid']
BOOT_SEED=CFG['bootstrap']['seed']; BOOT_N=CFG['bootstrap']['replicates']
METHODS=['zero_fill','complete_case','train_mean','train_median','knn','chained_equations','random_forest_imputation','independent_bounds','EBMUS_evidence_bounds']
POINT=set(['zero_fill','complete_case','train_mean','train_median','knn','random_forest_imputation'])
INTERVAL=set(['chained_equations','independent_bounds','EBMUS_evidence_bounds'])

NUMCOLS=['input_tokens','output_tokens','step_index','n_messages','session_relative_position']
CATCOLS=['model','provider','harness','step_trigger']

def cat_hash_matrix(df):
    out=np.zeros((len(df),HASH_DIMS),dtype=np.float32)
    for col in CATCOLS:
        vals=df[col].fillna('UNKNOWN').astype(str).values
        uniq,inv=np.unique(vals,return_inverse=True)
        uv=np.zeros((len(uniq),HASH_DIMS),dtype=np.float32)
        for i,v in enumerate(uniq):
            d=hashlib.sha256((col+'='+v).encode()).digest(); idx=int.from_bytes(d[:4],'big')%HASH_DIMS; sign=1.0 if d[4]%2==0 else -1.0
            uv[i,idx]+=sign
        out+=uv[inv]
    return out

def numeric_matrix(df):
    a=[]
    for c in NUMCOLS:
        x=pd.to_numeric(df[c],errors='coerce').fillna(0).to_numpy(dtype=np.float64)
        if c in ('input_tokens','output_tokens','n_messages'): x=np.log1p(np.maximum(x,0))
        a.append(x)
    return np.column_stack(a).astype(np.float32)

def raw_features(df): return np.column_stack([numeric_matrix(df),cat_hash_matrix(df)]).astype(np.float32)

def encode_splits(df):
    tr=df[df.split=='train'].copy(); va=df[df.split=='validation'].copy(); te=df[df.split=='test'].copy()
    Xtr=raw_features(tr); Xva=raw_features(va); Xte=raw_features(te)
    mu=Xtr.mean(axis=0,dtype=np.float64); sd=Xtr.std(axis=0,dtype=np.float64); sd[sd<1e-8]=1.0
    Xtr=((Xtr-mu)/sd).astype(np.float32); Xva=((Xva-mu)/sd).astype(np.float32); Xte=((Xte-mu)/sd).astype(np.float32)
    ytr=tr.cache_read_tokens.to_numpy(dtype=np.float64); yva=va.cache_read_tokens.to_numpy(dtype=np.float64); yte=te.cache_read_tokens.to_numpy(dtype=np.float64)
    return tr,va,te,Xtr,Xva,Xte,ytr,yva,yte,mu,sd

def neighbors_predict(nn,Xq,ytr,k,chunk=20000):
    out=np.empty(len(Xq),dtype=np.float64)
    for s in range(0,len(Xq),chunk):
        dist,idx=nn.kneighbors(Xq[s:s+chunk],n_neighbors=k,return_distance=True)
        out[s:s+chunk]=ytr[idx].mean(axis=1)
    return out

def fit_predictions(dataset,df):
    tr,va,te,Xtr,Xva,Xte,ytr,yva,yte,mu,sd=encode_splits(df)
    Pte=te.input_tokens.to_numpy(dtype=np.float64)
    pred={}
    pred['train_mean']=np.minimum(Pte,np.maximum(0.0,np.full(len(te),float(np.mean(ytr)))))
    pred['train_median']=np.minimum(Pte,np.maximum(0.0,np.full(len(te),float(np.median(ytr)))))
    # KNN validation grid. Fit neighbor index once at maximum K.
    maxk=min(max(KGRID),len(ytr)); nn=NearestNeighbors(n_neighbors=maxk,algorithm='kd_tree',leaf_size=40,n_jobs=-1).fit(Xtr)
    val_scores={}
    for k in KGRID:
        kk=min(k,len(ytr)); pv=neighbors_predict(nn,Xva,ytr,kk); val_scores[str(k)]=float(mean_absolute_error(yva,np.minimum(va.input_tokens.to_numpy(float),np.maximum(0,pv))))
    bestk=min(KGRID,key=lambda k:(val_scores[str(k)],k)); pred['knn']=np.minimum(Pte,np.maximum(0,neighbors_predict(nn,Xte,ytr,min(bestk,len(ytr)))))
    # Frozen two-point RF validation grid.
    rf_scores=[]; best=None; best_score=None
    for gi,g in enumerate(RFGRID):
        rf=RandomForestRegressor(n_estimators=g['n_estimators'],max_depth=g['max_depth'],min_samples_leaf=g['min_samples_leaf'],max_features=g['max_features'],random_state=20261109+gi,n_jobs=-1)
        rf.fit(Xtr,ytr); pv=np.minimum(va.input_tokens.to_numpy(float),np.maximum(0,rf.predict(Xva))); sc=float(mean_absolute_error(yva,pv)); rf_scores.append({'grid':g,'validation_mae':sc})
        if best_score is None or sc<best_score: best_score=sc; best=(g,rf)
    pred['random_forest_imputation']=np.minimum(Pte,np.maximum(0,best[1].predict(Xte)))
    # Single-target chained-equations special case: Bayesian conditional regression with posterior predictive draws.
    br=BayesianRidge(compute_score=False); br.fit(Xtr,ytr); mean,std=br.predict(Xte,return_std=True)
    draws=[]
    for s in MI_SEEDS:
        rng=np.random.default_rng(s); d=mean+rng.normal(0.0,std,size=len(mean)); draws.append(np.minimum(Pte,np.maximum(0,d)))
    draws=np.asarray(draws,dtype=np.float64)
    contract={'dataset':dataset,'train_rows':len(tr),'validation_rows':len(va),'test_rows':len(te),'feature_dim':int(Xtr.shape[1]),
              'knn_validation_mae':val_scores,'chosen_k':int(bestk),'rf_validation_grid':rf_scores,'chosen_rf':best[0],
              'chained_equations_implementation':'single-target MICE-style BayesianRidge conditional equation; 20 posterior predictive draws; not exact R mice',
              'multiple_imputation_seeds':MI_SEEDS,'target_clipped_to_input_range':True}
    atomic_json(CHECKS/('predictor_validation_%s.json'%dataset.lower()),contract)
    return te.reset_index(drop=True),pred,draws,contract

def make_masks(dataset,te):
    # canonical order is frozen here and all method code consumes only these saved arrays.
    te=te.sort_values(['eval_session_id','step_index','row_local_id'],kind='mergesort').reset_index(drop=True)
    by={sid:idx.to_numpy(dtype=np.int64) for sid,idx in te.groupby('eval_session_id').groups.items()}
    arrays={}; manifest={}
    for mech in MECHS:
      for q in QS:
       for seed in MASK_SEEDS:
        rng=np.random.default_rng(seed+stable_u64('%s|%s|%.2f'%(dataset,mech,q))%(2**31-1))
        mask=[]
        if mech=='MCAR':
            choose=np.where(rng.random(len(te))<q)[0].astype(np.int64); mask=choose.tolist()
        elif mech=='BLOCK':
            for sid,idx in by.items():
                n=len(idx); k=max(1,min(n,int(round(q*n)))); start=int(rng.integers(0,n-k+1)); mask.extend(idx[start:start+k].tolist())
        elif mech=='CONTEXT_BIASED':
            P=te.input_tokens.to_numpy(dtype=float)
            for sid,idx in by.items():
                n=len(idx); k=max(1,min(n,int(round(q*n)))); w=np.log1p(np.maximum(P[idx],0));
                if not np.isfinite(w).all() or w.sum()<=0: w=np.ones(n)
                prob=w/w.sum(); mask.extend(rng.choice(idx,size=k,replace=False,p=prob).tolist())
        key='%s_q%02d_s%d'%(mech,int(round(q*100)),seed); arr=np.asarray(sorted(set(mask)),dtype=np.int32); arrays[key]=arr
        manifest[key]={'count':int(len(arr)),'fraction_of_test_rows':float(len(arr)/max(len(te),1)),'sha256_raw_int32':hashlib.sha256(arr.tobytes()).hexdigest()}
    npz=MASKS/('masks_%s.npz'%dataset.lower()); np.savez_compressed(npz,**arrays)
    atomic_json(MASKS/('mask_manifest_%s.json'%dataset.lower()),{'dataset':dataset,'canonical_test_order':'sorted eval_session_id, step_index, row_local_id','test_rows':len(te),'masks':manifest,'npz_sha256':sha256_file(npz)})
    return te,arrays

def row_cost(P,O,C): return (C+10.0*np.maximum(P-C,0.0)+50.0*O)/1e6

def eval_dataset(dataset,df):
    # fit uses original dataframe; then re-sort test and predictions consistently by stable row key
    te0,pred0,draws0,contract=fit_predictions(dataset,df)
    # attach predictions before canonical reordering
    for k,v in pred0.items(): te0['_pred_'+k]=v
    for j in range(draws0.shape[0]): te0['_mice_%02d'%j]=draws0[j]
    te,arrays=make_masks(dataset,te0)
    pred={k:te['_pred_'+k].to_numpy(float) for k in pred0}
    draws=np.vstack([te['_mice_%02d'%j].to_numpy(float) for j in range(draws0.shape[0])])
    P=te.input_tokens.to_numpy(float); O=te.output_tokens.to_numpy(float); C=te.cache_read_tokens.to_numpy(float)
    true_row=row_cost(P,O,C); truth_hash_before=hashlib.sha256(C.tobytes()).hexdigest()
    sess_ids,scode=np.unique(te.eval_session_id.astype(str).to_numpy(),return_inverse=True); ns=len(sess_ids)
    true_sess=np.bincount(scode,weights=true_row,minlength=ns); total_rows=np.bincount(scode,minlength=ns)
    rows=[]
    for key,idx in arrays.items():
        prefix,sstr=key.rsplit('_s',1); mech,qstr=prefix.rsplit('_q',1); q=int(qstr)/100.0; seed=int(sstr)
        mask=np.zeros(len(te),dtype=bool); mask[idx]=True; masked_count=np.bincount(scode,weights=mask.astype(float),minlength=ns); active=masked_count>0
        lo_row=true_row.copy(); hi_row=true_row.copy(); lo=(P[idx]+50.0*O[idx])/1e6; hi=(10.0*P[idx]+50.0*O[idx])/1e6; lo_row[idx]=lo; hi_row[idx]=hi
        lo_s=np.bincount(scode,weights=lo_row,minlength=ns); hi_s=np.bincount(scode,weights=hi_row,minlength=ns); exact_w=(hi_s-lo_s)/np.maximum((hi_s+lo_s)/2.0,1e-12)
        def point_rows(method,predcache=None,complete_case=False):
            pr=true_row.copy()
            if complete_case: pr[idx]=0.0
            else:
                cc=np.zeros(len(idx)) if predcache is None else predcache[idx]
                pr[idx]=row_cost(P[idx],O[idx],cc)
            ps=np.bincount(scode,weights=pr,minlength=ns)
            # per-session masked target MAE
            if complete_case: terr=np.full(len(idx),np.nan)
            else:
                cc=np.zeros(len(idx)) if predcache is None else predcache[idx]; terr=np.abs(cc-C[idx])
            tsum=np.bincount(scode[idx],weights=np.nan_to_num(terr,nan=0.0),minlength=ns); tcnt=np.bincount(scode[idx],weights=np.isfinite(terr).astype(float),minlength=ns)
            for s in np.where(active)[0]:
                true=true_sess[s]; center=ps[s]; rel=abs(center-true)/max(abs(true),1e-12); signed=(center-true)/max(abs(true),1e-12)
                rows.append({'dataset':dataset,'mechanism':mech,'q':q,'mask_seed':seed,'method':method,'output_type':'point','session_id':sess_ids[s],
                  'true_settlement':true,'center':center,'lower':np.nan,'upper':np.nan,'absolute_error':abs(center-true),'relative_error':rel,'signed_relative_error':signed,
                  'covered':np.nan,'relative_width':0.0,'unsupported_point_at_1pct':int(exact_w[s]>0.01),'final_at_1pct':1,
                  'masked_rows':int(masked_count[s]),'session_rows':int(total_rows[s]),'request_target_mae':(tsum[s]/tcnt[s] if tcnt[s]>0 else np.nan)})
        point_rows('zero_fill',None,False); point_rows('complete_case',None,True)
        for m in ('train_mean','train_median','knn','random_forest_imputation'): point_rows(m,pred[m],False)
        # chained equations: aggregate 20 imputed session settlements.
        vals=[]
        for j in range(draws.shape[0]):
            pr=true_row.copy(); pr[idx]=row_cost(P[idx],O[idx],draws[j,idx]); vals.append(np.bincount(scode,weights=pr,minlength=ns))
        vals=np.asarray(vals); lower=np.quantile(vals,0.025,axis=0); upper=np.quantile(vals,0.975,axis=0); center=np.median(vals,axis=0); rw=(upper-lower)/np.maximum((upper+lower)/2.0,1e-12)
        for s in np.where(active)[0]:
            true=true_sess[s]
            rows.append({'dataset':dataset,'mechanism':mech,'q':q,'mask_seed':seed,'method':'chained_equations','output_type':'interval','session_id':sess_ids[s],
             'true_settlement':true,'center':center[s],'lower':lower[s],'upper':upper[s],'absolute_error':abs(center[s]-true),'relative_error':abs(center[s]-true)/max(abs(true),1e-12),'signed_relative_error':(center[s]-true)/max(abs(true),1e-12),
             'covered':int(lower[s]<=true<=upper[s]),'relative_width':rw[s],'unsupported_point_at_1pct':np.nan,'final_at_1pct':int(rw[s]<=0.01),'masked_rows':int(masked_count[s]),'session_rows':int(total_rows[s]),'request_target_mae':np.nan})
        # exact evidence bounds: independent atomic cache partition == EBMUS on this separable real masking task.
        for method in ('independent_bounds','EBMUS_evidence_bounds'):
            for s in np.where(active)[0]:
                true=true_sess[s]; rwv=exact_w[s]
                rows.append({'dataset':dataset,'mechanism':mech,'q':q,'mask_seed':seed,'method':method,'output_type':'interval','session_id':sess_ids[s],
                 'true_settlement':true,'center':(lo_s[s]+hi_s[s])/2.0,'lower':lo_s[s],'upper':hi_s[s],'absolute_error':np.nan,'relative_error':np.nan,'signed_relative_error':np.nan,
                 'covered':int(lo_s[s]-1e-12<=true<=hi_s[s]+1e-12),'relative_width':rwv,'unsupported_point_at_1pct':np.nan,'final_at_1pct':int(rwv<=0.01),'masked_rows':int(masked_count[s]),'session_rows':int(total_rows[s]),'request_target_mae':np.nan})
    raw=pd.DataFrame(rows); out=RAW/('%s_session_metrics.csv.gz'%dataset.lower()); raw.to_csv(out,index=False,compression='gzip')
    truth_hash_after=hashlib.sha256(C.tobytes()).hexdigest(); atomic_json(CHECKS/('truth_preservation_%s.json'%dataset.lower()),{'before':truth_hash_before,'after':truth_hash_after,'match':truth_hash_before==truth_hash_after,'masked_copy_only':True})
    return raw

def bootstrap_ci(vals,stat='median'):
    x=np.asarray(vals,dtype=float); x=x[np.isfinite(x)]
    if len(x)<2:return (None,None)
    rng=np.random.default_rng(BOOT_SEED); bs=[]
    for _ in range(BOOT_N):
        z=x[rng.integers(0,len(x),len(x))]; bs.append(float(np.median(z) if stat=='median' else np.mean(z)))
    return float(np.quantile(bs,.025)),float(np.quantile(bs,.975))

def aggregate(raw):
    records=[]
    for (ds,mech,q,method,otype),g in raw.groupby(['dataset','mechanism','q','method','output_type']):
        # session-average across mask seeds before primary distribution statistics
        numcols=['relative_error','signed_relative_error','covered','relative_width','unsupported_point_at_1pct','final_at_1pct']
        sg=g.groupby('session_id')[numcols].mean(numeric_only=True).reset_index()
        rec={'dataset':ds,'mechanism':mech,'q':q,'method':method,'output_type':otype,'sessions':len(sg),'session_mask_rows':len(g)}
        if otype=='point':
            vals=sg.relative_error.dropna().to_numpy(); rec.update({'median_relative_error':float(np.median(vals)),'p25_relative_error':float(np.quantile(vals,.25)),'p75_relative_error':float(np.quantile(vals,.75)),'p95_relative_error':float(np.quantile(vals,.95)),'unsupported_point_rate':float(sg.unsupported_point_at_1pct.mean())})
            lo,hi=bootstrap_ci(vals,'median'); rec['primary_metric']='median_relative_error'; rec['bootstrap95_low']=lo; rec['bootstrap95_high']=hi
            seedmed=g.groupby('mask_seed').relative_error.median(); rec['mean_across_seed_medians']=float(seedmed.mean())
        else:
            vals=sg.relative_width.dropna().to_numpy(); rec.update({'coverage':float(sg.covered.mean()),'median_relative_width':float(np.median(vals)),'p25_relative_width':float(np.quantile(vals,.25)),'p75_relative_width':float(np.quantile(vals,.75)),'p95_relative_width':float(np.quantile(vals,.95)),'finality_1pct_fraction':float(sg.final_at_1pct.mean())})
            lo,hi=bootstrap_ci(vals,'median'); rec['primary_metric']='median_relative_width'; rec['bootstrap95_low']=lo; rec['bootstrap95_high']=hi
            seedmed=g.groupby('mask_seed').relative_width.median(); rec['mean_across_seed_medians']=float(seedmed.mean())
        records.append(rec)
    return pd.DataFrame(records)

def build_table(agg,datasets):
    rows=[]
    for m in METHODS:
        r={'method':m,'output_type':'point' if m in POINT else 'interval'}
        for ds in datasets:
          for mech in MECHS:
            x=agg[(agg.dataset==ds)&(agg.mechanism==mech)&(np.isclose(agg.q,.20))&(agg.method==m)]
            prefix='%s__%s20'%(ds,mech)
            if len(x)==1:
                z=x.iloc[0]
                if z.output_type=='point':
                    r[prefix+'__median_relative_error']=z.get('median_relative_error'); r[prefix+'__p95_relative_error']=z.get('p95_relative_error'); r[prefix+'__unsupported_point_rate']=z.get('unsupported_point_rate')
                else:
                    r[prefix+'__coverage']=z.get('coverage'); r[prefix+'__median_relative_width']=z.get('median_relative_width'); r[prefix+'__finality_1pct_fraction']=z.get('finality_1pct_fraction')
        rows.append(r)
    pd.DataFrame(rows).to_csv(PAPER/'paper_table_main_comparison.csv',index=False)

def provider_generalization(raw):
    # raw session metrics do not contain provider; join test-session dominant provider from compact datasets.
    out=[]
    for ds in raw.dataset.unique():
        p=CACHE/(ds.lower()+'_complete.parquet')
        if not p.exists():continue
        df=pd.read_parquet(p,columns=['eval_session_id','provider','split']); df=df[df.split=='test']; dom=df.groupby('eval_session_id').provider.agg(lambda x:x.mode().iloc[0] if len(x.mode()) else 'UNKNOWN')
        x=raw[(raw.dataset==ds)&(np.isclose(raw.q,.20))].copy(); x['provider']=x.session_id.map(dom)
        for (mech,method,provider),g in x.groupby(['mechanism','method','provider']):
            rec={'dataset':ds,'mechanism':mech,'q':.2,'method':method,'provider':provider,'sessions':g.session_id.nunique()}
            if g.output_type.iloc[0]=='point': rec.update({'metric':'median_relative_error','value':float(g.relative_error.median())})
            else: rec.update({'metric':'median_relative_width','value':float(g.relative_width.median()),'coverage':float(g.covered.mean())})
            out.append(rec)
    pd.DataFrame(out).to_csv(PAPER/'fig_real_trace_generalization.csv',index=False)

def main():
    prep=read_json(RUNS/'prepared_datasets.json'); allraw=[]; datasets=[]
    fi=pd.read_parquet(prep['freeinference']['path']); allraw.append(eval_dataset('FreeInference',fi)); datasets.append('FreeInference')
    if prep['syfi'].get('schema_pass'):
        sy=pd.read_parquet(prep['syfi']['path']); allraw.append(eval_dataset('SyFI',sy)); datasets.append('SyFI')
    raw=pd.concat(allraw,ignore_index=True); agg=aggregate(raw); agg.to_csv(PAPER/'fig_main_comparison.csv',index=False); build_table(agg,datasets); provider_generalization(raw)
    atomic_json(CHECKS/'main_comparison_mechanical.json',{'datasets':datasets,'methods':METHODS,'all_methods_use_single_saved_mask_npz_per_dataset':True,'target_leakage_prohibited':True,'truth_preservation_files':[str(x.name) for x in CHECKS.glob('truth_preservation_*.json')],'scientific_outcome':'NOT_ADJUDICATED'})
    print('MAIN_COMPARISON_COMPLETE datasets=%s raw_rows=%d'%(datasets,len(raw)))
if __name__=='__main__': main()
