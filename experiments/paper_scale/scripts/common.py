#!/usr/bin/env python3
from __future__ import annotations
import csv, gzip, hashlib, json, math, os, platform, random, resource, statistics, time
from pathlib import Path
from typing import Iterable, Dict, Any, List, Tuple

PKG=Path(__file__).resolve().parent.parent
RUNS=PKG/'runs'; CACHE=PKG/'cache'; DATA=PKG/'data'; CONTRACTS=PKG/'data_contracts'; MASKS=PKG/'masks'; RAW=PKG/'raw_metrics'; PAPER=PKG/'paper_facing'; CHECKS=PKG/'mechanical_checks'
for p in (RUNS,CACHE,DATA,CONTRACTS,MASKS,RAW,PAPER,CHECKS): p.mkdir(parents=True,exist_ok=True)
RATES=(1.0,10.0,50.0); TPM=1_000_000.0

def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''): h.update(b)
    return h.hexdigest()

def atomic_json(path:Path,obj:Any):
    tmp=Path(str(path)+'.tmp')
    with tmp.open('w',encoding='utf-8',newline='\n') as f:
        json.dump(obj,f,indent=2,sort_keys=True,ensure_ascii=False); f.write('\n')
    tmp.replace(path)

def read_json(path:Path): return json.load(path.open(encoding='utf-8'))

def now_utc():
    import datetime as dt
    return dt.datetime.utcnow().replace(microsecond=0).isoformat()+'Z'

def stable_u64(text:str)->int:
    return int.from_bytes(hashlib.sha256(text.encode('utf-8')).digest()[:8],'big')

def percentile(vals,q):
    vals=sorted(float(x) for x in vals)
    if not vals: return None
    if len(vals)==1: return vals[0]
    x=(len(vals)-1)*q; lo=int(math.floor(x)); hi=int(math.ceil(x)); w=x-lo
    return vals[lo]*(1-w)+vals[hi]*w

def midpoint_wrel(L,U):
    den=(float(L)+float(U))/2.0
    return (float(U)-float(L))/max(den,1e-12)

def exact_cost(P,O,C):
    P=float(P); O=float(O); C=float(C); fresh=max(P-C,0.0)
    return (1.0*C+10.0*fresh+50.0*O)/TPM

def bound_cost(P,O):
    P=float(P); O=float(O)
    return ((1.0*P+50.0*O)/TPM,(10.0*P+50.0*O)/TPM)

def clamp_cache(pred,P):
    return max(0.0,min(float(P),float(pred)))

def proc_peak_rss_mb():
    r=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(r)/(1024.0 if platform.system()!='Darwin' else 1024.0*1024.0)

def csv_gz_writer(path:Path,fields):
    f=gzip.open(str(path),'wt',encoding='utf-8',newline='')
    return f,csv.DictWriter(f,fieldnames=fields)
