#!/usr/bin/env python3
from __future__ import print_function
import hashlib,json,os,shutil,sys,time,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent; PKG=HERE.parent
RUNS=PKG/'runs'; DATA=PKG/'data'; RUNS.mkdir(exist_ok=True); DATA.mkdir(exist_ok=True)
MIRROR=os.environ.get('EBMUS_HF_ENDPOINT','https://huggingface.co').rstrip('/')
FI_REPO='harvardMadsys/freeinference_agentic_trace'
SYFI_REPO='UW-SyFI/TraceLab'
SYFI_REL='data/v0.0.2/rounds/train.parquet'
SYFI_REV='7256bbf77e76d5cb83f4ddbd5ad31d8faee9dc7a'
SYFI_SHA='cb8c2b1d23307090c8057c747f955663bacde16a49f84a3c5071a1e2e96a7ab8'
FI_HASHES=json.load(open(PKG/'freeinference_frozen_hashes.json'))['weeks']
MANUAL=RUNS/'MANUAL_RELAY_REQUIRED.json'

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(16*1024*1024),b''): h.update(b)
 return h.hexdigest()

def atomic(p,o):
 q=Path(str(p)+'.tmp'); q.write_text(json.dumps(o,indent=2,sort_keys=True)+'\n'); q.replace(p)

def download(url,dest):
 part=Path(str(dest)+'.part'); part.parent.mkdir(parents=True,exist_ok=True)
 if part.exists(): part.unlink()
 req=urllib.request.Request(url,headers={'User-Agent':'EBMUS-repro/1.0'})
 with urllib.request.urlopen(req,timeout=180) as r, part.open('wb') as out:
  while True:
   b=r.read(8*1024*1024)
   if not b: break
   out.write(b)
 part.replace(dest)

def valid_files(base,expected):
 facts={}; ok=True
 for rel,rec in expected.items():
  p=base/rel; present=p.is_file(); actual=sha(p) if present else None
  m=present and actual==rec['sha256']
  facts[rel]={'path':str(p),'present':present,'bytes':p.stat().st_size if present else None,'sha256':actual,'expected_sha256':rec['sha256'],'match':m}
  ok=ok and m
 return ok,facts

def candidate_dirs(week):
    roots=[]
    external=os.environ.get('EBMUS_FI_DATA_ROOT','').strip()
    if external:
        roots.append(Path(external)/week)
    roots.append(DATA/'freeinference'/'release'/week)
    return roots

def acquire_fi(prov):
 loc={}
 for i,(week,expected) in enumerate(sorted(FI_HASHES.items()),1):
  found=None; facts=None
  for c in candidate_dirs(week):
   ok,ff=valid_files(c,expected)
   if ok: found=c; facts=ff; break
  if found is not None:
   print('[FI %02d/16] REUSE %s %s'%(i,week,found),flush=True)
   loc[week]=str(found); prov['freeinference'][week]={'channel':'reused_verified_local','base':str(found),'files':facts}; continue
  base=DATA/'freeinference'/'release'/week
  print('[FI %02d/16] ACQUIRE %s'%(i,week),flush=True)
  attempts=[]
  for rel,rec in expected.items():
   p=base/rel; p.parent.mkdir(parents=True,exist_ok=True)
   success=False
   for a in (1,2,3):
    url='%s/datasets/%s/resolve/main/release/%s/%s'%(MIRROR,FI_REPO,week,rel)
    try:
     download(url,p); actual=sha(p); good=actual==rec['sha256']
     attempts.append({'file':rel,'attempt':a,'url':url,'sha256':actual,'expected':rec['sha256'],'match':good})
     if not good: raise RuntimeError('frozen SHA256 mismatch')
     success=True; break
    except Exception as e:
     attempts.append({'file':rel,'attempt':a,'error':'%s: %s'%(e.__class__.__name__,e)})
     try:
      if p.exists(): p.unlink()
     except Exception: pass
     if a<3: time.sleep(5*a)
   if not success:
    atomic(MANUAL,{'kind':'FREEINFERENCE','week':week,'failed_file':rel,'target_directory':str(base),'required_files':list(expected),'attempts':attempts})
    prov['freeinference'][week]={'channel':'mirror_failed_manual_required','attempts':attempts}; atomic(RUNS/'source_acquisition.json',prov)
    return None,20
  ok,facts=valid_files(base,expected)
  if not ok:
   atomic(MANUAL,{'kind':'FREEINFERENCE','week':week,'reason':'post_acquisition_hash_failure','target_directory':str(base),'required_files':list(expected)})
   return None,20
  loc[week]=str(base); prov['freeinference'][week]={'channel':'domestic_hf_mirror','base':str(base),'files':facts,'attempts':attempts}
  atomic(RUNS/'source_acquisition.json',prov)
 return loc,0

def acquire_syfi(prov):
 p=DATA/'syfi'/SYFI_REL
 if p.is_file() and sha(p)==SYFI_SHA:
  prov['syfi']={'channel':'preexisting_exact','path':str(p),'revision':SYFI_REV,'sha256':SYFI_SHA,'bytes':p.stat().st_size}; return str(p),0
 url='%s/datasets/%s/resolve/%s/%s'%(MIRROR,SYFI_REPO,SYFI_REV,SYFI_REL)
 attempts=[]
 for a in (1,2,3):
  try:
   print('[SYFI] acquire attempt',a,flush=True); download(url,p); actual=sha(p); good=actual==SYFI_SHA
   attempts.append({'attempt':a,'url':url,'sha256':actual,'expected':SYFI_SHA,'match':good,'bytes':p.stat().st_size})
   if not good: raise RuntimeError('SyFI frozen parquet SHA mismatch')
   prov['syfi']={'channel':'domestic_hf_mirror','path':str(p),'revision':SYFI_REV,'sha256':actual,'bytes':p.stat().st_size,'attempts':attempts}; return str(p),0
  except Exception as e:
   attempts.append({'attempt':a,'error':'%s: %s'%(e.__class__.__name__,e)})
   try:
    if p.exists(): p.unlink()
   except Exception: pass
   if a<3: time.sleep(5*a)
 atomic(MANUAL,{'kind':'SYFI','failed_file':SYFI_REL,'expected_sha256':SYFI_SHA,'target_path':str(p),'attempts':attempts})
 prov['syfi']={'channel':'mirror_failed_manual_required','attempts':attempts}; atomic(RUNS/'source_acquisition.json',prov)
 return None,20

def main():
 prov={'mirror':MIRROR,'freeinference':{},'syfi':{}}
 # honor existing manual relay if it has since been supplied
 if MANUAL.exists():
  m=json.load(open(MANUAL))
  if m.get('kind')=='SYFI':
   p=Path(m['target_path'])
   if p.is_file() and sha(p)==m['expected_sha256']: MANUAL.unlink()
   else:
    print('AWAITING_MANUAL_RELAY SYFI',p); return 20
  elif m.get('kind')=='FREEINFERENCE':
   base=Path(m['target_directory']); exp=FI_HASHES[m['week']]; ok,_=valid_files(base,exp)
   if ok: MANUAL.unlink()
   else:
    print('AWAITING_MANUAL_RELAY FREEINFERENCE',m['week'],base); return 20
 loc,rc=acquire_fi(prov)
 if rc: return rc
 syfi,rc=acquire_syfi(prov)
 if rc: return rc
 prov['freeinference_locations']=loc; prov['syfi_path']=syfi
 atomic(RUNS/'source_acquisition.json',prov)
 print('ALL_PUBLIC_SOURCES_READY')
 return 0
if __name__=='__main__': sys.exit(main())
