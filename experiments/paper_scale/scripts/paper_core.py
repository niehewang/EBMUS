#!/usr/bin/env python3
from __future__ import annotations
import itertools, math, random, time
from collections import defaultdict
import numpy as np
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
from scipy.sparse import lil_matrix, csr_matrix

RATE_VECTORS={'low':(10.0,11.0,12.0),'medium':(1.0,2.0,4.0),'high':(1.0,4.0,16.0)}

def children(parent):
    ch=[[] for _ in parent]
    for v in range(1,len(parent)): ch[parent[v]].append(v)
    return ch

def subtree_sums(parent,local):
    s=list(local)
    for v in range(len(parent)-1,0,-1): s[parent[v]]+=s[v]
    return s

def make_tree(rng,n): return [-1]+[rng.randrange(v) for v in range(1,n)]

def generate_instance(rng,n,missing_p=.10,hidden_p=.10,summary_cov=.50,rate_label='medium',ambiguity_density='none'):
    parent=make_tree(rng,n); ch=children(parent); cap=[rng.choice((8,12,16)) for _ in range(n)]; q=[rng.randint(0,cap[i]) for i in range(n)]; true_k=[rng.randrange(3) for _ in range(n)]
    status=[]
    for _ in range(n):
        x=rng.random(); status.append('missing' if x<missing_p else ('hidden' if x<missing_p+hidden_p else 'exact'))
    ss=subtree_sums(parent,q); ev={0:ss[0]}
    chosen_internal=[]
    for v in range(1,n):
        if ch[v] and rng.random()<summary_cov: ev[v]=ss[v]; chosen_internal.append(v)
    ng={'none':0,'low':1,'medium':3}[ambiguity_density]
    groups=[]
    for gid in range(ng):
        if rng.random()<.5:
            amt=rng.choice((1,2,3)); k=rng.randrange(3); bit=rng.randrange(2)
            groups.append({'group_id':gid,'type':'duplicate_distinct','amount':amt,'rate_class':k,'true_bit':bit})
        else:
            groups.append({'group_id':gid,'type':'correction_branch','branch0_amount':rng.choice((1,2,3)),'branch0_rate_class':rng.randrange(3),'branch1_amount':rng.choice((1,2,3)),'branch1_rate_class':rng.randrange(3),'true_bit':rng.randrange(2)})
    return {'n':n,'parent':parent,'q':q,'true_k':true_k,'cap':cap,'status':status,'subtree_evidence':ev,'internal_evidence_nodes':chosen_internal,'rates':RATE_VECTORS[rate_label],'rate_label':rate_label,'groups':groups}

def ambiguity_cost_one(g,bit,rates):
    if g['type']=='duplicate_distinct':
        one=g['amount']*rates[g['rate_class']]; return one*(1+bit)
    if g['type']=='correction_branch':
        return (g['branch0_amount']*rates[g['branch0_rate_class']] if bit==0 else g['branch1_amount']*rates[g['branch1_rate_class']])
    raise ValueError(g['type'])

def ambiguity_bounds(groups,rates,resolved=None):
    resolved=resolved or {}; L=U=0.0
    for g in groups:
        gid=g['group_id']
        if gid in resolved:
            c=ambiguity_cost_one(g,resolved[gid],rates); L+=c; U+=c
        else:
            a=ambiguity_cost_one(g,0,rates); b=ambiguity_cost_one(g,1,rates); L+=min(a,b); U+=max(a,b)
    return L,U

def true_cost(inst):
    rates=inst['rates']; base=sum(inst['q'][v]*rates[inst['true_k'][v]] for v in range(inst['n'])); amb=sum(ambiguity_cost_one(g,g['true_bit'],rates) for g in inst.get('groups',[])); return base+amb

def _local_bounds_coeff(inst,mode):
    rates=inst['rates']; lo=[]; hi=[]; c=[]
    for q,k,cap,status in zip(inst['q'],inst['true_k'],inst['cap'],inst['status']):
        if status=='missing': lo.append(0.0); hi.append(float(cap)); c.append(rates[k])
        elif status=='hidden': lo.append(float(q)); hi.append(float(q)); c.append(min(rates) if mode=='min' else max(rates))
        elif status=='exact': lo.append(float(q)); hi.append(float(q)); c.append(rates[k])
        else: raise ValueError(status)
    return lo,hi,c

def laminar_flow_solve(inst,mode):
    if mode not in ('min','max'): raise ValueError(mode)
    n=inst['n']; parent=inst['parent']; ev=inst['subtree_evidence']; rates=inst['rates']
    if 0 not in ev:return {'feasible':False,'reason':'root_total_missing'}
    constrained=set(ev); current=[None]*n; region_nodes=defaultdict(list); cp={}
    for v in range(n):
        prev=current[parent[v]] if v else None
        if v in constrained: cp[v]=prev; current[v]=v
        else: current[v]=prev
        if current[v] is None:return {'feasible':False,'reason':'uncovered_root'}
        region_nodes[current[v]].append(v)
    cc=defaultdict(list)
    for c,p in cp.items():
        if p is not None:cc[p].append(c)
    lo,hi,coeff=_local_bounds_coeff(inst,mode); amount=[None]*n
    for c in constrained:
        residual=ev[c]-sum(ev[d] for d in cc[c]); nodes=region_nodes[c]; slo=sum(lo[v] for v in nodes); shi=sum(hi[v] for v in nodes)
        if residual<slo-1e-9 or residual>shi+1e-9:return {'feasible':False,'reason':'region_capacity'}
        for v in nodes:amount[v]=lo[v]
        extra=residual-slo; order=sorted(nodes,key=lambda v:(coeff[v] if mode=='min' else -coeff[v],v))
        for v in order:
            add=min(extra,hi[v]-lo[v]); amount[v]+=add; extra-=add
            if abs(extra)<1e-9:break
        if abs(extra)>1e-7:return {'feasible':False,'reason':'allocation_error'}
    return {'feasible':True,'cost':sum(amount[v]*coeff[v] for v in range(n)),'amounts':amount}

def evidence_bounds(inst,resolved=None):
    lo=laminar_flow_solve(inst,'min'); hi=laminar_flow_solve(inst,'max')
    if not lo.get('feasible') or not hi.get('feasible'): return {'feasible':False}
    aL,aU=ambiguity_bounds(inst.get('groups',[]),inst['rates'],resolved)
    return {'feasible':True,'L':lo['cost']+aL,'U':hi['cost']+aU}

def independent_bounds(inst,resolved=None):
    rates=inst['rates']; L=U=0.0
    for q,k,cap,s in zip(inst['q'],inst['true_k'],inst['cap'],inst['status']):
        if s=='exact':L+=q*rates[k];U+=q*rates[k]
        elif s=='missing':L+=0;U+=cap*rates[k]
        elif s=='hidden':L+=q*min(rates);U+=q*max(rates)
    a,b=ambiguity_bounds(inst.get('groups',[]),rates,resolved); return {'feasible':True,'L':L+a,'U':U+b}

def _generic_matrix(inst):
    n=inst['n']; ch=children(inst['parent']); ev=inst['subtree_evidence']; m=n+len(ev); A=lil_matrix((m,2*n),dtype=float); b=np.zeros(m); r=0
    for v in range(n):
        A[r,n+v]=1.0; A[r,v]=-1.0
        for c in ch[v]:A[r,n+c]-=1.0
        r+=1
    for v,total in sorted(ev.items()): A[r,n+v]=1.0; b[r]=float(total); r+=1
    return csr_matrix(A),b

def generic_lp(inst,mode,time_limit=60):
    n=inst['n']; lo,hi,cq=_local_bounds_coeff(inst,mode); A,b=_generic_matrix(inst); c=np.zeros(2*n); c[:n]=np.array(cq)*(1.0 if mode=='min' else -1.0)
    bounds=list(zip(lo,hi))+[(0,None)]*n
    t=time.perf_counter(); res=linprog(c,A_eq=A,b_eq=b,bounds=bounds,method='highs',options={'time_limit':float(time_limit)}); dt=time.perf_counter()-t
    if not res.success:return {'feasible':False,'success':False,'status':str(res.message),'runtime':dt,'timeout':'time' in str(res.message).lower()}
    val=float(res.fun)*(1.0 if mode=='min' else -1.0); return {'feasible':True,'success':True,'cost':val,'runtime':dt,'status':str(res.message),'timeout':False}

def generic_milp(inst,mode,time_limit=60,resolved=None):
    resolved=resolved or {}; groups=inst.get('groups',[]); unresolved=[g for g in groups if g['group_id'] not in resolved]
    n=inst['n']; lo,hi,cq=_local_bounds_coeff(inst,mode); Aeq,beq=_generic_matrix(inst); nb=len(unresolved); N=2*n+nb
    c=np.zeros(N); c[:n]=np.array(cq)
    const=0.0; rates=inst['rates']
    for g in groups:
        gid=g['group_id']
        if gid in resolved: const+=ambiguity_cost_one(g,resolved[gid],rates)
    for j,g in enumerate(unresolved):
        c0=ambiguity_cost_one(g,0,rates); c1=ambiguity_cost_one(g,1,rates); const+=c0; c[2*n+j]=c1-c0
    sign=1.0 if mode=='min' else -1.0; c*=sign
    A=lil_matrix((Aeq.shape[0],N)); A[:,:2*n]=Aeq; A=csr_matrix(A)
    lower=np.array(lo+[0.0]*n+[0.0]*nb,float); upper=np.array(hi+[np.inf]*n+[1.0]*nb,float)
    integ=np.array([0]*(2*n)+[1]*nb,int)  # accounting flow is continuous LP; only scoped ambiguity variables are binary
    cons=LinearConstraint(A,beq,beq); t=time.perf_counter(); res=milp(c,integrality=integ,bounds=Bounds(lower,upper),constraints=cons,options={'time_limit':float(time_limit),'presolve':True}); dt=time.perf_counter()-t
    if not res.success:return {'feasible':False,'success':False,'runtime':dt,'status':str(res.message),'timeout':'time' in str(res.message).lower()}
    val=float(res.fun)*sign+const; return {'feasible':True,'success':True,'cost':val,'runtime':dt,'status':str(res.message),'timeout':False}

def brute_force(inst,resolved=None):
    resolved=resolved or {}; n=inst['n']; rates=inst['rates']; missing=[i for i,s in enumerate(inst['status']) if s=='missing']; hidden=[i for i,s in enumerate(inst['status']) if s=='hidden']; groups=inst.get('groups',[]); unresolved=[g for g in groups if g['group_id'] not in resolved]
    L=U=None; count=0
    ad=[range(inst['cap'][v]+1) for v in missing]; kd=[range(3) for _ in hidden]
    for av in itertools.product(*ad) if ad else [()]:
      q=list(inst['q'])
      for v,a in zip(missing,av):q[v]=a
      ss=subtree_sums(inst['parent'],q)
      if any(abs(ss[v]-t)>1e-9 for v,t in inst['subtree_evidence'].items()):continue
      for kv in itertools.product(*kd) if kd else [()]:
       ks=list(inst['true_k'])
       for v,k in zip(hidden,kv):ks[v]=k
       base=sum(q[v]*rates[ks[v]] for v in range(n))
       for bits in itertools.product((0,1),repeat=len(unresolved)) if unresolved else [()]:
        m=dict(resolved); m.update({g['group_id']:b for g,b in zip(unresolved,bits)}); val=base+sum(ambiguity_cost_one(g,m[g['group_id']],rates) for g in groups); count+=1
        L=val if L is None or val<L else L; U=val if U is None or val>U else U
    return {'feasible':L is not None,'L':L,'U':U,'completion_count':count}

def refinement_conditions(inst):
    # Construct nested evidence sets from one latent world, independent of observed outcomes.
    base={k:(list(v) if isinstance(v,list) else dict(v) if isinstance(v,dict) else v) for k,v in inst.items() if k not in ('groups',)}; base['groups']=[dict(g) for g in inst.get('groups',[])]
    ss=subtree_sums(inst['parent'],inst['q']); root_only={0:ss[0]}; full_ev={0:ss[0]};
    for v in inst.get('internal_evidence_nodes',[]): full_ev[v]=ss[v]
    out=[]
    e0=dict(base);e0['subtree_evidence']=root_only; out.append(('E0',e0,{}))
    e1=dict(base);e1['subtree_evidence']=full_ev; out.append(('E1',e1,{}))
    e2=dict(e1);e2['status']=['exact' if s=='hidden' else s for s in inst['status']]; out.append(('E2',e2,{}))
    dup={g['group_id']:g['true_bit'] for g in inst.get('groups',[]) if g['type']=='duplicate_distinct'}; out.append(('E3',dict(e2),dup))
    e4=dict(e2);e4['status']=['exact']*inst['n']; corr={g['group_id']:g['true_bit'] for g in inst.get('groups',[])}; out.append(('E4',e4,corr))
    return out
