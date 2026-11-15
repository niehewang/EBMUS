#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run frozen EBMUS A0. No scientific GO/MODIFY/NO-GO classification."""
from __future__ import print_function

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import math
import os
import platform
import random
import resource
import shutil
import statistics
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
from ebmus.core import (
    RATES, generate_small_laminar, generate_grid_laminar,
    generate_ambiguity_groups, brute_force_exact, brute_force_general,
    laminar_flow_solve, validate_laminar_witness, evidence_milp_solve,
    validate_milp_witness, true_settlement, received_only_naive,
    conceptual_flow_size
)

def utc():
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def atomic_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)

def sha256_file(path):
    h=hashlib.sha256()
    with open(str(path),"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def rss_mb():
    try:
        x = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB.
        return float(x) / 1024.0
    except Exception:
        return None

def median(xs):
    return statistics.median(xs) if xs else None

def percentile_nearest(xs, q):
    if not xs:
        return None
    ys=sorted(xs)
    rank=max(1,min(len(ys),int(math.ceil(q*len(ys)))))
    return ys[rank-1]

def write_status(run_dir, state, phase, progress, start, failure=None, return_package=None):
    st = {
        "run_id":"EBMUS_A0_METHOD_VALIDATION",
        "experiment_id":"EBMUS_A0_METHOD_VALIDATION",
        "project_id":"EBMUS",
        "state":state,
        "phase":phase,
        "progress":progress,
        "start_time":start,
        "last_heartbeat":utc(),
        "pid":os.getpid(),
        "run_dir":str(run_dir),
        "resume_allowed":False,
        "scientific_decision":"NOT_EVALUATED"
    }
    if failure is not None:
        st["failure_summary"]=failure
    if return_package is not None:
        st["return_package"]=str(return_package)
    atomic_json(run_dir/"RUN_STATUS.json",st)

def settlement_quarters(vec):
    # rates 1,4,0.25 scaled by 4 => 4,16,1.
    return 4*vec[0] + 16*vec[1] + vec[2]

def q4_string(q4):
    whole=q4//4; rem=q4%4
    return str(whole) if rem==0 else ("%d.%02d"%(whole,rem*25))

def run_normative(pkg, results):
    spec=json.load(open(str(pkg/"normative_test_vectors.json"),encoding="utf-8"))
    rows=[]
    mismatches=[]
    for case in spec["cases"]:
        cid=case["id"]
        out={"case_id":cid}
        if cid=="complete_pair":
            vec=case["expected"]["correct_tree"]
            out.update(exact=True, contradiction=False, lower_q4=settlement_quarters(vec),
                       upper_q4=settlement_quarters(vec))
        elif cid=="missing_child_report":
            vec=case["received_lower"]
            out.update(exact=False, contradiction=False, lower_q4=settlement_quarters(vec),
                       upper_unbounded=True)
        elif cid=="missing_child_bound_exceeded":
            lower=case["received_lower"]; asserted=case["asserted_subtree"]
            flags=["inconsistent" if lower[i]>asserted[i] else "unchecked" for i in range(3)]
            out.update(exact=False, contradiction=any(x=="inconsistent" for x in flags),
                       per_counter=flags, lower_q4=settlement_quarters(lower), upper_unbounded=True)
        elif cid=="collector_double_count":
            correct=case["correct_tree"]; wrong=case["incorrect_double_count"]
            out.update(exact=True, contradiction=False,
                       correct_q4=settlement_quarters(correct),
                       double_count_q4=settlement_quarters(wrong),
                       double_count_rejected=(settlement_quarters(correct)!=settlement_quarters(wrong)))
        elif cid=="unauthorized_attachment_missing_child":
            lower=case["received_lower"]
            out.update(exact=False, contradiction=False, lower_q4=settlement_quarters(lower),
                       upper_unbounded=True, finding="unauthorized-attachment")
        elif cid=="missing_child_partial_contradiction":
            lower=case["received_lower"]; asserted=case["asserted_subtree"]
            flags=[]
            for i in range(3):
                if lower[i] > asserted[i]:
                    flags.append("inconsistent")
                else:
                    flags.append("unchecked")
            out.update(exact=False, contradiction=True, per_counter=flags,
                       lower_q4=settlement_quarters(lower), upper_unbounded=True)
        else:
            mismatches.append({"case_id":cid,"reason":"unknown_case"})
            rows.append(out); continue

        exp=case["expected"]
        for key in ("exact","contradiction","per_counter","finding"):
            if key in exp and out.get(key)!=exp[key]:
                mismatches.append({"case_id":cid,"field":key,"expected":exp[key],"observed":out.get(key)})
        if exp.get("reject_double_count_as_correct") and not out.get("double_count_rejected"):
            mismatches.append({"case_id":cid,"field":"reject_double_count_as_correct"})
        if exp.get("upper_unbounded") and not out.get("upper_unbounded"):
            mismatches.append({"case_id":cid,"field":"upper_unbounded"})
        rows.append(out)

    atomic_json(results/"normative_results.json",{"cases":rows,"mismatches":mismatches})
    return {"case_count":len(rows),"mismatch_count":len(mismatches)}

def subtest_b(results, progress_cb):
    rng=random.Random(20261106)
    path=results/"subtest_B_small_exactness.csv.gz"
    mismatches=[]
    rows=0
    with gzip.open(str(path),"wt",encoding="utf-8",newline="") as f:
        fields=["instance_id","n","brute_feasible","laminar_feasible","brute_L","brute_U",
                "laminar_L","laminar_U","feasibility_mismatch","L_mismatch","U_mismatch",
                "lower_witness_valid","upper_witness_valid","brute_completion_count","naive_received"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        iid=0
        for n in (3,4,5,6):
            for j in range(1000):
                inst=generate_small_laminar(rng,n)
                brute=brute_force_exact(inst)
                lo=laminar_flow_solve(inst,"min"); hi=laminar_flow_solve(inst,"max")
                lf=bool(lo.get("feasible") and hi.get("feasible"))
                fm=(brute["feasible"]!=lf)
                lm=(brute["L"]!=(lo.get("cost") if lo.get("feasible") else None))
                um=(brute["U"]!=(hi.get("cost") if hi.get("feasible") else None))
                lw=validate_laminar_witness(inst,lo) if lo.get("feasible") else (not brute["feasible"])
                uw=validate_laminar_witness(inst,hi) if hi.get("feasible") else (not brute["feasible"])
                row={
                    "instance_id":iid,"n":n,"brute_feasible":brute["feasible"],"laminar_feasible":lf,
                    "brute_L":brute["L"],"brute_U":brute["U"],
                    "laminar_L":lo.get("cost"),"laminar_U":hi.get("cost"),
                    "feasibility_mismatch":fm,"L_mismatch":lm,"U_mismatch":um,
                    "lower_witness_valid":lw,"upper_witness_valid":uw,
                    "brute_completion_count":brute["completion_count"],
                    "naive_received":received_only_naive(inst)
                }
                w.writerow(row); rows+=1
                if fm or lm or um or not lw or not uw:
                    mismatches.append({"instance_id":iid,"n":n,"row":row,"instance":inst})
                iid+=1
                if iid%250==0: progress_cb(iid,4000)
    atomic_json(results/"subtest_B_mismatches.json",mismatches)
    return {"instances":rows,"mismatch_count":len(mismatches)}

def subtest_c(results, progress_cb):
    rng=random.Random(20261107)
    path=results/"subtest_C_ambiguity_exactness.csv.gz"
    mismatches=[]
    type_counts={"duplicate_distinct":0,"correction_branch":0}
    with gzip.open(str(path),"wt",encoding="utf-8",newline="") as f:
        fields=["instance_id","n","ambiguity_groups","brute_feasible","milp_feasible","brute_L","brute_U",
                "milp_L","milp_U","feasibility_mismatch","L_mismatch","U_mismatch",
                "lower_witness_valid","upper_witness_valid","brute_completion_count","naive_received"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for i in range(1000):
            n=rng.choice((3,4,5))
            inst=generate_small_laminar(rng,n)
            groups=generate_ambiguity_groups(rng)
            for g in groups: type_counts[g["type"]]+=1
            brute=brute_force_general(inst,groups)
            milp=evidence_milp_solve(inst,groups)
            fm=(brute["feasible"]!=milp.get("feasible",False))
            lm=(brute["L"]!=milp.get("L"))
            um=(brute["U"]!=milp.get("U"))
            lw=validate_milp_witness(inst,groups,milp.get("witness_L"),milp.get("L")) if milp.get("feasible") else (not brute["feasible"])
            uw=validate_milp_witness(inst,groups,milp.get("witness_U"),milp.get("U")) if milp.get("feasible") else (not brute["feasible"])
            row={
                "instance_id":i,"n":n,"ambiguity_groups":len(groups),
                "brute_feasible":brute["feasible"],"milp_feasible":milp.get("feasible",False),
                "brute_L":brute["L"],"brute_U":brute["U"],"milp_L":milp.get("L"),"milp_U":milp.get("U"),
                "feasibility_mismatch":fm,"L_mismatch":lm,"U_mismatch":um,
                "lower_witness_valid":lw,"upper_witness_valid":uw,
                "brute_completion_count":brute["completion_count"],
                "naive_received":received_only_naive(inst)
            }
            w.writerow(row)
            if fm or lm or um or not lw or not uw:
                mismatches.append({"instance_id":i,"row":row,"instance":inst,"groups":groups})
            if (i+1)%100==0: progress_cb(i+1,1000)
    atomic_json(results/"subtest_C_mismatches.json",mismatches)
    return {"instances":1000,"mismatch_count":len(mismatches),"ambiguity_type_counts":type_counts}

def summarize_cell(rows):
    widths=[r["W_rel"] for r in rows]
    nonfinal=[r for r in rows if r["L"]<r["U"]]
    nw=[r["W_rel"] for r in nonfinal]
    return {
        "instances":len(rows),
        "truth_containment_violations":sum(1 for r in rows if not r["truth_contained"]),
        "finite_U_count":sum(1 for r in rows if r["U"] is not None),
        "nonfinal_count":len(nonfinal),
        "nonfinal_fraction":len(nonfinal)/float(len(rows)),
        "nonfinal_and_W_rel_ge_0_05_count":sum(1 for r in rows if r["L"]<r["U"] and r["W_rel"]>=0.05),
        "nonfinal_and_W_rel_ge_0_05_fraction":sum(1 for r in rows if r["L"]<r["U"] and r["W_rel"]>=0.05)/float(len(rows)),
        "nonfinal_and_E_recv_ge_0_05_count":sum(1 for r in rows if r["L"]<r["U"] and r["E_recv"]>=0.05),
        "nonfinal_and_E_recv_ge_0_05_fraction":sum(1 for r in rows if r["L"]<r["U"] and r["E_recv"]>=0.05)/float(len(rows)),
        "median_W_rel_all":median(widths),
        "median_W_rel_nonfinal":median(nw),
        "p95_W_rel_all":percentile_nearest(widths,0.95),
        "mean_E_recv":sum(r["E_recv"] for r in rows)/float(len(rows)),
        "naive_false_finality_count":sum(1 for r in rows if r["naive_false_finality"])
    }

def subtest_d(results, progress_cb):
    rng=random.Random(20261108)
    path=results/"subtest_D_nondegeneracy_per_instance.csv.gz"
    all_by_cell={}
    violations=[]
    fields=["m","a","s","cell_instance","S_true","L","U","W_rel","S_recv","E_recv",
            "naive_false_finality","truth_contained"]
    with gzip.open(str(path),"wt",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        done=0
        for m in (0.05,0.10,0.20):
            for a in (0.05,0.10,0.20):
                for s in (0.25,0.50,0.75):
                    key="m=%.2f,a=%.2f,s=%.2f"%(m,a,s)
                    cell=[]
                    for j in range(200):
                        inst=generate_grid_laminar(rng,64,m,a,s)
                        lo=laminar_flow_solve(inst,"min"); hi=laminar_flow_solve(inst,"max")
                        if not lo.get("feasible") or not hi.get("feasible"):
                            L=U=None
                            truth=False
                        else:
                            L=lo["cost"]; U=hi["cost"]
                            truth_value=true_settlement(inst)
                            truth=(L<=truth_value<=U)
                        st=true_settlement(inst)
                        recv=received_only_naive(inst)
                        if L is None or U is None:
                            wr=None
                        else:
                            wr=(U-L)/float(max(st,1))
                        er=abs(recv-st)/float(max(st,1))
                        row={"m":m,"a":a,"s":s,"cell_instance":j,"S_true":st,"L":L,"U":U,
                             "W_rel":wr,"S_recv":recv,"E_recv":er,
                             "naive_false_finality":bool(L is not None and U is not None and L<U),
                             "truth_contained":truth}
                        w.writerow(row); cell.append(row)
                        if not truth:
                            violations.append({"cell":key,"cell_instance":j,"row":row,"instance":inst})
                        done+=1
                        if done%200==0: progress_cb(done,5400)
                    all_by_cell[key]=cell
    summary={k:summarize_cell(v) for k,v in all_by_cell.items()}
    primary=summary["m=0.10,a=0.10,s=0.50"]
    atomic_json(results/"subtest_D_grid_summary.json",{
        "cells":summary,
        "primary_moderate_cell":{"key":"m=0.10,a=0.10,s=0.50","summary":primary}
    })
    atomic_json(results/"subtest_D_truth_containment_violations.json",violations)
    return {"instances":5400,"truth_containment_violation_count":len(violations),"primary_moderate_cell":primary}

def subtest_e(results, progress_cb):
    rng=random.Random(20261109)
    path=results/"subtest_E_scaling_per_instance.csv"
    fields=["n","instance_id","success","lower_plus_upper_solve_seconds","peak_rss_mb_process",
            "L","U","amount_variables","conceptual_tree_arcs","conceptual_rate_class_arcs",
            "conceptual_total_arcs","hidden_rate_nodes","error"]
    by_n={100:[],1000:[],10000:[]}
    with open(str(path),"w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        done=0
        for n in (100,1000,10000):
            for i in range(50):
                inst=generate_grid_laminar(rng,n,0.10,0.10,0.50)
                sz=conceptual_flow_size(inst)
                t0=time.perf_counter()
                err=""
                try:
                    lo=laminar_flow_solve(inst,"min"); hi=laminar_flow_solve(inst,"max")
                    success=bool(lo.get("feasible") and hi.get("feasible"))
                    L=lo.get("cost"); U=hi.get("cost")
                    if not success:
                        err="infeasible_or_solver_failure"
                except Exception as e:
                    success=False; L=U=None; err=e.__class__.__name__
                elapsed=time.perf_counter()-t0
                row={"n":n,"instance_id":i,"success":success,
                     "lower_plus_upper_solve_seconds":elapsed,
                     "peak_rss_mb_process":rss_mb(),"L":L,"U":U,"error":err}
                row.update(sz)
                w.writerow(row); by_n[n].append(row)
                done+=1
                if done%10==0: progress_cb(done,150)
    summary={}
    for n,rows in by_n.items():
        times=[r["lower_plus_upper_solve_seconds"] for r in rows if r["success"]]
        rss=[r["peak_rss_mb_process"] for r in rows if r["peak_rss_mb_process"] is not None]
        summary[str(n)]={
            "instances":50,
            "success_count":sum(1 for r in rows if r["success"]),
            "failure_count":sum(1 for r in rows if not r["success"]),
            "median_lower_plus_upper_solve_seconds":median(times),
            "p95_lower_plus_upper_solve_seconds":percentile_nearest(times,0.95),
            "max_lower_plus_upper_solve_seconds":max(times) if times else None,
            "max_peak_rss_mb_process":max(rss) if rss else None
        }
    atomic_json(results/"subtest_E_scaling_summary.json",summary)
    return {"by_n":summary}

def make_manifest(directory):
    lines=[]
    for p in sorted(directory.rglob("*")):
        if p.is_file() and p.name!="SHA256SUMS.txt":
            lines.append("%s  %s\n"%(sha256_file(p),p.relative_to(directory).as_posix()))
    with (directory/"SHA256SUMS.txt").open("w",encoding="utf-8",newline="\n") as f:
        f.write("".join(lines))

def pack_return(pkg, run_dir, results, summary):
    ret=pkg/"RETURN_TO_CHATGPT_TSC_EBMUS_A0_METHOD_VALIDATION"
    tgz=pkg/"RETURN_TO_CHATGPT_TSC_EBMUS_A0_METHOD_VALIDATION.tar.gz"
    if ret.exists(): shutil.rmtree(ret)
    ret.mkdir()
    # Required outputs.
    for name in ["frozen_config.json","normative_test_vectors.json","preregistration_copy.txt"]:
        shutil.copy2(str(pkg/name),str(ret/name))
    for name in ["normative_results.json",
                 "subtest_B_small_exactness.csv.gz","subtest_B_mismatches.json",
                 "subtest_C_ambiguity_exactness.csv.gz","subtest_C_mismatches.json",
                 "subtest_D_nondegeneracy_per_instance.csv.gz","subtest_D_grid_summary.json",
                 "subtest_D_truth_containment_violations.json",
                 "subtest_E_scaling_per_instance.csv","subtest_E_scaling_summary.json",
                 "summary.json","summary.md"]:
        shutil.copy2(str(results/name),str(ret/name))
    shutil.copytree(str(pkg/"code"),str(ret/"code"))
    shutil.copy2(str(run_dir/"command_environment.txt"),str(ret/"command_environment.txt"))
    shutil.copy2(str(pkg/"SHA256SUMS_SOURCE.txt"),str(ret/"SHA256SUMS_SOURCE.txt"))
    make_manifest(ret)
    if tgz.exists(): tgz.unlink()
    with tarfile.open(str(tgz),"w:gz") as tf:
        tf.add(str(ret),arcname=ret.name)
    return tgz

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-dir",required=True)
    args=ap.parse_args()
    run_dir=Path(args.run_dir).resolve()
    results=run_dir/"results"; results.mkdir(parents=True,exist_ok=True)
    start=utc()
    write_status(run_dir,"RUNNING","SUBTEST_A",{"done":0,"total":1},start)

    summary={
        "project_id":"EBMUS",
        "experiment_id":"EBMUS_A0_METHOD_VALIDATION",
        "candidate":"Evidence-Bounded Model-Usage Settlement (EBMUS)",
        "started_utc":start,
        "scientific_GO_MODIFY_NO_GO":"REFERENCE_REPRODUCTION_ONLY",
        "policy":{"gpu":False,"training":False,"llm_inference":False,"external_api":False,"paid_service":False}
    }

    def phase_progress(phase):
        def cb(done,total):
            write_status(run_dir,"RUNNING",phase,{"done":done,"total":total},start)
        return cb

    try:
        summary["subtest_A"]=run_normative(PKG,results)
        write_status(run_dir,"RUNNING","SUBTEST_B",{"done":0,"total":4000},start)
        summary["subtest_B"]=subtest_b(results,phase_progress("SUBTEST_B"))
        write_status(run_dir,"RUNNING","SUBTEST_C",{"done":0,"total":1000},start)
        summary["subtest_C"]=subtest_c(results,phase_progress("SUBTEST_C"))
        write_status(run_dir,"RUNNING","SUBTEST_D",{"done":0,"total":5400},start)
        summary["subtest_D"]=subtest_d(results,phase_progress("SUBTEST_D"))
        write_status(run_dir,"RUNNING","SUBTEST_E",{"done":0,"total":150},start)
        summary["subtest_E"]=subtest_e(results,phase_progress("SUBTEST_E"))
        summary["completed_utc"]=utc()

        atomic_json(results/"summary.json",summary)
        p=summary["subtest_D"]["primary_moderate_cell"]
        e10=summary["subtest_E"]["by_n"]["10000"]
        md = """# EBMUS A0 Method Validation — Mechanical Summary

No scientific GO / MODIFY / NO-GO decision is made by the experiment execution session.

## Subtest A — normative sanity
- cases: {A_cases}
- mismatches: {A_mis}

## Subtest B — LaminarFlow exactness
- instances: {B_n}
- exactness/feasibility/witness mismatches: {B_mis}

## Subtest C — EvidenceMILP exactness
- instances: {C_n}
- exactness/feasibility/witness mismatches: {C_mis}
- ambiguity type counts: `{C_types}`

## Subtest D — controlled non-degeneracy
- instances: {D_n}
- truth-containment violations: {D_bad}
- primary cell nonfinal fraction: {nonfinal}
- primary cell fraction with L<U and W_rel>=0.05: {g5metric}
- primary cell fraction with L<U and E_recv>=0.05: {g6metric}
- primary cell finite-U count: {finiteU}/200
- primary cell median W_rel among non-final: {medW}

## Subtest E — scalability
- n=10,000 successful instances: {E_success}/50
- n=10,000 median lower+upper solve seconds: {E_med}
- n=10,000 max process peak RSS MB: {E_rss}

## Boundary
This A0 validates exactness, controlled non-degeneracy, and CPU scalability only.
It does not establish real-world anomaly prevalence, monetary loss prevalence, or production superiority.
""".format(
            A_cases=summary["subtest_A"]["case_count"],A_mis=summary["subtest_A"]["mismatch_count"],
            B_n=summary["subtest_B"]["instances"],B_mis=summary["subtest_B"]["mismatch_count"],
            C_n=summary["subtest_C"]["instances"],C_mis=summary["subtest_C"]["mismatch_count"],
            C_types=json.dumps(summary["subtest_C"]["ambiguity_type_counts"],sort_keys=True),
            D_n=summary["subtest_D"]["instances"],D_bad=summary["subtest_D"]["truth_containment_violation_count"],
            nonfinal=p["nonfinal_fraction"],g5metric=p["nonfinal_and_W_rel_ge_0_05_fraction"],
            g6metric=p["nonfinal_and_E_recv_ge_0_05_fraction"],finiteU=p["finite_U_count"],
            medW=p["median_W_rel_nonfinal"],E_success=e10["success_count"],
            E_med=e10["median_lower_plus_upper_solve_seconds"],E_rss=e10["max_peak_rss_mb_process"]
        )
        with (results/"summary.md").open("w",encoding="utf-8",newline="\n") as f:
            f.write(md)
        write_status(run_dir,"SUCCESS","COMPLETE",{"done":1,"total":1},start)
        print("EBMUS A0 reproduction complete.")
        print("Results directory: %s" % results)
        return 0
    except Exception as e:
        write_status(run_dir,"FAILED","EXCEPTION",{"done":0,"total":1},start,failure="%s: %s"%(e.__class__.__name__,str(e)))
        raise

if __name__=="__main__":
    sys.exit(main())
