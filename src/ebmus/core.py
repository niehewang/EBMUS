#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EBMUS A0 exact solvers and frozen generators. Python 3.7+ stdlib only."""

from __future__ import print_function
import itertools
import math
import random
from collections import defaultdict

RATES = (1, 2, 4)

def children_from_parent(parent):
    ch = [[] for _ in parent]
    for v in range(1, len(parent)):
        ch[parent[v]].append(v)
    return ch

def subtree_sums(parent, local):
    s = list(local)
    for v in range(len(parent)-1, 0, -1):
        s[parent[v]] += s[v]
    return s

def true_settlement(inst):
    return sum(inst["q"][v] * RATES[inst["true_k"][v]] for v in range(inst["n"]))

def received_only_naive(inst):
    # Frozen baseline: only directly attributed observed local records are rated.
    # Missing local is zero-filled; hidden-attribution amount is not directly rate-attributed.
    return sum(
        inst["q"][v] * RATES[inst["true_k"][v]]
        for v in range(inst["n"]) if inst["status"][v] == "exact"
    )

def make_tree(rng, n):
    return [-1] + [rng.randrange(v) for v in range(1, n)]

def _status_draw(rng, p_exact, p_missing, p_hidden):
    x = rng.random()
    if x < p_exact:
        return "exact"
    if x < p_exact + p_missing:
        return "missing"
    return "hidden"

def generate_small_laminar(rng, n):
    parent = make_tree(rng, n)
    children = children_from_parent(parent)
    q = [rng.randint(0, 3) for _ in range(n)]
    true_k = [rng.randrange(3) for _ in range(n)]
    cap = [rng.choice([c for c in (3,4,5) if c >= qv]) for qv in q]
    status = [_status_draw(rng, 0.65, 0.20, 0.15) for _ in range(n)]
    ss = subtree_sums(parent, q)
    evidence = {0: ss[0]}
    for v in range(1, n):
        if children[v] and rng.random() < 0.50:
            evidence[v] = ss[v]
    return {
        "n": n, "parent": parent, "q": q, "true_k": true_k, "cap": cap,
        "status": status, "subtree_evidence": evidence
    }

def generate_grid_laminar(rng, n, m, a, s):
    parent = make_tree(rng, n)
    children = children_from_parent(parent)
    cap = [rng.choice((8,12,16)) for _ in range(n)]
    q = [rng.randint(0, cap[v]) for v in range(n)]
    true_k = [rng.randrange(3) for _ in range(n)]
    status = []
    for _ in range(n):
        x = rng.random()
        if x < m:
            status.append("missing")
        elif x < m + a:
            status.append("hidden")
        else:
            status.append("exact")
    ss = subtree_sums(parent, q)
    evidence = {0: ss[0]}
    for v in range(1, n):
        if children[v] and rng.random() < s:
            evidence[v] = ss[v]
    return {
        "n": n, "parent": parent, "q": q, "true_k": true_k, "cap": cap,
        "status": status, "subtree_evidence": evidence
    }

def generate_ambiguity_groups(rng):
    gcount = 1 if rng.random() < 0.5 else 2
    groups = []
    for gid in range(gcount):
        if rng.random() < 0.5:
            amount = rng.choice((1,2,3))
            k = rng.randrange(3)
            groups.append({
                "group_id": gid,
                "type": "duplicate_distinct",
                "amount": amount,
                "rate_class": k
            })
        else:
            groups.append({
                "group_id": gid,
                "type": "correction_branch",
                "branch0_amount": rng.choice((1,2,3)),
                "branch0_rate_class": rng.randrange(3),
                "branch1_amount": rng.choice((1,2,3)),
                "branch1_rate_class": rng.randrange(3)
            })
    return groups

def ambiguity_cost(groups, bits):
    total = 0
    for g, bit in zip(groups, bits):
        if bit not in (0,1):
            raise ValueError("binary ambiguity variable must be 0/1")
        if g["type"] == "duplicate_distinct":
            one = g["amount"] * RATES[g["rate_class"]]
            total += one * (1 + bit)
        elif g["type"] == "correction_branch":
            if bit == 0:
                total += g["branch0_amount"] * RATES[g["branch0_rate_class"]]
            else:
                total += g["branch1_amount"] * RATES[g["branch1_rate_class"]]
        else:
            raise ValueError("unknown ambiguity group type")
    return total

def _bounds_and_costs(inst, mode):
    lo, hi, coeff, witness_k = [], [], [], []
    for q, k, cap, status in zip(inst["q"], inst["true_k"], inst["cap"], inst["status"]):
        if status == "missing":
            lo.append(0); hi.append(cap)
            coeff.append(RATES[k]); witness_k.append(k)
        elif status == "hidden":
            lo.append(q); hi.append(q)
            kk = 0 if mode == "min" else 2
            coeff.append(RATES[kk]); witness_k.append(kk)
        elif status == "exact":
            lo.append(q); hi.append(q)
            coeff.append(RATES[k]); witness_k.append(k)
        else:
            raise ValueError("unknown local evidence status")
    return lo, hi, coeff, witness_k

def laminar_flow_solve(inst, mode):
    """Exact min/max-cost-flow for the laminar exact-subtree-sum special case.

    The laminar equality system decomposes into regions between nested exact
    subtree constraints. Each region has one exact residual flow demand and
    bounded local arcs. Greedy residual allocation by arc cost is the exact
    min/max-cost-flow solution for each independent region.
    """
    if mode not in ("min","max"):
        raise ValueError("mode")
    n = inst["n"]
    parent = inst["parent"]
    ev = inst["subtree_evidence"]
    if 0 not in ev:
        return {"feasible": False, "reason": "root_total_missing"}

    constrained = set(ev.keys())
    current = [None] * n
    region_nodes = defaultdict(list)
    constrained_parent = {}

    for v in range(n):
        prev = current[parent[v]] if v else None
        if v in constrained:
            constrained_parent[v] = prev
            current[v] = v
        else:
            current[v] = prev
        if current[v] is None:
            return {"feasible": False, "reason": "uncovered_root"}
        region_nodes[current[v]].append(v)

    constrained_children = defaultdict(list)
    for c, p in constrained_parent.items():
        if p is not None:
            constrained_children[p].append(c)

    lo, hi, coeff, witness_k = _bounds_and_costs(inst, mode)
    amount = [None] * n

    for c in constrained:
        residual = ev[c] - sum(ev[d] for d in constrained_children[c])
        nodes = region_nodes[c]
        sum_lo = sum(lo[v] for v in nodes)
        sum_hi = sum(hi[v] for v in nodes)
        if residual < sum_lo or residual > sum_hi:
            return {
                "feasible": False,
                "reason": "region_capacity",
                "region_root": c,
                "residual": residual,
                "sum_lo": sum_lo,
                "sum_hi": sum_hi
            }
        for v in nodes:
            amount[v] = lo[v]
        extra = residual - sum_lo
        order = sorted(
            nodes,
            key=lambda v: ((coeff[v] if mode == "min" else -coeff[v]), v)
        )
        for v in order:
            add = min(extra, hi[v] - lo[v])
            amount[v] += add
            extra -= add
            if extra == 0:
                break
        if extra != 0:
            return {"feasible": False, "reason": "allocation_internal_error"}

    cost = sum(amount[v] * coeff[v] for v in range(n))
    return {
        "feasible": True,
        "cost": cost,
        "amounts": amount,
        "rate_classes": witness_k
    }

def validate_laminar_witness(inst, sol):
    if not sol.get("feasible"):
        return False
    amount = sol["amounts"]
    ks = sol["rate_classes"]
    if len(amount) != inst["n"] or len(ks) != inst["n"]:
        return False
    for v in range(inst["n"]):
        status = inst["status"][v]
        q = inst["q"][v]
        k = inst["true_k"][v]
        cap = inst["cap"][v]
        a = amount[v]
        wk = ks[v]
        if status == "exact":
            if a != q or wk != k:
                return False
        elif status == "missing":
            if not (0 <= a <= cap) or wk != k:
                return False
        elif status == "hidden":
            if a != q or wk not in (0,1,2):
                return False
        else:
            return False
    ss = subtree_sums(inst["parent"], amount)
    for v, total in inst["subtree_evidence"].items():
        if ss[v] != total:
            return False
    witness_cost = sum(amount[v] * RATES[ks[v]] for v in range(inst["n"]))
    return witness_cost == sol["cost"]

def iter_bruteforce_completions(inst):
    missing = [v for v,s in enumerate(inst["status"]) if s == "missing"]
    hidden = [v for v,s in enumerate(inst["status"]) if s == "hidden"]

    amount_domains = [range(inst["cap"][v] + 1) for v in missing]
    class_domains = [range(3) for _ in hidden]
    amount_products = itertools.product(*amount_domains) if amount_domains else [()]

    for vals in amount_products:
        amount = list(inst["q"])
        for v, a in zip(missing, vals):
            amount[v] = a
        ss = subtree_sums(inst["parent"], amount)
        if any(ss[v] != total for v,total in inst["subtree_evidence"].items()):
            continue

        class_products = itertools.product(*class_domains) if class_domains else [()]
        for kvals in class_products:
            ks = list(inst["true_k"])
            for v,k in zip(hidden, kvals):
                ks[v] = k
            cost = sum(amount[v] * RATES[ks[v]] for v in range(inst["n"]))
            yield cost, list(amount), list(ks)

def brute_force_exact(inst):
    L = U = None
    wL = wU = None
    count = 0
    for cost, amount, ks in iter_bruteforce_completions(inst):
        count += 1
        if L is None or cost < L:
            L, wL = cost, {"amounts":amount, "rate_classes":ks}
        if U is None or cost > U:
            U, wU = cost, {"amounts":amount, "rate_classes":ks}
    return {
        "feasible": L is not None,
        "L": L,
        "U": U,
        "witness_L": wL,
        "witness_U": wU,
        "completion_count": count
    }

def brute_force_general(inst, groups):
    L = U = None
    wL = wU = None
    count = 0
    bit_products = list(itertools.product((0,1), repeat=len(groups)))
    for base_cost, amount, ks in iter_bruteforce_completions(inst):
        for bits in bit_products:
            count += 1
            total = base_cost + ambiguity_cost(groups, bits)
            if L is None or total < L:
                L = total
                wL = {"amounts":list(amount), "rate_classes":list(ks), "ambiguity_bits":list(bits)}
            if U is None or total > U:
                U = total
                wU = {"amounts":list(amount), "rate_classes":list(ks), "ambiguity_bits":list(bits)}
    return {
        "feasible": L is not None,
        "L": L, "U": U,
        "witness_L": wL, "witness_U": wU,
        "completion_count": count
    }

def evidence_milp_solve(inst, groups):
    """Exact MILP layer.

    Binary ambiguity variables are branched exactly (1-2 variables under the
    frozen generator). At each binary leaf, the integer laminar flow subproblem
    is solved exactly by LaminarFlow. This is exact branch-and-bound, not a
    heuristic and does not enumerate local usage worlds.
    """
    lo = laminar_flow_solve(inst, "min")
    hi = laminar_flow_solve(inst, "max")
    if not lo.get("feasible") or not hi.get("feasible"):
        return {"feasible": False}

    L = U = None
    wL = wU = None
    for bits in itertools.product((0,1), repeat=len(groups)):
        amb = ambiguity_cost(groups, bits)
        lower = lo["cost"] + amb
        upper = hi["cost"] + amb
        if L is None or lower < L:
            L = lower
            wL = {
                "amounts": list(lo["amounts"]),
                "rate_classes": list(lo["rate_classes"]),
                "ambiguity_bits": list(bits)
            }
        if U is None or upper > U:
            U = upper
            wU = {
                "amounts": list(hi["amounts"]),
                "rate_classes": list(hi["rate_classes"]),
                "ambiguity_bits": list(bits)
            }
    return {"feasible": True, "L":L, "U":U, "witness_L":wL, "witness_U":wU}

def validate_milp_witness(inst, groups, witness, expected_cost):
    if witness is None:
        return False
    bits = witness.get("ambiguity_bits")
    if bits is None or len(bits) != len(groups) or any(b not in (0,1) for b in bits):
        return False
    sol = {
        "feasible": True,
        "amounts": witness["amounts"],
        "rate_classes": witness["rate_classes"],
        "cost": sum(
            witness["amounts"][v] * RATES[witness["rate_classes"][v]]
            for v in range(inst["n"])
        )
    }
    if not validate_laminar_witness(inst, sol):
        return False
    total = sol["cost"] + ambiguity_cost(groups, bits)
    return total == expected_cost

def conceptual_flow_size(inst):
    unknown_class = sum(1 for s in inst["status"] if s == "hidden")
    class_arcs = sum(3 if s == "hidden" else 1 for s in inst["status"])
    return {
        "amount_variables": inst["n"],
        "conceptual_tree_arcs": max(0, inst["n"]-1),
        "conceptual_rate_class_arcs": class_arcs,
        "hidden_rate_nodes": unknown_class,
        "conceptual_total_arcs": max(0, inst["n"]-1) + inst["n"] + class_arcs
    }
