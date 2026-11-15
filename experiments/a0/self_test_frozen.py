#!/usr/bin/env python3
from __future__ import print_function
import random, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path.insert(0,str(REPO/'src'))
from ebmus.core import *

def main():
    rng=random.Random(123)
    for n in (3,4,5,6):
        for _ in range(50):
            inst=generate_small_laminar(rng,n)
            b=brute_force_exact(inst)
            lo=laminar_flow_solve(inst,"min"); hi=laminar_flow_solve(inst,"max")
            assert b["feasible"] == bool(lo.get("feasible") and hi.get("feasible"))
            assert b["L"] == lo.get("cost")
            assert b["U"] == hi.get("cost")
            assert validate_laminar_witness(inst,lo)
            assert validate_laminar_witness(inst,hi)
    rng=random.Random(456)
    for _ in range(100):
        inst=generate_small_laminar(rng,rng.choice((3,4,5)))
        groups=generate_ambiguity_groups(rng)
        b=brute_force_general(inst,groups)
        m=evidence_milp_solve(inst,groups)
        assert b["feasible"]==m["feasible"]
        assert b["L"]==m["L"] and b["U"]==m["U"]
        assert validate_milp_witness(inst,groups,m["witness_L"],m["L"])
        assert validate_milp_witness(inst,groups,m["witness_U"],m["U"])
    rng=random.Random(789)
    for _ in range(100):
        inst=generate_grid_laminar(rng,64,0.10,0.10,0.50)
        lo=laminar_flow_solve(inst,"min"); hi=laminar_flow_solve(inst,"max")
        st=true_settlement(inst)
        assert lo["cost"] <= st <= hi["cost"]
    print("SELF_TEST_PASS: exactness, MILP ambiguity, and truth containment synthetic smoke tests")

if __name__=="__main__":
    main()
