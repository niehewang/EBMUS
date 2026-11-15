#!/usr/bin/env python3
from __future__ import annotations
import random
from common import exact_cost,bound_cost,midpoint_wrel
from paper_core import generate_instance,laminar_flow_solve,generic_lp,generic_milp,evidence_bounds,brute_force,refinement_conditions
assert abs(exact_cost(100,20,40)-0.00164)<1e-12
L,U=bound_cost(100,20); assert abs(L-.0011)<1e-12 and abs(U-.002)<1e-12 and midpoint_wrel(L,U)>.01
rng=random.Random(7); inst=generate_instance(rng,5,.2,.1,.5,'medium','low'); br=brute_force(inst); eb=evidence_bounds(inst); assert br['feasible']==eb['feasible']; assert abs(br['L']-eb['L'])<1e-7 and abs(br['U']-eb['U'])<1e-7
# continuous LP agrees on a clean instance
inst2=generate_instance(random.Random(8),20,.1,.1,.5,'medium','none'); lfL=laminar_flow_solve(inst2,'min');lfU=laminar_flow_solve(inst2,'max');lpL=generic_lp(inst2,'min',10);lpU=generic_lp(inst2,'max',10);assert lfL['feasible'] and lpL['success'];assert abs(lfL['cost']-lpL['cost'])<1e-6 and abs(lfU['cost']-lpU['cost'])<1e-6
# refinement monotonicity
prev=None
for c,x,res in refinement_conditions(inst):
 z=evidence_bounds(x,res); assert z['feasible']
 if prev: assert z['L']>=prev[0]-1e-7 and z['U']<=prev[1]+1e-7
 prev=(z['L'],z['U'])
print('SELF_TEST_PASS: rating, evidence bounds, brute-force exactness, generic LP agreement, refinement nesting')
