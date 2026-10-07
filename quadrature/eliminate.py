#!/usr/bin/env python3
"""Greedy orbit elimination starting from an existing rule.

Repeatedly tries to delete one orbit (largest symmetry class first, smallest weight
first) and re-solve the moment equations with Levenberg-Marquardt. Accepts the first
deletion that converges. Runs candidate attempts in parallel.

usage: eliminate.py <domain> <degree> <start_rule.csv> <outdir> [--maxit N] [--workers W]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import time
import math
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qsearch import DOMAINS, System, Rule, load_expanded  # noqa: E402

TOL = 1e-13


def lm(system, rule, maxit, tol=TOL, lam=1e-4):
    theta = system.pack(rule)
    cur = rule
    r, J = system.res_jac(cur)
    nr = float(np.linalg.norm(r))
    hist = [nr]
    for it in range(maxit):
        if nr < tol:
            break
        D = np.linalg.norm(J, axis=0) + 1e-300
        Js = J / D
        A = Js.T @ Js
        g = Js.T @ r
        A[np.diag_indices_from(A)] += lam * (1 + np.diag(A))
        try:
            step = -np.linalg.solve(A, g) / D
        except np.linalg.LinAlgError:
            lam *= 10
            continue
        smax = np.max(np.abs(step))
        if smax > 1.0:
            step *= 1.0 / smax
        cand = system.unpack(rule, theta + step)
        rc = system.residual(cand)
        nc = float(np.linalg.norm(rc))
        if np.isfinite(nc) and nc < nr:
            theta = theta + step
            cur = cand
            r, J = system.res_jac(cur)
            nr = nc
            lam = max(lam / 5, 1e-15)
        else:
            lam = min(lam * 5, 1e10)
        hist.append(nr)
        # give up if no 10x progress over the last 150 iterations (once well past the start)
        if it > 300 and hist[-150] < 10 * nr and nr > 1e-9:
            break
    return cur, nr


_SYS = {}


def attempt(args):
    dname, q, rule_orbits, drop, maxit = args
    key = (dname, q)
    if key not in _SYS:
        _SYS[key] = System(DOMAINS[dname](), q)
    S = _SYS[key]
    dom = S.dom
    orbits = [o for i, o in enumerate(rule_orbits) if i != drop]
    rule = Rule(dom, q, orbits)
    t0 = time.time()
    out, nr = lm(S, rule, maxit)
    return drop, nr, out.orbits, time.time() - t0


def save_rule(rule, path):
    P, W = rule.expand()
    with open(path, "w") as f:
        f.write("x,y,z,weight\n" if P.shape[1] == 3 else "x,y,weight\n")
        for p, w in zip(P, W):
            f.write(",".join(repr(float(v)) for v in p) + "," + repr(float(w)) + "\n")


def candidate_order(dom, rule):
    # largest symmetry class first; inside a class smallest weight first
    pri = {t: -dom.orbit_types[t][0] for t in dom.orbit_types}
    idx = list(range(len(rule.orbits)))
    idx.sort(key=lambda i: (pri[rule.orbits[i][0]], rule.orbits[i][2]))
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("domain"); ap.add_argument("q", type=int); ap.add_argument("start"); ap.add_argument("outdir")
    ap.add_argument("--maxit", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-candidates", type=int, default=10**9,
                    help="give up after this many failed candidates in a row")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    dom = DOMAINS[a.domain]()
    S = System(dom, a.q)
    rule = load_expanded(dom, a.q, a.start)
    r = S.residual(rule)
    print(f"start: nodes={rule.nodes()} counts={rule.counts()} DOF={rule.dof()} E={S.E} |r|={np.linalg.norm(r):.2e}", flush=True)
    import multiprocessing as mp
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex:
        while True:
            order = candidate_order(dom, rule)
            # only candidates that keep DOF >= E
            order = [i for i in order if rule.dof() - (dom.orbit_types[rule.orbits[i][0]][1] + 1) >= S.E]
            if not order:
                print("no candidate keeps DOF >= E; stopping", flush=True)
                break
            success = None
            tried = 0
            for start in range(0, len(order), a.workers):
                batch = order[start:start + a.workers]
                futs = [ex.submit(attempt, (a.domain, a.q, rule.orbits, i, a.maxit)) for i in batch]
                results = sorted((f.result() for f in futs), key=lambda x: batch.index(x[0]))
                for drop, nr, orbs, dt in results:
                    t = rule.orbits[drop][0]
                    print(f"  drop orbit {drop:4d} ({t}, w={math.exp(rule.orbits[drop][2]):.2e}) -> |r|={nr:.2e}  [{dt:.0f}s]", flush=True)
                    if nr < TOL and success is None:
                        success = orbs
                tried += len(batch)
                if success is not None or tried >= a.max_candidates:
                    break
            if success is None:
                print("no orbit could be removed; stopping", flush=True)
                break
            rule = Rule(dom, a.q, success)
            n = rule.nodes()
            path = os.path.join(a.outdir, f"{a.domain}_q{a.q}_n{n}.csv")
            save_rule(rule, path)
            print(f"ACCEPTED: nodes={n} counts={rule.counts()} DOF={rule.dof()} E={S.E} -> {path}", flush=True)


if __name__ == "__main__":
    main()
