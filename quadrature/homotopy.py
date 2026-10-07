#!/usr/bin/env python3
"""Orbit removal by continuation along the manifold of exact rules.

When a rule has more parameters than equations (DOF > E) and the Jacobian has full
row rank, the exact rules near it form a smooth manifold of dimension DOF - E.
We drive the weight of a chosen orbit continuously to zero while Newton-correcting
(minimum-norm Gauss-Newton) back onto the manifold. If the path reaches weight 0
with all other nodes still interior and weights positive, the orbit is deleted and
the rule remains exact.

usage: homotopy.py <domain> <degree> <start.csv> <outdir> [--workers W] [--rounds R]
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import sys
import math
import time
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qsearch import DOMAINS, System, Rule, load_expanded  # noqa: E402

TOL = 1e-13
UMAX = 9.0        # |u| beyond this = node within ~1e-8 of boundary/degeneracy -> reject
OMIN = math.log(1e-14)


def newton_correct(S, rule, theta, fixed_col, fixed_val, maxit=12):
    """Solve r=0 holding orbit 'fixed_col' weight at fixed_val (absolute weight).
    theta: packed params; the fixed orbit's log-weight entry is ignored and replaced."""
    for it in range(maxit):
        theta = theta.copy()
        if fixed_val > 0:
            theta[fixed_col] = math.log(fixed_val)
        cur = S.unpack(rule, theta)
        r, J = S.res_jac(cur)
        if fixed_val <= 0:
            # orbit removed: zero its contribution by removing from residual via exact subtraction
            pass
        nr = float(np.linalg.norm(r))
        if nr < TOL:
            return theta, nr, True
        sc = S.param_scales(cur)
        free = sc > 1e-5          # freeze coordinates pinned against the boundary (tanh saturated)
        if fixed_col >= 0:
            free[fixed_col] = False
        Jf = J[:, free]
        Dn = 1.0 / sc[free]       # step measured in physical units
        Js = Jf / Dn
        # minimum norm solution of Js d = -r  via  d = Js^T (Js Js^T)^-1 (-r)
        G = Js @ Js.T
        try:
            y = np.linalg.solve(G, -r)
        except np.linalg.LinAlgError:
            return theta, nr, False
        d = (Js.T @ y) / Dn
        sm = np.max(np.abs(d / Dn))   # physical displacement / relative weight change
        if sm > 0.05:
            d *= 0.05 / sm
        # damped: accept the first fraction that reduces the residual
        accepted = False
        for frac in (1.0, 0.5, 0.25, 0.125):
            th2 = theta.copy(); th2[free] += frac * d
            if np.max(th2[free]) > 50 or not np.all(np.isfinite(th2)):
                continue
            n2 = float(np.linalg.norm(S.residual(S.unpack(rule, th2))))
            if n2 < nr:
                theta, accepted = th2, True
                break
        if not accepted:
            return theta, nr, False
    cur = S.unpack(rule, theta)
    nr = float(np.linalg.norm(S.residual(cur)))
    return theta, nr, nr < TOL


def weight_col(S, rule, k):
    i = 0
    for j, (t, _, _) in enumerate(rule.orbits):
        kk = S.dom.orbit_types[t][1]
        if j == k:
            return i + kk
        i += kk + 1
    raise IndexError


def feasible(S, rule, theta, skip_col):
    cur = S.unpack(rule, theta)
    m = S.margins(cur)
    if m.min() < 1e-10:
        return False, f"orbit {int(m.argmin())} reaches the boundary"
    i = 0
    for j, (t, _, _) in enumerate(rule.orbits):
        kk = S.dom.orbit_types[t][1]
        if i + kk != skip_col and theta[i + kk] < OMIN:
            return False, f"orbit {j} weight vanishes"
        i += kk + 1
    return True, ""


_S = {}


def remove_orbit(args):
    dname, q, orbits, k = args
    if (dname, q) not in _S:
        _S[(dname, q)] = System(DOMAINS[dname](), q)
    S = _S[(dname, q)]
    rule = Rule(S.dom, q, orbits)
    theta = S.pack(rule)
    col = weight_col(S, rule, k)
    w0 = math.exp(theta[col])
    t, dt = 0.0, 0.25
    t0 = time.time()
    steps = 0
    while t < 1.0 and dt > 1e-4 and steps < 400:
        steps += 1
        tn = min(1.0, t + dt)
        target = w0 * (1 - tn)
        if tn >= 1.0:
            # final: remove orbit and correct without it
            sub = Rule(S.dom, q, [o for j, o in enumerate(S.unpack(rule, theta).orbits) if j != k])
            th2 = S.pack(sub)
            th2n, nr, ok = newton_correct(S, sub, th2, -1, -1)
            if ok:
                ok2, why = feasible(S, sub, th2n, -1)
                if ok2:
                    return k, True, S.unpack(sub, th2n).orbits, f"removed in {steps} steps, {time.time()-t0:.0f}s"
            dt /= 2
            continue
        thn, nr, ok = newton_correct(S, rule, theta, col, target)
        if ok:
            ok2, why = feasible(S, rule, thn, col)
            if ok2:
                theta, t = thn, tn
                dt = min(dt * 1.5, 0.5)
                continue
        dt /= 2
    return k, False, None, f"stuck at t={t:.3f} after {steps} steps, {time.time()-t0:.0f}s"


def save_rule(rule, path):
    P, W = rule.expand()
    with open(path, "w") as f:
        f.write("x,y,z,weight\n" if P.shape[1] == 3 else "x,y,weight\n")
        for p, w in zip(P, W):
            f.write(",".join(repr(float(v)) for v in p) + "," + repr(float(w)) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("domain"); ap.add_argument("q", type=int); ap.add_argument("start"); ap.add_argument("outdir")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8, help="candidates tried per round")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    dom = DOMAINS[a.domain]()
    S = System(dom, a.q)
    rule = load_expanded(dom, a.q, a.start)
    print(f"start: nodes={rule.nodes()} counts={rule.counts()} DOF={rule.dof()} E={S.E} "
          f"|r|={np.linalg.norm(S.residual(rule)):.2e}", flush=True)
    import multiprocessing as mp
    failed = set()
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex:
        while True:
            size = {t: dom.orbit_types[t][0] for t in dom.orbit_types}
            # candidates: removal keeps DOF >= E; prefer most nodes per DOF removed, then small weight
            cands = []
            for j, (t, u, om) in enumerate(rule.orbits):
                d = dom.orbit_types[t][1] + 1
                if rule.dof() - d < S.E:
                    continue
                key = (t, round(om, 6))
                if key in failed:
                    continue
                cands.append((-size[t] / d, om, j))
            cands.sort()
            if not cands:
                print("no candidates left; stopping", flush=True)
                break
            batch = [j for _, _, j in cands[:a.batch]]
            futs = [ex.submit(remove_orbit, (a.domain, a.q, rule.orbits, j)) for j in batch]
            res = [f.result() for f in futs]
            best = None
            for k, ok, orbs, msg in res:
                t, _, om = rule.orbits[k]
                print(f"  orbit {k:4d} ({t}, w={math.exp(om):.2e}): {'OK ' if ok else 'no '} {msg}", flush=True)
                if ok:
                    cand = Rule(dom, a.q, orbs)
                    if best is None or cand.nodes() < best.nodes():
                        best = cand
                else:
                    failed.add((t, round(om, 6)))
            if best is None:
                continue
            rule = best
            failed = set()  # rule changed: previous failures may now succeed
            path = os.path.join(a.outdir, f"{a.domain}_q{a.q}_n{rule.nodes()}.csv")
            save_rule(rule, path)
            print(f"ACCEPTED: nodes={rule.nodes()} counts={rule.counts()} DOF={rule.dof()} E={S.E} -> {path}", flush=True)


if __name__ == "__main__":
    main()
