#!/usr/bin/env python3
"""Orbit removal by *steered* continuation on the manifold of exact rules.

Minimise   F = omega_k + mu * Barrier   subject to   r(theta) = 0
by projected-gradient steps along the manifold (tangent step in the null space of the
Jacobian, measured in physical units) followed by minimum-norm Newton correction.
The barrier keeps nodes away from the boundary and other weights away from zero,
so the path can bend around obstacles instead of stopping at them. When
w_k/w_k0 < 1e-10 the orbit is deleted and the rule re-corrected.

usage: steer.py <domain> <degree> <start.csv> <outdir> [--workers W] [--batch B]
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
PIN = 1e-7  # coordinates whose physical sensitivity is below this are frozen


# linear interior constraints  c_j(p) = a_j . p + b_j > 0
def constraints(name):
    if name == "pyr":
        A = np.array([[0, 0, -1], [0, 0, 1], [-1, 0, -.5], [1, 0, -.5], [0, -1, -.5], [0, 1, -.5]], float)
        b = np.array([1, 1, .5, .5, .5, .5], float)
    elif name == "pri":
        A = np.array([[-.5, -.5, 0], [.5, 0, 0], [0, .5, 0], [0, 0, -1], [0, 0, 1]], float)
        b = np.array([0, .5, .5, 1, 1], float)
    elif name == "cube":
        A = np.vstack([np.eye(3), -np.eye(3)]); b = np.ones(6)
    elif name == "sqr":
        A = np.vstack([np.eye(2), -np.eye(2)]); b = np.ones(4)
    return A, b


def layout(S, rule):
    """For each orbit: (start index in theta, number of coordinate params)."""
    out, i = [], 0
    for t, _, _ in rule.orbits:
        k = S.dom.orbit_types[t][1]
        out.append((i, k)); i += k + 1
    return out


def barrier_grad(S, rule, theta, k_skip, wfloor):
    """Value and gradient (w.r.t. theta) of
       B = sum_orbits sum_j -log c_j(rep)  +  sum_{i != k} wfloor / w_i"""
    cur = S.unpack(rule, theta)
    pts, jacs = S.reps(cur)
    A, b = constraints(S.dom.name)
    g = np.zeros_like(theta)
    val = 0.0
    for i, (start, k) in enumerate(layout(S, rule)):
        c = A @ pts[i] + b
        val += -np.sum(np.log(np.maximum(c, 1e-300)))
        if k:
            dc = A @ jacs[i]               # (nc, k)
            g[start:start + k] += -(dc / c[:, None]).sum(axis=0)
        if i != k_skip:
            w = math.exp(theta[start + k])
            val += wfloor / w
            g[start + k] += -wfloor / w
    return val, g


def newton(S, rule, theta, maxit=15):
    for _ in range(maxit):
        cur = S.unpack(rule, theta)
        r, J = S.res_jac(cur)
        nr = float(np.linalg.norm(r))
        if nr < TOL:
            return theta, nr, True
        sc = S.param_scales(cur)
        free = sc > PIN
        Js = J[:, free] / sc[free]          # columns per unit physical displacement
        try:
            y = np.linalg.solve(Js @ Js.T, -r)
        except np.linalg.LinAlgError:
            return theta, nr, False
        d = Js.T @ y                        # physical displacement (min norm)
        m = np.max(np.abs(d))
        if m > 0.02:
            d *= 0.02 / m
        ok = False
        for frac in (1.0, 0.5, 0.25, 0.1):
            th2 = theta.copy(); th2[free] += frac * d / sc[free]
            if not np.all(np.isfinite(th2)) or np.max(th2) > 60:
                continue
            n2 = float(np.linalg.norm(S.residual(S.unpack(rule, th2))))
            if n2 < nr:
                theta, ok = th2, True
                break
        if not ok:
            return theta, nr, False
    nr = float(np.linalg.norm(S.residual(S.unpack(rule, theta))))
    return theta, nr, nr < TOL


def interior_ok(S, rule, theta, k_skip, wmin=1e-13):
    cur = S.unpack(rule, theta)
    if S.margins(cur).min() <= 1e-12:
        return False
    for i, (start, k) in enumerate(layout(S, rule)):
        if i != k_skip and theta[start + k] < math.log(wmin):
            return False
    return True


_S = {}


def steer_remove(args):
    dname, q, orbits, kk, mu, maxsteps = args
    if (dname, q) not in _S:
        _S[(dname, q)] = System(DOMAINS[dname](), q)
    S = _S[(dname, q)]
    rule = Rule(S.dom, q, orbits)
    theta = S.pack(rule)
    lay = layout(S, rule)
    col = lay[kk][0] + lay[kk][1]
    om0 = theta[col]
    weights = np.exp([o[2] for o in rule.orbits])
    wfloor = 0.05 * float(np.min(np.delete(weights, kk)))
    h = 0.02  # physical step length
    t0 = time.time()
    steps = 0
    while steps < maxsteps:
        steps += 1
        if theta[col] - om0 < math.log(1e-10):
            sub = Rule(S.dom, q, [o for j, o in enumerate(S.unpack(rule, theta).orbits) if j != kk])
            th2, nr, ok = newton(S, sub, S.pack(sub))
            if ok and interior_ok(S, sub, th2, -1):
                return kk, True, S.unpack(sub, th2).orbits, f"removed after {steps} steps, {time.time()-t0:.0f}s"
            return kk, False, None, f"final deletion failed (|r|={nr:.1e}) after {steps} steps"
        cur = S.unpack(rule, theta)
        r, J = S.res_jac(cur)
        sc = S.param_scales(cur)
        free = sc > PIN
        _, gB = barrier_grad(S, rule, theta, kk, wfloor)
        gF = mu * gB
        gF[col] += 1.0
        G = gF[free] / sc[free]
        Js = J[:, free] / sc[free]
        try:
            proj = Js.T @ np.linalg.solve(Js @ Js.T, Js @ G)
        except np.linalg.LinAlgError:
            return kk, False, None, "singular Jacobian"
        v = -(G - proj)
        nv = np.linalg.norm(v)
        if nv < 1e-14:
            return kk, False, None, f"stationary point at w/w0={math.exp(theta[col]-om0):.2e}"
        v /= np.max(np.abs(v))
        moved = False
        while h > 1e-7:
            th2 = theta.copy(); th2[free] += h * v / sc[free]
            th2, nr, ok = newton(S, rule, th2)
            if ok and interior_ok(S, rule, th2, kk):
                Fo = theta[col] + mu * barrier_grad(S, rule, theta, kk, wfloor)[0]
                Fn = th2[col] + mu * barrier_grad(S, rule, th2, kk, wfloor)[0]
                if Fn < Fo:
                    theta, moved = th2, True
                    h = min(h * 1.5, 0.2)
                    break
            h /= 2
        if not moved:
            return kk, False, None, f"no descent at w/w0={math.exp(theta[col]-om0):.2e} after {steps} steps"
    return kk, False, None, f"step limit, w/w0={math.exp(theta[col]-om0):.2e}"


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
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--mu", type=float, default=1e-3)
    ap.add_argument("--maxsteps", type=int, default=600)
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
            cands = []
            for j, (t, u, om) in enumerate(rule.orbits):
                d = dom.orbit_types[t][1] + 1
                if rule.dof() - d < S.E or j in failed:
                    continue
                cands.append((-dom.orbit_types[t][0] / d, om, j))
            cands.sort()
            if not cands:
                print("no candidates left; stopping", flush=True)
                break
            batch = [j for _, _, j in cands[:a.batch]]
            res = list(ex.map(steer_remove, [(a.domain, a.q, rule.orbits, j, a.mu, a.maxsteps) for j in batch]))
            best = None
            for k, ok, orbs, msg in res:
                t, _, om = rule.orbits[k]
                print(f"  orbit {k:4d} ({t}, w={math.exp(om):.2e}): {'OK ' if ok else 'no '} {msg}", flush=True)
                if ok:
                    c = Rule(dom, a.q, orbs)
                    if best is None or c.nodes() < best.nodes():
                        best = c
                else:
                    failed.add(k)
            if best is None:
                continue
            rule = best
            failed = set()
            path = os.path.join(a.outdir, f"{a.domain}_q{a.q}_n{rule.nodes()}.csv")
            save_rule(rule, path)
            print(f"ACCEPTED: nodes={rule.nodes()} counts={rule.counts()} DOF={rule.dof()} E={S.E} -> {path}", flush=True)


if __name__ == "__main__":
    main()
