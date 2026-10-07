#!/usr/bin/env python3
"""Orbit collapse by continuation.

A collapse moves an orbit continuously onto a symmetry set (e.g. a general prism orbit
of 12 nodes onto the mid-plane z=0, where it becomes an orbit of 6 nodes). It costs only
one degree of freedom, so a rule with DOF - E = s can in principle absorb up to s
collapses. We impose the goal  c(theta) = s(t)  as an extra equation, move s from its
current value to the target in adaptive steps, and Newton-correct (minimum-norm, physical
units) the augmented system. On arrival the orbit is snapped onto the symmetry set,
re-classified (lower symmetry type, same total weight) and the rule re-corrected.

usage: collapse.py <domain> <degree> <start.csv> <outdir> [--workers W] [--batch B]
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
from qsearch import DOMAINS, System, Rule, load_expanded, rep_jac  # noqa: E402

TOL = 1e-13
PIN = 1e-7
MIN_MARGIN = 1e-5   # every node must stay at least this far inside the domain
A13 = math.atanh(1 / 3)

# goals per (domain, orbit type): (coefficients over the orbit's u-params, target, label)
GOALS = {
    "pri": {
        "S6": [((0, 0, 1), 0.0, "z->0"), ((1, -1, 0), 0.0, "median A=B"), ((1, 0, 0), 0.0, "median A=C"),
               ((0, 1, 0), 0.0, "median B=C")],
        "S4": [((0, 1), 0.0, "z->0"), ((1, 0), A13, "->centroid")],
        "S5": [((1, -1), 0.0, "median A=B"), ((1, 0), 0.0, "median A=C"), ((0, 1), 0.0, "median B=C")],
        "S3": [((1,), A13, "->centroid")],
        "S2": [((1,), 0.0, "z->0")],
    },
    "pyr": {
        "S4": [((0, 1, -1), 0.0, "a=b"), ((0, 1, 1), 0.0, "a=-b"), ((0, 0, 1), 0.0, "b->0"), ((0, 1, 0), 0.0, "a->0")],
    },
    "cube": {
        "S7": [((1, -1, 0), 0.0, "a=b"), ((1, 1, 0), 0.0, "a=-b"), ((0, 1, -1), 0.0, "b=c"), ((0, 1, 1), 0.0, "b=-c"),
               ((1, 0, -1), 0.0, "a=c"), ((1, 0, 1), 0.0, "a=-c"), ((1, 0, 0), 0.0, "a->0"), ((0, 1, 0), 0.0, "b->0"),
               ((0, 0, 1), 0.0, "c->0")],
        "S5": [((1, -1), 0.0, "a=b"), ((1, 1), 0.0, "a=-b"), ((0, 1), 0.0, "b->0"), ((1, 0), 0.0, "a->0")],
        "S6": [((1, -1), 0.0, "a=b"), ((1, 1), 0.0, "a=-b"), ((1, 0), 0.0, "a->0"), ((0, 1), 0.0, "b->0")],
        "S2": [((1,), 0.0, "->0")], "S3": [((1,), 0.0, "->0")], "S4": [((1,), 0.0, "->0")],
    },
    "sqr": {
        "S4": [((1, -1), 0.0, "a=b"), ((1, 1), 0.0, "a=-b"), ((1, 0), 0.0, "a->0"), ((0, 1), 0.0, "b->0")],
        "S2": [((1,), 0.0, "->0")], "S3": [((1,), 0.0, "->0")],
    },
}


def layout(S, rule):
    out, i = [], 0
    for t, _, _ in rule.orbits:
        k = S.dom.orbit_types[t][1]
        out.append((i, k)); i += k + 1
    return out


def newton_aug(S, rule, theta, gvec, gtarget, maxit=15):
    """Solve r(theta)=0 and gvec.theta = gtarget (gvec may be None)."""
    for _ in range(maxit):
        cur = S.unpack(rule, theta)
        r, J = S.res_jac(cur)
        if gvec is not None:
            r = np.append(r, gvec @ theta - gtarget)
            J = np.vstack([J, gvec])
        nr = float(np.linalg.norm(r))
        if nr < TOL:
            return theta, nr, True
        sc = S.param_scales(cur)
        free = sc > PIN
        if gvec is not None:
            free |= gvec != 0
        Js = J[:, free] / sc[free]
        try:
            y = np.linalg.solve(Js @ Js.T, -r)
        except np.linalg.LinAlgError:
            return theta, nr, False
        d = Js.T @ y
        m = np.max(np.abs(d))
        if m > 0.02:
            d *= 0.02 / m
        ok = False
        for frac in (1.0, 0.5, 0.25, 0.1):
            th2 = theta.copy(); th2[free] += frac * d / sc[free]
            if not np.all(np.isfinite(th2)) or np.max(np.abs(th2)) > 60:
                continue
            r2 = S.residual(S.unpack(rule, th2))
            if gvec is not None:
                r2 = np.append(r2, gvec @ th2 - gtarget)
            n2 = float(np.linalg.norm(r2))
            if n2 < nr:
                theta, ok = th2, True
                break
        if not ok:
            return theta, nr, False
    cur = S.unpack(rule, theta)
    r = S.residual(cur)
    if gvec is not None:
        r = np.append(r, gvec @ theta - gtarget)
    nr = float(np.linalg.norm(r))
    return theta, nr, nr < TOL


def interior_ok(S, rule, theta, wmin=1e-13):
    cur = S.unpack(rule, theta)
    if S.margins(cur).min() < MIN_MARGIN:
        return False
    for i, (start, k) in enumerate(layout(S, rule)):
        if theta[start + k] < math.log(wmin):
            return False
    return True


def snap(S, rule, theta, kk):
    """Replace orbit kk (now on a symmetry set) by its lower-symmetry orbit."""
    cur = S.unpack(rule, theta)
    t, u, om = cur.orbits[kk]
    p = rep_jac(S.dom, t, np.asarray(u)[None, :])[0][0]
    p = tuple(float(v) for v in p)
    nt = S.dom.classify(p, tol=1e-7)
    if nt == t:
        return None
    # symmetrise the point exactly by projecting via the lower type's own parameterisation
    nu = S.dom.unrep(nt, p)
    w_total = math.exp(om) * S.dom.orbit_types[t][0]
    new = (nt, np.asarray(nu, dtype=float), math.log(w_total / S.dom.orbit_types[nt][0]))
    orbs = list(cur.orbits); orbs[kk] = new
    return Rule(S.dom, rule.q, orbs)


_S = {}


def collapse(args):
    dname, q, orbits, kk, gi, maxsteps = args
    if (dname, q) not in _S:
        _S[(dname, q)] = System(DOMAINS[dname](), q)
    S = _S[(dname, q)]
    rule = Rule(S.dom, q, orbits)
    theta = S.pack(rule)
    start, k = layout(S, rule)[kk]
    t = rule.orbits[kk][0]
    coeffs, target, label = GOALS[dname][t][gi]
    gvec = np.zeros_like(theta); gvec[start:start + k] = coeffs
    c0 = float(gvec @ theta)
    s, ds = 0.0, 0.1
    t0 = time.time()
    steps = 0
    while steps < maxsteps and ds > 1e-5:
        steps += 1
        sn = min(1.0, s + ds)
        g = c0 + (target - c0) * sn
        if sn >= 1.0:
            g = target + 1e-9 * np.sign(c0 - target)   # stop just short, then snap
        th2, nr, ok = newton_aug(S, rule, theta, gvec, g)
        if ok and interior_ok(S, rule, th2):
            theta, s = th2, sn
            ds = min(ds * 1.5, 0.3)
            if s >= 1.0:
                new = snap(S, rule, theta, kk)
                if new is None:
                    return kk, gi, False, None, "snap failed"
                th3, nr, ok = newton_aug(S, new, S.pack(new), None, 0.0)
                if ok and interior_ok(S, new, th3):
                    out = S.unpack(new, th3)
                    return kk, gi, True, out.orbits, f"{label}: collapsed in {steps} steps, {time.time()-t0:.0f}s"
                return kk, gi, False, None, f"{label}: post-snap correction failed |r|={nr:.1e}"
        else:
            ds /= 2
    return kk, gi, False, None, f"{label}: stuck at s={s:.3f} after {steps} steps, {time.time()-t0:.0f}s"


def save_rule(rule, path):
    P, W = rule.expand()
    with open(path, "w") as f:
        f.write("x,y,z,weight\n" if P.shape[1] == 3 else "x,y,weight\n")
        for p, w in zip(P, W):
            f.write(",".join(repr(float(v)) for v in p) + "," + repr(float(w)) + "\n")


def candidates(S, rule, failed):
    """(priority, orbit index, goal index): most nodes saved first, then nearest goal."""
    out = []
    lay = layout(S, rule)
    theta = S.pack(rule)
    for kk, (t, u, om) in enumerate(rule.orbits):
        for gi, (coeffs, target, label) in enumerate(GOALS[S.dom.name].get(t, [])):
            if (kk, gi) in failed:
                continue
            start, k = lay[kk]
            dist = abs(float(np.dot(coeffs, theta[start:start + k])) - target)
            out.append((dist, kk, gi))
    # nodes saved depends on resulting type; approximate by current orbit size
    out.sort(key=lambda x: (-S.dom.orbit_types[rule.orbits[x[1]][0]][0], x[0]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("domain"); ap.add_argument("q", type=int); ap.add_argument("start"); ap.add_argument("outdir")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--maxsteps", type=int, default=300)
    ap.add_argument("--min-margin", type=float, default=1e-5)
    a = ap.parse_args()
    global MIN_MARGIN
    MIN_MARGIN = a.min_margin
    os.makedirs(a.outdir, exist_ok=True)
    dom = DOMAINS[a.domain]()
    S = System(dom, a.q)
    rule = load_expanded(dom, a.q, a.start)
    print(f"start: nodes={rule.nodes()} counts={rule.counts()} DOF={rule.dof()} E={S.E} "
          f"|r|={np.linalg.norm(S.residual(rule)):.2e}", flush=True)
    import multiprocessing as mp
    failed = set()
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex:
        while rule.dof() > S.E:
            cands = candidates(S, rule, failed)
            if not cands:
                print("no candidates left; stopping", flush=True)
                break
            batch = cands[:a.batch]
            res = list(ex.map(collapse, [(a.domain, a.q, rule.orbits, kk, gi, a.maxsteps) for _, kk, gi in batch]))
            best = None
            for kk, gi, ok, orbs, msg in res:
                t, _, om = rule.orbits[kk]
                print(f"  orbit {kk:4d} ({t}, w={math.exp(om):.2e}): {'OK ' if ok else 'no '} {msg}", flush=True)
                if ok:
                    c = Rule(dom, a.q, orbs)
                    if best is None or c.nodes() < best.nodes():
                        best = c
                else:
                    failed.add((kk, gi))
            if best is None:
                continue
            rule = best
            failed = set()
            path = os.path.join(a.outdir, f"{a.domain}_q{a.q}_n{rule.nodes()}.csv")
            save_rule(rule, path)
            print(f"ACCEPTED: nodes={rule.nodes()} counts={rule.counts()} DOF={rule.dof()} E={S.E} -> {path}", flush=True)
        else:
            print("DOF == E: no slack left; stopping", flush=True)


if __name__ == "__main__":
    main()
