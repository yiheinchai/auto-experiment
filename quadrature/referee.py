#!/usr/bin/env python3
"""Independent referee for fully symmetric, positive, interior (f-SPI) quadrature rules.

Shares no code with the search. Given a rule (list of nodes + weights) on one of the
reference domains, it checks:

  1. EXACTNESS   - every monomial x^a y^b z^c with a+b+c <= degree is integrated
                   exactly. Exact integrals are computed with rational arithmetic
                   (Python Fractions); the rule is evaluated with mpmath at 100 digits.
  2. POSITIVE    - every weight > 0.
  3. INTERIOR    - every node strictly inside the domain.
  4. SYMMETRIC   - the weighted point set is invariant under the domain's full symmetry group.
  5. NODE COUNT  - number of distinct nodes.

Domains (same conventions as Diallo & Worku 2026, arXiv:2601.14488):
  sqr  : |x|,|y| < 1
  cube : |x|,|y|,|z| < 1
  pri  : -1 < y < -x < 1,  |z| < 1     (triangle with vertices (-1,-1),(1,-1),(-1,1)) x (-1,1)
  pyr  : |z| < 1,  |x|,|y| < (1-z)/2   (square base at z=-1, apex at z=1)

Usage:  python3 referee.py <domain> <degree> <rule.csv>
CSV columns: x,y[,z],weight  (a header line is allowed).
"""
import sys
import csv
import itertools
from fractions import Fraction
from math import comb, factorial

import mpmath as mp

mp.mp.dps = 100


# ---------------------------------------------------------------- exact integrals
def line_moment(a):
    """Integral of x^a over (-1,1)."""
    return Fraction(0) if a % 2 else Fraction(2, a + 1)


def triangle_moment(a, b):
    """Integral of x^a y^b over triangle (-1,-1),(1,-1),(-1,1).
    x = 2u-1, y = 2v-1, (u,v) in unit simplex, dx dy = 4 du dv,
    int u^i v^j over simplex = i! j! / (i+j+2)!"""
    total = Fraction(0)
    for i in range(a + 1):
        for j in range(b + 1):
            coef = comb(a, i) * 2**i * (-1) ** (a - i) * comb(b, j) * 2**j * (-1) ** (b - j)
            total += coef * Fraction(factorial(i) * factorial(j), factorial(i + j + 2))
    return 4 * total


def pyramid_moment(a, b, c):
    """Integral of x^a y^b z^c over the pyramid.
    = int_{-1}^{1} z^c * [2 h^{a+1}/(a+1)] [2 h^{b+1}/(b+1)] dz, h=(1-z)/2 (a,b even)."""
    if a % 2 or b % 2:
        return Fraction(0)
    n = a + b + 2
    # ((1-z)/2)^n = 2^-n sum_k C(n,k) (-z)^k
    s = Fraction(0)
    for k in range(n + 1):
        s += comb(n, k) * (-1) ** k * line_moment(c + k)
    return Fraction(4, (a + 1) * (b + 1)) * s / 2**n


def exact_moment(domain, e):
    if domain == "sqr":
        return line_moment(e[0]) * line_moment(e[1])
    if domain == "cube":
        return line_moment(e[0]) * line_moment(e[1]) * line_moment(e[2])
    if domain == "pri":
        return triangle_moment(e[0], e[1]) * line_moment(e[2])
    if domain == "pyr":
        return pyramid_moment(*e)
    raise ValueError(domain)


DIM = {"sqr": 2, "cube": 3, "pri": 3, "pyr": 3}


def exponents(dim, q):
    for tot in range(q + 1):
        for e in itertools.product(range(tot + 1), repeat=dim):
            if sum(e) == tot:
                yield e


# ---------------------------------------------------------------- domain tests
def interior_margin(domain, p):
    """Smallest distance-like margin to the boundary (positive = strictly inside)."""
    if domain == "sqr":
        return min(1 - abs(p[0]), 1 - abs(p[1]))
    if domain == "cube":
        return min(1 - abs(p[0]), 1 - abs(p[1]), 1 - abs(p[2]))
    if domain == "pri":
        x, y, z = p
        return min(y + 1, x + 1, -(x + y), 1 - abs(z))
    if domain == "pyr":
        x, y, z = p
        h = (1 - z) / 2
        return min(1 - abs(z), h - abs(x), h - abs(y))
    raise ValueError(domain)


def symmetry_group(domain):
    """Return list of functions mapping a point to its image."""
    if domain == "sqr":
        ops = []
        for swap in (False, True):
            for sx, sy in itertools.product((1, -1), repeat=2):
                ops.append(lambda p, swap=swap, sx=sx, sy=sy:
                           (sx * (p[1] if swap else p[0]), sy * (p[0] if swap else p[1])))
        return ops
    if domain == "cube":
        ops = []
        for perm in itertools.permutations(range(3)):
            for s in itertools.product((1, -1), repeat=3):
                ops.append(lambda p, perm=perm, s=s: tuple(s[i] * p[perm[i]] for i in range(3)))
        return ops
    if domain == "pyr":
        sq = symmetry_group("sqr")
        return [lambda p, g=g: (*g((p[0], p[1])), p[2]) for g in sq]
    if domain == "pri":
        # barycentric coords of triangle A=(-1,-1), B=(1,-1), C=(-1,1)
        def to_bary(x, y):
            return (-(x + y) / 2, (x + 1) / 2, (y + 1) / 2)

        def from_bary(l):
            return (2 * l[1] - 1, 2 * l[2] - 1)

        ops = []
        for perm in itertools.permutations(range(3)):
            for sz in (1, -1):
                ops.append(lambda p, perm=perm, sz=sz:
                           (*from_bary(tuple(to_bary(p[0], p[1])[perm[i]] for i in range(3))), sz * p[2]))
        return ops
    raise ValueError(domain)


# ---------------------------------------------------------------- main check
def load_rule(path):
    pts, wts = [], []
    with open(path) as f:
        for row in csv.reader(f):
            if not row:
                continue
            try:
                vals = [mp.mpf(v.strip()) for v in row]
            except (ValueError, TypeError):
                continue  # header
            pts.append(tuple(vals[:-1]))
            wts.append(vals[-1])
    return pts, wts


def check(domain, q, pts, wts, sym_tol=None, verbose=True):
    dim = DIM[domain]
    assert all(len(p) == dim for p in pts), "wrong dimension"
    n = len(pts)

    # node scale for tolerance reporting
    vol = exact_moment(domain, (0,) * dim)

    # 1. exactness (relative to the integral of |monomial| scale ~ vol)
    max_err = mp.mpf(0)
    worst = None
    for e in exponents(dim, q):
        exact = exact_moment(domain, e)
        approx = mp.fsum(w * mp.fprod(p[i] ** e[i] for i in range(dim)) for p, w in zip(pts, wts))
        err = abs(approx - mp.mpf(exact.numerator) / exact.denominator)
        if err > max_err:
            max_err, worst = err, e
    # 2. positivity
    min_w = min(wts)
    # 3. interiority
    min_margin = min(interior_margin(domain, p) for p in pts)
    # 4. symmetry: every image of every node must coincide with a node of equal weight
    if sym_tol is None:
        sym_tol = mp.mpf(10) ** -10
    sym_bad = 0
    keys = [(p, w) for p, w in zip(pts, wts)]
    for g in symmetry_group(domain):
        for p, w in keys:
            gp = g(p)
            if not any(max(abs(gp[i] - p2[i]) for i in range(dim)) < sym_tol and abs(w - w2) < sym_tol
                       for p2, w2 in keys):
                sym_bad += 1
                break
    # 5. distinct nodes
    distinct = n
    for i in range(n):
        for j in range(i):
            if max(abs(pts[i][k] - pts[j][k]) for k in range(dim)) < sym_tol:
                distinct -= 1
                break

    res = dict(nodes=n, distinct=distinct, max_moment_error=max_err, worst_monomial=worst,
               min_weight=min_w, min_interior_margin=min_margin, symmetry_violations=sym_bad,
               volume=vol)
    if verbose:
        print(f"domain={domain} degree={q} nodes={n} (distinct {distinct})")
        print(f"  exactness : max |error| over all monomials of degree<={q} = {mp.nstr(max_err, 3)}"
              f"  (worst monomial exponents {worst})")
        print(f"  positive  : min weight = {mp.nstr(min_w, 6)}  -> {'OK' if min_w > 0 else 'FAIL'}")
        print(f"  interior  : min margin to boundary = {mp.nstr(min_margin, 6)}  -> {'OK' if min_margin > 0 else 'FAIL'}")
        print(f"  symmetric : {'OK' if sym_bad == 0 else f'FAIL ({sym_bad} group elements broken)'}")
    return res


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(2)
    domain, q, path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    pts, wts = load_rule(path)
    r = check(domain, q, pts, wts)
    tol = mp.mpf(10) ** -12
    ok = (r["max_moment_error"] < tol and r["min_weight"] > 0 and r["min_interior_margin"] > 0
          and r["symmetry_violations"] == 0 and r["distinct"] == r["nodes"])
    print("VERDICT:", "PASS" if ok else "FAIL", f"(exactness tolerance {mp.nstr(tol, 2)})")
    sys.exit(0 if ok else 1)
