"""Numpy implementations of the orthonormal invariant bases with analytic gradients.

basis_grad(dom_name, q) returns a function  pts (R,dim) -> (Phi (E,R), dPhi (dim,E,R)).
Index ordering matches qsearch.<Domain>.invariant_index(q).
"""
import math
import itertools
import numpy as np


def legendre_with_deriv(x, n):
    """P_k(x), P_k'(x) for k=0..n.  x: (R,)"""
    P = np.zeros((n + 1, x.size)); D = np.zeros_like(P)
    P[0] = 1.0
    if n >= 1:
        P[1] = x; D[1] = 1.0
    for k in range(1, n):
        P[k + 1] = ((2 * k + 1) * x * P[k] - k * P[k - 1]) / (k + 1)
        D[k + 1] = ((2 * k + 1) * (P[k] + x * D[k]) - k * D[k - 1]) / (k + 1)
    return P, D


def scaled_legendre_with_deriv(x, h, n):
    """Q_k = h^k P_k(x/h), dQ/dx, dQ/dh for k=0..n."""
    Q = np.zeros((n + 1, x.size)); Dx = np.zeros_like(Q); Dh = np.zeros_like(Q)
    Q[0] = 1.0
    if n >= 1:
        Q[1] = x; Dx[1] = 1.0
    for k in range(1, n):
        Q[k + 1] = ((2 * k + 1) * x * Q[k] - k * h * h * Q[k - 1]) / (k + 1)
        Dx[k + 1] = ((2 * k + 1) * (Q[k] + x * Dx[k]) - k * h * h * Dx[k - 1]) / (k + 1)
        Dh[k + 1] = ((2 * k + 1) * x * Dh[k] - k * (2 * h * Q[k - 1] + h * h * Dh[k - 1])) / (k + 1)
    return Q, Dx, Dh


def jacobi_a0_with_deriv(z, alpha, n):
    P = np.zeros((n + 1, z.size)); D = np.zeros_like(P)
    P[0] = 1.0
    if n >= 1:
        P[1] = ((alpha + 2) * z + alpha) / 2; D[1] = (alpha + 2) / 2
    a = alpha
    for k in range(2, n + 1):
        c1 = 2 * k * (k + a) * (2 * k + a - 2)
        c2a = (2 * k + a - 1) * (2 * k + a) * (2 * k + a - 2)
        c2b = (2 * k + a - 1) * a * a
        c3 = 2 * (k + a - 1) * (k - 1) * (2 * k + a)
        P[k] = ((c2a * z + c2b) * P[k - 1] - c3 * P[k - 2]) / c1
        D[k] = (c2a * P[k - 1] + (c2a * z + c2b) * D[k - 1] - c3 * D[k - 2]) / c1
    return P, D


def _segment_sum(vals, rows, E):
    out = np.zeros((E,) + vals.shape[1:])
    np.add.at(out, rows, vals)
    return out


def make_cube(q, index):
    rows, A, B, C, scale = [], [], [], [], []
    for r, (a, b, c) in enumerate(index):
        perms = sorted(set(itertools.permutations((a, b, c))))
        nrm = math.sqrt(len(perms) * (2 / (2 * a + 1)) * (2 / (2 * b + 1)) * (2 / (2 * c + 1)))
        for p in perms:
            rows.append(r); A.append(p[0]); B.append(p[1]); C.append(p[2]); scale.append(1 / nrm)
    rows, A, B, C = map(np.array, (rows, A, B, C)); scale = np.array(scale)[:, None]
    E = len(index)

    def f(pts):
        Lx, Dx = legendre_with_deriv(pts[:, 0], q)
        Ly, Dy = legendre_with_deriv(pts[:, 1], q)
        Lz, Dz = legendre_with_deriv(pts[:, 2], q)
        X, Y, Z = Lx[A], Ly[B], Lz[C]
        Phi = _segment_sum(scale * X * Y * Z, rows, E)
        g = np.stack([_segment_sum(scale * Dx[A] * Y * Z, rows, E),
                      _segment_sum(scale * X * Dy[B] * Z, rows, E),
                      _segment_sum(scale * X * Y * Dz[C], rows, E)])
        return Phi, g
    return f


def make_sqr(q, index):
    rows, A, B, scale = [], [], [], []
    for r, (a, b) in enumerate(index):
        perms = sorted(set(itertools.permutations((a, b))))
        nrm = math.sqrt(len(perms) * (2 / (2 * a + 1)) * (2 / (2 * b + 1)))
        for p in perms:
            rows.append(r); A.append(p[0]); B.append(p[1]); scale.append(1 / nrm)
    rows, A, B = map(np.array, (rows, A, B)); scale = np.array(scale)[:, None]
    E = len(index)

    def f(pts):
        Lx, Dx = legendre_with_deriv(pts[:, 0], q)
        Ly, Dy = legendre_with_deriv(pts[:, 1], q)
        X, Y = Lx[A], Ly[B]
        Phi = _segment_sum(scale * X * Y, rows, E)
        g = np.stack([_segment_sum(scale * Dx[A] * Y, rows, E),
                      _segment_sum(scale * X * Dy[B], rows, E)])
        return Phi, g
    return f


def make_pyr(q, index):
    rows, A, B, Jrow, scale = [], [], [], [], []
    jkeys = {}
    for s in range(0, q + 1, 2):
        for c in range(0, q - s + 1):
            jkeys[(s, c)] = len(jkeys)
    for r, (a, b, c) in enumerate(index):
        s = a + b
        perms = sorted(set([(a, b), (b, a)]))
        nrm = math.sqrt(len(perms) * (2 / (2 * a + 1)) * (2 / (2 * b + 1)) * (2 / (2 * c + 2 * s + 3)))
        for p in perms:
            rows.append(r); A.append(p[0]); B.append(p[1]); Jrow.append(jkeys[(s, c)]); scale.append(1 / nrm)
    rows, A, B, Jrow = map(np.array, (rows, A, B, Jrow)); scale = np.array(scale)[:, None]
    E = len(index)

    def f(pts):
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        h = (1 - z) / 2
        Qx, Qxx, Qxh = scaled_legendre_with_deriv(x, h, q)
        Qy, Qyy, Qyh = scaled_legendre_with_deriv(y, h, q)
        Jall = np.zeros((len(jkeys), x.size)); Jd = np.zeros_like(Jall)
        for s in range(0, q + 1, 2):
            P, D = jacobi_a0_with_deriv(z, 2 * s + 2, q - s)
            for c in range(q - s + 1):
                Jall[jkeys[(s, c)]] = P[c]; Jd[jkeys[(s, c)]] = D[c]
        X, Y, J = Qx[A], Qy[B], Jall[Jrow]
        Phi = _segment_sum(scale * X * Y * J, rows, E)
        gx = _segment_sum(scale * Qxx[A] * Y * J, rows, E)
        gy = _segment_sum(scale * X * Qyy[B] * J, rows, E)
        gz = _segment_sum(scale * ((Qxh[A] * Y + X * Qyh[B]) * (-0.5) * J + X * Y * Jd[Jrow]), rows, E)
        return Phi, np.stack([gx, gy, gz])
    return f


MAKERS = {"cube": make_cube, "sqr": make_sqr, "pyr": make_pyr}


# ---------------------------------------------------------------- prism (triangle x line)
def dubiner_with_deriv(r, s, n):
    """Orthonormal PKD/Dubiner basis on triangle (-1,-1),(1,-1),(-1,1), all degrees <= n.
    Returns (B, Br, Bs, degs): arrays (M,R) and degree of each basis function, graded order."""
    xp = r + (1 + s) / 2
    h = (1 - s) / 2
    Q, Qx, Qh = scaled_legendre_with_deriv(xp, h, n)
    rows, dr, ds, degs = [], [], [], []
    for d in range(n + 1):
        for i in range(d + 1):
            j = d - i
            P, D = jacobi_a0_with_deriv(s, 2 * i + 1, j)
            nrm = math.sqrt((2 / (2 * i + 1)) * (2 / (2 * i + 2 * j + 2)))
            rows.append(Q[i] * P[j] / nrm)
            dr.append(Qx[i] * P[j] / nrm)
            ds.append(((0.5 * Qx[i] - 0.5 * Qh[i]) * P[j] + Q[i] * D[j]) / nrm)
            degs.append(d)
    return np.array(rows), np.array(dr), np.array(ds), np.array(degs)


def _tri_ops():
    ops = []
    for perm in itertools.permutations(range(3)):
        def op(x, y, perm=perm):
            l = (-(x + y) / 2, (x + 1) / 2, (y + 1) / 2)
            m = [l[perm[0]], l[perm[1]], l[perm[2]]]
            return 2 * m[1] - 1, 2 * m[2] - 1
        ops.append(op)
    return ops


_TRI_INV = {}


def triangle_invariant(n):
    """Matrix U (M x E_tri) whose columns span the D3-invariant subspace of the Dubiner basis,
    computed per degree block; returns (U, degree of each invariant function)."""
    if n in _TRI_INV:
        return _TRI_INV[n]
    rng = np.random.default_rng(12345)
    # random interior points
    l = rng.dirichlet((1, 1, 1), size=4 * (n + 1) * (n + 2))
    x, y = 2 * l[:, 1] - 1, 2 * l[:, 2] - 1
    B, _, _, degs = dubiner_with_deriv(x, y, n)
    M = len(degs)
    Rey = np.zeros((M, M))
    pinv = np.linalg.pinv(B.T)
    for op in _tri_ops():
        xs, ys = op(x, y)
        Bs, _, _, _ = dubiner_with_deriv(xs, ys, n)
        Rey += (pinv @ Bs.T).T  # B(sigma x) = M_sigma B(x)
    Rey /= 6
    cols, cdeg = [], []
    for d in range(n + 1):
        idx = np.where(degs == d)[0]
        blk = Rey[np.ix_(idx, idx)]
        blk = (blk + blk.T) / 2
        w, V = np.linalg.eigh(blk)
        for k in np.where(w > 0.5)[0]:
            v = np.zeros(M); v[idx] = V[:, k]
            cols.append(v); cdeg.append(d)
    U = np.array(cols).T
    _TRI_INV[n] = (U, np.array(cdeg))
    return _TRI_INV[n]


def make_pri(q, index):
    """index: list of (k, c) = (triangle invariant function k, Legendre degree c even)."""
    U, cdeg = triangle_invariant(q)
    K = np.array([k for k, c in index]); C = np.array([c for k, c in index])
    scale = np.array([1 / math.sqrt(2 / (2 * c + 1)) for k, c in index])[:, None]

    def f(pts):
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        B, Br, Bs, _ = dubiner_with_deriv(x, y, q)
        T, Tr, Ts = U.T @ B, U.T @ Br, U.T @ Bs
        L, Ld = legendre_with_deriv(z, q)
        Phi = scale * T[K] * L[C]
        g = np.stack([scale * Tr[K] * L[C], scale * Ts[K] * L[C], scale * T[K] * Ld[C]])
        return Phi, g
    return f


MAKERS["pri"] = make_pri
