"""Search engine for fully symmetric positive interior (f-SPI) quadrature rules.

Formulation: a symmetric rule is a list of orbits. Each orbit has a type (which
symmetry class), unconstrained parameters u (mapped smoothly into the interior of
the domain) and a log-weight (weight = exp(omega) > 0). Positivity and interiority
therefore hold by construction.

Moment equations are written in an orthonormal basis of the *invariant* polynomials
of degree <= q, so each orbit contributes  size * weight * phi(representative).

Domains implemented: pyr (pyramid), cube, sqr. Conventions match referee.py.
"""
import math
import itertools
import numpy as np
try:  # jax only needed for the reference (slow) basis implementation used in tests
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
except ImportError:  # pragma: no cover
    jax = jnp = None


# ---------------------------------------------------------------- 1D polynomial helpers
def scaled_legendre(x, h, n):
    """[h^k P_k(x/h) for k=0..n] (polynomials in x and h, no division)."""
    Q = [jnp.ones_like(x), x]
    for k in range(1, n):
        Q.append(((2 * k + 1) * x * Q[k] - k * h * h * Q[k - 1]) / (k + 1))
    return Q[: n + 1]


def legendre(x, n):
    return scaled_legendre(x, jnp.ones_like(x), n)


def jacobi_a0(z, alpha, n):
    """[P_k^{(alpha,0)}(z) for k=0..n]."""
    P = [jnp.ones_like(z)]
    if n >= 1:
        P.append(((alpha + 2) * z + alpha) / 2)
    for k in range(2, n + 1):
        a = alpha
        c1 = 2 * k * (k + a) * (2 * k + a - 2)
        c2 = (2 * k + a - 1) * ((2 * k + a) * (2 * k + a - 2) * z + a * a)
        c3 = 2 * (k + a - 1) * (k - 1) * (2 * k + a)
        P.append((c2 * P[k - 1] - c3 * P[k - 2]) / c1)
    return P[: n + 1]


# ---------------------------------------------------------------- domains
class Domain:
    name = None
    dim = None
    orbit_types = None  # dict name -> (size, nparams)

    def invariant_index(self, q):
        raise NotImplementedError

    def basis(self, pts, q):
        """pts: (R, dim) jnp array -> (E, R) orthonormal invariant basis values."""
        raise NotImplementedError

    def rep(self, otype, u):
        """unconstrained params u (k,) -> representative point (dim,)."""
        raise NotImplementedError

    def unrep(self, otype, p):
        raise NotImplementedError

    def classify(self, p, tol=1e-9):
        raise NotImplementedError

    def group(self):
        raise NotImplementedError


def _atanh(v):
    v = np.clip(v, -1 + 1e-15, 1 - 1e-15)
    return np.arctanh(v)


class Pyramid(Domain):
    """|z|<1, |x|,|y| < h=(1-z)/2.  Group C4v on (x,y)."""
    name = "pyr"
    dim = 3
    # S1 (0,0,z) | S2 (a,0,z) | S3 (a,a,z) | S4 (a,b,z)
    orbit_types = {"S1": (1, 1), "S2": (4, 2), "S3": (4, 2), "S4": (8, 3)}
    order = ["S1", "S2", "S3", "S4"]

    def invariant_index(self, q):
        idx = []
        for a in range(0, q + 1, 2):
            for b in range(0, a + 1, 2):
                for c in range(0, q - a - b + 1):
                    idx.append((a, b, c))
        return idx

    def basis(self, pts, q):
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        h = (1 - z) / 2
        Lx = scaled_legendre(x, h, q)
        Ly = scaled_legendre(y, h, q)
        out = []
        jac_cache = {}
        for (a, b, c) in self.invariant_index(q):
            s = a + b
            if s not in jac_cache:
                jac_cache[s] = jacobi_a0(z, 2 * s + 2, q - s)
            J = jac_cache[s][c]
            if a == b:
                xy = Lx[a] * Ly[b]
                nrm2 = (2 / (2 * a + 1)) * (2 / (2 * b + 1)) * (2 / (2 * c + 2 * s + 3))
            else:
                xy = Lx[a] * Ly[b] + Lx[b] * Ly[a]
                nrm2 = 2 * (2 / (2 * a + 1)) * (2 / (2 * b + 1)) * (2 / (2 * c + 2 * s + 3))
            out.append(xy * J / math.sqrt(nrm2))
        return jnp.stack(out)

    def volume(self):
        return 8.0 / 3.0

    def rep(self, otype, u):
        z = jnp.tanh(u[0])
        h = (1 - z) / 2
        if otype == "S1":
            return jnp.stack([0 * z, 0 * z, z])
        if otype == "S2":
            a = h * (1 + jnp.tanh(u[1])) / 2          # a in (0, h)
            return jnp.stack([a, 0 * z, z])
        if otype == "S3":
            a = h * (1 + jnp.tanh(u[1])) / 2
            return jnp.stack([a, a, z])
        if otype == "S4":
            a = h * jnp.tanh(u[1])
            b = h * jnp.tanh(u[2])
            return jnp.stack([a, b, z])
        raise ValueError(otype)

    def unrep(self, otype, p):
        x, y, z = p
        h = (1 - z) / 2
        u0 = _atanh(z)
        if otype == "S1":
            return np.array([u0])
        if otype == "S2":
            return np.array([u0, _atanh(2 * max(abs(x), abs(y)) / h - 1)])
        if otype == "S3":
            return np.array([u0, _atanh(2 * abs(x) / h - 1)])
        if otype == "S4":
            return np.array([u0, _atanh(x / h), _atanh(y / h)])
        raise ValueError(otype)

    def classify(self, p, tol=1e-9):
        x, y, z = abs(p[0]), abs(p[1]), p[2]
        if x < tol and y < tol:
            return "S1"
        if x < tol or y < tol:
            return "S2"
        if abs(x - y) < tol:
            return "S3"
        return "S4"

    def group(self):
        ops = []
        for swap in (False, True):
            for sx, sy in itertools.product((1, -1), repeat=2):
                ops.append((swap, sx, sy))

        def apply(op, p):
            swap, sx, sy = op
            x, y = (p[1], p[0]) if swap else (p[0], p[1])
            return (sx * x, sy * y, p[2])
        return [lambda p, op=op: apply(op, p) for op in ops]

    def margin(self, p):
        x, y, z = p
        h = (1 - z) / 2
        return min(1 - abs(z), h - abs(x), h - abs(y))


class Cube(Domain):
    """|x|,|y|,|z| < 1. Group Oh (48)."""
    name = "cube"
    dim = 3
    # S1 (0,0,0) S2 (a,0,0) S3 (a,a,a) S4 (a,a,0) S5 (a,a,b) S6 (a,b,0) S7 (a,b,c)
    orbit_types = {"S1": (1, 0), "S2": (6, 1), "S3": (8, 1), "S4": (12, 1),
                   "S5": (24, 2), "S6": (24, 2), "S7": (48, 3)}
    order = ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]

    def invariant_index(self, q):
        idx = []
        for a in range(0, q + 1, 2):
            for b in range(0, a + 1, 2):
                for c in range(0, b + 1, 2):
                    if a + b + c <= q:
                        idx.append((a, b, c))
        return idx

    def basis(self, pts, q):
        L = [legendre(pts[:, i], q) for i in range(3)]
        out = []
        for (a, b, c) in self.invariant_index(q):
            perms = set(itertools.permutations((a, b, c)))
            val = sum(L[0][p[0]] * L[1][p[1]] * L[2][p[2]] for p in perms)
            nrm2 = len(perms) * (2 / (2 * a + 1)) * (2 / (2 * b + 1)) * (2 / (2 * c + 1))
            out.append(val / math.sqrt(nrm2))
        return jnp.stack(out)

    def volume(self):
        return 8.0

    def rep(self, otype, u):
        t = jnp.tanh
        z0 = 0.0 * u[0] if u.shape[0] else jnp.float64(0.0)
        if otype == "S1":
            return jnp.zeros(3)
        if otype == "S2":
            a = t(u[0]); return jnp.stack([a, z0, z0])
        if otype == "S3":
            a = t(u[0]); return jnp.stack([a, a, a])
        if otype == "S4":
            a = t(u[0]); return jnp.stack([a, a, z0])
        if otype == "S5":
            a, b = t(u[0]), t(u[1]); return jnp.stack([a, a, b])
        if otype == "S6":
            a, b = t(u[0]), t(u[1]); return jnp.stack([a, b, z0])
        if otype == "S7":
            return jnp.stack([t(u[0]), t(u[1]), t(u[2])])
        raise ValueError(otype)

    def unrep(self, otype, p):
        v = sorted((abs(c) for c in p), reverse=True)  # a>=b>=c
        if otype == "S1":
            return np.zeros(0)
        if otype in ("S2", "S3", "S4"):
            return _atanh(np.array([v[0]]))
        if otype == "S5":
            # two equal, one different
            if abs(v[0] - v[1]) < 1e-9:
                return _atanh(np.array([v[0], v[2]]))
            return _atanh(np.array([v[1], v[0]]))
        if otype == "S6":
            return _atanh(np.array([v[0], v[1]]))
        if otype == "S7":
            return _atanh(np.array(v))
        raise ValueError(otype)

    def classify(self, p, tol=1e-9):
        v = sorted((abs(c) for c in p), reverse=True)
        zeros = sum(1 for c in v if c < tol)
        if zeros == 3:
            return "S1"
        if zeros == 2:
            return "S2"
        if zeros == 1:
            return "S4" if abs(v[0] - v[1]) < tol else "S6"
        eq = int(abs(v[0] - v[1]) < tol) + int(abs(v[1] - v[2]) < tol)
        if eq == 2:
            return "S3"
        if eq == 1:
            return "S5"
        return "S7"

    def group(self):
        ops = []
        for perm in itertools.permutations(range(3)):
            for s in itertools.product((1, -1), repeat=3):
                ops.append(lambda p, perm=perm, s=s: tuple(s[i] * p[perm[i]] for i in range(3)))
        return ops

    def margin(self, p):
        return min(1 - abs(c) for c in p)


class Square(Domain):
    name = "sqr"
    dim = 2
    orbit_types = {"S1": (1, 0), "S2": (4, 1), "S3": (4, 1), "S4": (8, 2)}
    order = ["S1", "S2", "S3", "S4"]

    def invariant_index(self, q):
        return [(a, b) for a in range(0, q + 1, 2) for b in range(0, a + 1, 2) if a + b <= q]

    def basis(self, pts, q):
        L = [legendre(pts[:, i], q) for i in range(2)]
        out = []
        for (a, b) in self.invariant_index(q):
            if a == b:
                val = L[0][a] * L[1][b]; nrm2 = (2 / (2 * a + 1)) * (2 / (2 * b + 1))
            else:
                val = L[0][a] * L[1][b] + L[0][b] * L[1][a]; nrm2 = 2 * (2 / (2 * a + 1)) * (2 / (2 * b + 1))
            out.append(val / math.sqrt(nrm2))
        return jnp.stack(out)

    def volume(self):
        return 4.0

    def rep(self, otype, u):
        t = jnp.tanh
        if otype == "S1":
            return jnp.zeros(2)
        if otype == "S2":
            a = t(u[0]); return jnp.stack([a, 0 * a])
        if otype == "S3":
            a = t(u[0]); return jnp.stack([a, a])
        if otype == "S4":
            return jnp.stack([t(u[0]), t(u[1])])
        raise ValueError(otype)

    def unrep(self, otype, p):
        v = sorted((abs(c) for c in p), reverse=True)
        if otype == "S1":
            return np.zeros(0)
        if otype in ("S2", "S3"):
            return _atanh(np.array([v[0]]))
        return _atanh(np.array(v))

    def classify(self, p, tol=1e-9):
        v = sorted((abs(c) for c in p), reverse=True)
        if v[0] < tol:
            return "S1"
        if v[1] < tol:
            return "S2"
        if abs(v[0] - v[1]) < tol:
            return "S3"
        return "S4"

    def group(self):
        ops = []
        for swap in (False, True):
            for sx, sy in itertools.product((1, -1), repeat=2):
                ops.append(lambda p, swap=swap, sx=sx, sy=sy:
                           (sx * (p[1] if swap else p[0]), sy * (p[0] if swap else p[1])))
        return ops

    def margin(self, p):
        return min(1 - abs(c) for c in p)


class Prism(Domain):
    """Triangle (-1,-1),(1,-1),(-1,1) x (-1,1). Group D3h (12).
    S1 centroid,z=0 (1) | S2 centroid,+-z (2) | S3 median,z=0 (3) | S4 median,+-z (6)
    S5 general,z=0 (6)  | S6 general,+-z (12)"""
    name = "pri"
    dim = 3
    orbit_types = {"S1": (1, 0), "S2": (2, 1), "S3": (3, 1), "S4": (6, 2), "S5": (6, 2), "S6": (12, 3)}
    order = ["S1", "S2", "S3", "S4", "S5", "S6"]

    def invariant_index(self, q):
        from fastbasis import triangle_invariant
        _, cdeg = triangle_invariant(q)
        return [(k, c) for k in range(len(cdeg)) for c in range(0, q + 1, 2) if cdeg[k] + c <= q]

    def volume(self):
        return 4.0

    @staticmethod
    def bary(x, y):
        return (-(x + y) / 2, (x + 1) / 2, (y + 1) / 2)

    def classify(self, p, tol=1e-9):
        l = sorted(self.bary(p[0], p[1]))
        zero = abs(p[2]) < tol
        if abs(l[0] - l[2]) < tol:
            return "S1" if zero else "S2"
        if abs(l[0] - l[1]) < tol or abs(l[1] - l[2]) < tol:
            return "S3" if zero else "S4"
        return "S5" if zero else "S6"

    def unrep(self, otype, p):
        l = sorted(self.bary(p[0], p[1]))
        uz = [_atanh(abs(p[2]))] if otype in ("S2", "S4", "S6") else []
        if otype in ("S1", "S2"):
            return np.array(uz, dtype=float)
        if otype in ("S3", "S4"):
            alpha = l[1]  # repeated value
            return np.array([_atanh(4 * alpha - 1)] + uz, dtype=float)
        return np.array([math.log(l[0] / l[2]), math.log(l[1] / l[2])] + uz, dtype=float)

    def group(self):
        ops = []
        for perm in itertools.permutations(range(3)):
            for sz in (1, -1):
                def op(p, perm=perm, sz=sz):
                    l = self.bary(p[0], p[1])
                    m = [l[perm[0]], l[perm[1]], l[perm[2]]]
                    return (2 * m[1] - 1, 2 * m[2] - 1, sz * p[2])
                ops.append(op)
        return ops

    def margin(self, p):
        return min(min(self.bary(p[0], p[1])) * 2, 1 - abs(p[2]))


DOMAINS = {"pyr": Pyramid, "cube": Cube, "sqr": Square, "pri": Prism}


# ---------------------------------------------------------------- numpy rep maps with Jacobians
def rep_jac(dom, otype, U):
    """Vectorised representative map. U: (n,k) -> P (n,dim), J (n,dim,k)."""
    U = np.asarray(U, dtype=float).reshape(len(U), -1)
    n = len(U)
    if dom.name == "pyr":
        z = np.tanh(U[:, 0]); dz = 1 - z * z; h = (1 - z) / 2; dh = -dz / 2
        P = np.zeros((n, 3)); P[:, 2] = z
        k = U.shape[1]; J = np.zeros((n, 3, k)); J[:, 2, 0] = dz
        if otype == "S1":
            return P, J
        if otype in ("S2", "S3"):
            t = np.tanh(U[:, 1]); s = (1 + t) / 2; a = h * s
            da0 = dh * s; da1 = h * (1 - t * t) / 2
            P[:, 0] = a; J[:, 0, 0] = da0; J[:, 0, 1] = da1
            if otype == "S3":
                P[:, 1] = a; J[:, 1, 0] = da0; J[:, 1, 1] = da1
            return P, J
        if otype == "S4":
            t1 = np.tanh(U[:, 1]); t2 = np.tanh(U[:, 2])
            P[:, 0] = h * t1; P[:, 1] = h * t2
            J[:, 0, 0] = dh * t1; J[:, 0, 1] = h * (1 - t1 * t1)
            J[:, 1, 0] = dh * t2; J[:, 1, 2] = h * (1 - t2 * t2)
            return P, J
    if dom.name in ("cube", "sqr"):
        # pattern: list over coordinates of parameter index (or None for zero)
        pats = {"cube": {"S1": (None, None, None), "S2": (0, None, None), "S3": (0, 0, 0), "S4": (0, 0, None),
                         "S5": (0, 0, 1), "S6": (0, 1, None), "S7": (0, 1, 2)},
                "sqr": {"S1": (None, None), "S2": (0, None), "S3": (0, 0), "S4": (0, 1)}}[dom.name]
        pat = pats[otype]
        T = np.tanh(U); dT = 1 - T * T
        P = np.zeros((n, dom.dim)); J = np.zeros((n, dom.dim, U.shape[1]))
        for c, pi in enumerate(pat):
            if pi is not None:
                P[:, c] = T[:, pi]; J[:, c, pi] = dT[:, pi]
        return P, J
    if dom.name == "pri":
        k = U.shape[1]
        P = np.zeros((n, 3)); J = np.zeros((n, 3, k))
        has_z = otype in ("S2", "S4", "S6")
        if has_z:
            z = np.tanh(U[:, -1]); P[:, 2] = z; J[:, 2, k - 1] = 1 - z * z
        if otype in ("S1", "S2"):
            P[:, 0] = -1 / 3; P[:, 1] = -1 / 3
            return P, J
        if otype in ("S3", "S4"):
            t = np.tanh(U[:, 0]); alpha = (1 + t) / 4; da = (1 - t * t) / 4
            # barycentric (1-2a, a, a): x = 2a-1, y = 2a-1
            P[:, 0] = 2 * alpha - 1; P[:, 1] = 2 * alpha - 1
            J[:, 0, 0] = 2 * da; J[:, 1, 0] = 2 * da
            return P, J
        # general: barycentric softmax(u1, u2, 0) -> (lA, lB, lC); x = 2 lB - 1, y = 2 lC - 1
        e = np.exp(np.stack([U[:, 0], U[:, 1], np.zeros(n)], axis=1) - np.maximum(0, U[:, :2].max(axis=1))[:, None])
        L = e / e.sum(axis=1, keepdims=True)
        P[:, 0] = 2 * L[:, 1] - 1; P[:, 1] = 2 * L[:, 2] - 1
        # dL_i/du_j = L_i (delta_ij - L_j), j in {0,1}
        for j in range(2):
            J[:, 0, j] = 2 * L[:, 1] * ((1 if j == 1 else 0) - L[:, j])
            J[:, 1, j] = 2 * L[:, 2] * (0 - L[:, j])
        return P, J
    raise ValueError((dom.name, otype))


# ---------------------------------------------------------------- rule representation
class Rule:
    """orbits: list of (otype, u (np array), omega (float))."""

    def __init__(self, dom, q, orbits):
        self.dom, self.q, self.orbits = dom, q, list(orbits)

    def copy(self):
        return Rule(self.dom, self.q, [(t, np.array(u, dtype=float), float(w)) for t, u, w in self.orbits])

    def nodes(self):
        return sum(self.dom.orbit_types[t][0] for t, _, _ in self.orbits)

    def dof(self):
        return sum(self.dom.orbit_types[t][1] + 1 for t, _, _ in self.orbits)

    def counts(self):
        return [sum(1 for t, _, _ in self.orbits if t == s) for s in self.dom.order]

    def expand(self):
        """Return (points, weights) of all nodes (numpy, float64)."""
        pts, wts = [], []
        G = self.dom.group()
        for t, u, om in self.orbits:
            r = tuple(float(v) for v in rep_jac(self.dom, t, np.asarray(u, dtype=float)[None, :])[0][0])
            imgs = []
            for g in G:
                gp = g(r)
                if not any(max(abs(gp[i] - p[i]) for i in range(len(gp))) < 1e-12 for p in imgs):
                    imgs.append(gp)
            assert len(imgs) == self.dom.orbit_types[t][0], (t, r, len(imgs))
            for p in imgs:
                pts.append(p); wts.append(math.exp(om))
        return np.array(pts), np.array(wts)


def load_expanded(dom, q, path, tol=1e-8):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    pts, wts = data[:, :-1], data[:, -1]
    G = dom.group()
    used = np.zeros(len(pts), bool)
    orbits = []
    for i in range(len(pts)):
        if used[i]:
            continue
        p = tuple(pts[i])
        # mark images
        for g in G:
            gp = np.array(g(p))
            d = np.max(np.abs(pts - gp), axis=1)
            used |= d < tol
        t = dom.classify(p, tol)
        orbits.append((t, dom.unrep(t, p), math.log(wts[i])))
    return Rule(dom, q, orbits)


# ---------------------------------------------------------------- residual and Jacobian
class System:
    def __init__(self, dom, q):
        self.dom, self.q = dom, q
        self.E = len(dom.invariant_index(q))
        # target: integral of orthonormal basis = 0 except constant = vol/sqrt(vol)
        f = np.zeros(self.E)
        f[0] = math.sqrt(dom.volume())
        self.f = f
        from fastbasis import MAKERS
        self._fast = MAKERS[dom.name](q, dom.invariant_index(q))
        self._basis = lambda P: self._fast(np.asarray(P))[0]
        self._basis_grad = lambda P: self._fast(np.asarray(P))[1]

    def pack(self, rule):
        return np.concatenate([np.concatenate([u, [om]]) for _, u, om in rule.orbits])

    def unpack(self, rule, theta):
        out, i = [], 0
        for t, _, _ in rule.orbits:
            k = self.dom.orbit_types[t][1]
            out.append((t, theta[i:i + k].copy(), float(theta[i + k])))
            i += k + 1
        return Rule(self.dom, self.q, out)

    def reps(self, rule):
        """Representative points (R, dim) and their d rep / d u (per orbit list)."""
        R = len(rule.orbits)
        pts = np.zeros((R, self.dom.dim))
        jacs = [None] * R
        by_type = {}
        for i, (t, u, _) in enumerate(rule.orbits):
            by_type.setdefault(t, []).append(i)
        for t, ids in by_type.items():
            k = self.dom.orbit_types[t][1]
            U = np.stack([rule.orbits[i][1] for i in ids]) if k else np.zeros((len(ids), 0))
            P, JJ = rep_jac(self.dom, t, U)
            pts[ids] = P
            if k:
                for j, i in enumerate(ids):
                    jacs[i] = JJ[j]
        return pts, jacs

    def param_scales(self, rule):
        """Physical sensitivity of each packed parameter: |d rep/d u_j| for coordinates, 1 for log-weights."""
        _, jacs = self.reps(rule)
        out = []
        for i, (t, u, om) in enumerate(rule.orbits):
            k = self.dom.orbit_types[t][1]
            if k:
                out.extend(np.linalg.norm(jacs[i], axis=0) + 1e-300)
            out.append(1.0)
        return np.array(out)

    def margins(self, rule):
        pts, _ = self.reps(rule)
        return np.array([self.dom.margin(tuple(p)) for p in pts])

    def residual(self, rule):
        pts, _ = self.reps(rule)
        Phi = self._basis(pts)  # (E, R)
        sw = np.array([self.dom.orbit_types[t][0] * math.exp(om) for t, _, om in rule.orbits])
        return Phi @ sw - self.f

    def res_jac(self, rule):
        pts, jacs = self.reps(rule)
        P = jnp.asarray(pts)
        Phi, dPhi = self._fast(pts)  # (E, R), (dim, E, R)
        sw = np.array([self.dom.orbit_types[t][0] * math.exp(om) for t, _, om in rule.orbits])
        r = Phi @ sw - self.f
        cols = []
        for i, (t, u, om) in enumerate(rule.orbits):
            k = self.dom.orbit_types[t][1]
            if k:
                g = np.einsum("de,dk->ek", dPhi[:, :, i], jacs[i]) * sw[i]  # (E, k)
                cols.append(g)
            cols.append((Phi[:, i] * sw[i])[:, None])
        return r, np.hstack(cols)


def solve(system, rule, maxit=2000, tol=1e-14, lam=1e-3, verbose=False, stall=200):
    """Levenberg-Marquardt on unconstrained variables. Returns (rule, ||r||)."""
    theta = system.pack(rule)
    cur = system.unpack(rule, theta)
    r, J = system.res_jac(cur)
    nr = np.linalg.norm(r)
    best_hist = [nr]
    for it in range(maxit):
        if nr < tol:
            break
        # LM step via augmented least squares (scaled by column norms)
        D = np.linalg.norm(J, axis=0) + 1e-12
        Js = J / D
        A = np.vstack([Js, math.sqrt(lam) * np.eye(Js.shape[1])])
        b = np.concatenate([-r, np.zeros(Js.shape[1])])
        step = np.linalg.lstsq(A, b, rcond=None)[0] / D
        # limit step size in unconstrained space for safety
        smax = np.max(np.abs(step))
        if smax > 2.0:
            step *= 2.0 / smax
        cand = system.unpack(rule, theta + step)
        rc = system.residual(cand)
        nc = np.linalg.norm(rc)
        if np.isfinite(nc) and nc < nr:
            theta = theta + step
            cur = cand
            r, J = system.res_jac(cur)
            nr = nc
            lam = max(lam / 3, 1e-12)
        else:
            lam = min(lam * 4, 1e12)
        best_hist.append(nr)
        if verbose and it % 50 == 0:
            print(f"    it {it} |r|={nr:.3e} lam={lam:.1e}")
        if len(best_hist) > stall and best_hist[-stall] < 1.0001 * nr and nr > 1e3 * tol:
            break  # stalled
    return cur, nr
