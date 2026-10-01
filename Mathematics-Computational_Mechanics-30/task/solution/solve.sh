#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np


def neighbor_family(coords, i, delta, dx):
    """Discrete family of node i on a uniform grid, with unmodified midpoint volumes.

    Returns an (n_F, 5) float64 array with columns [j, xi_x, xi_y, |xi|, V_j] for every
    neighbour j with 0 < |xi_j| <= delta, in increasing node-index order (j stored as float).
    V_j = dx^2 for every neighbour: the source's numerical study uses unmodified midpoint
    quadrature, with NO partial-volume correction for bonds cut by the horizon.
    """
    import numpy as np
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] < 2:
        raise ValueError("coords must be an (N, 2) array with N >= 2")
    if int(i) != i or not (0 <= i < coords.shape[0]):
        raise ValueError("i must be a valid node index")
    if delta <= 0.0 or dx <= 0.0:
        raise ValueError("delta and dx must be positive")
    xi = coords - coords[int(i)]
    # Membership is decided on the INTEGER lattice offsets, not on float distances: on a
    # uniform grid F_i = {j : 0 < |xi_j| <= delta} is exactly {(a,b) : 0 < a^2+b^2 <= m^2}
    # with m = delta/dx, and a node sitting exactly on the horizon (e.g. (m,0) or (3,4)
    # for m = 5) must not drop in or out with the last bit of a hypot().
    off = np.rint(xi / float(dx))
    if not np.allclose(off * float(dx), xi, rtol=0.0, atol=1e-9 * float(dx)):
        raise ValueError("coords must lie on a uniform lattice of spacing dx")
    a2b2 = off[:, 0] ** 2 + off[:, 1] ** 2
    m2 = (float(delta) / float(dx)) ** 2
    mask = (a2b2 > 0.0) & (a2b2 <= m2 * (1.0 + 1e-9))
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        raise ValueError("node has no neighbours inside the horizon")
    r = np.hypot(xi[idx, 0], xi[idx, 1])
    V = np.full(idx.size, float(dx) * float(dx))
    return np.column_stack([idx.astype(np.float64), xi[idx, 0], xi[idx, 1], r, V])

import numpy as np


def scaled_monomial_basis(xi, delta):
    """Eq (49): the complete quadratic monomial basis with coordinates scaled by delta.

    Returns an (n, 6) array with columns [1, x/d, y/d, x^2/d^2, xy/d^2, y^2/d^2].
    """
    import numpy as np
    xi = np.asarray(xi, dtype=np.float64)
    if xi.ndim != 2 or xi.shape[1] != 2 or xi.shape[0] < 1:
        raise ValueError("xi must be a non-empty (n, 2) array")
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    x = xi[:, 0] / float(delta)
    y = xi[:, 1] / float(delta)
    return np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])

import numpy as np


def derivative_block(family, basis, basis0, delta, q):
    """Eqs (56)-(57): the x- and y-derivative operator matrices, stacked as (2*n_T, n_p).

    Row (l, k): sum_j  (p(xi_j) + p(0))/2  *  xi_{j,l} / |xi_j|^q  *  (phi_k(xi_j) - phi_k(0))  * V_j
    for the n_T = 5 non-constant scaled monomial tests (the constant test gives an
    identically zero row and is omitted). The bond-symmetrized average (p(xi_j) + p(0))/2
    is what makes the fitted weights enter every bond symmetrically.
    """
    import numpy as np
    fam = np.asarray(family, dtype=np.float64)
    P = np.asarray(basis, dtype=np.float64)
    p0 = np.asarray(basis0, dtype=np.float64).reshape(-1)
    if fam.ndim != 2 or fam.shape[1] != 5:
        raise ValueError("family must be (n_F, 5)")
    if P.ndim != 2 or P.shape[0] != fam.shape[0] or P.shape[1] != p0.size or P.shape[1] < 2:
        raise ValueError("basis must be (n_F, n_p) and basis0 must have n_p entries")
    if delta <= 0.0 or q <= 0.0:
        raise ValueError("delta and q must be positive")
    Pav = 0.5 * (P + p0[None, :])
    dphi = (P - p0[None, :])[:, 1:]            # the n_p - 1 non-constant tests
    r = fam[:, 3]
    V = fam[:, 4]
    blocks = []
    for l in (1, 2):                            # xi_x, xi_y columns
        kern = fam[:, l] / r ** q * V
        blocks.append((Pav[:, None, :] * (kern[:, None] * dphi)[:, :, None]).sum(axis=0))
    return np.vstack(blocks)

import numpy as np


def energy_block(family, basis, basis0, delta, q):
    """Eq (59): the energy operator rows for the vector tests phi_k e_l, stacked (2*n_T, n_p).

    Row (l, k): sum_j  (p(xi_j) + p(0))/2  *  ((phi_k(xi_j) - phi_k(0)) * xi_{j,l})^2 / |xi_j|^q  * V_j
    x-lifted tests first, then y-lifted, for the 5 non-constant scaled monomials. Any
    constant prefactor (the micromodulus, the 1/4 of the source's eq 45) is common to the
    row and its target and is removed by the per-row normalization of step 6.
    """
    import numpy as np
    fam = np.asarray(family, dtype=np.float64)
    P = np.asarray(basis, dtype=np.float64)
    p0 = np.asarray(basis0, dtype=np.float64).reshape(-1)
    if fam.ndim != 2 or fam.shape[1] != 5:
        raise ValueError("family must be (n_F, 5)")
    if P.ndim != 2 or P.shape[0] != fam.shape[0] or P.shape[1] != p0.size or P.shape[1] < 2:
        raise ValueError("basis must be (n_F, n_p) and basis0 must have n_p entries")
    if delta <= 0.0 or q <= 0.0:
        raise ValueError("delta and q must be positive")
    Pav = 0.5 * (P + p0[None, :])
    dphi = (P - p0[None, :])[:, 1:]
    r = fam[:, 3]
    V = fam[:, 4]
    blocks = []
    for l in (1, 2):
        kern = fam[:, l] ** 2 / r ** q * V
        blocks.append((Pav[:, None, :] * (kern[:, None] * dphi ** 2)[:, :, None]).sum(axis=0))
    return np.vstack(blocks)

import numpy as np


def full_ball_targets(delta, q):
    """Eqs (60)-(63): the operators applied to the tests over the FULL ball, analytically.

    Returns a 1-D array of length 4*n_T = 20: [d_x*, d_y*, e_x*, e_y*] for the five
    non-constant scaled monomial tests x/d, y/d, x^2/d^2, xy/d^2, y^2/d^2. Uses the
    closed-form ball moments
        I_q(r1, r2) = int_{|xi|<=d} xi_x^r1 xi_y^r2 / |xi|^q dA
                    = d^(r1+r2+2-q) / (r1+r2+2-q) * int_0^{2pi} cos^r1 sin^r2 dtheta,
    which is the source's eq (63); it vanishes unless both exponents are even.
    """
    import numpy as np
    from math import gamma
    if delta <= 0.0 or q <= 0.0:
        raise ValueError("delta and q must be positive")
    mono = [(1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]

    def angular(p, s):
        if p % 2 or s % 2:
            return 0.0
        return 2.0 * gamma((p + 1) / 2.0) * gamma((s + 1) / 2.0) / gamma((p + s) / 2.0 + 1.0)

    def moment(r1, r2):
        e = r1 + r2 + 2 - q
        if e <= 0:
            raise ValueError("non-integrable ball moment")
        return delta ** e / e * angular(r1, r2)

    d = [[], []]
    e = [[], []]
    for a, b in mono:
        sc = delta ** (a + b)
        d[0].append(moment(a + 1, b) / sc)
        d[1].append(moment(a, b + 1) / sc)
        e[0].append(moment(2 * a + 2, 2 * b) / sc ** 2)
        e[1].append(moment(2 * a, 2 * b + 2) / sc ** 2)
    out = np.array(d[0] + d[1] + e[0] + e[1], dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError("targets must be finite")
    return out

import numpy as np


def local_coupling_row(basis, m_deriv, m_energy, targets, lam):
    """Eqs (64)-(68) plus the source's per-row normalization: one node's row of (69).

    M = [M_c; M_D; M_E] with M_c the basis at the neighbours (collocation, target = the
    neighbours' weights w_j, which are UNKNOWN) and the operator blocks with the analytic
    targets. Every row of M, and its target, is scaled to unit Euclidean norm; then
    a = (M~^T M~ + lam I)^-1 M~^T b~ and the node's own weight is w_i = p(0)^T a = a_0.
    Reading off the first row of the generalized inverse gives
        w_i = sum_j gamma_j w_j + f_i,
    where gamma_j multiplies neighbour j's weight and f_i collects the target
    contributions. Returns the (n_F + 1,) array [f_i, gamma_1, ..., gamma_nF].
    """
    import numpy as np
    Mc = np.asarray(basis, dtype=np.float64)
    MD = np.asarray(m_deriv, dtype=np.float64)
    ME = np.asarray(m_energy, dtype=np.float64)
    t = np.asarray(targets, dtype=np.float64).reshape(-1)
    if Mc.ndim != 2 or MD.ndim != 2 or ME.ndim != 2:
        raise ValueError("blocks must be 2-D")
    n_p = Mc.shape[1]
    if MD.shape[1] != n_p or ME.shape[1] != n_p:
        raise ValueError("all blocks must share the basis dimension")
    if t.size != MD.shape[0] + ME.shape[0]:
        raise ValueError("targets must hold one entry per derivative and energy row")
    if lam < 0.0:
        raise ValueError("lam must be nonnegative")
    n_F = Mc.shape[0]
    M = np.vstack([Mc, MD, ME])
    nr = np.linalg.norm(M, axis=1)
    nr = np.where(nr > 1e-20, nr, 1.0)          # an all-zero row stays zero, untouched
    Mn = M / nr[:, None]
    A = Mn.T @ Mn + float(lam) * np.eye(n_p)
    Mplus = np.linalg.solve(A, Mn.T)            # (n_p, n_rows) regularized generalized inverse
    row0 = Mplus[0] / nr                         # scaling the target by 1/nr as well
    gamma = row0[:n_F]
    f_i = float(row0[n_F:] @ t)
    return np.concatenate([[f_i], gamma])

import numpy as np


def corner_influence_weight(nx, ny, dx, m, q, lam):
    """ORCHESTRATOR. Optimized influence weight at the corner node (0, 0) of an nx-by-ny grid.

    Calls every earlier step directly and uses each result. For EVERY node of the lattice:
      1 family with midpoint volumes, 2 scaled basis at the neighbours and at the node,
      3 derivative block, 4 energy block, 5 full-ball targets (once), 6 the normalized local
      fit, read off as w_i - sum_j gamma_ij w_j = f_i. Those N rows form the global sparse
      system K_w w = F_w of the source's eq (69); it is solved directly and w[0] returned.
    """
    import numpy as np
    if int(nx) != nx or int(ny) != ny or nx < 2 or ny < 2:
        raise ValueError("nx and ny must be integers >= 2")
    if dx <= 0.0 or q <= 0.0 or lam < 0.0:
        raise ValueError("dx and q must be positive, lam nonnegative")
    if int(m) != m or m < 1:
        raise ValueError("m must be a positive integer")
    nx, ny, m = int(nx), int(ny), int(m)
    delta = m * float(dx)
    if delta > (min(nx, ny) - 1) * dx:
        raise ValueError("horizon exceeds the grid; the corner family would be clipped by extent, not by the boundary")
    X, Y = np.meshgrid(np.arange(nx) * dx, np.arange(ny) * dx, indexing="ij")
    coords = np.column_stack([X.ravel(), Y.ravel()])
    N = coords.shape[0]

    p0 = scaled_monomial_basis(np.zeros((1, 2)), delta)[0]
    tgt = full_ball_targets(delta, q)
    K = np.eye(N)
    F = np.zeros(N)
    for i in range(N):
        fam = neighbor_family(coords, i, delta, dx)
        P = scaled_monomial_basis(fam[:, 1:3], delta)
        MD = derivative_block(fam, P, p0, delta, q)
        ME = energy_block(fam, P, p0, delta, q)
        row = local_coupling_row(P, MD, ME, tgt, lam)
        idx = fam[:, 0].astype(int)
        K[i, idx] -= row[1:]
        F[i] = row[0]
    w = np.linalg.solve(K, F)
    return float(w[0])
SCICODE_GOLD_EOF
