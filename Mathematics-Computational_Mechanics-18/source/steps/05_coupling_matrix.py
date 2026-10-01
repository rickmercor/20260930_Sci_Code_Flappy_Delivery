"""
Assemble the coupling stiffness between the global B-spline patch and the overlaid local Lagrange (bilinear Q4) mesh in the s-version isogeometric method (the paper's Eq. 46): the integral over the local domain of the global strain-displacement operator transpose times the plane-strain elasticity matrix times the local strain-displacement operator. Integrate element by element over the local mesh with the standard three-point Gauss rule per direction; the global geometry is the identity map, so the global basis is evaluated at the physical Gauss location. Because the global B-spline basis is smooth across its knot spans, no sub-element partitioning is used. Recover the coupling operator and the plane-strain matrix from the paper. Equation 46 is an integral over the whole local domain, so the result is assembled and not a collection of per-element blocks: element contributions are scatter-added into the degrees of freedom of the local nodes they share, and a node used by several elements carries one pair of columns in total. Local nodes are numbered by first appearance while scanning the element list, each contributing its x degree of freedom before its y, so the operator is (2*nG, 2*nL) with nL the number of distinct local nodes.

The s-version (superposition) method adds a fine local field on top of a coarse global one; the cross stiffness that couples the two discretisations is the term that makes the superposition consistent, and evaluating it correctly is what distinguishes the isogeometric global basis from a Lagrange one.

Returns
-------
return (2*nG, 2*nL) float64: the assembled global-local coupling stiffness, nL distinct local nodes
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def coupling_matrix(gknx, gkny, p, loc_elems, E, nu):
    """gknx, gkny: open knot vectors of the global quadratic B-spline patch (identity
    geometry); p: global degree; loc_elems: list of (4, 2) physical node arrays of the
    local Q4 elements; E, nu: plane-strain elastic constants. Returns a float64 array
    (2*nG, 2*nL) holding the ASSEMBLED global-local coupling stiffness of paper Eq. 46,
    integrated with the standard 3x3 Gauss rule per local element; nG is the number of
    global basis functions and nL the number of distinct local nodes. Element
    contributions are scatter-added into the degrees of freedom of the shared local
    nodes, so a node used by several elements carries one pair of columns and not one
    pair per element. Local nodes are numbered by first appearance while scanning
    loc_elems in order, each contributing its x degree of freedom before its y."""
    nG = (len(gknx) - p - 1) * (len(gkny) - p - 1)
    seen = []
    for xy in loc_elems:
        for x, y in np.asarray(xy, dtype=np.float64):
            k = (round(float(x), 9), round(float(y), 9))
            if k not in seen:
                seen.append(k)
    return np.zeros((2 * nG, 2 * len(seen)))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 5: global(B-spline)-local(Lagrange) coupling stiffness (Eq. 46)."""

import numpy as np


def _psD(E, nu):
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return c * np.array([[1.0 - nu, nu, 0.0],
                         [nu, 1.0 - nu, 0.0],
                         [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0]], dtype=np.float64)


_GPT = np.array([-(3.0 / 5.0) ** 0.5, 0.0, (3.0 / 5.0) ** 0.5], dtype=np.float64)
_GWT = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=np.float64)


def _q4b(xi, eta):
    N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                         (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)], dtype=np.float64)
    dNr = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)], dtype=np.float64)
    dNs = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)], dtype=np.float64)
    return N, dNr, dNs

def _bspl(knots, p, xi):
    knots = np.asarray(knots, dtype=np.float64)
    m = len(knots) - 1
    n = m - p - 1
    N = np.zeros((n + 1, p + 1), dtype=np.float64)
    for i in range(n + 1):
        hi = knots[i + 1]
        if (knots[i] <= xi < hi) or (xi == knots[-1] and knots[i] <= xi <= hi and hi == knots[-1]):
            N[i, 0] = 1.0
    for q in range(1, p + 1):
        for i in range(n + 1):
            a = 0.0
            d1 = knots[i + q] - knots[i]
            if d1 > 0:
                a = (xi - knots[i]) / d1 * N[i, q - 1]
            b = 0.0
            if i + 1 <= n:
                d2 = knots[i + q + 1] - knots[i + 1]
                if d2 > 0:
                    b = (knots[i + q + 1] - xi) / d2 * N[i + 1, q - 1]
            N[i, q] = a + b
    vals = N[:, p].copy()
    der = np.zeros(n + 1, dtype=np.float64)
    for i in range(n + 1):
        t1 = 0.0
        d1 = knots[i + p] - knots[i]
        if d1 > 0:
            t1 = p / d1 * N[i, p - 1]
        t2 = 0.0
        if i + 1 <= n:
            d2 = knots[i + p + 1] - knots[i + 1]
            if d2 > 0:
                t2 = p / d2 * N[i + 1, p - 1]
        der[i] = t1 - t2
    return vals, der


def _oracle_coupling_matrix(gknx, gkny, p, loc_elems, E, nu):
    if not isinstance(p, (int, np.integer)) or isinstance(p, bool) or p < 1:
        raise ValueError("p must be a positive integer degree")
    if len(gknx) < p + 2 or len(gkny) < p + 2:
        raise ValueError("knot vector too short for the requested degree")
    if len(loc_elems) < 1:
        raise ValueError("no local elements supplied")
    if not np.isfinite(E) or E <= 0 or not (-1.0 < float(nu) < 0.5):
        raise ValueError("invalid plane-strain constants")
    D = _psD(E, nu)
    nGx = len(gknx) - p - 1
    nGy = len(gkny) - p - 1
    nG = nGx * nGy
    node_ids = []
    conn = []
    for xy in loc_elems:
        ids = []
        for x, y in np.asarray(xy, dtype=np.float64):
            key = (round(float(x), 9), round(float(y), 9))
            if key not in node_ids:
                node_ids.append(key)
            ids.append(node_ids.index(key))
        conn.append(ids)
    nL = len(node_ids)
    K_full = np.zeros((2 * nG, 2 * nL), dtype=np.float64)
    for e, xy in enumerate(loc_elems):
        xy = np.asarray(xy, dtype=np.float64)
        Ke = np.zeros((2 * nG, 8), dtype=np.float64)
        for a in range(3):
            for b in range(3):
                xi, eta, w = _GPT[a], _GPT[b], _GWT[a] * _GWT[b]
                N, dNr, dNs = _q4b(xi, eta)
                J00 = dNr @ xy[:, 0]; J01 = dNr @ xy[:, 1]
                J10 = dNs @ xy[:, 0]; J11 = dNs @ xy[:, 1]
                detJ = J00 * J11 - J01 * J10
                dNx = (J11 * dNr - J01 * dNs) / detJ
                dNy = (-J10 * dNr + J00 * dNs) / detJ
                BL = np.zeros((3, 8))
                for k in range(4):
                    BL[0, 2 * k] = dNx[k]; BL[1, 2 * k + 1] = dNy[k]
                    BL[2, 2 * k] = dNy[k]; BL[2, 2 * k + 1] = dNx[k]
                px = N @ xy[:, 0]; py = N @ xy[:, 1]
                Nx, dNxg = _bspl(gknx, p, px)
                Ny, dNyg = _bspl(gkny, p, py)
                BG = np.zeros((3, 2 * nG))
                idx = 0
                for jy in range(nGy):
                    for ix in range(nGx):
                        gx = dNxg[ix] * Ny[jy]
                        gy = Nx[ix] * dNyg[jy]
                        BG[0, 2 * idx] = gx; BG[1, 2 * idx + 1] = gy
                        BG[2, 2 * idx] = gy; BG[2, 2 * idx + 1] = gx
                        idx += 1
                Ke += (BG.T @ D @ BL) * detJ * w
        for k in range(4):
            gk = conn[e][k]
            K_full[:, 2 * gk] += Ke[:, 2 * k]
            K_full[:, 2 * gk + 1] += Ke[:, 2 * k + 1]
    return K_full

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as _n\n_xs=_n.array([-1.5,-0.5,0.5,1.5])\nloc=[]\nfor _j in range(3):\n    for _i in range(3):\n        loc.append(_n.array([[_xs[_i],_xs[_j]],[_xs[_i+1],_xs[_j]],[_xs[_i+1],_xs[_j+1]],[_xs[_i],_xs[_j+1]]]))\nE=1.0;nu=0.3;p=2\ngknx=_n.array([-1.5,-1.5,-1.5,0.,1.5,1.5,1.5]);gkny=gknx.copy()', "call": 'coupling_matrix(gknx, gkny, p, loc, E, nu)', "gold_call": '_oracle_coupling_matrix(gknx, gkny, p, loc, E, nu)', "tol": 1e-08},
        {"setup": 'import numpy as _n\n_xs=_n.array([-1.5,-0.5,0.5,1.5])\nloc=[]\nfor _j in range(3):\n    for _i in range(3):\n        loc.append(_n.array([[_xs[_i],_xs[_j]],[_xs[_i+1],_xs[_j]],[_xs[_i+1],_xs[_j+1]],[_xs[_i],_xs[_j+1]]]))\nE=2.0;nu=0.25;p=2\ngknx=_n.array([-1.5,-1.5,-1.5,0.,1.5,1.5,1.5]);gkny=gknx.copy()', "call": 'coupling_matrix(gknx, gkny, p, loc, E, nu)', "gold_call": '_oracle_coupling_matrix(gknx, gkny, p, loc, E, nu)', "tol": 1e-08},
        {"setup": 'import numpy as _n\n_xs=_n.array([-1.5,-0.5,0.5,1.5])\nloc=[]\nfor _j in range(3):\n    for _i in range(3):\n        loc.append(_n.array([[_xs[_i],_xs[_j]],[_xs[_i+1],_xs[_j]],[_xs[_i+1],_xs[_j+1]],[_xs[_i],_xs[_j+1]]]))\nE=1.5;nu=0.35;p=2\ngknx=_n.array([-1.5,-1.5,-1.5,0.,1.5,1.5,1.5]);gkny=gknx.copy()', "call": 'coupling_matrix(gknx, gkny, p, loc, E, nu)', "gold_call": '_oracle_coupling_matrix(gknx, gkny, p, loc, E, nu)', "tol": 1e-08},
    ]
