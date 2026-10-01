"""
Assemble the residual and analytic sparse Jacobian of the coupled DDFV-HA Poisson and carrier-continuity equations, with Dirichlet nodes held fixed.

The discrete system imposes three balances on every primal node (triangle or boundary edge) and on every vertex. For the potential, the sum of outgoing electrostatic fluxes minus the cell area times the scaled space charge p - n + N must vanish; for each carrier the sum of outgoing carrier fluxes must vanish, since there is no recombination. A primal flux leaves node K and enters node L, and a dual flux leaves Ks and enters Ls, so it is added with a plus sign to the first node's balance and with a minus sign to the second. Boundary-edge nodes have zero area; on an insulating side their three balances reduce to zero normal flux through that edge, which is exactly the homogeneous Neumann condition of the DDFV scheme, and the side vertices keep their dual balances with nothing crossing the boundary. Nodes on the Ohmic contacts carry fixed values, so their rows are replaced by identity rows with zero residual. Newton's method needs the Jacobian: the electrostatic fluxes are linear in u, the carrier fluxes are linear in n and p with Bernoulli coefficients, and their u-derivatives use B'(t) = B(t)(1 - B(-t))/t, which tends to -1/2 as t tends to 0. An analytic Jacobian is essential, because finite-difference Jacobians lose accuracy when the densities span fourteen orders of magnitude.

Returns
-------
tuple (R, J): R float array (3*Nn,), J scipy.sparse matrix (3*Nn, 3*Nn).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def assemble_system(geom: dict, u: "np.ndarray", n: "np.ndarray", p: "np.ndarray",
                    Nd: "np.ndarray", dirichlet: "np.ndarray",
                    lam2: float, Dn: float, Dp: float) -> tuple:
    """Residual and Jacobian of the scaled DDFV-HA drift-diffusion system.

    Parameters
    ----------
    geom : dict
        Output of ddfv_geometry.
    u, n, p, Nd : "np.ndarray"
        Node values, shape (Nn,) with Nn = T + Eb + V (triangles, boundary
        edges, vertices): potential in thermal voltages, scaled carrier densities
        and scaled net doping.
    dirichlet : "np.ndarray"
        Bool, shape (Nn,), True for nodes whose u, n, p are held fixed.
    lam2, Dn, Dp : float
        As in ddfv_ha_fluxes.

    Returns
    -------
    R : "np.ndarray"
        Shape (3*Nn,), unknowns interleaved as index 3*node + (0: u, 1: n, 2: p).
        For each node: R_u = sum(signed electrostatic fluxes) - area*(p - n + Nd),
        R_n = sum(signed electron fluxes), R_p = sum(signed hole fluxes), where a
        diamond's primal fluxes count +1 at K and -1 at L, its dual fluxes +1 at
        Ks and -1 at Ls, and area is tri_area for triangles, 0 for boundary edges
        and dual_area for vertices. Entries of Dirichlet nodes are 0.
    J : scipy.sparse matrix
        Shape (3*Nn, 3*Nn), the exact Jacobian dR/d(u, n, p), with the rows of
        Dirichlet nodes replaced by identity rows.

    Notes
    -----
    Include every import your implementation needs inside the function body.
    """
    return R, J  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import scipy.sparse as sp


def _bernoulli_derivative(t: "np.ndarray") -> "np.ndarray":
    """Derivative B'(t) = B(t)(1 - B(-t))/t, with the removable limit -1/2 at t = 0."""
    t = np.asarray(t, dtype=float)
    out = np.empty_like(t)
    small = np.abs(t) < 1e-5
    out[small] = -0.5 + t[small] / 6.0
    tb = t[~small]
    out[~small] = _oracle_bernoulli(tb) * (1.0 - _oracle_bernoulli(-tb)) / tb
    return out


def _oracle_assemble_system(geom: dict, u: "np.ndarray", n: "np.ndarray", p: "np.ndarray",
                            Nd: "np.ndarray", dirichlet: "np.ndarray",
                            lam2: float, Dn: float, Dp: float) -> tuple:
    u, n, p, Nd = (np.asarray(v, dtype=float) for v in (u, n, p, Nd))
    dia = geom["diamonds"]
    nT, nE = len(geom["tri_area"]), len(geom["bnd_edges"])
    NP = nT + nE
    Nn = len(u)
    area = np.concatenate([geom["tri_area"], np.zeros(nE), geom["dual_area"]])
    K, L, A, B = dia[:, 0], dia[:, 1], NP + dia[:, 2], NP + dia[:, 3]
    al, be, ga = geom["alpha"], geom["beta"], geom["gamma"]
    fl = _oracle_ddfv_ha_fluxes(geom, u, n, p, lam2, Dn, Dp)
    R = np.zeros(3 * Nn)
    for k in range(3):
        fp, fd = fl[2 * k], fl[2 * k + 1]
        np.add.at(R, 3 * K + k, fp)
        np.add.at(R, 3 * L + k, -fp)
        np.add.at(R, 3 * A + k, fd)
        np.add.at(R, 3 * B + k, -fd)
    R[0::3] -= area * (p - n + Nd)

    a, b = u[K] - u[L], u[A] - u[B]
    Ba, Bma, Bb, Bmb = (_oracle_bernoulli(x) for x in (a, -a, b, -b))
    dBa, dBma, dBb, dBmb = (_bernoulli_derivative(x) for x in (a, -a, b, -b))
    one = np.ones_like(a)
    sa = dBa * n[K] + dBma * n[L]
    sb = dBb * n[A] + dBmb * n[B]
    ta = -dBma * p[K] - dBa * p[L]
    tb = -dBmb * p[A] - dBb * p[B]
    d_prim = {0: [(K, 0, one), (L, 0, -one)],
              1: [(K, 0, sa), (L, 0, -sa), (K, 1, Ba), (L, 1, -Bma)],
              2: [(K, 0, ta), (L, 0, -ta), (K, 2, Bma), (L, 2, -Ba)]}
    d_dual = {0: [(A, 0, one), (B, 0, -one)],
              1: [(A, 0, sb), (B, 0, -sb), (A, 1, Bb), (B, 1, -Bmb)],
              2: [(A, 0, tb), (B, 0, -tb), (A, 2, Bmb), (B, 2, -Bb)]}
    coef = {0: lam2, 1: Dn, 2: Dp}
    rows, cols, vals = [], [], []
    for k in range(3):
        for plus, minus, cp, cd in ((K, L, al, be), (A, B, be, ga)):
            for parts, c in ((d_prim[k], cp), (d_dual[k], cd)):
                for node, var, v in parts:
                    w = coef[k] * c * v
                    rows += [3 * plus + k, 3 * minus + k]
                    cols += [3 * node + var, 3 * node + var]
                    vals += [w, -w]
    node = np.arange(Nn)
    rows += [3 * node, 3 * node]
    cols += [3 * node + 1, 3 * node + 2]
    vals += [area, -area]
    J = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(3 * Nn, 3 * Nn)).tocsr()
    fixed = np.repeat(np.asarray(dirichlet, dtype=bool), 3)
    R[fixed] = 0.0
    J = (sp.diags((~fixed).astype(float)) @ J + sp.diags(fixed.astype(float))).tocsr()
    return R, J

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nimport scipy.sparse as sp\nP, T = _oracle_build_mesh(4)\ndef _fx_cross2d(u, v):\n    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]\n\n\ndef _fx_ddfv_geometry(points, triangles):\n    P = np.asarray(points, dtype=float)\n    T = np.asarray(triangles, dtype=int)\n    nT = len(T)\n    tri_area = 0.5 * np.abs(_fx_cross2d(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]))\n    tri_cent = P[T].mean(axis=1)\n    edge_tris = {}\n    for t, (a, b, c) in enumerate(T):\n        for u2, v2 in ((a, b), (b, c), (c, a)):\n            edge_tris.setdefault((min(u2, v2), max(u2, v2)), []).append(t)\n    keys = sorted(edge_tris)\n    bnd = [k for k in keys if len(edge_tris[k]) == 1]\n    bnd_index = {k: nT + m for m, k in enumerate(bnd)}\n    bnd_edges = np.array(bnd, dtype=int).reshape(-1, 2)\n    bnd_mid = P[bnd_edges].mean(axis=1)\n    primal_xy = np.vstack([tri_cent, bnd_mid])\n    dia = np.empty((len(keys), 4), dtype=int)\n    for m, k in enumerate(keys):\n        ts = edge_tris[k]\n        dia[m] = (ts[0], ts[1] if len(ts) == 2 else bnd_index[k], k[0], k[1])\n    xK, xL = primal_xy[dia[:, 0]], primal_xy[dia[:, 1]]\n    xA, xB = P[dia[:, 2]], P[dia[:, 3]]\n    s, ss = xB - xA, xL - xK\n    ls, lss = np.linalg.norm(s, axis=1), np.linalg.norm(ss, axis=1)\n    nKL = np.column_stack([s[:, 1], -s[:, 0]]) / ls[:, None]\n    nKL *= np.sign(np.sum(nKL * ss, axis=1))[:, None]\n    nAB = np.column_stack([ss[:, 1], -ss[:, 0]]) / lss[:, None]\n    nAB *= np.sign(np.sum(nAB * s, axis=1))[:, None]\n    c = np.sum(nKL * nAB, axis=1)\n    dA = 0.5 * np.abs(_fx_cross2d(ss, s))\n    dual_area = np.zeros(len(P))\n    np.add.at(dual_area, dia[:, 2], 0.5 * np.abs(_fx_cross2d(xL - xK, xA - xK)))\n    np.add.at(dual_area, dia[:, 3], 0.5 * np.abs(_fx_cross2d(xL - xK, xB - xK)))\n    return {"tri_area": tri_area, "tri_centroid": tri_cent,\n            "bnd_edges": bnd_edges, "bnd_mid": bnd_mid,\n            "dual_area": dual_area, "diamonds": dia,\n            "alpha": ls ** 2 / (2 * dA), "beta": ls * lss * c / (2 * dA),\n            "gamma": lss ** 2 / (2 * dA), "diamond_area": dA}\ng = _fx_ddfv_geometry(P, T)\nNn = len(T) + len(g[\'bnd_edges\']) + len(P)\nrng = np.random.default_rng(3)\nu = rng.normal(0, 2, Nn)\nn = np.exp(rng.normal(0, 1, Nn))\np = np.exp(rng.normal(0, 1, Nn))\nNd = rng.normal(0, 1, Nn)\ndirichlet = np.zeros(Nn, bool)\ndirichlet[-5:] = True\n', 'call': 'assemble_system(g, u, n, p, Nd, dirichlet, 1.6715e-3, 36.632284, 12.163366)[0]', 'gold_call': '_oracle_assemble_system(g, u, n, p, Nd, dirichlet, 1.6715e-3, 36.632284, 12.163366)[0]'},
        {'setup': 'import numpy as np\nimport scipy.sparse as sp\nP, T = _oracle_build_mesh(4)\ndef _fx_cross2d(u, v):\n    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]\n\n\ndef _fx_ddfv_geometry(points, triangles):\n    P = np.asarray(points, dtype=float)\n    T = np.asarray(triangles, dtype=int)\n    nT = len(T)\n    tri_area = 0.5 * np.abs(_fx_cross2d(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]))\n    tri_cent = P[T].mean(axis=1)\n    edge_tris = {}\n    for t, (a, b, c) in enumerate(T):\n        for u2, v2 in ((a, b), (b, c), (c, a)):\n            edge_tris.setdefault((min(u2, v2), max(u2, v2)), []).append(t)\n    keys = sorted(edge_tris)\n    bnd = [k for k in keys if len(edge_tris[k]) == 1]\n    bnd_index = {k: nT + m for m, k in enumerate(bnd)}\n    bnd_edges = np.array(bnd, dtype=int).reshape(-1, 2)\n    bnd_mid = P[bnd_edges].mean(axis=1)\n    primal_xy = np.vstack([tri_cent, bnd_mid])\n    dia = np.empty((len(keys), 4), dtype=int)\n    for m, k in enumerate(keys):\n        ts = edge_tris[k]\n        dia[m] = (ts[0], ts[1] if len(ts) == 2 else bnd_index[k], k[0], k[1])\n    xK, xL = primal_xy[dia[:, 0]], primal_xy[dia[:, 1]]\n    xA, xB = P[dia[:, 2]], P[dia[:, 3]]\n    s, ss = xB - xA, xL - xK\n    ls, lss = np.linalg.norm(s, axis=1), np.linalg.norm(ss, axis=1)\n    nKL = np.column_stack([s[:, 1], -s[:, 0]]) / ls[:, None]\n    nKL *= np.sign(np.sum(nKL * ss, axis=1))[:, None]\n    nAB = np.column_stack([ss[:, 1], -ss[:, 0]]) / lss[:, None]\n    nAB *= np.sign(np.sum(nAB * s, axis=1))[:, None]\n    c = np.sum(nKL * nAB, axis=1)\n    dA = 0.5 * np.abs(_fx_cross2d(ss, s))\n    dual_area = np.zeros(len(P))\n    np.add.at(dual_area, dia[:, 2], 0.5 * np.abs(_fx_cross2d(xL - xK, xA - xK)))\n    np.add.at(dual_area, dia[:, 3], 0.5 * np.abs(_fx_cross2d(xL - xK, xB - xK)))\n    return {"tri_area": tri_area, "tri_centroid": tri_cent,\n            "bnd_edges": bnd_edges, "bnd_mid": bnd_mid,\n            "dual_area": dual_area, "diamonds": dia,\n            "alpha": ls ** 2 / (2 * dA), "beta": ls * lss * c / (2 * dA),\n            "gamma": lss ** 2 / (2 * dA), "diamond_area": dA}\ng = _fx_ddfv_geometry(P, T)\nNn = len(T) + len(g[\'bnd_edges\']) + len(P)\nrng = np.random.default_rng(3)\nu = rng.normal(0, 2, Nn)\nn = np.exp(rng.normal(0, 1, Nn))\np = np.exp(rng.normal(0, 1, Nn))\nNd = rng.normal(0, 1, Nn)\ndirichlet = np.zeros(Nn, bool)\ndirichlet[-5:] = True\n', 'call': 'sp.csr_matrix(assemble_system(g, u, n, p, Nd, dirichlet, 1.6715e-3, 36.632284, 12.163366)[1]).toarray()', 'gold_call': '_oracle_assemble_system(g, u, n, p, Nd, dirichlet, 1.6715e-3, 36.632284, 12.163366)[1].toarray()'},
        {'setup': 'import numpy as np\nimport scipy.sparse as sp\nP, T = _oracle_build_mesh(4)\ndef _fx_cross2d(u, v):\n    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]\n\n\ndef _fx_ddfv_geometry(points, triangles):\n    P = np.asarray(points, dtype=float)\n    T = np.asarray(triangles, dtype=int)\n    nT = len(T)\n    tri_area = 0.5 * np.abs(_fx_cross2d(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]))\n    tri_cent = P[T].mean(axis=1)\n    edge_tris = {}\n    for t, (a, b, c) in enumerate(T):\n        for u2, v2 in ((a, b), (b, c), (c, a)):\n            edge_tris.setdefault((min(u2, v2), max(u2, v2)), []).append(t)\n    keys = sorted(edge_tris)\n    bnd = [k for k in keys if len(edge_tris[k]) == 1]\n    bnd_index = {k: nT + m for m, k in enumerate(bnd)}\n    bnd_edges = np.array(bnd, dtype=int).reshape(-1, 2)\n    bnd_mid = P[bnd_edges].mean(axis=1)\n    primal_xy = np.vstack([tri_cent, bnd_mid])\n    dia = np.empty((len(keys), 4), dtype=int)\n    for m, k in enumerate(keys):\n        ts = edge_tris[k]\n        dia[m] = (ts[0], ts[1] if len(ts) == 2 else bnd_index[k], k[0], k[1])\n    xK, xL = primal_xy[dia[:, 0]], primal_xy[dia[:, 1]]\n    xA, xB = P[dia[:, 2]], P[dia[:, 3]]\n    s, ss = xB - xA, xL - xK\n    ls, lss = np.linalg.norm(s, axis=1), np.linalg.norm(ss, axis=1)\n    nKL = np.column_stack([s[:, 1], -s[:, 0]]) / ls[:, None]\n    nKL *= np.sign(np.sum(nKL * ss, axis=1))[:, None]\n    nAB = np.column_stack([ss[:, 1], -ss[:, 0]]) / lss[:, None]\n    nAB *= np.sign(np.sum(nAB * s, axis=1))[:, None]\n    c = np.sum(nKL * nAB, axis=1)\n    dA = 0.5 * np.abs(_fx_cross2d(ss, s))\n    dual_area = np.zeros(len(P))\n    np.add.at(dual_area, dia[:, 2], 0.5 * np.abs(_fx_cross2d(xL - xK, xA - xK)))\n    np.add.at(dual_area, dia[:, 3], 0.5 * np.abs(_fx_cross2d(xL - xK, xB - xK)))\n    return {"tri_area": tri_area, "tri_centroid": tri_cent,\n            "bnd_edges": bnd_edges, "bnd_mid": bnd_mid,\n            "dual_area": dual_area, "diamonds": dia,\n            "alpha": ls ** 2 / (2 * dA), "beta": ls * lss * c / (2 * dA),\n            "gamma": lss ** 2 / (2 * dA), "diamond_area": dA}\ng = _fx_ddfv_geometry(P, T)\nNn = len(T) + len(g[\'bnd_edges\']) + len(P)\nrng = np.random.default_rng(3)\nu = rng.normal(0, 2, Nn)\nn = np.exp(rng.normal(0, 1, Nn))\np = np.exp(rng.normal(0, 1, Nn))\nNd = rng.normal(0, 1, Nn)\ndirichlet = np.zeros(Nn, bool)\ndirichlet[-5:] = True\nu = 1e-9 * rng.normal(size=Nn)\n', 'call': 'sp.csr_matrix(assemble_system(g, u, n, p, Nd, dirichlet, 1.0, 1.0, 1.0)[1]).toarray()', 'gold_call': '_oracle_assemble_system(g, u, n, p, Nd, dirichlet, 1.0, 1.0, 1.0)[1].toarray()'},
        {'setup': 'import numpy as np\nimport scipy.sparse as sp\nP, T = _oracle_build_mesh(4)\ndef _fx_cross2d(u, v):\n    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]\n\n\ndef _fx_ddfv_geometry(points, triangles):\n    P = np.asarray(points, dtype=float)\n    T = np.asarray(triangles, dtype=int)\n    nT = len(T)\n    tri_area = 0.5 * np.abs(_fx_cross2d(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]))\n    tri_cent = P[T].mean(axis=1)\n    edge_tris = {}\n    for t, (a, b, c) in enumerate(T):\n        for u2, v2 in ((a, b), (b, c), (c, a)):\n            edge_tris.setdefault((min(u2, v2), max(u2, v2)), []).append(t)\n    keys = sorted(edge_tris)\n    bnd = [k for k in keys if len(edge_tris[k]) == 1]\n    bnd_index = {k: nT + m for m, k in enumerate(bnd)}\n    bnd_edges = np.array(bnd, dtype=int).reshape(-1, 2)\n    bnd_mid = P[bnd_edges].mean(axis=1)\n    primal_xy = np.vstack([tri_cent, bnd_mid])\n    dia = np.empty((len(keys), 4), dtype=int)\n    for m, k in enumerate(keys):\n        ts = edge_tris[k]\n        dia[m] = (ts[0], ts[1] if len(ts) == 2 else bnd_index[k], k[0], k[1])\n    xK, xL = primal_xy[dia[:, 0]], primal_xy[dia[:, 1]]\n    xA, xB = P[dia[:, 2]], P[dia[:, 3]]\n    s, ss = xB - xA, xL - xK\n    ls, lss = np.linalg.norm(s, axis=1), np.linalg.norm(ss, axis=1)\n    nKL = np.column_stack([s[:, 1], -s[:, 0]]) / ls[:, None]\n    nKL *= np.sign(np.sum(nKL * ss, axis=1))[:, None]\n    nAB = np.column_stack([ss[:, 1], -ss[:, 0]]) / lss[:, None]\n    nAB *= np.sign(np.sum(nAB * s, axis=1))[:, None]\n    c = np.sum(nKL * nAB, axis=1)\n    dA = 0.5 * np.abs(_fx_cross2d(ss, s))\n    dual_area = np.zeros(len(P))\n    np.add.at(dual_area, dia[:, 2], 0.5 * np.abs(_fx_cross2d(xL - xK, xA - xK)))\n    np.add.at(dual_area, dia[:, 3], 0.5 * np.abs(_fx_cross2d(xL - xK, xB - xK)))\n    return {"tri_area": tri_area, "tri_centroid": tri_cent,\n            "bnd_edges": bnd_edges, "bnd_mid": bnd_mid,\n            "dual_area": dual_area, "diamonds": dia,\n            "alpha": ls ** 2 / (2 * dA), "beta": ls * lss * c / (2 * dA),\n            "gamma": lss ** 2 / (2 * dA), "diamond_area": dA}\ng = _fx_ddfv_geometry(P, T)\nNn = len(T) + len(g[\'bnd_edges\']) + len(P)\nrng = np.random.default_rng(3)\nu = rng.normal(0, 2, Nn)\nn = np.exp(rng.normal(0, 1, Nn))\np = np.exp(rng.normal(0, 1, Nn))\nNd = rng.normal(0, 1, Nn)\ndirichlet = np.zeros(Nn, bool)\ndirichlet[-5:] = True\ndirichlet[:] = True\n', 'call': 'np.concatenate([assemble_system(g, u, n, p, Nd, dirichlet, 1.0, 1.0, 1.0)[0], sp.csr_matrix(assemble_system(g, u, n, p, Nd, dirichlet, 1.0, 1.0, 1.0)[1]).diagonal()])', 'gold_call': 'np.concatenate([_oracle_assemble_system(g, u, n, p, Nd, dirichlet, 1.0, 1.0, 1.0)[0], _oracle_assemble_system(g, u, n, p, Nd, dirichlet, 1.0, 1.0, 1.0)[1].diagonal()])'},
    ]
