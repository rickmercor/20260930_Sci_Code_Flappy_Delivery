"""
Compute the DDFV-HA electrostatic, electron and hole fluxes across the primal and dual edge of every diamond.

Written with the Slotboom variable Phi_n = n exp(-psi) (psi in thermal voltages), the stationary electron equation becomes a pure diffusion problem, -div(D_n exp(psi) grad Phi_n) = 0, with a coefficient that varies by many orders of magnitude across a junction. The DDFV-HA scheme freezes that coefficient on each edge as the harmonic average of exp(psi) along a linear potential profile, which is exactly what produces the Scharfetter-Gummel form with Bernoulli weights. Pairing each exponential factor with the density difference along the same edge direction gives, per diamond, a primal difference s_K = B(a) n_K - B(-a) n_L with a = u_K - u_L, and a dual difference s_Ks = B(b) n_Ks - B(-b) n_Ls with b = u_Ks - u_Ls. The electron fluxes are then F_KL = D_n (alpha s_K + beta s_Ks) across the primal edge and F_KsLs = D_n (beta s_K + gamma s_Ks) across the dual edge. Holes drift the other way, so their differences use the reversed potential drop: t_K = B(-a) p_K - B(a) p_L and t_Ks = B(-b) p_Ks - B(b) p_Ls, with G = D_p (same combinations). The electrostatic fluxes use the same coefficients with plain differences, lam2 (alpha a + beta b) and lam2 (beta a + gamma b). Two properties follow directly: each flux is antisymmetric under swapping its two nodes, which gives exact local conservation, and each bracket vanishes when n = C exp(u) and p = C' exp(-u), because B(-t) = exp(t) B(t), so thermal equilibrium carries exactly zero current. When beta = 0 the scheme reduces to two ordinary Scharfetter-Gummel schemes.

Returns
-------
np.ndarray of shape (6, ND), float: electrostatic, electron and hole fluxes, primal then dual for each.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ddfv_ha_fluxes(geom: dict, u: "np.ndarray", n: "np.ndarray", p: "np.ndarray",
                   lam2: float, Dn: float, Dp: float) -> "np.ndarray":
    """DDFV-HA fluxes on every diamond.

    Parameters
    ----------
    geom : dict
        Output of ddfv_geometry.
    u, n, p : "np.ndarray"
        Node values of shape (T + Eb + V,): primal nodes (triangles, then
        boundary edges) followed by vertices. u is the potential in thermal
        voltages; n and p are scaled carrier densities.
    lam2 : float
        Scaled permittivity multiplying the electrostatic fluxes.
    Dn, Dp : float
        Electron and hole diffusion coefficients.

    Returns
    -------
    fluxes : "np.ndarray"
        Shape (6, ND), rows: electrostatic primal, electrostatic dual, electron
        primal, electron dual, hole primal, hole dual. Primal fluxes point from
        K to L, dual fluxes from Ks to Ls. With a = u_K - u_L, b = u_Ks - u_Ls,
        B the Bernoulli function and (alpha, beta, gamma) from geom:
          s_K = B(a) n_K - B(-a) n_L,  s_Ks = B(b) n_Ks - B(-b) n_Ls,
          t_K = B(-a) p_K - B(a) p_L,  t_Ks = B(-b) p_Ks - B(b) p_Ls,
          rows = [lam2 (alpha a + beta b), lam2 (beta a + gamma b),
                  Dn (alpha s_K + beta s_Ks), Dn (beta s_K + gamma s_Ks),
                  Dp (alpha t_K + beta t_Ks), Dp (beta t_K + gamma t_Ks)].

    Notes
    -----
    Include every import your implementation needs inside the function body.
    """
    return fluxes  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_ddfv_ha_fluxes(geom: dict, u: "np.ndarray", n: "np.ndarray", p: "np.ndarray",
                           lam2: float, Dn: float, Dp: float) -> "np.ndarray":
    dia = geom["diamonds"]
    NP = len(geom["tri_area"]) + len(geom["bnd_edges"])
    K, L, A, B = dia[:, 0], dia[:, 1], NP + dia[:, 2], NP + dia[:, 3]
    al, be, ga = geom["alpha"], geom["beta"], geom["gamma"]
    u, n, p = (np.asarray(v, dtype=float) for v in (u, n, p))
    a, b = u[K] - u[L], u[A] - u[B]
    Ba, Bma = _oracle_bernoulli(a), _oracle_bernoulli(-a)
    Bb, Bmb = _oracle_bernoulli(b), _oracle_bernoulli(-b)
    sK = Ba * n[K] - Bma * n[L]
    sA = Bb * n[A] - Bmb * n[B]
    tK = Bma * p[K] - Ba * p[L]
    tA = Bmb * p[A] - Bb * p[B]
    return np.stack([lam2 * (al * a + be * b), lam2 * (be * a + ga * b),
                     Dn * (al * sK + be * sA), Dn * (be * sK + ga * sA),
                     Dp * (al * tK + be * tA), Dp * (be * tK + ga * tA)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nP, T = _oracle_build_mesh(4)\ndef _fx_cross2d(u, v):\n    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]\n\n\ndef _fx_ddfv_geometry(points, triangles):\n    P = np.asarray(points, dtype=float)\n    T = np.asarray(triangles, dtype=int)\n    nT = len(T)\n    tri_area = 0.5 * np.abs(_fx_cross2d(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]))\n    tri_cent = P[T].mean(axis=1)\n    edge_tris = {}\n    for t, (a, b, c) in enumerate(T):\n        for u2, v2 in ((a, b), (b, c), (c, a)):\n            edge_tris.setdefault((min(u2, v2), max(u2, v2)), []).append(t)\n    keys = sorted(edge_tris)\n    bnd = [k for k in keys if len(edge_tris[k]) == 1]\n    bnd_index = {k: nT + m for m, k in enumerate(bnd)}\n    bnd_edges = np.array(bnd, dtype=int).reshape(-1, 2)\n    bnd_mid = P[bnd_edges].mean(axis=1)\n    primal_xy = np.vstack([tri_cent, bnd_mid])\n    dia = np.empty((len(keys), 4), dtype=int)\n    for m, k in enumerate(keys):\n        ts = edge_tris[k]\n        dia[m] = (ts[0], ts[1] if len(ts) == 2 else bnd_index[k], k[0], k[1])\n    xK, xL = primal_xy[dia[:, 0]], primal_xy[dia[:, 1]]\n    xA, xB = P[dia[:, 2]], P[dia[:, 3]]\n    s, ss = xB - xA, xL - xK\n    ls, lss = np.linalg.norm(s, axis=1), np.linalg.norm(ss, axis=1)\n    nKL = np.column_stack([s[:, 1], -s[:, 0]]) / ls[:, None]\n    nKL *= np.sign(np.sum(nKL * ss, axis=1))[:, None]\n    nAB = np.column_stack([ss[:, 1], -ss[:, 0]]) / lss[:, None]\n    nAB *= np.sign(np.sum(nAB * s, axis=1))[:, None]\n    c = np.sum(nKL * nAB, axis=1)\n    dA = 0.5 * np.abs(_fx_cross2d(ss, s))\n    dual_area = np.zeros(len(P))\n    np.add.at(dual_area, dia[:, 2], 0.5 * np.abs(_fx_cross2d(xL - xK, xA - xK)))\n    np.add.at(dual_area, dia[:, 3], 0.5 * np.abs(_fx_cross2d(xL - xK, xB - xK)))\n    return {"tri_area": tri_area, "tri_centroid": tri_cent,\n            "bnd_edges": bnd_edges, "bnd_mid": bnd_mid,\n            "dual_area": dual_area, "diamonds": dia,\n            "alpha": ls ** 2 / (2 * dA), "beta": ls * lss * c / (2 * dA),\n            "gamma": lss ** 2 / (2 * dA), "diamond_area": dA}\ng = _fx_ddfv_geometry(P, T)\nNn = len(T) + len(g[\'bnd_edges\']) + len(P)\nrng = np.random.default_rng(1)\nu = rng.normal(0, 3, Nn)\nn = np.exp(rng.normal(0, 2, Nn))\np = np.exp(rng.normal(0, 2, Nn))\n', 'call': 'ddfv_ha_fluxes(g, u, n, p, 1.6715e-3, 36.632284, 12.163366)', 'gold_call': '_oracle_ddfv_ha_fluxes(g, u, n, p, 1.6715e-3, 36.632284, 12.163366)'},
        {'setup': 'import numpy as np\nP, T = _oracle_build_mesh(8)\ndef _fx_cross2d(u, v):\n    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]\n\n\ndef _fx_ddfv_geometry(points, triangles):\n    P = np.asarray(points, dtype=float)\n    T = np.asarray(triangles, dtype=int)\n    nT = len(T)\n    tri_area = 0.5 * np.abs(_fx_cross2d(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]))\n    tri_cent = P[T].mean(axis=1)\n    edge_tris = {}\n    for t, (a, b, c) in enumerate(T):\n        for u2, v2 in ((a, b), (b, c), (c, a)):\n            edge_tris.setdefault((min(u2, v2), max(u2, v2)), []).append(t)\n    keys = sorted(edge_tris)\n    bnd = [k for k in keys if len(edge_tris[k]) == 1]\n    bnd_index = {k: nT + m for m, k in enumerate(bnd)}\n    bnd_edges = np.array(bnd, dtype=int).reshape(-1, 2)\n    bnd_mid = P[bnd_edges].mean(axis=1)\n    primal_xy = np.vstack([tri_cent, bnd_mid])\n    dia = np.empty((len(keys), 4), dtype=int)\n    for m, k in enumerate(keys):\n        ts = edge_tris[k]\n        dia[m] = (ts[0], ts[1] if len(ts) == 2 else bnd_index[k], k[0], k[1])\n    xK, xL = primal_xy[dia[:, 0]], primal_xy[dia[:, 1]]\n    xA, xB = P[dia[:, 2]], P[dia[:, 3]]\n    s, ss = xB - xA, xL - xK\n    ls, lss = np.linalg.norm(s, axis=1), np.linalg.norm(ss, axis=1)\n    nKL = np.column_stack([s[:, 1], -s[:, 0]]) / ls[:, None]\n    nKL *= np.sign(np.sum(nKL * ss, axis=1))[:, None]\n    nAB = np.column_stack([ss[:, 1], -ss[:, 0]]) / lss[:, None]\n    nAB *= np.sign(np.sum(nAB * s, axis=1))[:, None]\n    c = np.sum(nKL * nAB, axis=1)\n    dA = 0.5 * np.abs(_fx_cross2d(ss, s))\n    dual_area = np.zeros(len(P))\n    np.add.at(dual_area, dia[:, 2], 0.5 * np.abs(_fx_cross2d(xL - xK, xA - xK)))\n    np.add.at(dual_area, dia[:, 3], 0.5 * np.abs(_fx_cross2d(xL - xK, xB - xK)))\n    return {"tri_area": tri_area, "tri_centroid": tri_cent,\n            "bnd_edges": bnd_edges, "bnd_mid": bnd_mid,\n            "dual_area": dual_area, "diamonds": dia,\n            "alpha": ls ** 2 / (2 * dA), "beta": ls * lss * c / (2 * dA),\n            "gamma": lss ** 2 / (2 * dA), "diamond_area": dA}\ng = _fx_ddfv_geometry(P, T)\nxy = np.vstack([g[\'tri_centroid\'], g[\'bnd_mid\'], P])\nu = 30.0 * np.tanh(8.0 * (xy[:, 1] - 0.5)) + np.sin(5 * xy[:, 0])\nn = 1e-3 * np.exp(u)\np = 2e-3 * np.exp(-u)\n', 'call': 'bool(np.max(np.abs(ddfv_ha_fluxes(g, u, n, p, 1.0, 1.0, 1.0)[2:])) < 1e-10 * max(n.max(), p.max()))', 'gold_call': 'bool(np.max(np.abs(_oracle_ddfv_ha_fluxes(g, u, n, p, 1.0, 1.0, 1.0)[2:])) < 1e-10 * max(n.max(), p.max()))'},
        {'setup': 'import numpy as np\nP, T = _oracle_build_mesh(4)\ndef _fx_cross2d(u, v):\n    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]\n\n\ndef _fx_ddfv_geometry(points, triangles):\n    P = np.asarray(points, dtype=float)\n    T = np.asarray(triangles, dtype=int)\n    nT = len(T)\n    tri_area = 0.5 * np.abs(_fx_cross2d(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]))\n    tri_cent = P[T].mean(axis=1)\n    edge_tris = {}\n    for t, (a, b, c) in enumerate(T):\n        for u2, v2 in ((a, b), (b, c), (c, a)):\n            edge_tris.setdefault((min(u2, v2), max(u2, v2)), []).append(t)\n    keys = sorted(edge_tris)\n    bnd = [k for k in keys if len(edge_tris[k]) == 1]\n    bnd_index = {k: nT + m for m, k in enumerate(bnd)}\n    bnd_edges = np.array(bnd, dtype=int).reshape(-1, 2)\n    bnd_mid = P[bnd_edges].mean(axis=1)\n    primal_xy = np.vstack([tri_cent, bnd_mid])\n    dia = np.empty((len(keys), 4), dtype=int)\n    for m, k in enumerate(keys):\n        ts = edge_tris[k]\n        dia[m] = (ts[0], ts[1] if len(ts) == 2 else bnd_index[k], k[0], k[1])\n    xK, xL = primal_xy[dia[:, 0]], primal_xy[dia[:, 1]]\n    xA, xB = P[dia[:, 2]], P[dia[:, 3]]\n    s, ss = xB - xA, xL - xK\n    ls, lss = np.linalg.norm(s, axis=1), np.linalg.norm(ss, axis=1)\n    nKL = np.column_stack([s[:, 1], -s[:, 0]]) / ls[:, None]\n    nKL *= np.sign(np.sum(nKL * ss, axis=1))[:, None]\n    nAB = np.column_stack([ss[:, 1], -ss[:, 0]]) / lss[:, None]\n    nAB *= np.sign(np.sum(nAB * s, axis=1))[:, None]\n    c = np.sum(nKL * nAB, axis=1)\n    dA = 0.5 * np.abs(_fx_cross2d(ss, s))\n    dual_area = np.zeros(len(P))\n    np.add.at(dual_area, dia[:, 2], 0.5 * np.abs(_fx_cross2d(xL - xK, xA - xK)))\n    np.add.at(dual_area, dia[:, 3], 0.5 * np.abs(_fx_cross2d(xL - xK, xB - xK)))\n    return {"tri_area": tri_area, "tri_centroid": tri_cent,\n            "bnd_edges": bnd_edges, "bnd_mid": bnd_mid,\n            "dual_area": dual_area, "diamonds": dia,\n            "alpha": ls ** 2 / (2 * dA), "beta": ls * lss * c / (2 * dA),\n            "gamma": lss ** 2 / (2 * dA), "diamond_area": dA}\ng = _fx_ddfv_geometry(P, T)\nNn = len(T) + len(g[\'bnd_edges\']) + len(P)\nrng = np.random.default_rng(7)\nu = rng.normal(0, 150, Nn)\nn = np.full(Nn, 1.0)\np = np.full(Nn, 1.0)\n', 'call': 'ddfv_ha_fluxes(g, u, n, p, 1.0, 2.0, 0.5)', 'gold_call': '_oracle_ddfv_ha_fluxes(g, u, n, p, 1.0, 2.0, 0.5)'},
        {'setup': 'import numpy as np\nP, T = _oracle_build_mesh(8)\ndef _fx_cross2d(u, v):\n    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]\n\n\ndef _fx_ddfv_geometry(points, triangles):\n    P = np.asarray(points, dtype=float)\n    T = np.asarray(triangles, dtype=int)\n    nT = len(T)\n    tri_area = 0.5 * np.abs(_fx_cross2d(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]))\n    tri_cent = P[T].mean(axis=1)\n    edge_tris = {}\n    for t, (a, b, c) in enumerate(T):\n        for u2, v2 in ((a, b), (b, c), (c, a)):\n            edge_tris.setdefault((min(u2, v2), max(u2, v2)), []).append(t)\n    keys = sorted(edge_tris)\n    bnd = [k for k in keys if len(edge_tris[k]) == 1]\n    bnd_index = {k: nT + m for m, k in enumerate(bnd)}\n    bnd_edges = np.array(bnd, dtype=int).reshape(-1, 2)\n    bnd_mid = P[bnd_edges].mean(axis=1)\n    primal_xy = np.vstack([tri_cent, bnd_mid])\n    dia = np.empty((len(keys), 4), dtype=int)\n    for m, k in enumerate(keys):\n        ts = edge_tris[k]\n        dia[m] = (ts[0], ts[1] if len(ts) == 2 else bnd_index[k], k[0], k[1])\n    xK, xL = primal_xy[dia[:, 0]], primal_xy[dia[:, 1]]\n    xA, xB = P[dia[:, 2]], P[dia[:, 3]]\n    s, ss = xB - xA, xL - xK\n    ls, lss = np.linalg.norm(s, axis=1), np.linalg.norm(ss, axis=1)\n    nKL = np.column_stack([s[:, 1], -s[:, 0]]) / ls[:, None]\n    nKL *= np.sign(np.sum(nKL * ss, axis=1))[:, None]\n    nAB = np.column_stack([ss[:, 1], -ss[:, 0]]) / lss[:, None]\n    nAB *= np.sign(np.sum(nAB * s, axis=1))[:, None]\n    c = np.sum(nKL * nAB, axis=1)\n    dA = 0.5 * np.abs(_fx_cross2d(ss, s))\n    dual_area = np.zeros(len(P))\n    np.add.at(dual_area, dia[:, 2], 0.5 * np.abs(_fx_cross2d(xL - xK, xA - xK)))\n    np.add.at(dual_area, dia[:, 3], 0.5 * np.abs(_fx_cross2d(xL - xK, xB - xK)))\n    return {"tri_area": tri_area, "tri_centroid": tri_cent,\n            "bnd_edges": bnd_edges, "bnd_mid": bnd_mid,\n            "dual_area": dual_area, "diamonds": dia,\n            "alpha": ls ** 2 / (2 * dA), "beta": ls * lss * c / (2 * dA),\n            "gamma": lss ** 2 / (2 * dA), "diamond_area": dA}\ng = _fx_ddfv_geometry(P, T)\nxy = np.vstack([g[\'tri_centroid\'], g[\'bnd_mid\'], P])\nu = np.zeros(len(xy))\nn = 1.0 + xy[:, 0] + 2.0 * xy[:, 1]\np = 3.0 - xy[:, 1]\n', 'call': 'ddfv_ha_fluxes(g, u, n, p, 1.0, 1.0, 1.0)', 'gold_call': '_oracle_ddfv_ha_fluxes(g, u, n, p, 1.0, 1.0, 1.0)'},
    ]
