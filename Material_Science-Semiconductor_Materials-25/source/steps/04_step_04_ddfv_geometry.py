"""
Build the discrete duality finite volume (DDFV) primal, dual and diamond meshes from a triangulation, with the geometric coefficients of every diamond.

DDFV places unknowns on two interlocking meshes. The primal unknowns sit at the barycentres of the triangles, and every boundary edge is treated as a degenerate primal cell whose unknown sits at the edge midpoint. The dual unknowns sit at the vertices, and the dual cell of a vertex is the polygon through the barycentres of the surrounding triangles (plus the adjacent boundary-edge midpoints and the vertex itself on the boundary). Each primal edge sigma = [x_K*, x_L*] shared by primal cells K and L defines a diamond D with corners x_K, x_K*, x_L, x_L*, whose dual edge sigma* is the segment [x_K, x_L]. On a diamond the gradient is reconstructed from both differences, grad_D u = (|sigma|(u_L - u_K) n_KL + |sigma*|(u_L* - u_K*) n_K*L*)/(2|D|), where n_KL is the unit normal to sigma pointing from K to L, n_K*L* is the unit normal to sigma* pointing from K* to L*, and |D| = |sigma||sigma*| sin(theta_D)/2 is half the magnitude of the cross product of the two diagonals. Every flux the scheme needs reduces to three diamond coefficients: alpha = |sigma|^2/(2|D|), beta = |sigma||sigma*|(n_KL . n_K*L*)/(2|D|) and gamma = |sigma*|^2/(2|D|). On a mesh whose primal and dual edges are orthogonal beta vanishes and the primal and dual problems decouple; with barycentres, beta is nonzero even on an undistorted right-triangle mesh, and on the distorted mesh it carries the whole effect of the distortion. The dual-cell area of a vertex is the sum, over its diamonds, of the triangles (x_K, x_L, vertex).

Returns
-------
dict of numpy arrays: tri_area, tri_centroid, bnd_edges, bnd_mid, dual_area, diamonds, alpha, beta, gamma, diamond_area.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ddfv_geometry(points: "np.ndarray", triangles: "np.ndarray") -> dict:
    """DDFV mesh data for a 2D triangulation.

    Parameters
    ----------
    points : "np.ndarray"
        Shape (V, 2), vertex coordinates.
    triangles : "np.ndarray"
        Shape (T, 3), int vertex indices.

    Returns
    -------
    geom : dict with these numpy arrays
        "tri_area"     (T,)   triangle areas.
        "tri_centroid" (T, 2) triangle barycentres.
        "bnd_edges"    (Eb, 2) int, boundary edges (edges belonging to one
                       triangle) as (vmin, vmax), sorted lexicographically.
        "bnd_mid"      (Eb, 2) midpoints of those edges.
        "dual_area"    (V,)   dual-cell area of each vertex.
        "diamonds"     (ND, 4) int, one row [K, L, Ks, Ls] per edge, rows in
                       lexicographic order of (vmin, vmax) over all edges.
                       Ks = vmin, Ls = vmax. K is the lowest-index triangle
                       containing the edge. L is the other triangle, or, for a
                       boundary edge, T + m where m is its row in "bnd_edges".
                       Primal node indices run over triangles 0..T-1, then
                       boundary edges T..T+Eb-1.
        "alpha", "beta", "gamma", "diamond_area"  (ND,) each, with
                       alpha = |s|^2/(2|D|), beta = |s||s*| (n_KL . n_KsLs)/(2|D|),
                       gamma = |s*|^2/(2|D|), where s = x_Ls - x_Ks, s* = x_L - x_K,
                       n_KL is the unit normal to s with n_KL . s* > 0, n_KsLs is
                       the unit normal to s* with n_KsLs . s > 0, and
                       |D| = |s* x s| / 2 (the 2D cross product of the diagonals).

    Notes
    -----
    Include every import your implementation needs inside the function body.
    """
    return geom  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _cross2d(u: "np.ndarray", v: "np.ndarray") -> "np.ndarray":
    """Scalar cross product of stacks of 2D vectors."""
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def _oracle_ddfv_geometry(points: "np.ndarray", triangles: "np.ndarray") -> dict:
    P = np.asarray(points, dtype=float)
    T = np.asarray(triangles, dtype=int)
    nT = len(T)

    tri_area = 0.5 * np.abs(_cross2d(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]))
    tri_cent = P[T].mean(axis=1)
    edge_tris = {}
    for t, (a, b, c) in enumerate(T):
        for u, v in ((a, b), (b, c), (c, a)):
            edge_tris.setdefault((min(u, v), max(u, v)), []).append(t)
    keys = sorted(edge_tris)
    bnd = [k for k in keys if len(edge_tris[k]) == 1]
    bnd_index = {k: nT + m for m, k in enumerate(bnd)}
    bnd_edges = np.array(bnd, dtype=int).reshape(-1, 2)
    bnd_mid = P[bnd_edges].mean(axis=1)
    primal_xy = np.vstack([tri_cent, bnd_mid])
    dia = np.empty((len(keys), 4), dtype=int)
    for m, k in enumerate(keys):
        ts = edge_tris[k]
        dia[m] = (ts[0], ts[1] if len(ts) == 2 else bnd_index[k], k[0], k[1])
    xK, xL = primal_xy[dia[:, 0]], primal_xy[dia[:, 1]]
    xA, xB = P[dia[:, 2]], P[dia[:, 3]]
    s, ss = xB - xA, xL - xK
    ls, lss = np.linalg.norm(s, axis=1), np.linalg.norm(ss, axis=1)
    nKL = np.column_stack([s[:, 1], -s[:, 0]]) / ls[:, None]
    nKL *= np.sign(np.sum(nKL * ss, axis=1))[:, None]
    nAB = np.column_stack([ss[:, 1], -ss[:, 0]]) / lss[:, None]
    nAB *= np.sign(np.sum(nAB * s, axis=1))[:, None]
    c = np.sum(nKL * nAB, axis=1)
    dA = 0.5 * np.abs(_cross2d(ss, s))
    dual_area = np.zeros(len(P))
    np.add.at(dual_area, dia[:, 2], 0.5 * np.abs(_cross2d(xL - xK, xA - xK)))
    np.add.at(dual_area, dia[:, 3], 0.5 * np.abs(_cross2d(xL - xK, xB - xK)))
    return {"tri_area": tri_area, "tri_centroid": tri_cent,
            "bnd_edges": bnd_edges, "bnd_mid": bnd_mid,
            "dual_area": dual_area, "diamonds": dia,
            "alpha": ls ** 2 / (2 * dA), "beta": ls * lss * c / (2 * dA),
            "gamma": lss ** 2 / (2 * dA), "diamond_area": dA}

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Normal: diamond connectivity [K, L, Ks, Ls] on the small distorted mesh.
        {"setup": "import numpy as np\nP, T = _oracle_build_mesh(4)\n",
         "call": "ddfv_geometry(P, T)['diamonds'].astype(float)",
         "gold_call": "_oracle_ddfv_geometry(P, T)['diamonds'].astype(float)"},
        # Normal: the three diamond coefficients, including the signed coupling beta.
        {"setup": "import numpy as np\nP, T = _oracle_build_mesh(8)\n",
         "call": "np.concatenate([ddfv_geometry(P, T)[k] for k in ('alpha', 'beta', 'gamma')])",
         "gold_call": "np.concatenate([_oracle_ddfv_geometry(P, T)[k] for k in ('alpha', 'beta', 'gamma')])"},
        # Normal: dual-cell areas, including the truncated cells on the boundary.
        {"setup": "import numpy as np\nP, T = _oracle_build_mesh(8)\n",
         "call": "ddfv_geometry(P, T)['dual_area']",
         "gold_call": "_oracle_ddfv_geometry(P, T)['dual_area']"},
        # Boundary: boundary-edge midpoints and diamond areas on the coarsest mesh.
        {"setup": "import numpy as np\nP, T = _oracle_build_mesh(2)\n",
         "call": "np.concatenate([ddfv_geometry(P, T)['bnd_mid'].ravel(), ddfv_geometry(P, T)['diamond_area']])",
         "gold_call": "np.concatenate([_oracle_ddfv_geometry(P, T)['bnd_mid'].ravel(), _oracle_ddfv_geometry(P, T)['diamond_area']])"},
        # Edge: a single non-grid triangle pair with an obtuse angle.
        {"setup": "import numpy as np\nP = np.array([[0.0, 0.0], [1.0, 0.0], [0.9, 0.25], [0.1, 0.3]])\nT = np.array([[0, 1, 2], [0, 2, 3]])\n",
         "call": "np.concatenate([ddfv_geometry(P, T)[k].ravel().astype(float) for k in ('diamonds', 'alpha', 'beta', 'gamma', 'dual_area')])",
         "gold_call": "np.concatenate([_oracle_ddfv_geometry(P, T)[k].ravel().astype(float) for k in ('diamonds', 'alpha', 'beta', 'gamma', 'dual_area')])"},
    ]
