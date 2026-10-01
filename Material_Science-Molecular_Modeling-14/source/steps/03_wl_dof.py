"""
Enumerate the reciprocal degrees of freedom that survive the source's W and L truncations for a base momentum q: for each layer j the pairs of reciprocal vectors of the other two layers (in increasing layer order), with integer coordinates scanned over [-nmax, nmax]^2 in ascending lexicographic order, layer-major. Which distances the two criteria test, how the shifted momentum is prepared before the W test, and how the reference Dirac point is chosen, follow the source.

The four-dimensional reciprocal lattice must be cut down to a finite set; the source's two-stage truncation and its exact criteria are what make the resulting matrix both small and faithful.

Returns
-------
return (n_dof, 5) int64: [layer, nk(2), nl(2)] reciprocal degrees of freedom
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def wl_dof(q, geom, W, L, nmax):
    """q: (2,) base momentum; geom: (3, 14) packed layer geometry; W, L:
    truncation radii; nmax: integer scan bound. Returns (n_dof, 5) int64
    rows [j, nk1, nk2, nl1, nl2]: for layer j the integer coordinates of the
    reciprocal vectors of the other two layers in increasing layer order,
    keeping exactly the pairs the source's L and W criteria admit, rows
    ordered by layer then ascending lexicographic integer coordinates.
    Raises ValueError on nonpositive radii."""
    return np.zeros((1, 5), dtype=np.int64)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 3: WL-truncated reciprocal degrees of freedom."""

import numpy as np


def _oracle_wl_dof(q, geom, W, L, nmax):
    q = np.asarray(q, dtype=np.float64)
    geom = np.asarray(geom, dtype=np.float64)
    if not (np.isfinite(W) and np.isfinite(L)) or W <= 0 or L <= 0:
        raise ValueError("nonpositive truncation radius")
    rows = []
    rng = range(-int(nmax), int(nmax) + 1)
    for j in range(3):
        others = [t for t in range(3) if t != j]
        k, l = others[0], others[1]
        Bj = geom[j, 4:8]
        Bk = geom[k, 4:8].reshape(2, 2)
        Bl = geom[l, 4:8].reshape(2, 2)
        Kj = geom[j, 8:10]
        Kpj = geom[j, 10:12]
        qred = _oracle_cell_reduce(q[None, :], Bj)[0]
        Kt = Kj if np.linalg.norm(qred - Kj) <= np.linalg.norm(qred - Kpj) else Kpj
        for nk1 in rng:
            for nk2 in rng:
                Gk = Bk @ np.array([nk1, nk2], dtype=np.float64)
                if np.linalg.norm(Gk) >= L:
                    continue
                for nl1 in rng:
                    for nl2 in rng:
                        Gl = Bl @ np.array([nl1, nl2], dtype=np.float64)
                        if np.linalg.norm(Gl) >= L:
                            continue
                        shifted = _oracle_cell_reduce((q + Gk + Gl)[None, :], Bj)[0]
                        if np.linalg.norm(shifted - Kt) < W:
                            rows.append([j, nk1, nk2, nl1, nl2])
    return np.asarray(rows, dtype=np.int64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as _n\ngeom=layer_geometry(_n.array([-0.06,0.0,0.11]))\nq=geom[1,8:10]+_n.array([0.03,0.02])', "call": "wl_dof(q, geom, 1.15, 5.0, 3)", "gold_call": "_oracle_wl_dof(q, geom, 1.15, 5.0, 3)", "tol": 1e-09},
        {"setup": 'import numpy as _n\ngeom=layer_geometry(_n.array([-0.09,0.0,0.07]))\nq=geom[1,10:12]+_n.array([0.03,0.02])', "call": "wl_dof(q, geom, 1.05, 4.6, 3)", "gold_call": "_oracle_wl_dof(q, geom, 1.05, 4.6, 3)", "tol": 1e-09},
        {"setup": 'import numpy as _n\ngeom=layer_geometry(_n.array([-0.05,0.0,0.09]))\nq=geom[1,8:10]+_n.array([0.03,0.02])', "call": "wl_dof(q, geom, 1.1, 4.8, 3)", "gold_call": "_oracle_wl_dof(q, geom, 1.1, 4.8, 3)", "tol": 1e-09},
    ]
