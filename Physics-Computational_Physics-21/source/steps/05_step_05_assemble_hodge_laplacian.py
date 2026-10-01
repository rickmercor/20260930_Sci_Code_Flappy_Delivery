"""
Unnormalised Hodge Laplacian of the meshfree complex.

Assemble the unnormalised discrete Poisson operator from the coboundary and the virtual edge areas. The node-volume factors belong on the right-hand side of the conservation equation, so this step returns the transpose of the coboundary times the diagonal edge-area matrix times the coboundary. Return the dense N by N float matrix. It is symmetric and annihilates the constant vector.

Returns
-------
laplacian : np.ndarray, shape (N, N), float Unnormalised Hodge Laplacian.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def assemble_hodge_laplacian(coboundary_matrix, edge_areas) -> np.ndarray:
    '''Unnormalised Hodge Laplacian of the meshfree complex.

    Parameters
    ----------
    coboundary_matrix : array-like of shape (n_edges, N)
        Coboundary from the previous step.
    edge_areas : array-like of shape (n_edges,)
        Virtual edge areas.

    Returns
    -------
    laplacian : np.ndarray, shape (N, N), float
        Unnormalised Hodge Laplacian.

    Raises
    ------
    ValueError
        If the coboundary and edge-area inputs disagree on the edge count.
    '''
    return np.zeros((np.shape(coboundary_matrix)[1],) * 2)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_assemble_hodge_laplacian(coboundary_matrix, edge_areas):
    import numpy as np

    d0 = np.asarray(coboundary_matrix, dtype=float)
    a = np.asarray(edge_areas, dtype=float)
    if d0.ndim != 2 or a.shape != (d0.shape[0],):
        raise ValueError("coboundary and edge areas disagree on the edge count")
    return d0.T @ (a[:, None] * d0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    _base = (
                'import numpy as np\n'
                'def _cloud(k, amp=0.18):\n'
                '    xs = np.linspace(0.0, 1.0, k)\n'
                '    X, Y = np.meshgrid(xs, xs, indexing="ij")\n'
                '    P0 = np.stack([X.ravel(), Y.ravel()], axis=1)\n'
                '    h = 1.0 / (k - 1)\n'
                '    b = (np.isclose(P0[:, 0], 0.0) | np.isclose(P0[:, 0], 1.0) |\n'
                '         np.isclose(P0[:, 1], 0.0) | np.isclose(P0[:, 1], 1.0))\n'
                '    P = P0.copy()\n'
                '    P[~b, 0] += amp * h * np.sin(6.0 * P0[~b, 0] + 2.0 * P0[~b, 1])\n'
                '    P[~b, 1] += amp * h * np.cos(2.0 * P0[~b, 0] + 5.0 * P0[~b, 1])\n'
                '    return P, b, h\n'
                'OCT = np.array([[0.0,0.0],[0.5,0.0],[1.0,0.0],[1.0,0.5],[1.0,1.0],[0.5,1.0],\n'
                '                [0.0,1.0],[0.0,0.5],[0.37,0.41],[0.62,0.33],[0.55,0.68],[0.30,0.70]])\n'
                'OCTB = np.array([True]*8 + [False]*4)\n'
                'def _vol(P, b, eps, meas):\n'
                '    n = len(P)\n'
                '    E = np.array([(i, j) for i in range(n) for j in range(i + 1, n)\n'
                '                  if np.linalg.norm(P[j] - P[i]) < eps], dtype=int).reshape(-1, 2)\n'
                '    r = np.linalg.norm(P[E[:, 1]] - P[E[:, 0]], axis=1)\n'
                '    ph = np.clip(1.0 - r / eps, 0.0, None) ** 2\n'
                '    kap = np.zeros(n)\n'
                '    np.add.at(kap, E[:, 0], ph); np.add.at(kap, E[:, 1], ph)\n'
                '    m = np.zeros(n); inv = 1.0 / kap[~b]\n'
                '    m[~b] = inv / inv.sum() * float(meas)\n'
                '    return m, E, ph\n'
                'def _sys(P, b, eps, m):\n'
                '    _, E, _ = _vol(P, b, eps, 1.0)\n'
                '    inc = [[] for _ in range(len(P))]\n'
                '    for t, (p, q) in enumerate(E):\n'
                '        inc[p].append(t); inc[q].append(t)\n'
                '    idx = np.where(~b)[0]\n'
                '    pairs = [(0, 0), (0, 1), (1, 1)]\n'
                '    out = np.zeros((5 * len(idx), len(E) + 1))\n'
                '    for k, i in enumerate(idx):\n'
                '        r0 = 5 * k\n'
                '        for t in inc[i]:\n'
                '            p, q = E[t]\n'
                '            eta = P[q if p == i else p] - P[i]\n'
                '            out[r0, t] = eta[0]; out[r0 + 1, t] = eta[1]\n'
                '            for s, (c, d) in enumerate(pairs):\n'
                '                out[r0 + 2 + s, t] = eta[c] * eta[d]\n'
                '        for s, (c, d) in enumerate(pairs):\n'
                '            out[r0 + 2 + s, -1] = 2.0 * m[i] if c == d else 0.0\n'
                '    return out\n'
                'def _areas(P, eps, A):\n'
                '    _, E, ph = _vol(P, np.zeros(len(P), bool), eps, 1.0)\n'
                '    B, c = A[:, :-1], A[:, -1]\n'
                '    return ph * (B.T @ np.linalg.solve((B * ph) @ B.T, c))\n'
                'def _cob(P, eps):\n'
                '    n = len(P)\n'
                '    E = np.array([(i, j) for i in range(n) for j in range(i + 1, n)\n'
                '                  if np.linalg.norm(P[j] - P[i]) < eps], dtype=int).reshape(-1, 2)\n'
                '    d0 = np.zeros((len(E), n))\n'
                '    for t, (p, q) in enumerate(E):\n'
                '        d0[t, p] = -1.0; d0[t, q] = 1.0\n'
                '    return d0\n'
            )
    return [
        {
            "setup": _base + (
                '# case: normal\n'
                'P, b, h = _cloud(4)\n'
                'eps = 2.05 * h\n'
                'm, _, _ = _vol(P, b, eps, 1.0)\n'
                'A = _sys(P, b, eps, m)\n'
                'a = _areas(P, eps, A)\n'
                'd0 = _cob(P, eps)\n'
            ),
            "call": 'assemble_hodge_laplacian(d0, a)',
            "gold_call": '_oracle_assemble_hodge_laplacian(d0, a)',
        },
        {
            "setup": _base + (
                '# case: boundary\n'
                'P, b = OCT, OCTB\n'
                'm, _, _ = _vol(P, b, 0.7, 2.0)\n'
                'A = _sys(P, b, 0.7, m)\n'
                'a = _areas(P, 0.7, A)\n'
                'd0 = _cob(P, 0.7)\n'
            ),
            "call": 'assemble_hodge_laplacian(d0, a)',
            "gold_call": '_oracle_assemble_hodge_laplacian(d0, a)',
        },
        {
            "setup": _base + (
                'P, b, h = _cloud(5)\n'
                'eps = 2.3 * h\n'
                'm, _, _ = _vol(P, b, eps, 1.0)\n'
                'A = _sys(P, b, eps, m)\n'
                'a = _areas(P, eps, A)\n'
                'd0 = _cob(P, eps)\n'
            ),
            "call": 'assemble_hodge_laplacian(d0, a)',
            "gold_call": '_oracle_assemble_hodge_laplacian(d0, a)',
        },
        {
            "setup": _base + (
                '# case: edge\n'
                'P, b, h = _cloud(4)\n'
                'eps = 2.05 * h\n'
                'd0 = _cob(P, eps)\n'
                'a = np.ones(d0.shape[0])\n'
            ),
            "call": 'assemble_hodge_laplacian(d0, a)',
            "gold_call": '_oracle_assemble_hodge_laplacian(d0, a)',
        },
    ]
