"""
Virtual edge areas from constrained quadratic optimization.

Recover the edge-measure optimization for the same meshfree differential complex and apply it to the supplied moment constraints. Its objective uses the radial kernel from the node-volume construction. Derive the constrained minimizer and return its edge-area vector. Recompute the epsilon-ball edges from the points, retaining pairs at separation strictly less than epsilon and using lexicographic endpoint order with the smaller index first. This order must match the coefficient columns of moment_system. Return a float array with one entry per edge.

Returns
-------
areas : np.ndarray, shape (n_edges,), float Virtual edge areas, in the same edge order as the columns of ``moment_system``.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def solve_edge_areas(points, epsilon: float, moment_system) -> np.ndarray:
    '''Virtual edge areas from constrained quadratic optimization.

    Parameters
    ----------
    points : array-like of shape (N, 2)
        Point-cloud coordinates in the plane.
    epsilon : float
        Graph and kernel support radius.
    moment_system : array-like of shape (n_c, n_edges + 1)
        Augmented constraint system from the previous step.

    Returns
    -------
    areas : np.ndarray, shape (n_edges,), float
        Virtual edge areas, in the same edge order as the columns of
        ``moment_system``.

    Raises
    ------
    ValueError
        If ``moment_system`` does not have one coefficient column per graph edge.
    '''
    return np.zeros(np.shape(moment_system)[1] - 1)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_solve_edge_areas(points, epsilon, moment_system):
    import numpy as np

    P = np.asarray(points, dtype=float)
    A = np.asarray(moment_system, dtype=float)
    eps = float(epsilon)
    n = len(P)
    E = np.array([(i, j) for i in range(n) for j in range(i + 1, n)
                  if np.linalg.norm(P[j] - P[i]) < eps], dtype=int).reshape(-1, 2)
    if A.ndim != 2 or A.shape[1] != len(E) + 1:
        raise ValueError("moment system does not match the edge set of these points")
    r = np.linalg.norm(P[E[:, 1]] - P[E[:, 0]], axis=1)
    phi = np.clip(1.0 - r / eps, 0.0, None) ** 2
    B, c = A[:, :-1], A[:, -1]
    lam = np.linalg.solve((B * phi) @ B.T, c)
    return phi * (B.T @ lam)

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
            )
    return [
        {
            "setup": _base + (
                '# case: normal\n'
                'P, b, h = _cloud(4)\n'
                'eps = 2.05 * h\n'
                'm, _, _ = _vol(P, b, eps, 1.0)\n'
                'A = _sys(P, b, eps, m)\n'
            ),
            "call": 'solve_edge_areas(P, eps, A)',
            "gold_call": '_oracle_solve_edge_areas(P, eps, A)',
        },
        {
            "setup": _base + (
                'P, b, h = _cloud(5)\n'
                'eps = 2.3 * h\n'
                'm, _, _ = _vol(P, b, eps, 1.0)\n'
                'A = _sys(P, b, eps, m)\n'
            ),
            "call": 'solve_edge_areas(P, eps, A)',
            "gold_call": '_oracle_solve_edge_areas(P, eps, A)',
        },
        {
            "setup": _base + (
                '# case: boundary\n'
                'P, b = OCT, OCTB\n'
                'm, _, _ = _vol(P, b, 0.7, 2.0)\n'
                'A = _sys(P, b, 0.7, m)\n'
            ),
            "call": 'solve_edge_areas(P, 0.7, A)',
            "gold_call": '_oracle_solve_edge_areas(P, 0.7, A)',
        },
        {
            "setup": _base + (
                '# case: edge\n'
                'P, b, h = _cloud(6)\n'
                'eps = 2.6 * h\n'
                'm, _, _ = _vol(P, b, eps, 0.5)\n'
                'A = _sys(P, b, eps, m)\n'
            ),
            "call": 'solve_edge_areas(P, eps, A)',
            "gold_call": '_oracle_solve_edge_areas(P, eps, A)',
        },
    ]
