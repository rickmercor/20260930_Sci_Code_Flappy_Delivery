"""
Dirichlet solve of the discrete conservation law on the point cloud.

Solve the unnormalised discrete Poisson equation with node-volume-weighted forcing. Boundary values are prescribed. Restrict the Laplacian to interior rows and columns and subtract the boundary-column contribution from the interior right-hand side before solving. Return the full nodal field, with prescribed boundary entries and solved interior entries. Only entries of boundary_values at boundary nodes are meaningful; ignore every other entry. Return a length-N float array.

Returns
-------
solution : np.ndarray, shape (N,), float Nodal field with prescribed boundary data in place.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def solve_dirichlet_problem(
    laplacian_matrix, node_volumes, boundary_flags, forcing, boundary_values
) -> np.ndarray:
    '''Dirichlet solve of the discrete conservation law on the point cloud.

    Parameters
    ----------
    laplacian_matrix : array-like of shape (N, N)
        Unnormalised Hodge Laplacian.
    node_volumes : array-like of shape (N,)
        Virtual node volumes, zero on boundary nodes.
    boundary_flags : array-like of shape (N,), bool
        True at nodes lying on the domain boundary.
    forcing : array-like of shape (N,)
        Forcing sampled at the nodes.
    boundary_values : array-like of shape (N,)
        Prescribed values; only boundary entries are used.

    Returns
    -------
    solution : np.ndarray, shape (N,), float
        Nodal field with prescribed boundary data in place.

    Raises
    ------
    ValueError
        If the matrix and nodal inputs do not share one consistent node count.
    '''
    return np.zeros(len(node_volumes))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_solve_dirichlet_problem(laplacian_matrix, node_volumes, boundary_flags,
                                    forcing, boundary_values):
    import numpy as np

    K = np.asarray(laplacian_matrix, dtype=float)
    m = np.asarray(node_volumes, dtype=float)
    b = np.asarray(boundary_flags, dtype=bool)
    f = np.asarray(forcing, dtype=float)
    ub = np.asarray(boundary_values, dtype=float)
    n = len(m)
    if K.shape != (n, n) or b.shape != (n,) or f.shape != (n,) or ub.shape != (n,):
        raise ValueError("inconsistent shapes")
    u = np.zeros(n)
    u[b] = ub[b]
    rhs = (m * f) - K[:, b] @ u[b]
    u[~b] = np.linalg.solve(K[np.ix_(~b, ~b)], rhs[~b])
    return u

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
                'K = _cob(P, eps).T @ (a[:, None] * _cob(P, eps))\n'
                'ue = np.sin(np.pi*P[:,0])*np.sin(np.pi*P[:,1])\n'
                'f = 2*np.pi**2*ue\n'
            ),
            "call": 'solve_dirichlet_problem(K, m, b, f, ue)',
            "gold_call": '_oracle_solve_dirichlet_problem(K, m, b, f, ue)',
        },
        {
            "setup": _base + (
                '# case: boundary\n'
                'P, b = OCT, OCTB\n'
                'm, _, _ = _vol(P, b, 0.7, 2.0)\n'
                'A = _sys(P, b, 0.7, m)\n'
                'a = _areas(P, 0.7, A)\n'
                'K = _cob(P, 0.7).T @ (a[:, None] * _cob(P, 0.7))\n'
                'ue = P[:,0]**2 - P[:,1]**2\n'
                'f = np.zeros(len(P))\n'
            ),
            "call": 'solve_dirichlet_problem(K, m, b, f, ue)',
            "gold_call": '_oracle_solve_dirichlet_problem(K, m, b, f, ue)',
        },
        {
            "setup": _base + (
                'P, b, h = _cloud(5)\n'
                'eps = 2.3 * h\n'
                'm, _, _ = _vol(P, b, eps, 1.0)\n'
                'A = _sys(P, b, eps, m)\n'
                'a = _areas(P, eps, A)\n'
                'K = _cob(P, eps).T @ (a[:, None] * _cob(P, eps))\n'
                'ue = np.exp(P[:,0]) * np.cos(P[:,1])\n'
                'f = np.zeros(len(P))\n'
            ),
            "call": 'solve_dirichlet_problem(K, m, b, f, ue)',
            "gold_call": '_oracle_solve_dirichlet_problem(K, m, b, f, ue)',
        },
        {
            "setup": _base + (
                '# case: edge\n'
                'P, b, h = _cloud(4)\n'
                'eps = 2.3 * h\n'
                'm, _, _ = _vol(P, b, eps, 1.0)\n'
                'A = _sys(P, b, eps, m)\n'
                'a = _areas(P, eps, A)\n'
                'K = _cob(P, eps).T @ (a[:, None] * _cob(P, eps))\n'
                'ue = np.zeros(len(P))\n'
                'f = np.ones(len(P))\n'
            ),
            "call": 'solve_dirichlet_problem(K, m, b, f, ue)',
            "gold_call": '_oracle_solve_dirichlet_problem(K, m, b, f, ue)',
        },
    ]
