"""
Augmented moment system for consistent virtual edge areas.

Assemble the linear consistency constraints on virtual edge areas for the meshfree differential complex. The constraints enforce local reproduction of the divergence of affine vector fields at each interior node. This step assembles the equations without solving for the areas. Columns follow the epsilon-ball edges in lexicographic endpoint order, with the smaller index first; a pair is an edge when its separation is strictly less than epsilon. For a block owned by node i, take each incident displacement from i to its neighbor. Use products of those displacement components as the moment coefficients, with positive signs before the first- and second-moment sums. Rows are grouped into five consecutive rows per interior node in ascending node order: the two first-order conditions in coordinate order, then the second-order conditions in (x, x), (x, y), (y, y) order. Place the right-hand side in the last column. A row is supported only on edges incident to its owner. Omit boundary-node blocks. Return a float array of shape (5 * number of interior nodes, number of edges + 1).

Returns
-------
system : np.ndarray, shape (5 * n_interior, n_edges + 1), float Constraint coefficients in the first ``n_edges`` columns and the right-hand side in the last column, in the row and column order documented above.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def assemble_moment_constraint_system(
    points, boundary_flags, epsilon: float, node_volumes
) -> np.ndarray:
    '''Augmented moment system for consistent virtual edge areas.

    Parameters
    ----------
    points : array-like of shape (N, 2)
        Point-cloud coordinates in the plane.
    boundary_flags : array-like of shape (N,), bool
        True at nodes lying on the domain boundary.
    epsilon : float
        Graph radius; an unordered pair is an edge when its separation is
        strictly less than ``epsilon``.
    node_volumes : array-like of shape (N,)
        Virtual node volumes, zero on boundary nodes.

    Returns
    -------
    system : np.ndarray, shape (5 * n_interior, n_edges + 1), float
        Constraint coefficients in the first ``n_edges`` columns and the
        right-hand side in the last column, in the row and column order
        documented above.

    Raises
    ------
    ValueError
        If the point, flag, and node-volume inputs have inconsistent shapes.
    '''
    return np.zeros((5, 1))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_assemble_moment_constraint_system(points, boundary_flags, epsilon, node_volumes):
    import numpy as np

    P = np.asarray(points, dtype=float)
    b = np.asarray(boundary_flags, dtype=bool)
    m = np.asarray(node_volumes, dtype=float)
    if P.ndim != 2 or P.shape[1] != 2 or b.shape != (len(P),) or m.shape != (len(P),):
        raise ValueError("inconsistent point cloud, flags or volumes")
    eps = float(epsilon)
    n = len(P)
    E = np.array([(i, j) for i in range(n) for j in range(i + 1, n)
                  if np.linalg.norm(P[j] - P[i]) < eps], dtype=int).reshape(-1, 2)
    incident = [[] for _ in range(n)]
    for t, (p, q) in enumerate(E):
        incident[p].append(t)
        incident[q].append(t)
    interior = np.where(~b)[0]
    pairs = [(0, 0), (0, 1), (1, 1)]
    system = np.zeros((5 * len(interior), len(E) + 1))
    for k, i in enumerate(interior):
        r0 = 5 * k
        for t in incident[i]:
            p, q = E[t]
            eta = P[q if p == i else p] - P[i]
            system[r0, t] = eta[0]
            system[r0 + 1, t] = eta[1]
            for s, (c, d) in enumerate(pairs):
                system[r0 + 2 + s, t] = eta[c] * eta[d]
        for s, (c, d) in enumerate(pairs):
            system[r0 + 2 + s, -1] = 2.0 * m[i] if c == d else 0.0
    return system

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
            )
    return [
        {
            "setup": _base + (
                '# case: normal\n'
                'P, b, h = _cloud(4)\n'
                'eps = 2.05 * h\n'
                'm, _, _ = _vol(P, b, eps, 1.0)\n'
            ),
            "call": 'assemble_moment_constraint_system(P, b, eps, m)',
            "gold_call": '_oracle_assemble_moment_constraint_system(P, b, eps, m)',
        },
        {
            "setup": _base + (
                'P, b, h = _cloud(5)\n'
                'eps = 2.3 * h\n'
                'm, _, _ = _vol(P, b, eps, 1.0)\n'
            ),
            "call": 'assemble_moment_constraint_system(P, b, eps, m)',
            "gold_call": '_oracle_assemble_moment_constraint_system(P, b, eps, m)',
        },
        {
            "setup": _base + (
                '# case: boundary\n'
                'P, b = OCT, OCTB\n'
                'm, _, _ = _vol(P, b, 0.7, 2.0)\n'
            ),
            "call": 'assemble_moment_constraint_system(P, b, 0.7, m)',
            "gold_call": '_oracle_assemble_moment_constraint_system(P, b, 0.7, m)',
        },
        {
            "setup": _base + (
                'P, b, h = _cloud(6)\n'
                'eps = 2.6 * h\n'
                'm, _, _ = _vol(P, b, eps, 0.5)\n'
            ),
            "call": 'assemble_moment_constraint_system(P, b, eps, m)',
            "gold_call": '_oracle_assemble_moment_constraint_system(P, b, eps, m)',
        },
        {
            "setup": _base + (
                '# case: edge\n'
                'P, b, h = _cloud(4)\n'
                'eps = 2.3 * h\n'
                'm = np.zeros(len(P))\n'
            ),
            "call": 'assemble_moment_constraint_system(P, b, eps, m)',
            "gold_call": '_oracle_assemble_moment_constraint_system(P, b, eps, m)',
        },
    ]
