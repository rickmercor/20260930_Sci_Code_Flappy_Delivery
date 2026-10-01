"""
Nodal-to-edge coboundary of the epsilon-ball graph.

Build the coboundary that maps a nodal field to its oriented edge differences. An edge points from its smaller node index to its larger node index. Retain unordered pairs at separation strictly less than epsilon and list them lexicographically by endpoint pair. Each row has -1 at the tail and +1 at the head, with zeros elsewhere. Distances and kernel weights do not enter this topological map. Return a dense float matrix with one row per edge and one column per node.

Returns
-------
d0 : np.ndarray, shape (n_edges, N), float Dense coboundary matrix in the lexicographic edge order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def build_coboundary(points, epsilon: float) -> np.ndarray:
    '''Nodal-to-edge coboundary of the epsilon-ball graph.

    Parameters
    ----------
    points : array-like of shape (N, 2)
        Point-cloud coordinates in the plane.
    epsilon : float
        Graph radius.

    Returns
    -------
    d0 : np.ndarray, shape (n_edges, N), float
        Dense coboundary matrix in the lexicographic edge order.

    Raises
    ------
    ValueError
        If ``points`` is not a nonempty finite array of shape ``(N, 2)``,
        or if ``epsilon`` is not a finite positive scalar.
    '''
    return np.zeros((1, len(points)))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_build_coboundary(points, epsilon):
    import numpy as np

    try:
        P = np.asarray(points, dtype=float)
        eps = float(epsilon)
    except (TypeError, ValueError) as exc:
        raise ValueError("points and epsilon must be real-valued") from exc
    if (P.ndim != 2 or P.shape[1] != 2 or len(P) == 0
            or not np.all(np.isfinite(P))):
        raise ValueError("points must be a nonempty finite array of shape (N, 2)")
    if (isinstance(epsilon, (bool, np.bool_)) or np.ndim(epsilon) != 0
            or not np.isfinite(eps) or eps <= 0.0):
        raise ValueError("epsilon must be a finite positive scalar")
    n = len(P)
    E = np.array([(i, j) for i in range(n) for j in range(i + 1, n)
                  if np.linalg.norm(P[j] - P[i]) < eps], dtype=int).reshape(-1, 2)
    d0 = np.zeros((len(E), n))
    for t, (p, q) in enumerate(E):
        d0[t, p] = -1.0
        d0[t, q] = 1.0
    return d0

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
            )
    return [
        {
            "setup": _base + (
                '# case: normal\n'
                'P, b, h = _cloud(4)\n'
                'eps = 2.05 * h\n'
            ),
            "call": 'build_coboundary(P, eps)',
            "gold_call": '_oracle_build_coboundary(P, eps)',
        },
        {
            "setup": _base + (
                '# case: boundary\n'
                'P, b = OCT, OCTB\n'
            ),
            "call": 'build_coboundary(P, 0.7)',
            "gold_call": '_oracle_build_coboundary(P, 0.7)',
        },
        {
            "setup": _base + (
                'P, b, h = _cloud(5)\n'
                'eps = 2.3 * h\n'
            ),
            "call": 'build_coboundary(P, eps)',
            "gold_call": '_oracle_build_coboundary(P, eps)',
        },
        {
            "setup": _base + (
                'P, b, h = _cloud(4)\n'
                'eps = 1.2 * h\n'
            ),
            "call": 'build_coboundary(P, eps)',
            "gold_call": '_oracle_build_coboundary(P, eps)',
        },
        {
            "setup": _base + (
                '# case: edge\n'
                'def _value_error(thunk):\n'
                '    try:\n'
                '        thunk()\n'
                '    except ValueError:\n'
                '        return 1\n'
                '    except Exception:\n'
                '        return 2\n'
                '    return 0\n'
                'P = np.array([0.0, 1.0])\n'
            ),
            "call": '_value_error(lambda: build_coboundary(P, 1.0))',
            "gold_call": '_value_error(lambda: _oracle_build_coboundary(P, 1.0))',
        },
    ]
