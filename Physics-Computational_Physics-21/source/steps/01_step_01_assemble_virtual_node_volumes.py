"""
Virtual node volumes of a meshfree differential complex.

Equip the proximity graph with virtual nodal measure. Recover the compactly supported radial kernel and the node-volume rule for this meshfree differential complex from the research literature. Build the graph on the given points, keeping an unordered pair when its separation is strictly less than epsilon. Accumulate the edge kernel at both endpoints to obtain each node's kernel mass. The virtual volume is a function of this mass, normalized over interior nodes to the given domain measure. Use the same radial kernel in the edge-area optimization. Return a length-N float array with zero at each boundary node.

Returns
-------
volumes : np.ndarray, shape (N,), float Virtual node volumes, zero on boundary nodes, summing over the interior nodes to ``domain_measure``.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def assemble_virtual_node_volumes(
    points, boundary_flags, epsilon: float, domain_measure: float
) -> np.ndarray:
    '''Virtual node volumes of a meshfree differential complex.

    Parameters
    ----------
    points : array-like of shape (N, 2)
        Point-cloud coordinates in the plane.
    boundary_flags : array-like of shape (N,), bool
        True at nodes lying on the domain boundary.
    epsilon : float
        Graph and kernel support radius.
    domain_measure : float
        Measure of the domain that the interior volumes must reproduce.

    Returns
    -------
    volumes : np.ndarray, shape (N,), float
        Virtual node volumes, zero on boundary nodes, summing over the
        interior nodes to ``domain_measure``.

    Raises
    ------
    ValueError
        If the point, flag, and volume-domain inputs have inconsistent shapes.
    '''
    return np.zeros(len(points))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_assemble_virtual_node_volumes(points, boundary_flags, epsilon, domain_measure):
    import numpy as np

    P = np.asarray(points, dtype=float)
    b = np.asarray(boundary_flags, dtype=bool)
    if P.ndim != 2 or P.shape[1] != 2 or b.shape != (len(P),):
        raise ValueError("points must be (N, 2) and boundary_flags (N,)")
    eps = float(epsilon)
    n = len(P)
    E = np.array([(i, j) for i in range(n) for j in range(i + 1, n)
                  if np.linalg.norm(P[j] - P[i]) < eps], dtype=int).reshape(-1, 2)
    r = np.linalg.norm(P[E[:, 1]] - P[E[:, 0]], axis=1)
    phi = np.clip(1.0 - r / eps, 0.0, None) ** 2
    kappa = np.zeros(n)
    np.add.at(kappa, E[:, 0], phi)
    np.add.at(kappa, E[:, 1], phi)
    volumes = np.zeros(n)
    inv = 1.0 / kappa[~b]
    volumes[~b] = inv / inv.sum() * float(domain_measure)
    return volumes

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
            "call": 'assemble_virtual_node_volumes(P, b, eps, 1.0)',
            "gold_call": '_oracle_assemble_virtual_node_volumes(P, b, eps, 1.0)',
        },
        {
            "setup": _base + (
                'P, b, h = _cloud(5)\n'
                'eps = 2.3 * h\n'
            ),
            "call": 'assemble_virtual_node_volumes(P, b, eps, 1.0)',
            "gold_call": '_oracle_assemble_virtual_node_volumes(P, b, eps, 1.0)',
        },
        {
            "setup": _base + (
                '# case: boundary\n'
                'P, b = OCT, OCTB\n'
            ),
            "call": 'assemble_virtual_node_volumes(P, b, 0.7, 2.0)',
            "gold_call": '_oracle_assemble_virtual_node_volumes(P, b, 0.7, 2.0)',
        },
        {
            "setup": _base + (
                'P, b, h = _cloud(6)\n'
                'eps = 2.6 * h\n'
            ),
            "call": 'assemble_virtual_node_volumes(P, b, eps, 0.5)',
            "gold_call": '_oracle_assemble_virtual_node_volumes(P, b, eps, 0.5)',
        },
        {
            "setup": _base + (
                '# case: edge\n'
                'P, b, h = _cloud(4)\n'
                'eps = 2.3 * h\n'
            ),
            "call": 'assemble_virtual_node_volumes(P, b, eps, 3.0)',
            "gold_call": '_oracle_assemble_virtual_node_volumes(P, b, eps, 3.0)',
        },
    ]
