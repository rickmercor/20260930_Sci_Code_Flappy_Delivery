"""
Volume-weighted relative discrete L2 error over the interior nodes.

Compare the computed and reference nodal fields in the volume-weighted discrete L2 norm over interior nodes. Each norm is the square root of the sum of virtual node volume times squared field value. Return the norm of the difference divided by the norm of the reference field, as a native Python float.

Returns
-------
error : float Volume-weighted relative discrete L2 error over interior nodes.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def volume_weighted_relative_error(
    solution, reference, node_volumes, boundary_flags
) -> float:
    '''Volume-weighted relative discrete L2 error over the interior nodes.

    Parameters
    ----------
    solution : array-like of shape (N,)
        Computed nodal field.
    reference : array-like of shape (N,)
        Reference nodal field.
    node_volumes : array-like of shape (N,)
        Virtual node volumes.
    boundary_flags : array-like of shape (N,), bool
        True at nodes lying on the domain boundary.

    Returns
    -------
    error : float
        Volume-weighted relative discrete L2 error over interior nodes.

    Raises
    ------
    ValueError
        If the four inputs are not finite one-dimensional arrays of one common
        nonzero length, if a volume is negative, if ``boundary_flags`` is not
        boolean, or if the weighted reference norm is zero.
    '''
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_volume_weighted_relative_error(solution, reference, node_volumes, boundary_flags):
    import numpy as np

    try:
        u = np.asarray(solution, dtype=float)
        r = np.asarray(reference, dtype=float)
        m = np.asarray(node_volumes, dtype=float)
        raw_flags = np.asarray(boundary_flags)
    except (TypeError, ValueError) as exc:
        raise ValueError("inputs must be real-valued arrays") from exc
    if u.ndim != 1 or len(u) == 0 or r.shape != u.shape or m.shape != u.shape:
        raise ValueError("solution, reference and node_volumes must be same-length 1D arrays")
    if raw_flags.shape != u.shape or raw_flags.dtype.kind != "b":
        raise ValueError("boundary_flags must be a boolean 1D array of the same length")
    if (not np.all(np.isfinite(u)) or not np.all(np.isfinite(r))
            or not np.all(np.isfinite(m))):
        raise ValueError("solution, reference and node_volumes must be finite")
    if np.any(m < 0.0):
        raise ValueError("node volumes must be nonnegative")
    b = raw_flags.astype(bool, copy=False)
    denominator_sq = float(np.sum(m[~b] * r[~b] ** 2))
    if not np.isfinite(denominator_sq) or denominator_sq <= 0.0:
        raise ValueError("weighted reference norm must be finite and nonzero")
    numerator_sq = float(np.sum(m[~b] * (u[~b] - r[~b]) ** 2))
    if not np.isfinite(numerator_sq) or numerator_sq < 0.0:
        raise ValueError("weighted error norm must be finite")
    return float(np.sqrt(numerator_sq / denominator_sq))

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
                'u = np.sin(3*P[:,0])+P[:,1]\n'
                'r = np.sin(3*P[:,0])+P[:,1]+0.01*np.cos(5*P[:,1])\n'
            ),
            "call": 'volume_weighted_relative_error(u, r, m, b)',
            "gold_call": '_oracle_volume_weighted_relative_error(u, r, m, b)',
        },
        {
            "setup": _base + (
                '# case: boundary\n'
                'P, b = OCT, OCTB\n'
                'm, _, _ = _vol(P, b, 0.7, 2.0)\n'
                'u = P[:,0]*P[:,1]\n'
                'r = P[:,0]+P[:,1]\n'
            ),
            "call": 'volume_weighted_relative_error(u, r, m, b)',
            "gold_call": '_oracle_volume_weighted_relative_error(u, r, m, b)',
        },
        {
            "setup": _base + (
                'P, b, h = _cloud(5)\n'
                'eps = 2.3 * h\n'
                'm, _, _ = _vol(P, b, eps, 1.0)\n'
                'u = np.ones(len(P))\n'
                'r = np.ones(len(P))\n'
            ),
            "call": 'volume_weighted_relative_error(u, r, m, b)',
            "gold_call": '_oracle_volume_weighted_relative_error(u, r, m, b)',
        },
        {
            "setup": _base + (
                'P, b, h = _cloud(5)\n'
                'eps = 2.3 * h\n'
                'm, _, _ = _vol(P, b, eps, 1.0)\n'
                'u = np.zeros(len(P))\n'
                'r = np.exp(P[:,0])\n'
            ),
            "call": 'volume_weighted_relative_error(u, r, m, b)',
            "gold_call": '_oracle_volume_weighted_relative_error(u, r, m, b)',
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
                'P, b, h = _cloud(4)\n'
                'eps = 2.05 * h\n'
                'm, _, _ = _vol(P, b, eps, 1.0)\n'
                'u = np.zeros((len(P), 1))\n'
                'r = np.ones(len(P))\n'
            ),
            "call": '_value_error(lambda: volume_weighted_relative_error(u, r, m, b))',
            "gold_call": '_value_error(lambda: _oracle_volume_weighted_relative_error(u, r, m, b))',
        },
    ]
