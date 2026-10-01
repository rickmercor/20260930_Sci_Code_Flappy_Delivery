"""
Build the bond-associated deformation gradient of every bond from the two nodal gradients at its ends and the bond's own current image.

Evaluating the constitutive law at a quadrature point on each bond, with a gradient that reproduces the bond's actual deformation along its axis, removes the zero-energy modes of the node-wise correspondence formulation.

Returns
-------
np.ndarray: the (N_b, 3, 3) bond-associated deformation gradients.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_bond_deformation_gradients(
    nodal_gradients: "np.ndarray",
    ref_positions: "np.ndarray",
    cur_positions: "np.ndarray",
    bond_start: "np.ndarray",
    bond_end: "np.ndarray",
) -> "np.ndarray":
    """Return the bond-associated deformation gradient of every bond.

    For bond ``b`` from node ``k = bond_start[b]`` to ``n = bond_end[b]``
    let ``dX_b = X_n - X_k`` and ``dx_b = x_n - x_k`` and let
    ``F_avg = (F_k + F_n) / 2`` be the mean of the two nodal gradients. The
    bond gradient ``F_b`` is the unique 3x3 matrix that agrees with
    ``F_avg`` on every vector orthogonal to ``dX_b`` and maps ``dX_b``
    exactly onto ``dx_b``.

    Parameters
    ----------
    nodal_gradients : np.ndarray
        ``(N, 3, 3)`` nodal deformation gradients.
    ref_positions : np.ndarray
        ``(N, 3)`` reference positions.
    cur_positions : np.ndarray
        ``(N, 3)`` current positions.
    bond_start, bond_end : np.ndarray
        ``(N_b,)`` integer node indices of each bond.

    Returns
    -------
    np.ndarray
        ``(N_b, 3, 3)`` bond deformation gradients.

    Raises
    ------
    ValueError
        If the arrays are not finite with shapes ``(N, 3, 3)``, ``(N, 3)``
        and ``(N, 3)``, if the bond index arrays do not share one 1D length,
        if an index lies outside ``[0, N)``, or if a bond has zero reference
        length.
    """
    return bond_gradients

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_bond_deformation_gradients(
    nodal_gradients: "np.ndarray",
    ref_positions: "np.ndarray",
    cur_positions: "np.ndarray",
    bond_start: "np.ndarray",
    bond_end: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation (rank-one correction of the averaged gradient)."""
    import numpy as np

    grads = np.asarray(nodal_gradients, dtype=float)
    ref = np.asarray(ref_positions, dtype=float)
    cur = np.asarray(cur_positions, dtype=float)
    if ref.ndim != 2 or ref.shape[1] != 3 or cur.shape != ref.shape:
        raise ValueError("positions must be two (N, 3) arrays of equal shape")
    if grads.shape != (ref.shape[0], 3, 3):
        raise ValueError("nodal_gradients must have shape (N, 3, 3)")
    if not (np.all(np.isfinite(grads)) and np.all(np.isfinite(ref)) and np.all(np.isfinite(cur))):
        raise ValueError("inputs must be finite")
    start = np.asarray(bond_start)
    end = np.asarray(bond_end)
    if start.ndim != 1 or start.shape != end.shape:
        raise ValueError("bond index arrays must share one 1D length")
    if not (np.issubdtype(start.dtype, np.integer) and np.issubdtype(end.dtype, np.integer)):
        raise ValueError("bond indices must be integers")
    count = ref.shape[0]
    if start.size and (min(start.min(), end.min()) < 0 or max(start.max(), end.max()) >= count):
        raise ValueError("bond indices must lie in [0, N)")

    ref_bond = ref[end] - ref[start]
    cur_bond = cur[end] - cur[start]
    length_sq = np.einsum("bi,bi->b", ref_bond, ref_bond)
    if np.any(length_sq <= 0.0):
        raise ValueError("every bond needs a positive reference length")
    average = 0.5 * (grads[start] + grads[end])
    # Replace the averaged gradient's action along the bond by the bond's own
    # image: F_b = F_avg + (dx - F_avg dX) dX^T / |dX|^2.
    mismatch = cur_bond - np.einsum("bij,bj->bi", average, ref_bond)
    correction = mismatch[:, :, None] * ref_bond[:, None, :] / length_sq[:, None, None]
    return average + correction

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    """Return deterministic numerical comparisons with isolated inputs."""
    fixture_0 = (
        'import numpy as np\n'
        'rng = np.random.default_rng(11)\n'
        'def _run(fn, *args):\n'
        '    return fn(*[v.copy() if isinstance(v, np.ndarray) else v for v in args])\n'
        'X = rng.uniform(0.0, 2.0, size=(7, 3))\n'
        'x = X + 0.1 * rng.normal(size=(7, 3))\n'
        'G = np.eye(3) + 0.08 * rng.normal(size=(7, 3, 3))\n'
        'a = np.array([0, 0, 1, 2, 3, 4, 5, 6, 6, 2])\n'
        'b = np.array([1, 3, 2, 5, 6, 0, 4, 1, 3, 4])\n'
        'def _sig(result):\n'
        '    a = np.asarray(result, dtype=float)\n'
        '    assert a.ndim in (2, 3) and a.shape[-1] == 3\n'
        '    return a.ravel()\n'
    )
    fixture_1 = (
        'import numpy as np\n'
        'rng = np.random.default_rng(11)\n'
        'def _run(fn, *args):\n'
        '    return fn(*[v.copy() if isinstance(v, np.ndarray) else v for v in args])\n'
        'X = rng.uniform(0.0, 2.0, size=(7, 3))\n'
        'x = X + 0.1 * rng.normal(size=(7, 3))\n'
        'G = np.eye(3) + 0.08 * rng.normal(size=(7, 3, 3))\n'
        'a = np.array([0, 0, 1, 2, 3, 4, 5, 6, 6, 2])\n'
        'b = np.array([1, 3, 2, 5, 6, 0, 4, 1, 3, 4])\n'
        'def _sig(result):\n'
        '    a = np.asarray(result, dtype=float)\n'
        '    assert a.ndim in (2, 3) and a.shape[-1] == 3\n'
        '    return a.ravel()\n'
        'v = np.cross(X[b] - X[a], np.array([0.3, -0.7, 0.2]))\n'
    )
    fixture_2 = (
        'import numpy as np\n'
        'rng = np.random.default_rng(11)\n'
        'def _run(fn, *args):\n'
        '    return fn(*[v.copy() if isinstance(v, np.ndarray) else v for v in args])\n'
        'X = rng.uniform(0.0, 2.0, size=(7, 3))\n'
        'x = X + 0.1 * rng.normal(size=(7, 3))\n'
        'G = np.eye(3) + 0.08 * rng.normal(size=(7, 3, 3))\n'
        'a = np.array([0, 0, 1, 2, 3, 4, 5, 6, 6, 2])\n'
        'b = np.array([1, 3, 2, 5, 6, 0, 4, 1, 3, 4])\n'
        'def _sig(result):\n'
        '    a = np.asarray(result, dtype=float)\n'
        '    assert a.ndim in (2, 3) and a.shape[-1] == 3\n'
        '    return a.ravel()\n'
        'F0 = np.array([[1.2, -0.1, 0.0], [0.3, 0.95, 0.1], [0.0, 0.2, 1.1]])\n'
    )
    fixture_3 = (
        'import numpy as np\n'
        'rng = np.random.default_rng(11)\n'
        'def _run(fn, *args):\n'
        '    return fn(*[v.copy() if isinstance(v, np.ndarray) else v for v in args])\n'
        'X = rng.uniform(0.0, 2.0, size=(7, 3))\n'
        'x = X + 0.1 * rng.normal(size=(7, 3))\n'
        'G = np.eye(3) + 0.08 * rng.normal(size=(7, 3, 3))\n'
        'a = np.array([0, 0, 1, 2, 3, 4, 5, 6, 6, 2])\n'
        'b = np.array([1, 3, 2, 5, 6, 0, 4, 1, 3, 4])\n'
        'def _sig(result):\n'
        '    a = np.asarray(result, dtype=float)\n'
        '    assert a.ndim in (2, 3) and a.shape[-1] == 3\n'
        '    return a.ravel()\n'
        'def _status(fn):\n'
        '    try:\n'
        '        fn()\n'
        '        return 0\n'
        '    except ValueError:\n'
        '        return 1\n'
        '    except Exception:\n'
        '        return 2\n'
    )
    return [
        {
            "setup": fixture_0,
            'call': '_sig(_run(compute_bond_deformation_gradients, G, X, x, a, b))',
            'gold_call': '_sig(_run(_oracle_compute_bond_deformation_gradients, G, X, x, a, b))',
        },
        {
            "setup": fixture_1,
            'call': "_sig(np.einsum('bij,bj->bi', _run(compute_bond_deformation_gradients, G, X, x, a, b), v))",
            'gold_call': "_sig(np.einsum('bij,bj->bi', _run(_oracle_compute_bond_deformation_gradients, G, X, x, a, b), v))",
        },
        {
            "setup": fixture_2,
            'call': '_sig(_run(compute_bond_deformation_gradients, np.repeat(F0[None], 7, axis=0), X, X @ F0.T, a, b))',
            'gold_call': '_sig(_run(_oracle_compute_bond_deformation_gradients, np.repeat(F0[None], 7, axis=0), X, X @ F0.T, a, b))',
        },
        {
            "setup": fixture_0,
            'call': '_sig(_run(compute_bond_deformation_gradients, G, X, x + 2.5 * (X[:, :1] > 1.0) * np.array([0.0, 1.0, 0.5]), a, b))',
            'gold_call': '_sig(_run(_oracle_compute_bond_deformation_gradients, G, X, x + 2.5 * (X[:, :1] > 1.0) * np.array([0.0, 1.0, 0.5]), a, b))',
        },
        {
            "setup": fixture_0,
            'call': '_sig(_run(compute_bond_deformation_gradients, G[:2], X[:2], x[:2], np.array([1]), np.array([0])))',
            'gold_call': '_sig(_run(_oracle_compute_bond_deformation_gradients, G[:2], X[:2], x[:2], np.array([1]), np.array([0])))',
        },
        {
            "setup": fixture_3,
            'call': '_status(lambda: _run(compute_bond_deformation_gradients, G, X, x, np.array([2, 3]), np.array([2, 4])))',
            'gold_call': '_status(lambda: _run(_oracle_compute_bond_deformation_gradients, G, X, x, np.array([2, 3]), np.array([2, 4])))',
        },
        {
            "setup": fixture_3,
            'call': '_status(lambda: _run(compute_bond_deformation_gradients, G[:, :2], X, x, a, b))',
            'gold_call': '_status(lambda: _run(_oracle_compute_bond_deformation_gradients, G[:, :2], X, x, a, b))',
        },
    ]
