"""
Recover nodal deformation gradients by degraded reproducing-kernel fits on truncated neighborhoods, with selectable polynomial enrichment and nonuniform quadrature volumes.

The correspondence shape tensor is the linear special case of the reproducing-kernel moment matrix. Enriching the basis introduces coupled nuisance coefficients that must be eliminated when recovering the derivative at a node. Damage and the neighbor's quadrature volume enter the entire fit (Section 4.2 of the source paper).

Returns
-------
tuple[np.ndarray, np.ndarray]: the (N, 3, 3) degraded shape tensors and the (N, 3, 3) nodal deformation gradients.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def assemble_nodal_deformation_gradients(
    ref_positions: "np.ndarray",
    cur_positions: "np.ndarray",
    bond_start: "np.ndarray",
    bond_end: "np.ndarray",
    bond_weight: "np.ndarray",
    kinematic_weight: "np.ndarray",
    nodal_volume: "float | np.ndarray",
    basis: str = "C1",
) -> tuple:
    """Return physical shape tensors and derivatives of weighted local fits.

    For bond ``b`` from node ``k = bond_start[b]`` to ``n = bond_end[b]``
    let ``dX_b = X_n - X_k`` and ``dx_b = x_n - x_k`` (reference and current
    positions). Let ``V_n`` be the volume of the END node and
    ``c_b = bond_weight[b] * kinematic_weight[b] * V_n``. For every basis,
    return the physical 3x3 tensor ``K_k = sum_b c_b dX_b dX_b^T``.

    At each node, fit the three components of ``dx_b`` by polynomials of
    ``dX_b`` that minimize the weighted sum of squared residuals. ``C1``
    uses the span of ``(X, Y, Z)``; ``RK1`` uses ``(1, X, Y, Z)``; ``RK2``
    uses ``(1, X, Y, Z, X**2, Y**2, Z**2, X*Y, X*Z, Y*Z)``. Return the
    derivative of the fitted vector polynomial at ``dX = 0`` as ``F_k``;
    the constant and quadratic coefficients participate in the fit but
    are not themselves returned. Rows of ``F_k`` index current components
    and columns index reference derivatives. Rank of each node's weighted
    polynomial fit is judged on the dimensionless monomials obtained by
    dividing ``dX_b`` by that node's largest bond length. Implementations
    must rescale before the rank test. Physical-unit conditioning of the
    unscaled basis must not trigger the rank error. Returned derivatives
    must use physical units. The shape tensor is always 3x3, even for
    enriched fits.

    Parameters
    ----------
    ref_positions : np.ndarray
        ``(N, 3)`` reference positions.
    cur_positions : np.ndarray
        ``(N, 3)`` current positions.
    bond_start, bond_end : np.ndarray
        ``(N_b,)`` integer node indices of each bond.
    bond_weight : np.ndarray
        ``(N_b,)`` nonnegative kernel weights.
    kinematic_weight : np.ndarray
        ``(N_b,)`` nonnegative kinematic weights.
    nodal_volume : float or np.ndarray
        Positive common volume, or a positive finite ``(N,)`` array.
    basis : str
        One of ``"C1"``, ``"RK1"``, ``"RK2"``. Defaults to correspondence.

    Returns
    -------
    tuple
        ``(shape_tensors, nodal_gradients)``, two ``(N, 3, 3)`` arrays.

    Raises
    ------
    ValueError
        If the position arrays are not finite ``(N, 3)`` arrays of equal
        shape, if the bond arrays do not share one 1D length, if an index
        lies outside ``[0, N)``, if a weight is negative or not finite, if
        volumes have an invalid shape or nonpositive/nonfinite entries,
        ``basis`` is unsupported, or a node's weighted polynomial fit
        lacks full column rank on that dimensionless basis. Inputs must
        not be modified.
    """
    return shape_tensors, nodal_gradients

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_assemble_nodal_deformation_gradients(
    ref_positions: "np.ndarray",
    cur_positions: "np.ndarray",
    bond_start: "np.ndarray",
    bond_end: "np.ndarray",
    bond_weight: "np.ndarray",
    kinematic_weight: "np.ndarray",
    nodal_volume: "float | np.ndarray",
    basis: str = "C1",
) -> tuple:
    """Reference implementation (normal equations of the weighted fit)."""
    import numpy as np

    ref = np.asarray(ref_positions, dtype=float)
    cur = np.asarray(cur_positions, dtype=float)
    if ref.ndim != 2 or ref.shape[1] != 3 or cur.shape != ref.shape:
        raise ValueError("positions must be two (N, 3) arrays of equal shape")
    if not (np.all(np.isfinite(ref)) and np.all(np.isfinite(cur))):
        raise ValueError("positions must be finite")
    start = np.asarray(bond_start)
    end = np.asarray(bond_end)
    weight = np.asarray(bond_weight, dtype=float)
    kin = np.asarray(kinematic_weight, dtype=float)
    if start.ndim != 1 or not (start.shape == end.shape == weight.shape == kin.shape):
        raise ValueError("bond arrays must share one 1D length")
    if not (np.issubdtype(start.dtype, np.integer) and np.issubdtype(end.dtype, np.integer)):
        raise ValueError("bond indices must be integers")
    count = ref.shape[0]
    if start.size and (min(start.min(), end.min()) < 0 or max(start.max(), end.max()) >= count):
        raise ValueError("bond indices must lie in [0, N)")
    if not (np.all(np.isfinite(weight)) and np.all(np.isfinite(kin))):
        raise ValueError("weights must be finite")
    if np.any(weight < 0.0) or np.any(kin < 0.0):
        raise ValueError("weights must be nonnegative")
    volume = np.asarray(nodal_volume, dtype=float)
    if volume.ndim == 0:
        volume = np.full(count, float(volume))
    if volume.shape != (count,) or not np.all(np.isfinite(volume)) or np.any(volume <= 0.0):
        raise ValueError("nodal_volume must be positive and scalar or shape (N,)")
    if basis not in ("C1", "RK1", "RK2"):
        raise ValueError("unsupported polynomial basis")

    ref_bond = ref[end] - ref[start]
    cur_bond = cur[end] - cur[start]
    coef = weight * kin * volume[end]
    shape = np.zeros((count, 3, 3))
    moment = np.zeros((count, 3, 3))
    for a in range(3):
        for b in range(3):
            shape[:, a, b] = np.bincount(start, coef * ref_bond[:, a] * ref_bond[:, b], minlength=count)
            moment[:, a, b] = np.bincount(start, coef * cur_bond[:, a] * ref_bond[:, b], minlength=count)
    try:
        np.linalg.cholesky(shape)
    except np.linalg.LinAlgError:
        raise ValueError("every shape tensor must be positive definite") from None
    # Normal equations: F K = sum c dx dX^T, so F^T = K^{-1} (sum c dX dx^T).
    gradients = np.linalg.solve(shape, moment.transpose(0, 2, 1)).transpose(0, 2, 1)
    if basis != "C1":
        # Dimensionless monomials keep the mixed polynomial orders balanced.
        scale = np.zeros(count)
        np.maximum.at(scale, start, np.linalg.norm(ref_bond, axis=1))
        q = ref_bond / scale[start, None]
        columns = [np.ones(start.size), q[:, 0], q[:, 1], q[:, 2]]
        if basis == "RK2":
            columns.extend([q[:, 0] ** 2, q[:, 1] ** 2, q[:, 2] ** 2,
                            q[:, 0] * q[:, 1], q[:, 0] * q[:, 2], q[:, 1] * q[:, 2]])
        design = np.column_stack(columns)
        width = design.shape[1]
        moments = np.zeros((count, width, width))
        rhs = np.zeros((count, width, 3))
        for a in range(width):
            for b in range(width):
                moments[:, a, b] = np.bincount(start, coef * design[:, a] * design[:, b], minlength=count)
            for b in range(3):
                rhs[:, a, b] = np.bincount(start, coef * design[:, a] * cur_bond[:, b], minlength=count)
        if np.any(np.linalg.matrix_rank(moments) < width):
            raise ValueError("every weighted polynomial fit must have full column rank")
        coefficients = np.linalg.solve(moments, rhs)
        gradients = coefficients[:, 1:4, :].transpose(0, 2, 1) / scale[:, None, None]
    return shape, gradients

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    """Return deterministic numerical comparisons with isolated inputs."""
    fixture_0 = (
        'import itertools\n'
        'import numpy as np\n'
        'def _run(fn, *args):\n'
        '    return fn(*[v.copy() if isinstance(v, np.ndarray) else v for v in args])\n'
        'def _lattice(shape, spacing, horizon):\n'
        '    g = np.array(list(itertools.product(*[range(n) for n in shape])), dtype=float)\n'
        '    X = spacing * g\n'
        '    d = X[None, :, :] - X[:, None, :]\n'
        '    r = np.sqrt(np.sum(d * d, axis=2))\n'
        '    a, b = np.nonzero((r > 0.0) & (r <= horizon))\n'
        '    return X, a, b, (1.0 - r[a, b] / horizon) ** 2 + 0.1\n'
        'def _mats(result):\n'
        '    K, F = (np.asarray(m, dtype=float) for m in result)\n'
        '    assert K.shape == F.shape and K.ndim == 3 and K.shape[1:] == (3, 3)\n'
        '    return np.concatenate([K.ravel() / 100.0, F.ravel()])\n'
        'def _warp(X):\n'
        '    return X + 0.04 * np.stack([np.sin(X[:, 1] + 0.3 * X[:, 2]), X[:, 0] * X[:, 2] / 5.0,\n'
        '                               np.cos(0.8 * X[:, 0]) - 0.5 * X[:, 1] ** 2 / 4.0], axis=1)\n'
        'X, a, b, w = _lattice((3, 3, 3), 1.0, 1.8)\n'
        'cut = ((X[a, 1] <= 1.0) & (X[b, 1] >= 2.0)) | ((X[b, 1] <= 1.0) & (X[a, 1] >= 2.0))\n'
        'h = np.where(cut, 0.05, 1.0) * (1.0 - 0.3 * (X[b, 0] > X[a, 0]))\n'
    )
    fixture_1 = (
        'import itertools\n'
        'import numpy as np\n'
        'def _run(fn, *args):\n'
        '    return fn(*[v.copy() if isinstance(v, np.ndarray) else v for v in args])\n'
        'def _lattice(shape, spacing, horizon):\n'
        '    g = np.array(list(itertools.product(*[range(n) for n in shape])), dtype=float)\n'
        '    X = spacing * g\n'
        '    d = X[None, :, :] - X[:, None, :]\n'
        '    r = np.sqrt(np.sum(d * d, axis=2))\n'
        '    a, b = np.nonzero((r > 0.0) & (r <= horizon))\n'
        '    return X, a, b, (1.0 - r[a, b] / horizon) ** 2 + 0.1\n'
        'def _mats(result):\n'
        '    K, F = (np.asarray(m, dtype=float) for m in result)\n'
        '    assert K.shape == F.shape and K.ndim == 3 and K.shape[1:] == (3, 3)\n'
        '    return np.concatenate([K.ravel() / 100.0, F.ravel()])\n'
        'def _warp(X):\n'
        '    return X + 0.04 * np.stack([np.sin(X[:, 1] + 0.3 * X[:, 2]), X[:, 0] * X[:, 2] / 5.0,\n'
        '                               np.cos(0.8 * X[:, 0]) - 0.5 * X[:, 1] ** 2 / 4.0], axis=1)\n'
        'X, a, b, w = _lattice((3, 3, 3), 1.0, 1.8)\n'
        'cut = ((X[a, 1] <= 1.0) & (X[b, 1] >= 2.0)) | ((X[b, 1] <= 1.0) & (X[a, 1] >= 2.0))\n'
        'h = np.where(cut, 0.05, 1.0) * (1.0 - 0.3 * (X[b, 0] > X[a, 0]))\n'
        'F0 = np.array([[1.1, 0.2, -0.1], [0.05, 0.9, 0.3], [-0.2, 0.1, 1.05]])\n'
    )
    fixture_2 = (
        'import itertools\n'
        'import numpy as np\n'
        'def _run(fn, *args):\n'
        '    return fn(*[v.copy() if isinstance(v, np.ndarray) else v for v in args])\n'
        'def _lattice(shape, spacing, horizon):\n'
        '    g = np.array(list(itertools.product(*[range(n) for n in shape])), dtype=float)\n'
        '    X = spacing * g\n'
        '    d = X[None, :, :] - X[:, None, :]\n'
        '    r = np.sqrt(np.sum(d * d, axis=2))\n'
        '    a, b = np.nonzero((r > 0.0) & (r <= horizon))\n'
        '    return X, a, b, (1.0 - r[a, b] / horizon) ** 2 + 0.1\n'
        'def _mats(result):\n'
        '    K, F = (np.asarray(m, dtype=float) for m in result)\n'
        '    assert K.shape == F.shape and K.ndim == 3 and K.shape[1:] == (3, 3)\n'
        '    return np.concatenate([K.ravel() / 100.0, F.ravel()])\n'
        'def _warp(X):\n'
        '    return X + 0.04 * np.stack([np.sin(X[:, 1] + 0.3 * X[:, 2]), X[:, 0] * X[:, 2] / 5.0,\n'
        '                               np.cos(0.8 * X[:, 0]) - 0.5 * X[:, 1] ** 2 / 4.0], axis=1)\n'
        'X, a, b, w = _lattice((3, 3, 3), 1.0, 1.8)\n'
        'cut = ((X[a, 1] <= 1.0) & (X[b, 1] >= 2.0)) | ((X[b, 1] <= 1.0) & (X[a, 1] >= 2.0))\n'
        'h = np.where(cut, 0.05, 1.0) * (1.0 - 0.3 * (X[b, 0] > X[a, 0]))\n'
        'X2, a2, b2, w2 = _lattice((2, 2, 2), 0.5, 0.9)\n'
    )
    fixture_3 = (
        'import itertools\n'
        'import numpy as np\n'
        'def _run(fn, *args):\n'
        '    return fn(*[v.copy() if isinstance(v, np.ndarray) else v for v in args])\n'
        'def _lattice(shape, spacing, horizon):\n'
        '    g = np.array(list(itertools.product(*[range(n) for n in shape])), dtype=float)\n'
        '    X = spacing * g\n'
        '    d = X[None, :, :] - X[:, None, :]\n'
        '    r = np.sqrt(np.sum(d * d, axis=2))\n'
        '    a, b = np.nonzero((r > 0.0) & (r <= horizon))\n'
        '    return X, a, b, (1.0 - r[a, b] / horizon) ** 2 + 0.1\n'
        'def _mats(result):\n'
        '    K, F = (np.asarray(m, dtype=float) for m in result)\n'
        '    assert K.shape == F.shape and K.ndim == 3 and K.shape[1:] == (3, 3)\n'
        '    return np.concatenate([K.ravel() / 100.0, F.ravel()])\n'
        'def _warp(X):\n'
        '    return X + 0.04 * np.stack([np.sin(X[:, 1] + 0.3 * X[:, 2]), X[:, 0] * X[:, 2] / 5.0,\n'
        '                               np.cos(0.8 * X[:, 0]) - 0.5 * X[:, 1] ** 2 / 4.0], axis=1)\n'
        'X, a, b, w = _lattice((3, 3, 3), 1.0, 1.8)\n'
        'cut = ((X[a, 1] <= 1.0) & (X[b, 1] >= 2.0)) | ((X[b, 1] <= 1.0) & (X[a, 1] >= 2.0))\n'
        'h = np.where(cut, 0.05, 1.0) * (1.0 - 0.3 * (X[b, 0] > X[a, 0]))\n'
        'def _status(fn):\n'
        '    try:\n'
        '        fn()\n'
        '        return 0\n'
        '    except ValueError:\n'
        '        return 1\n'
        '    except Exception:\n'
        '        return 2\n'
        'X3, a3, b3, w3 = _lattice((3, 3, 1), 1.0, 1.5)\n'
    )
    fixture_4 = (
        'import itertools\n'
        'import numpy as np\n'
        'def _run(fn, *args):\n'
        '    return fn(*[v.copy() if isinstance(v, np.ndarray) else v for v in args])\n'
        'def _lattice(shape, spacing, horizon):\n'
        '    g = np.array(list(itertools.product(*[range(n) for n in shape])), dtype=float)\n'
        '    X = spacing * g\n'
        '    d = X[None, :, :] - X[:, None, :]\n'
        '    r = np.sqrt(np.sum(d * d, axis=2))\n'
        '    a, b = np.nonzero((r > 0.0) & (r <= horizon))\n'
        '    return X, a, b, (1.0 - r[a, b] / horizon) ** 2 + 0.1\n'
        'def _mats(result):\n'
        '    K, F = (np.asarray(m, dtype=float) for m in result)\n'
        '    assert K.shape == F.shape and K.ndim == 3 and K.shape[1:] == (3, 3)\n'
        '    return np.concatenate([K.ravel() / 100.0, F.ravel()])\n'
        'def _warp(X):\n'
        '    return X + 0.04 * np.stack([np.sin(X[:, 1] + 0.3 * X[:, 2]), X[:, 0] * X[:, 2] / 5.0,\n'
        '                               np.cos(0.8 * X[:, 0]) - 0.5 * X[:, 1] ** 2 / 4.0], axis=1)\n'
        'X, a, b, w = _lattice((3, 3, 3), 1.0, 1.8)\n'
        'cut = ((X[a, 1] <= 1.0) & (X[b, 1] >= 2.0)) | ((X[b, 1] <= 1.0) & (X[a, 1] >= 2.0))\n'
        'h = np.where(cut, 0.05, 1.0) * (1.0 - 0.3 * (X[b, 0] > X[a, 0]))\n'
        'def _status(fn):\n'
        '    try:\n'
        '        fn()\n'
        '        return 0\n'
        '    except ValueError:\n'
        '        return 1\n'
        '    except Exception:\n'
        '        return 2\n'
    )
    fixture_5 = (
        'import numpy as np\n'
        'g = np.indices((5, 4, 4)).reshape(3, -1).T.astype(float)\n'
        'U = g + 0.04 * np.sin(0.7 * g + np.arange(3))\n'
        'delta = U[None, :, :] - U[:, None, :]\n'
        'r = np.linalg.norm(delta, axis=2)\n'
        'a, b = np.nonzero((r > 0) & (r <= 2.8))\n'
        'w = (1.0 - r[a, b] / 2.9) ** 2\n'
        'h = 0.15 + 0.85 / (1.0 + (U[b, 0] - U[a, 0]) ** 2 + 0.7 * (U[a, 1] > 1.5))\n'
        'V = 0.3 + 0.013 * (np.arange(U.shape[0]) % 23)\n'
        'def warp(U):\n'
        '    x, y, z = U.T\n'
        '    return U + 0.025 * np.column_stack([x*y + 0.5*z*z, y*z - 0.3*x*x, x*z + 0.2*y*y])\n'
        'def derivative(U):\n'
        '    x, y, z = U.T\n'
        '    J = np.zeros((len(U), 3, 3))\n'
        '    J[:, 0, :] = np.column_stack([y, x, z])\n'
        '    J[:, 1, :] = np.column_stack([-0.6*x, z, y])\n'
        '    J[:, 2, :] = np.column_stack([z, 0.4*y, x])\n'
        '    return np.eye(3) + 0.025 * J\n'
        'def pack(result, length):\n'
        '    K, F = map(np.asarray, result)\n'
        '    assert K.shape == F.shape == (len(U), 3, 3)\n'
        '    return np.concatenate([K.ravel() / length**2, F.ravel()])\n'
        'def quadratic_check(result, length):\n'
        '    K, F = result\n'
        '    assert np.asarray(K).shape == np.asarray(F).shape == (len(U), 3, 3)\n'
        '    np.testing.assert_allclose(F, derivative(U), atol=2e-9, rtol=2e-9)\n'
        '    return pack(result, length)\n'
        'def safe(fn, *args):\n'
        '    copied = [x.copy() if isinstance(x, np.ndarray) else x for x in args]\n'
        '    before = [x.copy() if isinstance(x, np.ndarray) else x for x in copied]\n'
        '    result = fn(*copied)\n'
        '    for x, y in zip(copied, before):\n'
        '        if isinstance(x, np.ndarray):\n'
        '            np.testing.assert_array_equal(x, y)\n'
        '    return result\n'
    )
    fixture_6 = (
        'import numpy as np\n'
        'g = np.indices((5, 4, 4)).reshape(3, -1).T.astype(float)\n'
        'U = g + 0.04 * np.sin(0.7 * g + np.arange(3))\n'
        'delta = U[None, :, :] - U[:, None, :]\n'
        'r = np.linalg.norm(delta, axis=2)\n'
        'a, b = np.nonzero((r > 0) & (r <= 2.8))\n'
        'w = (1.0 - r[a, b] / 2.9) ** 2\n'
        'h = 0.15 + 0.85 / (1.0 + (U[b, 0] - U[a, 0]) ** 2 + 0.7 * (U[a, 1] > 1.5))\n'
        'V = 0.3 + 0.013 * (np.arange(U.shape[0]) % 23)\n'
        'def warp(U):\n'
        '    x, y, z = U.T\n'
        '    return U + 0.025 * np.column_stack([x*y + 0.5*z*z, y*z - 0.3*x*x, x*z + 0.2*y*y])\n'
        'def derivative(U):\n'
        '    x, y, z = U.T\n'
        '    J = np.zeros((len(U), 3, 3))\n'
        '    J[:, 0, :] = np.column_stack([y, x, z])\n'
        '    J[:, 1, :] = np.column_stack([-0.6*x, z, y])\n'
        '    J[:, 2, :] = np.column_stack([z, 0.4*y, x])\n'
        '    return np.eye(3) + 0.025 * J\n'
        'def pack(result, length):\n'
        '    K, F = map(np.asarray, result)\n'
        '    assert K.shape == F.shape == (len(U), 3, 3)\n'
        '    return np.concatenate([K.ravel() / length**2, F.ravel()])\n'
        'def quadratic_check(result, length):\n'
        '    K, F = result\n'
        '    assert np.asarray(K).shape == np.asarray(F).shape == (len(U), 3, 3)\n'
        '    np.testing.assert_allclose(F, derivative(U), atol=2e-9, rtol=2e-9)\n'
        '    return pack(result, length)\n'
        'def safe(fn, *args):\n'
        '    copied = [x.copy() if isinstance(x, np.ndarray) else x for x in args]\n'
        '    before = [x.copy() if isinstance(x, np.ndarray) else x for x in copied]\n'
        '    result = fn(*copied)\n'
        '    for x, y in zip(copied, before):\n'
        '        if isinstance(x, np.ndarray):\n'
        '            np.testing.assert_array_equal(x, y)\n'
        '    return result\n'
        'L = 1.0\n'
    )
    fixture_7 = (
        'import numpy as np\n'
        'g = np.indices((5, 4, 4)).reshape(3, -1).T.astype(float)\n'
        'U = g + 0.04 * np.sin(0.7 * g + np.arange(3))\n'
        'delta = U[None, :, :] - U[:, None, :]\n'
        'r = np.linalg.norm(delta, axis=2)\n'
        'a, b = np.nonzero((r > 0) & (r <= 2.8))\n'
        'w = (1.0 - r[a, b] / 2.9) ** 2\n'
        'h = 0.15 + 0.85 / (1.0 + (U[b, 0] - U[a, 0]) ** 2 + 0.7 * (U[a, 1] > 1.5))\n'
        'V = 0.3 + 0.013 * (np.arange(U.shape[0]) % 23)\n'
        'def warp(U):\n'
        '    x, y, z = U.T\n'
        '    return U + 0.025 * np.column_stack([x*y + 0.5*z*z, y*z - 0.3*x*x, x*z + 0.2*y*y])\n'
        'def derivative(U):\n'
        '    x, y, z = U.T\n'
        '    J = np.zeros((len(U), 3, 3))\n'
        '    J[:, 0, :] = np.column_stack([y, x, z])\n'
        '    J[:, 1, :] = np.column_stack([-0.6*x, z, y])\n'
        '    J[:, 2, :] = np.column_stack([z, 0.4*y, x])\n'
        '    return np.eye(3) + 0.025 * J\n'
        'def pack(result, length):\n'
        '    K, F = map(np.asarray, result)\n'
        '    assert K.shape == F.shape == (len(U), 3, 3)\n'
        '    return np.concatenate([K.ravel() / length**2, F.ravel()])\n'
        'def quadratic_check(result, length):\n'
        '    K, F = result\n'
        '    assert np.asarray(K).shape == np.asarray(F).shape == (len(U), 3, 3)\n'
        '    np.testing.assert_allclose(F, derivative(U), atol=2e-9, rtol=2e-9)\n'
        '    return pack(result, length)\n'
        'def safe(fn, *args):\n'
        '    copied = [x.copy() if isinstance(x, np.ndarray) else x for x in args]\n'
        '    before = [x.copy() if isinstance(x, np.ndarray) else x for x in copied]\n'
        '    result = fn(*copied)\n'
        '    for x, y in zip(copied, before):\n'
        '        if isinstance(x, np.ndarray):\n'
        '            np.testing.assert_array_equal(x, y)\n'
        '    return result\n'
        'L = 0.0004\n'
    )
    fixture_8 = (
        'import numpy as np\n'
        'g = np.indices((5, 4, 4)).reshape(3, -1).T.astype(float)\n'
        'U = g + 0.04 * np.sin(0.7 * g + np.arange(3))\n'
        'delta = U[None, :, :] - U[:, None, :]\n'
        'r = np.linalg.norm(delta, axis=2)\n'
        'a, b = np.nonzero((r > 0) & (r <= 2.8))\n'
        'w = (1.0 - r[a, b] / 2.9) ** 2\n'
        'h = 0.15 + 0.85 / (1.0 + (U[b, 0] - U[a, 0]) ** 2 + 0.7 * (U[a, 1] > 1.5))\n'
        'V = 0.3 + 0.013 * (np.arange(U.shape[0]) % 23)\n'
        'def warp(U):\n'
        '    x, y, z = U.T\n'
        '    return U + 0.025 * np.column_stack([x*y + 0.5*z*z, y*z - 0.3*x*x, x*z + 0.2*y*y])\n'
        'def derivative(U):\n'
        '    x, y, z = U.T\n'
        '    J = np.zeros((len(U), 3, 3))\n'
        '    J[:, 0, :] = np.column_stack([y, x, z])\n'
        '    J[:, 1, :] = np.column_stack([-0.6*x, z, y])\n'
        '    J[:, 2, :] = np.column_stack([z, 0.4*y, x])\n'
        '    return np.eye(3) + 0.025 * J\n'
        'def pack(result, length):\n'
        '    K, F = map(np.asarray, result)\n'
        '    assert K.shape == F.shape == (len(U), 3, 3)\n'
        '    return np.concatenate([K.ravel() / length**2, F.ravel()])\n'
        'def quadratic_check(result, length):\n'
        '    K, F = result\n'
        '    assert np.asarray(K).shape == np.asarray(F).shape == (len(U), 3, 3)\n'
        '    np.testing.assert_allclose(F, derivative(U), atol=2e-9, rtol=2e-9)\n'
        '    return pack(result, length)\n'
        'def safe(fn, *args):\n'
        '    copied = [x.copy() if isinstance(x, np.ndarray) else x for x in args]\n'
        '    before = [x.copy() if isinstance(x, np.ndarray) else x for x in copied]\n'
        '    result = fn(*copied)\n'
        '    for x, y in zip(copied, before):\n'
        '        if isinstance(x, np.ndarray):\n'
        '            np.testing.assert_array_equal(x, y)\n'
        '    return result\n'
        'L = 80.0\n'
    )
    fixture_9 = (
        'import numpy as np\n'
        'g = np.indices((5, 4, 4)).reshape(3, -1).T.astype(float)\n'
        'U = g + 0.04 * np.sin(0.7 * g + np.arange(3))\n'
        'delta = U[None, :, :] - U[:, None, :]\n'
        'r = np.linalg.norm(delta, axis=2)\n'
        'a, b = np.nonzero((r > 0) & (r <= 2.8))\n'
        'w = (1.0 - r[a, b] / 2.9) ** 2\n'
        'h = 0.15 + 0.85 / (1.0 + (U[b, 0] - U[a, 0]) ** 2 + 0.7 * (U[a, 1] > 1.5))\n'
        'V = 0.3 + 0.013 * (np.arange(U.shape[0]) % 23)\n'
        'def warp(U):\n'
        '    x, y, z = U.T\n'
        '    return U + 0.025 * np.column_stack([x*y + 0.5*z*z, y*z - 0.3*x*x, x*z + 0.2*y*y])\n'
        'def derivative(U):\n'
        '    x, y, z = U.T\n'
        '    J = np.zeros((len(U), 3, 3))\n'
        '    J[:, 0, :] = np.column_stack([y, x, z])\n'
        '    J[:, 1, :] = np.column_stack([-0.6*x, z, y])\n'
        '    J[:, 2, :] = np.column_stack([z, 0.4*y, x])\n'
        '    return np.eye(3) + 0.025 * J\n'
        'def pack(result, length):\n'
        '    K, F = map(np.asarray, result)\n'
        '    assert K.shape == F.shape == (len(U), 3, 3)\n'
        '    return np.concatenate([K.ravel() / length**2, F.ravel()])\n'
        'def quadratic_check(result, length):\n'
        '    K, F = result\n'
        '    assert np.asarray(K).shape == np.asarray(F).shape == (len(U), 3, 3)\n'
        '    np.testing.assert_allclose(F, derivative(U), atol=2e-9, rtol=2e-9)\n'
        '    return pack(result, length)\n'
        'def safe(fn, *args):\n'
        '    copied = [x.copy() if isinstance(x, np.ndarray) else x for x in args]\n'
        '    before = [x.copy() if isinstance(x, np.ndarray) else x for x in copied]\n'
        '    result = fn(*copied)\n'
        '    for x, y in zip(copied, before):\n'
        '        if isinstance(x, np.ndarray):\n'
        '            np.testing.assert_array_equal(x, y)\n'
        '    return result\n'
        'p = np.random.default_rng(27).permutation(a.size)\n'
    )
    fixture_10 = (
        'import itertools\n'
        'import numpy as np\n'
        'def _run(fn, *args):\n'
        '    return fn(*[v.copy() if isinstance(v, np.ndarray) else v for v in args])\n'
        'def _lattice(shape, spacing, horizon):\n'
        '    g = np.array(list(itertools.product(*[range(n) for n in shape])), dtype=float)\n'
        '    X = spacing * g\n'
        '    d = X[None, :, :] - X[:, None, :]\n'
        '    r = np.sqrt(np.sum(d * d, axis=2))\n'
        '    a, b = np.nonzero((r > 0.0) & (r <= horizon))\n'
        '    return X, a, b, (1.0 - r[a, b] / horizon) ** 2 + 0.1\n'
        'def _mats(result):\n'
        '    K, F = (np.asarray(m, dtype=float) for m in result)\n'
        '    assert K.shape == F.shape and K.ndim == 3 and K.shape[1:] == (3, 3)\n'
        '    return np.concatenate([K.ravel() / 100.0, F.ravel()])\n'
        'def _warp(X):\n'
        '    return X + 0.04 * np.stack([np.sin(X[:, 1] + 0.3 * X[:, 2]), X[:, 0] * X[:, 2] / 5.0,\n'
        '                               np.cos(0.8 * X[:, 0]) - 0.5 * X[:, 1] ** 2 / 4.0], axis=1)\n'
        'X, a, b, w = _lattice((3, 3, 3), 1.0, 1.8)\n'
        'cut = ((X[a, 1] <= 1.0) & (X[b, 1] >= 2.0)) | ((X[b, 1] <= 1.0) & (X[a, 1] >= 2.0))\n'
        'h = np.where(cut, 0.05, 1.0) * (1.0 - 0.3 * (X[b, 0] > X[a, 0]))\n'
        'def _status(fn):\n'
        '    try:\n'
        '        fn()\n'
        '        return 0\n'
        '    except ValueError:\n'
        '        return 1\n'
        '    except Exception:\n'
        '        return 2\n'
        'X2, a2, b2, w2 = _lattice((2, 2, 2), 1.0, 1.01)\n'
    )
    fixture_11 = (
        'import itertools\n'
        'import numpy as np\n'
        'def _run(fn, *args):\n'
        '    return fn(*[v.copy() if isinstance(v, np.ndarray) else v for v in args])\n'
        'def _lattice(shape, spacing, horizon):\n'
        '    g = np.array(list(itertools.product(*[range(n) for n in shape])), dtype=float)\n'
        '    X = spacing * g\n'
        '    d = X[None, :, :] - X[:, None, :]\n'
        '    r = np.sqrt(np.sum(d * d, axis=2))\n'
        '    a, b = np.nonzero((r > 0.0) & (r <= horizon))\n'
        '    return X, a, b, (1.0 - r[a, b] / horizon) ** 2 + 0.1\n'
        'def _mats(result):\n'
        '    K, F = (np.asarray(m, dtype=float) for m in result)\n'
        '    assert K.shape == F.shape and K.ndim == 3 and K.shape[1:] == (3, 3)\n'
        '    return np.concatenate([K.ravel() / 100.0, F.ravel()])\n'
        'def _warp(X):\n'
        '    return X + 0.04 * np.stack([np.sin(X[:, 1] + 0.3 * X[:, 2]), X[:, 0] * X[:, 2] / 5.0,\n'
        '                               np.cos(0.8 * X[:, 0]) - 0.5 * X[:, 1] ** 2 / 4.0], axis=1)\n'
        'X, a, b, w = _lattice((3, 3, 3), 1.0, 1.8)\n'
        'cut = ((X[a, 1] <= 1.0) & (X[b, 1] >= 2.0)) | ((X[b, 1] <= 1.0) & (X[a, 1] >= 2.0))\n'
        'h = np.where(cut, 0.05, 1.0) * (1.0 - 0.3 * (X[b, 0] > X[a, 0]))\n'
        'def _status(fn):\n'
        '    try:\n'
        '        fn()\n'
        '        return 0\n'
        '    except ValueError:\n'
        '        return 1\n'
        '    except Exception:\n'
        '        return 2\n'
        'X2, a2, b2, w2 = _lattice((2, 2, 2), 1.0, 1.8)\n'
    )
    return [
        {
            "setup": fixture_0,
            'call': '_mats(_run(assemble_nodal_deformation_gradients, X, _warp(X), a, b, w, np.ones(a.size), 1.0))',
            'gold_call': '_mats(_run(_oracle_assemble_nodal_deformation_gradients, X, _warp(X), a, b, w, np.ones(a.size), 1.0))',
        },
        {
            "setup": fixture_0,
            'call': '_mats(_run(assemble_nodal_deformation_gradients, X, _warp(X) + 0.3 * (X[:, 1:2] >= 2.0) * np.array([0.5, 1.0, 0.0]), a, b, w, h, 0.7))',
            'gold_call': '_mats(_run(_oracle_assemble_nodal_deformation_gradients, X, _warp(X) + 0.3 * (X[:, 1:2] >= 2.0) * np.array([0.5, 1.0, 0.0]), a, b, w, h, 0.7))',
        },
        {
            "setup": fixture_1,
            'call': 'float(np.max(np.abs(_run(assemble_nodal_deformation_gradients, X, X @ F0.T + 0.4, a, b, w, h, 1.3)[1] - F0)))',
            'gold_call': 'float(np.max(np.abs(_run(_oracle_assemble_nodal_deformation_gradients, X, X @ F0.T + 0.4, a, b, w, h, 1.3)[1] - F0)))',
        },
        {
            "setup": fixture_2,
            'call': '_mats(_run(assemble_nodal_deformation_gradients, X2, _warp(X2), a2, b2, w2, np.linspace(0.2, 1.0, a2.size), 2.0))',
            'gold_call': '_mats(_run(_oracle_assemble_nodal_deformation_gradients, X2, _warp(X2), a2, b2, w2, np.linspace(0.2, 1.0, a2.size), 2.0))',
        },
        {
            "setup": fixture_3,
            'call': '_status(lambda: _run(assemble_nodal_deformation_gradients, X3, X3, a3, b3, w3, np.ones(a3.size), 1.0))',
            'gold_call': '_status(lambda: _run(_oracle_assemble_nodal_deformation_gradients, X3, X3, a3, b3, w3, np.ones(a3.size), 1.0))',
        },
        {
            "setup": fixture_4,
            'call': '_status(lambda: _run(assemble_nodal_deformation_gradients, X, X, a, b, w, -h, 1.0))',
            'gold_call': '_status(lambda: _run(_oracle_assemble_nodal_deformation_gradients, X, X, a, b, w, -h, 1.0))',
        },
        {
            "setup": fixture_4,
            'call': '_status(lambda: _run(assemble_nodal_deformation_gradients, X, X, a, b + 1, w, h, 1.0))',
            'gold_call': '_status(lambda: _run(_oracle_assemble_nodal_deformation_gradients, X, X, a, b + 1, w, h, 1.0))',
        },
        {
            "setup": fixture_5,
            'call': "pack(safe(assemble_nodal_deformation_gradients, U, warp(U) + 0.007 * np.sin(2.1 * U), a, b, w, h, V, 'C1'), 1.0)",
            'gold_call': "pack(safe(_oracle_assemble_nodal_deformation_gradients, U, warp(U) + 0.007 * np.sin(2.1 * U), a, b, w, h, V, 'C1'), 1.0)",
            'tol': 1e-08,
        },
        {
            "setup": fixture_5,
            'call': "pack(safe(assemble_nodal_deformation_gradients, U, warp(U) + 0.007 * np.sin(2.1 * U), a, b, w, h, V, 'RK1'), 1.0)",
            'gold_call': "pack(safe(_oracle_assemble_nodal_deformation_gradients, U, warp(U) + 0.007 * np.sin(2.1 * U), a, b, w, h, V, 'RK1'), 1.0)",
            'tol': 1e-08,
        },
        {
            "setup": fixture_5,
            'call': "pack(safe(assemble_nodal_deformation_gradients, U, warp(U) + 0.007 * np.sin(2.1 * U), a, b, w, h, V, 'RK2'), 1.0)",
            'gold_call': "pack(safe(_oracle_assemble_nodal_deformation_gradients, U, warp(U) + 0.007 * np.sin(2.1 * U), a, b, w, h, V, 'RK2'), 1.0)",
            'tol': 1e-08,
        },
        {
            "setup": fixture_6,
            'call': "quadratic_check(safe(assemble_nodal_deformation_gradients, L*U + 2*L, L*warp(U) + np.array([3, -2, 1])*L, a, b, w, h, V, 'RK2'), L)",
            'gold_call': "quadratic_check(safe(_oracle_assemble_nodal_deformation_gradients, L*U + 2*L, L*warp(U) + np.array([3, -2, 1])*L, a, b, w, h, V, 'RK2'), L)",
            'tol': 1e-08,
        },
        {
            "setup": fixture_7,
            'call': "quadratic_check(safe(assemble_nodal_deformation_gradients, L*U + 2*L, L*warp(U) + np.array([3, -2, 1])*L, a, b, w, h, V, 'RK2'), L)",
            'gold_call': "quadratic_check(safe(_oracle_assemble_nodal_deformation_gradients, L*U + 2*L, L*warp(U) + np.array([3, -2, 1])*L, a, b, w, h, V, 'RK2'), L)",
            'tol': 1e-08,
        },
        {
            "setup": fixture_8,
            'call': "quadratic_check(safe(assemble_nodal_deformation_gradients, L*U + 2*L, L*warp(U) + np.array([3, -2, 1])*L, a, b, w, h, V, 'RK2'), L)",
            'gold_call': "quadratic_check(safe(_oracle_assemble_nodal_deformation_gradients, L*U + 2*L, L*warp(U) + np.array([3, -2, 1])*L, a, b, w, h, V, 'RK2'), L)",
            'tol': 1e-08,
        },
        {
            "setup": fixture_9,
            'call': "pack(safe(assemble_nodal_deformation_gradients, U, warp(U), a[p], b[p], w[p], h[p], V, 'RK2'), 1.0)",
            'gold_call': "pack(safe(_oracle_assemble_nodal_deformation_gradients, U, warp(U), a[p], b[p], w[p], h[p], V, 'RK2'), 1.0)",
            'tol': 1e-08,
        },
        {
            "setup": fixture_5,
            'call': "pack(safe(assemble_nodal_deformation_gradients, U, U @ np.array([[1.1, 0.2, -0.1], [0.1, 0.9, 0.3], [0.2, 0.0, 1.05]]).T + 0.8, a, b, w, h, 0.7, 'RK1'), 1.0)",
            'gold_call': "pack(safe(_oracle_assemble_nodal_deformation_gradients, U, U @ np.array([[1.1, 0.2, -0.1], [0.1, 0.9, 0.3], [0.2, 0.0, 1.05]]).T + 0.8, a, b, w, h, 0.7, 'RK1'), 1.0)",
            'tol': 1e-08,
        },
        {
            "setup": fixture_10,
            'call': "_status(lambda: _run(assemble_nodal_deformation_gradients, X2, X2, a2, b2, w2, np.ones(a2.size), 1.0, 'RK1'))",
            'gold_call': "_status(lambda: _run(_oracle_assemble_nodal_deformation_gradients, X2, X2, a2, b2, w2, np.ones(a2.size), 1.0, 'RK1'))",
        },
        {
            "setup": fixture_11,
            'call': "_status(lambda: _run(assemble_nodal_deformation_gradients, X2, X2, a2, b2, w2, np.ones(a2.size), 1.0, 'RK2'))",
            'gold_call': "_status(lambda: _run(_oracle_assemble_nodal_deformation_gradients, X2, X2, a2, b2, w2, np.ones(a2.size), 1.0, 'RK2'))",
        },
    ]
