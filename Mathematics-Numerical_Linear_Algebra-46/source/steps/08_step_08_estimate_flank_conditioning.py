"""
Compose every earlier step to obtain the spectral condition number of the kinematically degraded shape tensor at a probe node after a separation history.

The condition number of a node's shape tensor measures how stably its nonlocal deformation gradient can still be recovered once the bonds that cross a crack have lost kinematic weight.

Returns
-------
float: the 2-norm condition number of the probe node's kinematically degraded shape tensor after the last state.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def estimate_flank_conditioning(
    grid_shape: tuple = (10, 8, 5),
    spacing: float = 4.0e-4,
    horizon: float = 1.206e-3,
    profile_breaks: tuple = (0.0, 0.5, 1.0),
    profile_coeffs: tuple = ((1.0, 0.0, -6.0, 6.0), (2.0, -6.0, 6.0, -2.0)),
    youngs_modulus: float = 3.2e10,
    poisson_ratio: float = 0.25,
    fracture_energy: float = 3.0,
    kinematic_threshold: float = 0.95,
    separation_layer: int = 3,
    slip_angle: float = 60.0,
    openings: "tuple | np.ndarray" = (8.0e-7, 1.6e-6, 2.4e-6, 1.0e-6),
    probe_node: tuple = (5, 3, 0),
    basis: str = "C1",
    volume_factors: "np.ndarray | None" = None,
    initial_history: "np.ndarray | None" = None,
) -> float:
    """Return the spectral condition number of the probe node's degraded shape tensor.

    The separation history is run as in ``run_separation_history``. The
    probe node's shape tensor is then assembled, as in
    ``assemble_nodal_deformation_gradients`` with neighbor volumes
    ``spacing**3 * volume_factors`` and the selected ``basis``, from the
    kinematic weights left by the last state. The requested tensor is the
    physical 3x3 shape tensor for every basis, not the enriched moment
    matrix. Return
    its 2-norm condition number (largest over smallest eigenvalue). The
    defaults reproduce the problem statement.

    Parameters
    ----------
    grid_shape : tuple
        Lattice size ``(N_x, N_y, N_z)``.
    spacing, horizon : float
        Lattice spacing and horizon radius.
    profile_breaks, profile_coeffs : tuple
        Piecewise-polynomial kernel profile, as in ``build_bond_family``.
    youngs_modulus, poisson_ratio : float
        Saint Venant-Kirchhoff elastic constants.
    fracture_energy : float
        Griffith energy release rate ``G_c``.
    kinematic_threshold : float
        Threshold phase field ``s_c`` in ``[0, 1)``.
    separation_layer : int
        Last layer index ``j`` below the separation plane.
    slip_angle : float
        Angle of the separation direction from the x axis, in degrees.
    openings : tuple
        Nonempty nonnegative ``(T,)`` magnitudes or signed ``(T, 3)``
        absolute relative translations in the local slip frame, as in
        ``run_separation_history``.
    probe_node : tuple
        Lattice indices ``(i, j, k)`` of the probe node.
    basis : str
        ``"C1"``, ``"RK1"``, or ``"RK2"`` for the nodal reconstruction.
    volume_factors : np.ndarray or None
        Positive finite ``(N,)`` quadrature multipliers, defaulting to one.
    initial_history : np.ndarray or None
        Initial bond histories in directed bond order, defaulting to zero.
        See ``run_separation_history``. Inputs must not be modified.

    Returns
    -------
    float
        The condition number of the probe node's degraded shape tensor.

    Raises
    ------
    ValueError
        If ``probe_node`` is not three integers inside ``grid_shape``, or if
        any earlier step rejects its input.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_estimate_flank_conditioning(
    grid_shape: tuple = (10, 8, 5),
    spacing: float = 4.0e-4,
    horizon: float = 1.206e-3,
    profile_breaks: tuple = (0.0, 0.5, 1.0),
    profile_coeffs: tuple = ((1.0, 0.0, -6.0, 6.0), (2.0, -6.0, 6.0, -2.0)),
    youngs_modulus: float = 3.2e10,
    poisson_ratio: float = 0.25,
    fracture_energy: float = 3.0,
    kinematic_threshold: float = 0.95,
    separation_layer: int = 3,
    slip_angle: float = 60.0,
    openings: "tuple | np.ndarray" = (8.0e-7, 1.6e-6, 2.4e-6, 1.0e-6),
    probe_node: tuple = (5, 3, 0),
    basis: str = "C1",
    volume_factors: "np.ndarray | None" = None,
    initial_history: "np.ndarray | None" = None,
) -> float:
    """Reference orchestrator using only the reference function chain."""
    import numpy as np

    def _is_integer(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

    _, _, kinematic = _oracle_run_separation_history(
        grid_shape, spacing, horizon, profile_breaks, profile_coeffs,
        youngs_modulus, poisson_ratio, fracture_energy, kinematic_threshold,
        separation_layer, slip_angle, openings, basis, volume_factors, initial_history,
    )
    positions, start, end, weight = _oracle_build_bond_family(
        grid_shape, spacing, horizon, profile_breaks, profile_coeffs
    )
    try:
        probe = tuple(probe_node)
    except TypeError:
        raise ValueError("probe_node must hold three integers") from None
    shape = tuple(int(n) for n in grid_shape)
    if len(probe) != 3 or not all(_is_integer(p) and 0 <= p < n for p, n in zip(probe, shape)):
        raise ValueError("probe_node must hold three integers inside grid_shape")
    index = (probe[0] * shape[1] + probe[1]) * shape[2] + probe[2]

    # Only the shape tensors are needed, so the current configuration is
    # taken equal to the reference one.
    tensors, _ = _oracle_assemble_nodal_deformation_gradients(
        positions, positions, start, end, weight, kinematic,
        float(spacing) ** 3 * (np.ones(positions.shape[0]) if volume_factors is None else np.asarray(volume_factors)),
        basis,
    )
    tensor = 0.5 * (tensors[index] + tensors[index].T)
    eigenvalues = np.linalg.eigvalsh(tensor)
    if not eigenvalues[0] > 0.0:
        raise ValueError("the probe shape tensor must be positive definite")
    return float(eigenvalues[-1] / eigenvalues[0])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    """Return deterministic numerical comparisons with isolated inputs."""
    fixture_0 = (
        'import numpy as np\n'
        'cubic = ((0.0, 0.5, 1.0), ((1.0, 0.0, -6.0, 6.0), (2.0, -6.0, 6.0, -2.0)))\n'
    )
    fixture_1 = (
        'import numpy as np\n'
        'cubic = ((0.0, 0.5, 1.0), ((1.0, 0.0, -6.0, 6.0), (2.0, -6.0, 6.0, -2.0)))\n'
        'def _status(fn):\n'
        '    try:\n'
        '        fn()\n'
        '        return 0\n'
        '    except ValueError:\n'
        '        return 1\n'
        '    except Exception:\n'
        '        return 2\n'
    )
    fixture_2 = (
        'import numpy as np\n'
        'breaks = np.array([0., .5, 1.])\n'
        'coeffs = np.array([[1., 0., -6., 6.], [2., -6., 6., -2.]])\n'
        'shape = (5, 4, 4)\n'
        'g = np.indices(shape).reshape(3, -1).T\n'
        'r = np.linalg.norm(g[None, :, :] - g[:, None, :], axis=2)\n'
        'a, b = np.nonzero((r > 0) & (r <= 3.015))\n'
        'volumes = 0.7 + 0.017 * ((3*g[:, 0] + 5*g[:, 1] + 7*g[:, 2]) % 31)\n'
        'H0 = 2e-5 * (1.0 + (np.arange(a.size) % 19))\n'
        'path = np.array([[.018, .006, -.004], [.04, -.012, .009], [-.015, .025, .006], [.005, -.005, -.012]])\n'
        'def pack(result):\n'
        '    H, s, h = map(np.asarray, result)\n'
        '    assert H.shape == s.shape == h.shape == (len(a),)\n'
        '    return np.concatenate([H / 0.01, s, h])\n'
        'def run(fn, basis, weighted, restart, steps):\n'
        '    vf = volumes.copy() if weighted else None\n'
        '    initial = H0.copy() if restart else None\n'
        '    loads = np.asarray(steps).copy()\n'
        '    result = fn(shape, 1., 3.015, breaks.copy(), coeffs.copy(), 1.7, .27, 4.e-5, .88, 1, 37., loads, basis, vf, initial)\n'
        '    np.testing.assert_array_equal(loads, steps)\n'
        '    if weighted:\n'
        '        np.testing.assert_array_equal(vf, volumes)\n'
        '    if restart:\n'
        '        np.testing.assert_array_equal(initial, H0)\n'
        '    return pack(result)\n'
    )
    return [
        {
            "setup": fixture_0,
            'call': 'estimate_flank_conditioning()',
            'gold_call': '_oracle_estimate_flank_conditioning()',
        },
        {
            "setup": fixture_0,
            'call': 'estimate_flank_conditioning((7, 6, 4), 1.0, 2.015, *cubic, 1.0, 0.25, 3.0e-5, 0.95, 2, 45.0, (0.02, 0.05, 0.03), (3, 2, 0))',
            'gold_call': '_oracle_estimate_flank_conditioning((7, 6, 4), 1.0, 2.015, *cubic, 1.0, 0.25, 3.0e-5, 0.95, 2, 45.0, (0.02, 0.05, 0.03), (3, 2, 0))',
        },
        {
            "setup": fixture_0,
            'call': 'estimate_flank_conditioning((7, 6, 4), 1.0, 2.015, *cubic, 1.0, 0.25, 3.0e-5, 0.0, 2, 45.0, (0.02, 0.05, 0.03), (3, 3, 1))',
            'gold_call': '_oracle_estimate_flank_conditioning((7, 6, 4), 1.0, 2.015, *cubic, 1.0, 0.25, 3.0e-5, 0.0, 2, 45.0, (0.02, 0.05, 0.03), (3, 3, 1))',
        },
        {
            "setup": fixture_0,
            'call': 'estimate_flank_conditioning((6, 6, 3), 1.0, 2.015, *cubic, 1.0, 0.25, 3.0e-5, 0.95, 2, 90.0, (0.0,), (0, 2, 0))',
            'gold_call': '_oracle_estimate_flank_conditioning((6, 6, 3), 1.0, 2.015, *cubic, 1.0, 0.25, 3.0e-5, 0.95, 2, 90.0, (0.0,), (0, 2, 0))',
        },
        {
            "setup": fixture_0,
            'call': 'estimate_flank_conditioning((7, 6, 4), 1.0, 2.015, *cubic, 1.0, 0.25, 3.0e-5, 0.95, 2, 70.0, (0.03, 0.06), (4, 2, 0))',
            'gold_call': '_oracle_estimate_flank_conditioning((7, 6, 4), 1.0, 2.015, *cubic, 1.0, 0.25, 3.0e-5, 0.95, 2, 70.0, (0.03, 0.06), (4, 2, 0))',
        },
        {
            "setup": fixture_1,
            'call': '_status(lambda: estimate_flank_conditioning((7, 6, 4), 1.0, 2.015, *cubic, 1.0, 0.25, 3.0e-5, 0.95, 2, 45.0, (0.02,), (3, 6, 0)))',
            'gold_call': '_status(lambda: _oracle_estimate_flank_conditioning((7, 6, 4), 1.0, 2.015, *cubic, 1.0, 0.25, 3.0e-5, 0.95, 2, 45.0, (0.02,), (3, 6, 0)))',
        },
        {
            "setup": fixture_1,
            'call': '_status(lambda: estimate_flank_conditioning((7, 6, 4), 1.0, 2.015, *cubic, 1.0, 0.25, -1.0, 0.95, 2, 45.0, (0.02,), (3, 2, 0)))',
            'gold_call': '_status(lambda: _oracle_estimate_flank_conditioning((7, 6, 4), 1.0, 2.015, *cubic, 1.0, 0.25, -1.0, 0.95, 2, 45.0, (0.02,), (3, 2, 0)))',
        },
        {
            "setup": fixture_2,
            'call': "estimate_flank_conditioning(shape, 1., 3.015, breaks.copy(), coeffs.copy(), 1.7, .27, 4.e-5, .88, 1, 37., path.copy(), (2, 1, 0), 'C1', volumes.copy(), None)",
            'gold_call': "_oracle_estimate_flank_conditioning(shape, 1., 3.015, breaks.copy(), coeffs.copy(), 1.7, .27, 4.e-5, .88, 1, 37., path.copy(), (2, 1, 0), 'C1', volumes.copy(), None)",
            'tol': 3e-08,
        },
        {
            "setup": fixture_2,
            'call': "estimate_flank_conditioning(shape, 1., 3.015, breaks.copy(), coeffs.copy(), 1.7, .27, 4.e-5, .88, 1, 37., path.copy(), (0, 1, 0), 'RK1', None, None)",
            'gold_call': "_oracle_estimate_flank_conditioning(shape, 1., 3.015, breaks.copy(), coeffs.copy(), 1.7, .27, 4.e-5, .88, 1, 37., path.copy(), (0, 1, 0), 'RK1', None, None)",
            'tol': 3e-08,
        },
        {
            "setup": fixture_2,
            'call': "estimate_flank_conditioning(shape, 1., 3.015, breaks.copy(), coeffs.copy(), 1.7, .27, 4.e-5, .88, 1, 37., path.copy(), (2, 1, 0), 'RK2', volumes.copy(), None)",
            'gold_call': "_oracle_estimate_flank_conditioning(shape, 1., 3.015, breaks.copy(), coeffs.copy(), 1.7, .27, 4.e-5, .88, 1, 37., path.copy(), (2, 1, 0), 'RK2', volumes.copy(), None)",
            'tol': 3e-08,
        },
        {
            "setup": fixture_2,
            'call': "estimate_flank_conditioning(shape, 1., 3.015, breaks.copy(), coeffs.copy(), 1.7, .27, 4.e-5, .88, 1, 37., path.copy(), (2, 1, 0), 'RK2', volumes.copy(), H0.copy())",
            'gold_call': "_oracle_estimate_flank_conditioning(shape, 1., 3.015, breaks.copy(), coeffs.copy(), 1.7, .27, 4.e-5, .88, 1, 37., path.copy(), (2, 1, 0), 'RK2', volumes.copy(), H0.copy())",
            'tol': 3e-08,
        },
        {
            "setup": fixture_2,
            'call': "estimate_flank_conditioning(shape, 1., 3.015, breaks.copy(), coeffs.copy(), 1.7, .27, 4.e-5, .88, 1, 37., path[::-1].copy(), (2, 2, 1), 'RK1', volumes.copy(), H0.copy())",
            'gold_call': "_oracle_estimate_flank_conditioning(shape, 1., 3.015, breaks.copy(), coeffs.copy(), 1.7, .27, 4.e-5, .88, 1, 37., path[::-1].copy(), (2, 2, 1), 'RK1', volumes.copy(), H0.copy())",
            'tol': 3e-08,
        },
    ]
