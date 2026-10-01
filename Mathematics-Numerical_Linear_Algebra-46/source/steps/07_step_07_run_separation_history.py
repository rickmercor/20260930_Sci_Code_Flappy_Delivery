"""
Drive a lattice block through a sequence of rigid mixed-mode separations of its two halves and return every bond's history, phase field and kinematic weight after the last state.

In the force-evaluation algorithm the kinematics of a state use the kinematic weights left by the previous state, and only afterwards are the bond driving forces, histories and phase fields updated, so the damage reached depends on the loading path.

Returns
-------
tuple[np.ndarray, np.ndarray, np.ndarray]: the final history, bond phase field and kinematic weight of every bond.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_separation_history(
    grid_shape: tuple,
    spacing: float,
    horizon: float,
    profile_breaks: "np.ndarray",
    profile_coeffs: "np.ndarray",
    youngs_modulus: float,
    poisson_ratio: float,
    fracture_energy: float,
    kinematic_threshold: float,
    separation_layer: int,
    slip_angle: float,
    openings: "np.ndarray",
    basis: str = "C1",
    volume_factors: "np.ndarray | None" = None,
    initial_history: "np.ndarray | None" = None,
) -> tuple:
    """Return every bond's history, phase field and kinematic weight after the last state.

    Nodes, positions, bonds and kernel weights follow ``build_bond_family``;
    node ``(i, j, k)`` lies above the separation plane when
    ``j > separation_layer``. In state ``m`` every node above the plane sits
    at ``X + (lam_m / 2) d`` and every node below at ``X - (lam_m / 2) d``,
    with ``lam_m = openings[m]`` and
    ``d = (cos theta, sin theta, 0)``, ``theta = slip_angle`` in degrees.
    Alternatively, ``openings`` can have shape ``(T, 3)``. Each row holds
    signed components in the orthonormal frame ``(d, t, e_z)``, where
    ``t = (-sin theta, cos theta, 0)``. Its global relative translation
    replaces ``lam_m * d``. Rows are absolute states relative to the
    reference geometry, not increments. This permits changing mode mix,
    unloading and reversed shear with an out-of-plane component.

    Initial bond histories are zero unless ``initial_history`` is supplied;
    derive their initial phase fields and kinematic weights using the same
    constitutive law before evaluating the first state's kinematics. The
    states are
    processed in order, each by one pass of:

    1. nodal gradients from ``assemble_nodal_deformation_gradients`` with
       the kinematic weights left by the previous state and nodal volume
       ``spacing**3 * volume_factors`` (unit factors by default) and the
       selected polynomial ``basis``;
    2. bond gradients from ``compute_bond_deformation_gradients``;
    3. driving forces from ``compute_crack_driving_force``;
    4. history, phase field and kinematic weight from
       ``update_bond_phase_field``, with ``Y_c`` taken from
       ``compute_critical_driving_force``.

    Parameters
    ----------
    grid_shape : tuple
        Lattice size ``(N_x, N_y, N_z)``.
    spacing, horizon : float
        Lattice spacing and horizon radius.
    profile_breaks, profile_coeffs : np.ndarray
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
    openings : np.ndarray
        Nonempty nonnegative ``(T,)`` magnitudes, or finite signed
        ``(T, 3)`` local-frame components, one row per state.
    basis : str
        ``"C1"``, ``"RK1"``, or ``"RK2"`` as in the nodal reconstruction.
    volume_factors : np.ndarray or None
        Positive finite ``(N,)`` quadrature multipliers in lattice node
        order; defaults to one at every node.
    initial_history : np.ndarray or None
        Finite nonnegative ``(N_b,)`` history in the directed bond order
        of ``build_bond_family``. Enables restarting a loading path.
        All inputs are read-only; no state is retained between calls.

    Returns
    -------
    tuple
        ``(history, phase_field, kinematic_weight)``, three ``(N_b,)``
        float arrays in the bond order of ``build_bond_family``.

    Raises
    ------
    ValueError
        If ``openings`` has neither supported shape, is empty/nonfinite,
        or has negative entries in the 1D form; if the supplied volumes or
        history violate their contracts; if ``separation_layer`` is not an integer with
        ``0 <= separation_layer <= N_y - 2``, if ``slip_angle`` is not a
        finite number, or if any earlier step rejects its input.
    """
    return history, phase_field, kinematic_weight

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_run_separation_history(
    grid_shape: tuple,
    spacing: float,
    horizon: float,
    profile_breaks: "np.ndarray",
    profile_coeffs: "np.ndarray",
    youngs_modulus: float,
    poisson_ratio: float,
    fracture_energy: float,
    kinematic_threshold: float,
    separation_layer: int,
    slip_angle: float,
    openings: "np.ndarray",
    basis: str = "C1",
    volume_factors: "np.ndarray | None" = None,
    initial_history: "np.ndarray | None" = None,
) -> tuple:
    """Reference implementation (one lagged pass of the force-evaluation algorithm per state)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_integer(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

    try:
        loads = np.asarray(openings, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("openings must be numeric") from None
    if loads.size == 0 or not np.all(np.isfinite(loads)):
        raise ValueError("openings must be nonempty and finite")
    if loads.ndim == 1:
        if np.any(loads < 0.0):
            raise ValueError("1D openings must be nonnegative")
        loads = np.column_stack([loads, np.zeros_like(loads), np.zeros_like(loads)])
    elif loads.ndim != 2 or loads.shape[1] != 3:
        raise ValueError("openings must have shape (T,) or (T, 3)")
    if not _is_number(slip_angle):
        raise ValueError("slip_angle must be a finite number")

    _, critical = _oracle_compute_critical_driving_force(
        profile_breaks, profile_coeffs, fracture_energy, horizon
    )
    positions, start, end, weight = _oracle_build_bond_family(
        grid_shape, spacing, horizon, profile_breaks, profile_coeffs
    )
    ny, nz = int(grid_shape[1]), int(grid_shape[2])
    if not (_is_integer(separation_layer) and 0 <= separation_layer <= ny - 2):
        raise ValueError("separation_layer must be an integer in [0, N_y - 2]")
    layer = (np.arange(positions.shape[0]) // nz) % ny
    side = np.where(layer > separation_layer, 1.0, -1.0)[:, None]
    angle = np.radians(float(slip_angle))
    frame = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                      [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    factors = np.ones(positions.shape[0]) if volume_factors is None else np.asarray(volume_factors, dtype=float)
    if factors.shape != (positions.shape[0],) or not np.all(np.isfinite(factors)) or np.any(factors <= 0.0):
        raise ValueError("volume_factors must be positive finite shape (N,)")
    volume = float(spacing) ** 3 * factors
    history = np.zeros(start.size) if initial_history is None else np.asarray(initial_history, dtype=float).copy()
    if history.shape != (start.size,):
        raise ValueError("initial_history must have shape (N_b,)")
    history, phase, kinematic = _oracle_update_bond_phase_field(
        np.zeros(start.size), history, critical, kinematic_threshold
    )
    for local_translation in loads:
        current = positions + side * (0.5 * (frame @ local_translation))
        # Kinematics first, with the weights of the previous state.
        _, nodal = _oracle_assemble_nodal_deformation_gradients(
            positions, current, start, end, weight, kinematic, volume, basis
        )
        bond_gradients = _oracle_compute_bond_deformation_gradients(
            nodal, positions, current, start, end
        )
        force = _oracle_compute_crack_driving_force(bond_gradients, youngs_modulus, poisson_ratio)
        # Damage update last; its kinematic weight enters the next state.
        history, phase, kinematic = _oracle_update_bond_phase_field(
            force, history, critical, kinematic_threshold
        )
    return history, phase, kinematic

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    """Return deterministic numerical comparisons with isolated inputs."""
    fixture_0 = (
        'import numpy as np\n'
        'def cubic():\n'
        '    return np.array([0.0, 0.5, 1.0]), np.array([[1.0, 0.0, -6.0, 6.0], [2.0, -6.0, 6.0, -2.0]])\n'
        'def linear():\n'
        '    return np.array([0.0, 1.0]), np.array([[1.0, -1.0]])\n'
        'def _sig(result, scale):\n'
        '    H, s, h = (np.asarray(a, dtype=float) for a in result)\n'
        '    assert H.ndim == 1 and H.shape == s.shape == h.shape\n'
        '    return np.concatenate([H / scale, s, h])\n'
    )
    fixture_1 = (
        'import numpy as np\n'
        'def cubic():\n'
        '    return np.array([0.0, 0.5, 1.0]), np.array([[1.0, 0.0, -6.0, 6.0], [2.0, -6.0, 6.0, -2.0]])\n'
        'def linear():\n'
        '    return np.array([0.0, 1.0]), np.array([[1.0, -1.0]])\n'
        'def _sig(result, scale):\n'
        '    H, s, h = (np.asarray(a, dtype=float) for a in result)\n'
        '    assert H.ndim == 1 and H.shape == s.shape == h.shape\n'
        '    return np.concatenate([H / scale, s, h])\n'
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
            'call': '_sig(run_separation_history((5, 4, 3), 1.0, 2.015, *cubic(), 1.0, 0.25, 3.0e-5, 0.95, 1, 60.0, np.array([0.02, 0.05, 0.03])), 1.0e-3)',
            'gold_call': '_sig(_oracle_run_separation_history((5, 4, 3), 1.0, 2.015, *cubic(), 1.0, 0.25, 3.0e-5, 0.95, 1, 60.0, np.array([0.02, 0.05, 0.03])), 1.0e-3)',
        },
        {
            "setup": fixture_0,
            'call': '_sig(run_separation_history((4, 5, 3), 1.0, 2.015, *cubic(), 1.0, 0.25, 3.0e-5, 0.95, 2, 60.0, np.array([0.05, 0.02, 0.03])), 1.0e-3)',
            'gold_call': '_sig(_oracle_run_separation_history((4, 5, 3), 1.0, 2.015, *cubic(), 1.0, 0.25, 3.0e-5, 0.95, 2, 60.0, np.array([0.05, 0.02, 0.03])), 1.0e-3)',
        },
        {
            "setup": fixture_0,
            'call': '_sig(run_separation_history((4, 4, 4), 1.0, 1.8, *linear(), 2.0, 0.3, 5.0e-4, 0.0, 1, 20.0, np.array([0.04])), 1.0e-3)',
            'gold_call': '_sig(_oracle_run_separation_history((4, 4, 4), 1.0, 1.8, *linear(), 2.0, 0.3, 5.0e-4, 0.0, 1, 20.0, np.array([0.04])), 1.0e-3)',
        },
        {
            "setup": fixture_0,
            'call': '_sig(run_separation_history((5, 4, 3), 1.0, 2.015, *cubic(), 1.0, 0.25, 3.0e-5, 0.95, 1, 90.0, np.array([0.0, 0.06, 0.06, 0.01])), 1.0e-3)',
            'gold_call': '_sig(_oracle_run_separation_history((5, 4, 3), 1.0, 2.015, *cubic(), 1.0, 0.25, 3.0e-5, 0.95, 1, 90.0, np.array([0.0, 0.06, 0.06, 0.01])), 1.0e-3)',
        },
        {
            "setup": fixture_0,
            'call': '_sig(run_separation_history((5, 4, 3), 1.0, 2.015, *cubic(), 1.0, 0.25, 3.0e-5, 0.95, 1, 30.0, np.array([0.03, 0.05, 0.07])), 1.0e-3)',
            'gold_call': '_sig(_oracle_run_separation_history((5, 4, 3), 1.0, 2.015, *cubic(), 1.0, 0.25, 3.0e-5, 0.95, 1, 30.0, np.array([0.03, 0.05, 0.07])), 1.0e-3)',
        },
        {
            "setup": fixture_1,
            'call': '_status(lambda: run_separation_history((5, 4, 3), 1.0, 2.015, *cubic(), 1.0, 0.25, 3.0e-5, 0.95, 3, 60.0, np.array([0.02])))',
            'gold_call': '_status(lambda: _oracle_run_separation_history((5, 4, 3), 1.0, 2.015, *cubic(), 1.0, 0.25, 3.0e-5, 0.95, 3, 60.0, np.array([0.02])))',
        },
        {
            "setup": fixture_1,
            'call': '_status(lambda: run_separation_history((5, 4, 3), 1.0, 2.015, *cubic(), 1.0, 0.25, 3.0e-5, 0.95, 1, 60.0, np.array([0.02, -0.01])))',
            'gold_call': '_status(lambda: _oracle_run_separation_history((5, 4, 3), 1.0, 2.015, *cubic(), 1.0, 0.25, 3.0e-5, 0.95, 1, 60.0, np.array([0.02, -0.01])))',
        },
        {
            "setup": fixture_1,
            'call': '_status(lambda: run_separation_history((5, 4, 3), 1.0, 2.015, *cubic(), 1.0, 0.25, 3.0e-5, 1.0, 1, 60.0, np.array([0.02])))',
            'gold_call': '_status(lambda: _oracle_run_separation_history((5, 4, 3), 1.0, 2.015, *cubic(), 1.0, 0.25, 3.0e-5, 1.0, 1, 60.0, np.array([0.02])))',
        },
        {
            "setup": fixture_2,
            'call': "run(run_separation_history, 'C1', True, False, path)",
            'gold_call': "run(_oracle_run_separation_history, 'C1', True, False, path)",
            'tol': 3e-08,
        },
        {
            "setup": fixture_2,
            'call': "run(run_separation_history, 'RK1', False, False, path)",
            'gold_call': "run(_oracle_run_separation_history, 'RK1', False, False, path)",
            'tol': 3e-08,
        },
        {
            "setup": fixture_2,
            'call': "run(run_separation_history, 'RK2', True, False, path)",
            'gold_call': "run(_oracle_run_separation_history, 'RK2', True, False, path)",
            'tol': 3e-08,
        },
        {
            "setup": fixture_2,
            'call': "run(run_separation_history, 'RK2', True, True, path)",
            'gold_call': "run(_oracle_run_separation_history, 'RK2', True, True, path)",
            'tol': 3e-08,
        },
        {
            "setup": fixture_2,
            'call': "run(run_separation_history, 'C1', False, True, np.array([.01, .04, .02]))",
            'gold_call': "run(_oracle_run_separation_history, 'C1', False, True, np.array([.01, .04, .02]))",
            'tol': 3e-08,
        },
        {
            "setup": fixture_2,
            'call': "run(run_separation_history, 'RK1', True, True, path[::-1])",
            'gold_call': "run(_oracle_run_separation_history, 'RK1', True, True, path[::-1])",
            'tol': 3e-08,
        },
        {
            "setup": fixture_2,
            'call': "run(run_separation_history, 'RK2', False, False, np.array([0., .03, .01]))",
            'gold_call': "run(_oracle_run_separation_history, 'RK2', False, False, np.array([0., .03, .01]))",
            'tol': 3e-08,
        },
    ]
