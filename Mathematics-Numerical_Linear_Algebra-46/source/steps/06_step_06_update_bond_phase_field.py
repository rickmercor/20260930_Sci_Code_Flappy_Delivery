"""
Advance each bond's history variable, evaluate its closed-form phase field, and convert the phase field into the delayed kinematic weight used by the shape tensor.

An irreversible history of the crack driving force makes bond damage grow monotonically, and delaying the kinematic degradation until the phase field passes a threshold keeps the nonlocal kinematics intact while the bond's energy is already degrading.

Returns
-------
tuple[np.ndarray, np.ndarray, np.ndarray]: the updated history, the bond phase field and the kinematic weight of every bond.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def update_bond_phase_field(
    driving_force: "np.ndarray",
    history: "np.ndarray",
    critical_driving_force: float,
    kinematic_threshold: float,
) -> tuple:
    """Return the updated history, bond phase field and kinematic weight.

    The new history is ``H = max(history, driving_force)`` bond by bond. The
    bond phase field is ``s = min(1, H / (H + Y_c))`` with
    ``Y_c = critical_driving_force``. The kinematic weight is ``1`` where
    ``s <= s_c`` and ``((1 - s) / (1 - s_c))**2`` where ``s > s_c``, with
    ``s_c = kinematic_threshold``.

    Parameters
    ----------
    driving_force : np.ndarray
        ``(N_b,)`` nonnegative crack driving forces of the current state.
    history : np.ndarray
        ``(N_b,)`` nonnegative history values left by the previous state.
    critical_driving_force : float
        Positive critical crack driving force ``Y_c``.
    kinematic_threshold : float
        Threshold phase field ``s_c`` in ``[0, 1)``.

    Returns
    -------
    tuple
        ``(new_history, phase_field, kinematic_weight)``, three ``(N_b,)``
        float arrays.

    Raises
    ------
    ValueError
        If ``driving_force`` and ``history`` are not 1D arrays of equal
        length holding finite nonnegative numbers, if
        ``critical_driving_force`` is not a finite positive number, or if
        ``kinematic_threshold`` is not a finite number in ``[0, 1)``.
    """
    return new_history, phase_field, kinematic_weight

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_update_bond_phase_field(
    driving_force: "np.ndarray",
    history: "np.ndarray",
    critical_driving_force: float,
    kinematic_threshold: float,
) -> tuple:
    """Reference implementation (history maximum, closed-form law, delayed weight)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    try:
        force = np.asarray(driving_force, dtype=float)
        previous = np.asarray(history, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("driving_force and history must be numeric") from None
    if force.ndim != 1 or previous.shape != force.shape:
        raise ValueError("driving_force and history must be 1D arrays of equal length")
    if not (np.all(np.isfinite(force)) and np.all(np.isfinite(previous))):
        raise ValueError("driving_force and history must be finite")
    if np.any(force < 0.0) or np.any(previous < 0.0):
        raise ValueError("driving_force and history must be nonnegative")
    if not (_is_number(critical_driving_force) and critical_driving_force > 0.0):
        raise ValueError("critical_driving_force must be a finite positive number")
    if not (_is_number(kinematic_threshold) and 0.0 <= kinematic_threshold < 1.0):
        raise ValueError("kinematic_threshold must lie in [0, 1)")

    critical = float(critical_driving_force)
    threshold = float(kinematic_threshold)
    # Irreversibility: the phase field follows the largest force seen so far.
    new_history = np.maximum(previous, force)
    phase = np.minimum(1.0, new_history / (new_history + critical))
    # The kinematic weight stays at one until s passes s_c, then falls
    # quadratically to zero at s = 1; it is continuous at s = s_c.
    remaining = (1.0 - phase) / (1.0 - threshold)
    weight = np.where(phase <= threshold, 1.0, remaining ** 2)
    return new_history, phase, weight

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    """Return deterministic numerical comparisons with isolated inputs."""
    fixture_0 = (
        'import numpy as np\n'
        'def _run(fn, *args):\n'
        '    return fn(*[v.copy() if isinstance(v, np.ndarray) else v for v in args])\n'
        'def _sig(result):\n'
        '    H, s, h = (np.asarray(a, dtype=float) for a in result)\n'
        '    assert H.ndim == 1 and H.shape == s.shape == h.shape\n'
        '    return np.concatenate([H / 50.0, s, h])\n'
        'Y = np.array([0.0, 0.3, 1.0, 4.0, 19.0, 25.0, 60.0, 150.0, 2.0, 400.0])\n'
        'H0 = np.array([0.0, 0.5, 0.0, 3.0, 0.0, 40.0, 10.0, 0.0, 90.0, 0.0])\n'
    )
    fixture_1 = (
        'import numpy as np\n'
        'def _run(fn, *args):\n'
        '    return fn(*[v.copy() if isinstance(v, np.ndarray) else v for v in args])\n'
        'def _sig(result):\n'
        '    H, s, h = (np.asarray(a, dtype=float) for a in result)\n'
        '    assert H.ndim == 1 and H.shape == s.shape == h.shape\n'
        '    return np.concatenate([H / 50.0, s, h])\n'
        'Y = np.array([0.0, 0.3, 1.0, 4.0, 19.0, 25.0, 60.0, 150.0, 2.0, 400.0])\n'
        'H0 = np.array([0.0, 0.5, 0.0, 3.0, 0.0, 40.0, 10.0, 0.0, 90.0, 0.0])\n'
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
            'call': '_sig(_run(update_bond_phase_field, Y, H0, 1.0, 0.95))',
            'gold_call': '_sig(_run(_oracle_update_bond_phase_field, Y, H0, 1.0, 0.95))',
        },
        {
            "setup": fixture_0,
            'call': '_sig(_run(update_bond_phase_field, 0.4 * Y, H0, 2.5, 0.8))',
            'gold_call': '_sig(_run(_oracle_update_bond_phase_field, 0.4 * Y, H0, 2.5, 0.8))',
        },
        {
            "setup": fixture_0,
            'call': '_sig(_run(update_bond_phase_field, Y, np.zeros(10), 1.0, 0.0))',
            'gold_call': '_sig(_run(_oracle_update_bond_phase_field, Y, np.zeros(10), 1.0, 0.0))',
        },
        {
            "setup": fixture_0,
            'call': '_sig(_run(update_bond_phase_field, np.zeros(10), H0, 0.5, 0.99))',
            'gold_call': '_sig(_run(_oracle_update_bond_phase_field, np.zeros(10), H0, 0.5, 0.99))',
        },
        {
            "setup": fixture_0,
            'call': '_sig(_run(update_bond_phase_field, np.array([3.0e4, 1.0e-6]), np.array([1.0e4, 0.0]), 1.0, 0.95))',
            'gold_call': '_sig(_run(_oracle_update_bond_phase_field, np.array([3.0e4, 1.0e-6]), np.array([1.0e4, 0.0]), 1.0, 0.95))',
        },
        {
            "setup": fixture_1,
            'call': '_status(lambda: _run(update_bond_phase_field, Y, H0, 1.0, 1.0))',
            'gold_call': '_status(lambda: _run(_oracle_update_bond_phase_field, Y, H0, 1.0, 1.0))',
        },
        {
            "setup": fixture_1,
            'call': '_status(lambda: _run(update_bond_phase_field, Y - 1.0, H0, 1.0, 0.95))',
            'gold_call': '_status(lambda: _run(_oracle_update_bond_phase_field, Y - 1.0, H0, 1.0, 0.95))',
        },
        {
            "setup": fixture_1,
            'call': '_status(lambda: _run(update_bond_phase_field, Y, H0[:5], 1.0, 0.95))',
            'gold_call': '_status(lambda: _run(_oracle_update_bond_phase_field, Y, H0[:5], 1.0, 0.95))',
        },
    ]
