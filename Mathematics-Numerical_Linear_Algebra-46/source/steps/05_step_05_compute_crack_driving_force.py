"""
Evaluate the principal-stress crack driving force of every bond from its bond-associated deformation gradient for a Saint Venant-Kirchhoff material.

A stress-based damage criterion drives each bond by the energy density of uniaxial tension at its largest tensile principal Cauchy stress, computed from the undamaged constitutive response.

Returns
-------
np.ndarray: the (N_b,) crack driving forces.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_crack_driving_force(
    bond_gradients: "np.ndarray",
    youngs_modulus: float,
    poisson_ratio: float,
) -> "np.ndarray":
    """Return the crack driving force of every bond.

    For each bond gradient ``F`` the Green-Lagrange strain is
    ``E = (F^T F - I) / 2`` and the Saint Venant-Kirchhoff second
    Piola-Kirchhoff stress is ``S = lam * tr(E) * I + 2 * mu * E`` with
    ``lam = Y_mod * nu / ((1 + nu) * (1 - 2 * nu))`` and
    ``mu = Y_mod / (2 * (1 + nu))``. The Cauchy stress is
    ``sigma = F S F^T / det(F)`` and ``sigma_1`` is its largest eigenvalue.
    The driving force is ``max(sigma_1, 0)**2 / (2 * Y_mod)``.

    Parameters
    ----------
    bond_gradients : np.ndarray
        ``(N_b, 3, 3)`` bond deformation gradients.
    youngs_modulus : float
        Positive Young's modulus ``Y_mod``.
    poisson_ratio : float
        Poisson's ratio ``nu`` in ``(-1, 0.5)``.

    Returns
    -------
    np.ndarray
        ``(N_b,)`` nonnegative crack driving forces.

    Raises
    ------
    ValueError
        If ``bond_gradients`` is not a finite ``(N_b, 3, 3)`` array, if any
        gradient has a nonpositive determinant, if ``youngs_modulus`` is not
        a finite positive number, or if ``poisson_ratio`` is not a finite
        number in ``(-1, 0.5)``.
    """
    return driving_force

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_crack_driving_force(
    bond_gradients: "np.ndarray",
    youngs_modulus: float,
    poisson_ratio: float,
) -> "np.ndarray":
    """Reference implementation (Cauchy stress of the undamaged response)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    grads = np.asarray(bond_gradients, dtype=float)
    if grads.ndim != 3 or grads.shape[1:] != (3, 3):
        raise ValueError("bond_gradients must have shape (N_b, 3, 3)")
    if not np.all(np.isfinite(grads)):
        raise ValueError("bond_gradients must be finite")
    if not (_is_number(youngs_modulus) and youngs_modulus > 0.0):
        raise ValueError("youngs_modulus must be a finite positive number")
    if not (_is_number(poisson_ratio) and -1.0 < poisson_ratio < 0.5):
        raise ValueError("poisson_ratio must lie in (-1, 0.5)")
    jacobian = np.linalg.det(grads)
    if np.any(jacobian <= 0.0):
        raise ValueError("every bond gradient needs a positive determinant")

    modulus = float(youngs_modulus)
    nu = float(poisson_ratio)
    lame = modulus * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    shear = modulus / (2.0 * (1.0 + nu))
    identity = np.eye(3)
    green = 0.5 * (np.einsum("bki,bkj->bij", grads, grads) - identity)
    trace = np.trace(green, axis1=1, axis2=2)
    second_pk = lame * trace[:, None, None] * identity + 2.0 * shear * green
    cauchy = np.einsum("bij,bjk,blk->bil", grads, second_pk, grads) / jacobian[:, None, None]
    # Symmetrize against round-off before the symmetric eigen-solver.
    cauchy = 0.5 * (cauchy + cauchy.transpose(0, 2, 1))
    largest = np.linalg.eigvalsh(cauchy)[:, -1]
    tensile = np.maximum(largest, 0.0)
    return tensile ** 2 / (2.0 * modulus)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    """Return deterministic numerical comparisons with isolated inputs."""
    fixture_0 = (
        'import numpy as np\n'
        'rng = np.random.default_rng(5)\n'
        'def _run(fn, *args):\n'
        '    return fn(*[v.copy() if isinstance(v, np.ndarray) else v for v in args])\n'
        'G = np.eye(3) + 0.12 * rng.normal(size=(9, 3, 3))\n'
        'def _sig(y):\n'
        '    a = np.asarray(y, dtype=float)\n'
        '    assert a.ndim == 1\n'
        '    return a.copy()\n'
    )
    fixture_1 = (
        'import numpy as np\n'
        'rng = np.random.default_rng(5)\n'
        'def _run(fn, *args):\n'
        '    return fn(*[v.copy() if isinstance(v, np.ndarray) else v for v in args])\n'
        'G = np.eye(3) + 0.12 * rng.normal(size=(9, 3, 3))\n'
        'def _sig(y):\n'
        '    a = np.asarray(y, dtype=float)\n'
        '    assert a.ndim == 1\n'
        '    return a.copy()\n'
        'S = np.array([[[1.0, 0.25, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0], [-0.4, 1.0, 0.1], [0.0, 0.0, 0.9]]])\n'
    )
    fixture_2 = (
        'import numpy as np\n'
        'rng = np.random.default_rng(5)\n'
        'def _run(fn, *args):\n'
        '    return fn(*[v.copy() if isinstance(v, np.ndarray) else v for v in args])\n'
        'G = np.eye(3) + 0.12 * rng.normal(size=(9, 3, 3))\n'
        'def _sig(y):\n'
        '    a = np.asarray(y, dtype=float)\n'
        '    assert a.ndim == 1\n'
        '    return a.copy()\n'
        'R = np.array([[0.6, -0.8, 0.0], [0.8, 0.6, 0.0], [0.0, 0.0, 1.0]])\n'
    )
    fixture_3 = (
        'import numpy as np\n'
        'rng = np.random.default_rng(5)\n'
        'def _run(fn, *args):\n'
        '    return fn(*[v.copy() if isinstance(v, np.ndarray) else v for v in args])\n'
        'G = np.eye(3) + 0.12 * rng.normal(size=(9, 3, 3))\n'
        'def _sig(y):\n'
        '    a = np.asarray(y, dtype=float)\n'
        '    assert a.ndim == 1\n'
        '    return a.copy()\n'
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
            'call': '_sig(_run(compute_crack_driving_force, G, 50.0, 0.25))',
            'gold_call': '_sig(_run(_oracle_compute_crack_driving_force, G, 50.0, 0.25))',
        },
        {
            "setup": fixture_0,
            'call': '_sig(_run(compute_crack_driving_force, np.array([np.diag([1.15, 1.0, 1.0]), np.diag([0.85, 1.0, 1.0]), np.diag([0.9, 0.95, 0.97])]), 40.0, 0.3))',
            'gold_call': '_sig(_run(_oracle_compute_crack_driving_force, np.array([np.diag([1.15, 1.0, 1.0]), np.diag([0.85, 1.0, 1.0]), np.diag([0.9, 0.95, 0.97])]), 40.0, 0.3))',
        },
        {
            "setup": fixture_1,
            'call': '_sig(_run(compute_crack_driving_force, S, 25.0, -0.2))',
            'gold_call': '_sig(_run(_oracle_compute_crack_driving_force, S, 25.0, -0.2))',
        },
        {
            "setup": fixture_0,
            'call': '_sig(_run(compute_crack_driving_force, np.eye(3)[None] * 1.0, 10.0, 0.1))',
            'gold_call': '_sig(_run(_oracle_compute_crack_driving_force, np.eye(3)[None] * 1.0, 10.0, 0.1))',
        },
        {
            "setup": fixture_2,
            'call': '_sig(_run(compute_crack_driving_force, np.array([R @ np.diag([1.3, 1.05, 0.9]), R]), 60.0, 0.45))',
            'gold_call': '_sig(_run(_oracle_compute_crack_driving_force, np.array([R @ np.diag([1.3, 1.05, 0.9]), R]), 60.0, 0.45))',
        },
        {
            "setup": fixture_3,
            'call': '_status(lambda: _run(compute_crack_driving_force, np.array([np.diag([1.0, 1.0, -1.0])]), 10.0, 0.25))',
            'gold_call': '_status(lambda: _run(_oracle_compute_crack_driving_force, np.array([np.diag([1.0, 1.0, -1.0])]), 10.0, 0.25))',
        },
        {
            "setup": fixture_3,
            'call': '_status(lambda: _run(compute_crack_driving_force, G, 50.0, 0.5))',
            'gold_call': '_status(lambda: _run(_oracle_compute_crack_driving_force, G, 50.0, 0.5))',
        },
        {
            "setup": fixture_3,
            'call': '_status(lambda: _run(compute_crack_driving_force, G, 0.0, 0.25))',
            'gold_call': '_status(lambda: _run(_oracle_compute_crack_driving_force, G, 0.0, 0.25))',
        },
    ]
