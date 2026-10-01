"""
For every bond of a Morse chain under constant tension, integrate its Boltzmann weight over the intact range of reduced bond lengths at each polar angle of a Gauss-Legendre grid, and return the natural logarithm.

Treating a bond as intact only below its own rupture threshold restricts the configurational integral to the intact basin. With the length integrated out bond by bond, each bond contributes an angle-dependent local weight that the orientational propagation then carries along the chain.

Returns
-------
np.ndarray: float array of shape (n_bonds, n_theta) with the natural log of each bond's intact angular weight.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_intact_bond_weights(
    thresholds: "np.ndarray",
    force_reduced: float,
    beta_de: float,
    a_le: float,
    n_theta: int,
    n_length: int,
) -> "np.ndarray":
    """Return the log intact weight of every bond on the polar-angle grid.

    Lengths are reduced, ``x = l / l_e``. Bond ``j`` has Morse energy
    ``beta v(x) = beta_de * (1 - exp(-a_le * (x - 1)))**2`` and feels the
    reduced force ``force_reduced = f l_e / D_e``, so the force couples to the
    projection ``x cos(theta)`` of the bond on the force axis. Entry
    ``[j, k]`` is the natural log of the integral, over ``0.5 <= x <=
    thresholds[j]``, of the Boltzmann factor of the stretching energy and the
    force coupling times the polar part ``x**2 sin(theta_k)`` of the bond-vector
    volume element (no azimuthal factor). ``theta_k`` are the ``n_theta``
    Gauss-Legendre nodes of ``[0, pi]`` in ``leggauss`` order, and the length
    integral is the ``n_length``-node Gauss-Legendre rule on
    ``[0.5, thresholds[j]]``. Evaluate the quadrature in the log domain so
    the returned logarithms remain finite when the force bias or Morse depth
    makes the unnormalized Boltzmann factors overflow or underflow.

    Parameters
    ----------
    thresholds : np.ndarray
        One-dimensional reduced rupture thresholds, each finite and above 0.5.
    force_reduced : float
        Reduced force, finite and nonnegative.
    beta_de : float
        Reduced dissociation energy, finite and positive.
    a_le : float
        Reduced inverse Morse range, finite and positive.
    n_theta : int
        Number of polar-angle nodes, at least 2.
    n_length : int
        Number of bond-length nodes, at least 2.

    Returns
    -------
    np.ndarray
        Float array of shape ``(len(thresholds), n_theta)``.

    Raises
    ------
    ValueError
        If ``thresholds`` is not a non-empty one-dimensional array of finite
        values above 0.5, if ``force_reduced`` is negative or not finite, if
        ``beta_de`` or ``a_le`` is not finite and positive, or if ``n_theta``
        or ``n_length`` is not an integer of at least 2 (booleans rejected).
    """
    return log_weights

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_intact_bond_weights(
    thresholds: "np.ndarray",
    force_reduced: float,
    beta_de: float,
    a_le: float,
    n_theta: int,
    n_length: int,
) -> "np.ndarray":
    """Reference implementation with log-sum-exp length quadrature."""
    import numpy as np
    from scipy.special import logsumexp

    def _is_int(value):
        return not isinstance(value, bool) and isinstance(value, (int, np.integer))

    upper = np.asarray(thresholds, dtype=float)
    if upper.ndim != 1 or upper.size == 0 or not np.all(np.isfinite(upper)) or np.any(upper <= 0.5):
        raise ValueError("thresholds must be a non-empty 1D array of finite values above 0.5")
    if not np.isfinite(force_reduced) or force_reduced < 0:
        raise ValueError("force_reduced must be finite and nonnegative")
    for value in (beta_de, a_le):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("beta_de and a_le must be finite and positive")
    if not (_is_int(n_theta) and n_theta >= 2 and _is_int(n_length) and n_length >= 2):
        raise ValueError("n_theta and n_length must be integers of at least 2")
    angle_nodes, _ = np.polynomial.legendre.leggauss(int(n_theta))
    theta = 0.5 * np.pi * (angle_nodes + 1.0)
    length_nodes, length_weights = np.polynomial.legendre.leggauss(int(n_length))
    coupling = float(beta_de) * float(force_reduced)
    log_weights = np.empty((upper.size, int(n_theta)))
    for j, x_max in enumerate(upper):
        half = 0.5 * (x_max - 0.5)
        x = half * length_nodes + 0.5 + half
        stretch = float(beta_de) * (1.0 - np.exp(-float(a_le) * (x - 1.0))) ** 2
        radial = np.log(half * length_weights) + 2.0 * np.log(x) - stretch
        exponent = radial[None, :] + coupling * np.outer(np.cos(theta), x)
        log_weights[j] = np.log(np.sin(theta)) + logsumexp(exponent, axis=1)
    return log_weights

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    digest = (
        "import numpy as np\n"
        "def _sig(values):\n"
        "    arr = np.asarray(values, dtype=float)\n"
        "    flat = arr.ravel()\n"
        "    weight = 1.0 + 0.5 * np.cos(0.7 * np.arange(flat.size))\n"
        "    return float(arr.shape[0] + flat.size + np.sum(weight * flat) + 0.01 * np.sum(flat ** 2))\n"
    )
    raises = (
        "import numpy as np\n"
        "def _candidate():\n    try:\n        compute_intact_bond_weights({args})\n        return 0\n"
        "    except ValueError:\n        return 1\n    except Exception:\n        return 2\n"
        "def _reference():\n    try:\n        _oracle_compute_intact_bond_weights({args})\n        return 0\n"
        "    except ValueError:\n        return 1\n    except Exception:\n        return 2\n"
    )
    invalid = [
        "np.array([1.6, 0.45]), 0.9, 279.0, 2.15, 12, 40",
        "np.array([1.6]), -0.1, 279.0, 2.15, 12, 40",
        "np.array([[1.6, 1.7]]), 0.9, 279.0, 2.15, 12, 40",
        "np.array([1.6]), 0.9, 279.0, 2.15, 12, 1",
    ]
    return [
        {
            "setup": digest,
            "call": "_sig(compute_intact_bond_weights(np.array([1.58, 1.62, 1.61]), 0.9, 279.0, 2.15, 16, 120))",
            "gold_call": "_sig(_oracle_compute_intact_bond_weights(np.array([1.58, 1.62, 1.61]), 0.9, 279.0, 2.15, 16, 120))",
        },
        {
            "setup": digest,
            "call": "_sig(compute_intact_bond_weights(np.array([2.4]), 0.2, 279.0, 2.15, 10, 300))",
            "gold_call": "_sig(_oracle_compute_intact_bond_weights(np.array([2.4]), 0.2, 279.0, 2.15, 10, 300))",
        },
        {
            "setup": digest,
            "call": "_sig(compute_intact_bond_weights(np.array([2.0, 3.0]), 0.0, 279.0, 2.15, 8, 200))",
            "gold_call": "_sig(_oracle_compute_intact_bond_weights(np.array([2.0, 3.0]), 0.0, 279.0, 2.15, 8, 200))",
        },
        {
            "setup": digest,
            "call": "_sig(compute_intact_bond_weights(np.array([1.3, 1.9]), 0.5, 60.0, 1.5, 7, 64))",
            "gold_call": "_sig(_oracle_compute_intact_bond_weights(np.array([1.3, 1.9]), 0.5, 60.0, 1.5, 7, 64))",
        },
        {
            "setup": digest,
            "call": "_sig(compute_intact_bond_weights(np.array([1.7, 1.6, 0.51]), 1.1, 279.0, 2.15, 9, 2))",
            "gold_call": "_sig(_oracle_compute_intact_bond_weights(np.array([1.7, 1.6, 0.51]), 1.1, 279.0, 2.15, 9, 2))",
        },
        {
            "setup": digest,
            "call": "_sig(compute_intact_bond_weights(np.array([1.01, 1.8, 2.6]), 1.4, 1200.0, 3.2, 17, 160))",
            "gold_call": "_sig(_oracle_compute_intact_bond_weights(np.array([1.01, 1.8, 2.6]), 1.4, 1200.0, 3.2, 17, 160))",
        },
        {
            "setup": digest,
            "call": "_sig(compute_intact_bond_weights(np.array([0.5000001, 1.000001, 3.5]), 0.0, 1.0e-8, 0.7, 11, 96))",
            "gold_call": "_sig(_oracle_compute_intact_bond_weights(np.array([0.5000001, 1.000001, 3.5]), 0.0, 1.0e-8, 0.7, 11, 96))",
        },
    ] + [
        {"setup": raises.replace("{args}", args), "call": "_candidate()", "gold_call": "_reference()"}
        for args in invalid
    ]
