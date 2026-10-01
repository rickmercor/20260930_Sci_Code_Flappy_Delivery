"""
Evaluate the potential of mean force and its first two reduced-length derivatives for one bond of a tensioned Morse chain, given the angular weight imposed by the rest of the intact chain.

The free-energy profile for stretching one bond while the rest of the chain stays intact follows from integrating the constrained configurational weight over orientation. Its slope locates the bonded well and barrier, while the curvature distinguishes their stationary-point types; length-independent free-energy offsets remain arbitrary.

Returns
-------
np.ndarray: finite float array (n_x, 3) containing relative beta W, d(beta W)/dx, and d2(beta W)/dx2.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def evaluate_bond_pmf(
    x_values: "np.ndarray",
    angular_weight: "np.ndarray",
    force_reduced: float,
    beta_de: float,
    a_le: float,
) -> "np.ndarray":
    """Return the reduced bond-length PMF and its first two derivatives.

    The bond has reduced length ``x = l / l_e``, Morse energy
    ``beta v(x) = beta_de * (1 - exp(-a_le * (x - 1)))**2`` and feels the
    reduced force ``force_reduced = f l_e / D_e`` along the polar axis.
    ``angular_weight`` is the weight that the rest of the intact chain places
    on the bond's polar angle, tabulated at the Gauss-Legendre nodes of
    ``[0, pi]`` (``leggauss`` order, one node per entry); it does not include
    the bond's own ``sin(theta)`` measure, and every angular integral uses
    that rule. Column 0 is ``beta W(x) - beta W(x_values[0])`` in units of
    k_B T, where ``W`` is the potential of mean force of the bond length at
    fixed ``x`` in three dimensions. Columns 1 and 2 are the analytic first
    and second derivatives of ``beta W`` with respect to ``x`` at every
    requested value. The constant subtraction affects only column 0.

    Parameters
    ----------
    x_values : np.ndarray
        One-dimensional, non-empty, finite, positive reduced lengths.
    angular_weight : np.ndarray
        One-dimensional finite nonnegative weights, at least two entries and
        a positive sum.
    force_reduced : float
        Reduced force, finite and nonnegative.
    beta_de : float
        Reduced dissociation energy, finite and positive.
    a_le : float
        Reduced inverse Morse range, finite and positive.

    Returns
    -------
    np.ndarray
        Finite float array of shape ``(len(x_values), 3)``. Column 0 is the
        relative PMF and starts at zero; columns 1 and 2 are its slope and
        curvature with respect to reduced length.

    Raises
    ------
    ValueError
        If ``x_values`` is not a non-empty one-dimensional array of finite
        positive values, if ``angular_weight`` is not a one-dimensional finite
        nonnegative array with at least two entries and a positive sum, if
        ``force_reduced`` is negative or not finite, or if ``beta_de`` or
        ``a_le`` is not finite and positive.
    """
    return pmf

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_evaluate_bond_pmf(
    x_values: "np.ndarray",
    angular_weight: "np.ndarray",
    force_reduced: float,
    beta_de: float,
    a_le: float,
) -> "np.ndarray":
    """Reference PMF implementation with analytic angular cumulants."""
    import numpy as np
    from scipy.special import logsumexp

    x = np.asarray(x_values, dtype=float)
    weight = np.asarray(angular_weight, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)) or np.any(x <= 0):
        raise ValueError("x_values must be a non-empty 1D array of finite positive values")
    if (weight.ndim != 1 or weight.size < 2 or not np.all(np.isfinite(weight))
            or np.any(weight < 0) or not np.sum(weight) > 0):
        raise ValueError("angular_weight must be a finite nonnegative 1D array with a positive sum")
    if not np.isfinite(force_reduced) or force_reduced < 0:
        raise ValueError("force_reduced must be finite and nonnegative")
    for value in (beta_de, a_le):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("beta_de and a_le must be finite and positive")
    nodes, node_weights = np.polynomial.legendre.leggauss(weight.size)
    theta = 0.5 * np.pi * (nodes + 1.0)
    with np.errstate(divide="ignore"):
        log_angular = np.log(0.5 * np.pi * node_weights * weight * np.sin(theta))
    coupling = float(beta_de) * float(force_reduced)
    stretch = float(beta_de) * (1.0 - np.exp(-float(a_le) * (x - 1.0))) ** 2
    # The x**2 Jacobian and force-biased angular partition enter as entropy.
    cos_t = np.cos(theta)
    exponent = log_angular[None, :] + coupling * np.outer(x, cos_t)
    log_partition = logsumexp(exponent, axis=1)
    angular_share = np.exp(exponent - log_partition[:, None])
    mean_cos = angular_share @ cos_t
    variance_cos = angular_share @ (cos_t ** 2) - mean_cos ** 2
    pmf = stretch - 2.0 * np.log(x) - log_partition
    decay = np.exp(-float(a_le) * (x - 1.0))
    slope = (
        2.0 * float(beta_de) * float(a_le) * (1.0 - decay) * decay
        - 2.0 / x
        - coupling * mean_cos
    )
    curvature = (
        2.0 * float(beta_de) * float(a_le) ** 2 * (2.0 * decay ** 2 - decay)
        + 2.0 / x ** 2
        - coupling ** 2 * variance_cos
    )
    return np.column_stack((pmf - pmf[0], slope, curvature))

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
        "    shape_code = arr.ndim + sum((axis + 1) * size for axis, size in enumerate(arr.shape))\n"
        "    weight = 1.0 + 0.5 * np.cos(0.7 * np.arange(flat.size))\n"
        "    return float(1.0e3 * shape_code + flat.size + np.sum(weight * flat) + 0.01 * np.sum(flat ** 2) + 1.0e4 * flat[0])\n"
        "t, w = np.polynomial.legendre.leggauss(24)\n"
        "theta = 0.5 * np.pi * (t + 1.0)\n"
        "bump = np.exp(-((theta - 1.1) ** 2) / 0.05)\n"
        "xs = np.linspace(1.0, 2.2, 13)\n"
    )
    raises = (
        "import numpy as np\n"
        "def _candidate():\n    try:\n        evaluate_bond_pmf({args})\n        return 0\n"
        "    except ValueError:\n        return 1\n    except Exception:\n        return 2\n"
        "def _reference():\n    try:\n        _oracle_evaluate_bond_pmf({args})\n        return 0\n"
        "    except ValueError:\n        return 1\n    except Exception:\n        return 2\n"
    )
    invalid = [
        "np.array([1.0, 0.0]), np.ones(8), 0.9, 279.0, 2.15",
        "np.array([1.0, 1.5]), np.zeros(8), 0.9, 279.0, 2.15",
        "np.array([1.0, 1.5]), np.ones(8), 0.9, 279.0, -2.15",
        "np.array([1.0, 1.5]), -np.ones(8), 0.9, 279.0, 2.15",
    ]
    return [
        {
            "setup": digest,
            "call": "_sig(evaluate_bond_pmf(xs.copy(), bump.copy(), 0.9, 279.0, 2.15))",
            "gold_call": "_sig(_oracle_evaluate_bond_pmf(xs.copy(), bump.copy(), 0.9, 279.0, 2.15))",
        },
        {
            "setup": digest,
            "call": "_sig(evaluate_bond_pmf(xs[::-1].copy(), np.ones(24), 0.5, 279.0, 2.15))",
            "gold_call": "_sig(_oracle_evaluate_bond_pmf(xs[::-1].copy(), np.ones(24), 0.5, 279.0, 2.15))",
        },
        {
            "setup": digest,
            "call": "_sig(evaluate_bond_pmf(xs.copy(), bump.copy(), 0.0, 279.0, 2.15))",
            "gold_call": "_sig(_oracle_evaluate_bond_pmf(xs.copy(), bump.copy(), 0.0, 279.0, 2.15))",
        },
        {
            "setup": digest,
            "call": "_sig(evaluate_bond_pmf(np.array([1.3, 0.9, 1.8, 2.6]), bump + theta, 0.4, 80.0, 1.7))",
            "gold_call": "_sig(_oracle_evaluate_bond_pmf(np.array([1.3, 0.9, 1.8, 2.6]), bump + theta, 0.4, 80.0, 1.7))",
        },
        {
            "setup": digest,
            "call": "_sig(evaluate_bond_pmf(np.array([1.1, 1.6]), bump * (theta > 1.0), 1.0, 279.0, 2.15))",
            "gold_call": "_sig(_oracle_evaluate_bond_pmf(np.array([1.1, 1.6]), bump * (theta > 1.0), 1.0, 279.0, 2.15))",
        },
        {
            "setup": digest,
            "call": "_sig(evaluate_bond_pmf(np.array([1.37]), np.array([1.0e-250, 1.0e250]), 1.25, 279.0, 2.15))",
            "gold_call": "_sig(_oracle_evaluate_bond_pmf(np.array([1.37]), np.array([1.0e-250, 1.0e250]), 1.25, 279.0, 2.15))",
        },
    ] + [
        {"setup": raises.replace("{args}", args), "call": "_candidate()", "gold_call": "_reference()"}
        for args in invalid
    ]
