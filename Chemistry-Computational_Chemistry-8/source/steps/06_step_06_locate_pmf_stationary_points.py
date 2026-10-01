"""
For every bond of a tensioned chain, locate the bonded minimum and the barrier top of its bond-length potential of mean force from the angular weight imposed by the rest of the chain.

Under tension, a bond's free-energy profile keeps a metastable bonded state well separated from dissociation by a barrier, until the two merge at a critical force. The barrier top is the natural place for a dividing surface that minimizes the transition-state flux along the bond-length coordinate.

Returns
-------
np.ndarray: float array of shape (n_bonds, 2) with the reduced bonded-minimum and barrier-top lengths.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def locate_pmf_stationary_points(
    angular_weights: "np.ndarray",
    force_reduced: float,
    beta_de: float,
    a_le: float,
) -> "np.ndarray":
    """Return the bonded minimum and barrier top of each bond's PMF.

    Row ``i`` of ``angular_weights`` is the angular weight of bond ``i`` on the
    Gauss-Legendre nodes of ``[0, pi]`` (``leggauss`` order), and the PMF plus
    its first two derivatives are those returned by ``evaluate_bond_pmf`` with
    the same force and Morse parameters. Tabulate the derivative column at
    ``x = 0.9 + 0.002 k`` for
    ``k = 0, ..., 2050``. The bonded minimum lies in the first interval where
    the derivative goes from negative to nonnegative, and the barrier top in
    the first later interval where it goes from positive to nonpositive. Refine
    each root of the derivative inside its interval to an absolute accuracy of
    ``1e-12`` in ``x``. Verify from the analytic curvature column that the
    first root has positive curvature and the second has negative curvature.

    Parameters
    ----------
    angular_weights : np.ndarray
        Finite nonnegative array ``(n_bonds, n_theta)``, ``n_theta >= 2``, positive row sums.
    force_reduced : float
        Reduced force ``f l_e / D_e``, finite and nonnegative.
    beta_de : float
        Reduced dissociation energy, finite and positive.
    a_le : float
        Reduced inverse Morse range, finite and positive.

    Returns
    -------
    np.ndarray
        Float array of shape ``(n_bonds, 2)`` holding the reduced lengths of
        the bonded minimum and the barrier top of each bond.

    Raises
    ------
    ValueError
        If ``angular_weights`` is not a 2D finite nonnegative array with at least one
        row, two columns and positive row sums, if ``force_reduced`` is negative or not
        finite, if ``beta_de`` or ``a_le`` is not finite and positive, or if some bond
        has no bonded minimum followed by a barrier top on the tabulated range.
    """
    return points

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_locate_pmf_stationary_points(
    angular_weights: "np.ndarray",
    force_reduced: float,
    beta_de: float,
    a_le: float,
) -> "np.ndarray":
    """Reference implementation bracketing sign changes of the analytic derivative."""
    import numpy as np
    from scipy.optimize import brentq
    from scipy.special import logsumexp

    weights = np.asarray(angular_weights, dtype=float)
    if (weights.ndim != 2 or weights.shape[0] < 1 or weights.shape[1] < 2
            or not np.all(np.isfinite(weights)) or np.any(weights < 0)
            or not np.all(weights.sum(axis=1) > 0)):
        raise ValueError("angular_weights must be a finite nonnegative 2D array with positive rows")
    if not np.isfinite(force_reduced) or force_reduced < 0:
        raise ValueError("force_reduced must be finite and nonnegative")
    for value in (beta_de, a_le):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("beta_de and a_le must be finite and positive")
    nodes, node_weights = np.polynomial.legendre.leggauss(weights.shape[1])
    theta = 0.5 * np.pi * (nodes + 1.0)
    cos_t = np.cos(theta)
    coupling = float(beta_de) * float(force_reduced)
    grid = 0.9 + 0.002 * np.arange(2051)

    def _derivative(x, log_angular):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        decay = np.exp(-float(a_le) * (x - 1.0))
        exponent = log_angular[None, :] + coupling * np.outer(x, cos_t)
        share = np.exp(exponent - logsumexp(exponent, axis=1, keepdims=True))
        return 2.0 * float(beta_de) * float(a_le) * (1.0 - decay) * decay - 2.0 / x - coupling * share @ cos_t

    points = np.empty((weights.shape[0], 2))
    for i, row in enumerate(weights):
        with np.errstate(divide="ignore"):
            log_angular = np.log(0.5 * np.pi * node_weights * row * np.sin(theta))
        profile = _oracle_evaluate_bond_pmf(
            grid, row, force_reduced, beta_de, a_le
        )
        slope = profile[:, 1]
        rise = np.flatnonzero((slope[:-1] < 0) & (slope[1:] >= 0))
        fall = np.flatnonzero((slope[:-1] > 0) & (slope[1:] <= 0))
        fall = fall[fall > rise[0]] if rise.size else fall[:0]
        if fall.size == 0:
            raise ValueError("a bond has no bonded minimum followed by a barrier top")
        scalar = lambda x, la=log_angular: float(_derivative(x, la)[0])
        roots = np.array([
            brentq(scalar, grid[k], grid[k + 1], xtol=1e-14, maxiter=500)
            for k in (rise[0], fall[0])
        ])
        root_profile = _oracle_evaluate_bond_pmf(
            roots, row, force_reduced, beta_de, a_le
        )
        if not (root_profile[0, 2] > 0.0 and root_profile[1, 2] < 0.0):
            raise ValueError("stationary points do not form a bonded minimum and barrier top")
        points[i] = roots
    return points

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
        "    return float(arr.shape[0] + 100.0 * np.sum(weight * flat))\n"
        "t, w = np.polynomial.legendre.leggauss(24)\n"
        "theta = 0.5 * np.pi * (t + 1.0)\n"
        "bump = np.exp(-((theta - 1.1) ** 2) / 0.05)\n"
    )
    raises = (
        "import numpy as np\n"
        "def _candidate():\n    try:\n        locate_pmf_stationary_points({args})\n        return 0\n"
        "    except ValueError:\n        return 1\n    except Exception:\n        return 2\n"
        "def _reference():\n    try:\n        _oracle_locate_pmf_stationary_points({args})\n        return 0\n"
        "    except ValueError:\n        return 1\n    except Exception:\n        return 2\n"
    )
    invalid = [
        "np.ones((2, 16)), 1.3, 279.0, 2.15",
        "np.ones(16), 0.9, 279.0, 2.15",
        "-np.ones((1, 16)), 0.9, 279.0, 2.15",
        "np.ones((1, 16)), 0.9, 0.0, 2.15",
    ]
    return [
        {
            "setup": digest,
            "call": "_sig(locate_pmf_stationary_points(np.vstack([np.ones(24), bump]), 0.9, 279.0, 2.15))",
            "gold_call": "_sig(_oracle_locate_pmf_stationary_points(np.vstack([np.ones(24), bump]), 0.9, 279.0, 2.15))",
        },
        {
            "setup": digest,
            "call": "_sig(locate_pmf_stationary_points(bump[None, :].copy(), 0.6, 279.0, 2.15))",
            "gold_call": "_sig(_oracle_locate_pmf_stationary_points(bump[None, :].copy(), 0.6, 279.0, 2.15))",
        },
        {
            "setup": digest,
            "call": "_sig(locate_pmf_stationary_points(np.ones((1, 24)), 0.3, 279.0, 2.15))",
            "gold_call": "_sig(_oracle_locate_pmf_stationary_points(np.ones((1, 24)), 0.3, 279.0, 2.15))",
        },
        {
            "setup": digest,
            "call": "_sig(locate_pmf_stationary_points(np.vstack([bump * (theta < 1.2), bump + 0.1]), 1.04, 279.0, 2.15))",
            "gold_call": "_sig(_oracle_locate_pmf_stationary_points(np.vstack([bump * (theta < 1.2), bump + 0.1]), 1.04, 279.0, 2.15))",
        },
        {
            "setup": digest,
            "call": "_sig(locate_pmf_stationary_points(np.ones((1, 12)), 0.75, 150.0, 2.0))",
            "gold_call": "_sig(_oracle_locate_pmf_stationary_points(np.ones((1, 12)), 0.75, 150.0, 2.0))",
        },
    ] + [
        {"setup": raises.replace("{args}", args), "call": "_candidate()", "gold_call": "_reference()"}
        for args in invalid
    ]
