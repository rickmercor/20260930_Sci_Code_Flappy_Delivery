"""
Step description: Find the perturbation size that minimizes the exact mean squared error of a stencil-based Monte Carlo derivative estimator with a fixed number of replications, and report the minimized root mean square error.

Shrinking the perturbation lowers the finite-difference bias but inflates the Monte Carlo variance, so for a given simulation effort there is a best perturbation size.

Returns
-------
np.ndarray: float array [eps minimizing the exact mean square error, minimized root mean square error].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def minimize_stencil_rmse(
    generators: "np.ndarray",
    rates: "np.ndarray",
    initial_index: int,
    observable: "np.ndarray",
    horizon: float,
    channel: int,
    offsets: "np.ndarray",
    coefficients: "np.ndarray",
    derivative_order: int,
    replications: float,
    target: float,
    eps_bounds: "np.ndarray",
) -> "np.ndarray":
    """Return the perturbation size minimizing the exact mean square error and the minimized RMSE.

    For ``eps > 0`` let ``N_eps`` be the stencil numerator of
    ``compute_stencil_moments`` and ``k = derivative_order``. The estimator
    averages ``n = replications`` independent copies of ``N_eps / eps^k`` as
    an estimate of ``target``. Return the ``eps`` in the closed interval
    ``[lo, hi] = eps_bounds`` at which that estimator's exact mean square
    error is smallest, together with the square root of the smallest value.
    The mean square error is assumed to have a single local minimum on the
    interval. The returned ``eps`` must be accurate to a relative precision of
    ``1e-6`` and the root mean square error to ``1e-9``.

    Parameters
    ----------
    generators, rates, initial_index, observable, horizon, channel, offsets, coefficients
        As in ``compute_stencil_moments``.
    derivative_order : int
        Order ``k`` of the estimated derivative, at least 1.
    replications : float
        Finite positive number ``n`` of independent replications.
    target : float
        Finite value of the derivative being estimated.
    eps_bounds : np.ndarray
        Shape ``(2,)``, ``[lo, hi]`` with ``0 < lo < hi`` finite; every
        perturbed rate must be non-negative at ``eps = hi``.

    Returns
    -------
    np.ndarray
        Float array ``[eps_opt, rmse_min]``.

    Raises
    ------
    ValueError
        For any invalid input of ``compute_stencil_moments``, if
        ``derivative_order`` is not an integer of at least 1, if
        ``replications`` is not finite and positive, if ``target`` is not
        finite, or if ``eps_bounds`` is not an increasing pair of finite
        positive numbers.
    """
    return optimum

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_minimize_stencil_rmse(
    generators: "np.ndarray",
    rates: "np.ndarray",
    initial_index: int,
    observable: "np.ndarray",
    horizon: float,
    channel: int,
    offsets: "np.ndarray",
    coefficients: "np.ndarray",
    derivative_order: int,
    replications: float,
    target: float,
    eps_bounds: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation: logarithmic scan followed by bounded Brent refinement."""
    import numpy as np
    from scipy.optimize import minimize_scalar

    if isinstance(derivative_order, bool) or not isinstance(derivative_order, (int, np.integer)) or derivative_order < 1:
        raise ValueError("derivative_order must be an integer of at least 1")
    if not np.isfinite(float(replications)) or float(replications) <= 0.0 or not np.isfinite(float(target)):
        raise ValueError("replications must be finite and positive and target finite")
    bounds = np.asarray(eps_bounds, dtype=float)
    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or not 0.0 < bounds[0] < bounds[1]:
        raise ValueError("eps_bounds must be an increasing pair of finite positive numbers")
    k, n = int(derivative_order), float(replications)

    def _mse(eps):
        mean, second = _oracle_compute_stencil_moments(generators, rates, initial_index, observable, horizon,
                                                       channel, offsets, coefficients, float(eps))
        return (second - mean ** 2) / (n * eps ** (2 * k)) + (mean / eps ** k - float(target)) ** 2

    grid = np.geomspace(bounds[0], bounds[1], 17)
    scan = np.array([_mse(eps) for eps in grid])
    best = int(np.argmin(scan))
    left, right = grid[max(best - 1, 0)], grid[min(best + 1, grid.size - 1)]
    refined = minimize_scalar(_mse, bounds=(left, right), method="bounded", options={"xatol": 1e-10 * right})
    eps_opt, mse_min = (refined.x, refined.fun) if refined.fun <= scan[best] else (grid[best], scan[best])
    return np.array([float(eps_opt), float(np.sqrt(mse_min))])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    helpers = (
        "import numpy as np\n"
        "def _isomerization(total):\n"
        "    g = np.zeros((2, total + 1, total + 1))\n"
        "    for a in range(total + 1):\n"
        "        if a > 0:\n"
        "            g[0, a, a - 1], g[0, a, a] = a, -a\n"
        "        if a < total:\n"
        "            g[1, a, a + 1], g[1, a, a] = total - a, -(total - a)\n"
        "    return g\n"
        "def _dimerization(total):\n"
        "    g = np.zeros((2, total + 1, total + 1))\n"
        "    for c in range(total + 1):\n"
        "        if c < total:\n"
        "            g[0, c, c + 1], g[0, c, c] = (total - c) ** 2, -(total - c) ** 2\n"
        "        if c > 0:\n"
        "            g[1, c, c - 1], g[1, c, c] = c, -c\n"
        "    return g\n"
        "def _digest(v):\n"
        "    v = np.asarray(v, dtype=float)\n"
        "    if v.shape != (2,) or np.any(v <= 0):\n"
        "        return -1.0\n"
        "    return float(20.0 + np.arcsinh(1.0e6 * v[1]) + 1.0e-3 * np.log(v[0]))\n"
    )

    raises = (
        "def _candidate():\n    try:\n        minimize_stencil_rmse({args})\n        return 0\n"
        "    except ValueError:\n        return 1\n"
        "def _reference():\n    try:\n        _oracle_minimize_stencil_rmse({args})\n        return 0\n"
        "    except ValueError:\n        return 1\n"
    )
    iso = "G = _isomerization(3)\nf = np.arange(4.0)\n"
    return [
        {   # Normal: a centered first-derivative difference.
            "setup": helpers + iso,
            "call": "_digest(minimize_stencil_rmse(G.copy(), np.array([1.0, 0.6]), 3, f.copy(), 1.0, 0, np.array([1.0, -1.0]), np.array([0.5, -0.5]), 1, 400.0, -0.939722, np.array([0.01, 0.9])))",
            "gold_call": "_digest(_oracle_minimize_stencil_rmse(G.copy(), np.array([1.0, 0.6]), 3, f.copy(), 1.0, 0, np.array([1.0, -1.0]), np.array([0.5, -0.5]), 1, 400.0, -0.939722, np.array([0.01, 0.9])))",
        },
        {   # Boundary: the minimum sits at the upper end of the bracket.
            "setup": helpers + "G = _dimerization(2)\nf = np.arange(3.0)\n",
            "call": "_digest(minimize_stencil_rmse(G.copy(), np.array([0.8, 0.5]), 0, f.copy(), 2.0, 0, np.array([1.0, 0.0]), np.array([1.0, -1.0]), 1, 1.0, 0.538225, np.array([0.001, 0.75])))",
            "gold_call": "_digest(_oracle_minimize_stencil_rmse(G.copy(), np.array([0.8, 0.5]), 0, f.copy(), 2.0, 0, np.array([1.0, 0.0]), np.array([1.0, -1.0]), 1, 1.0, 0.538225, np.array([0.001, 0.75])))",
        },
        {   # Edge: a third derivative from the four-point stencil.
            "setup": helpers + iso,
            "call": "_digest(minimize_stencil_rmse(G.copy(), np.array([1.0, 0.6]), 3, f.copy(), 1.0, 0, np.array([2.0, 1.0, -1.0, -2.0]), np.array([0.5, -1.0, 1.0, -0.5]), 3, 1.0e6, -0.73557, np.array([0.02, 0.49])))",
            "gold_call": "_digest(_oracle_minimize_stencil_rmse(G.copy(), np.array([1.0, 0.6]), 3, f.copy(), 1.0, 0, np.array([2.0, 1.0, -1.0, -2.0]), np.array([0.5, -1.0, 1.0, -0.5]), 3, 1.0e6, -0.73557, np.array([0.02, 0.49])))",
        },
        {   # Edge: a huge replication count drives the optimum towards the lower end.
            "setup": helpers + iso,
            "call": "_digest(minimize_stencil_rmse(G.copy(), np.array([1.0, 0.6]), 3, f.copy(), 1.0, 0, np.array([1.0, -1.0]), np.array([0.5, -0.5]), 1, 1.0e10, -0.939722, np.array([1.0e-4, 0.9])))",
            "gold_call": "_digest(_oracle_minimize_stencil_rmse(G.copy(), np.array([1.0, 0.6]), 3, f.copy(), 1.0, 0, np.array([1.0, -1.0]), np.array([0.5, -0.5]), 1, 1.0e10, -0.939722, np.array([1.0e-4, 0.9])))",
        },
        {   # Invalid: the bracket does not increase.
            "setup": helpers + iso + raises.replace("{args}", "G.copy(), np.array([1.0, 0.6]), 3, f.copy(), 1.0, 0, np.array([1.0, -1.0]), np.array([0.5, -0.5]), 1, 400.0, 0.1, np.array([0.5, 0.2])"),
            "call": "_candidate()",
            "gold_call": "_reference()",
        },
        {   # Invalid: a replication count of zero.
            "setup": helpers + iso + raises.replace("{args}", "G.copy(), np.array([1.0, 0.6]), 3, f.copy(), 1.0, 0, np.array([1.0, -1.0]), np.array([0.5, -0.5]), 1, 0.0, 0.1, np.array([0.01, 0.5])"),
            "call": "_candidate()",
            "gold_call": "_reference()",
        },
    ]
