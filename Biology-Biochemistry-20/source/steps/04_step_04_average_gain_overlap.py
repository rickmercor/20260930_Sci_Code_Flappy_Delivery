"""
Evaluate, at a set of correlation lag values, the population-weighted three-Gaussian pair average of the response slope for an ensemble of constant regulatory inputs.

How strongly a gene passes on a small perturbation depends on the slope of its saturating response at its operating point, so the two-time average of that slope decides whether fluctuations of the collective state are amplified or damped.

Returns
-------
np.ndarray: population-weighted three-Gaussian pair averages of the response slope, one per lag value.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def average_gain_overlap(deltas: "np.ndarray", delta_zero: float, fields: "np.ndarray",
                         weights: "np.ndarray", beta: float, nodes: int = 241) -> "np.ndarray":
    """Return the population-weighted pair average of the response slope at each lag value.

    For each entry ``d`` of ``deltas``, use the signed-covariance construction
    from ``average_correlation_overlap``: ``a = sqrt(abs(d))``,
    ``b = sqrt(delta_zero - abs(d))`` and ``sigma = sign(d)`` with
    ``sigma = +1`` at zero. Entry ``i`` of the result is

    ``sum_j weights[j] * E[ s(b*z1 + a*z3 + fields[j])
                          * s(b*z2 + sigma*a*z3 + fields[j]) ]``

    with the slope ``s(x) = beta / cosh(beta * x)**2`` of ``tanh(beta * x)``.
    Expectations use the same equally spaced standard-normal rule as
    ``average_correlation_overlap``: abscissae ``x_k = -12 + 24 k / (nodes - 1)``
    and weights ``q_k = (24 / (nodes - 1)) * exp(-x_k**2 / 2) / sqrt(2 pi)`` in
    each variable. Only lag values with ``abs(d) <= delta_zero`` are admissible.

    Parameters
    ----------
    deltas : np.ndarray
        Non-empty one-dimensional array of finite lag values.
    delta_zero : float
        Nonnegative finite equal-time correlation.
    fields : np.ndarray
        Non-empty one-dimensional array of finite constant regulatory inputs.
    weights : np.ndarray
        Nonnegative population shares of the same length as ``fields``, summing
        to one within ``1e-9``.
    beta : float
        Positive finite response gain.
    nodes : int
        Number of abscissae per standard-normal variable, at least 3 (booleans
        are rejected).

    Returns
    -------
    np.ndarray
        Nonnegative float array of the same length as ``deltas``.

    Raises
    ------
    ValueError
        Under the same conditions as ``average_correlation_overlap``.
    """
    return overlaps

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_average_gain_overlap(deltas: "np.ndarray", delta_zero: float, fields: "np.ndarray",
                                 weights: "np.ndarray", beta: float,
                                 nodes: int = 241) -> "np.ndarray":
    """Reference implementation (same rule as the correlation overlap)."""
    import numpy as np

    lags, equal_time = _epi_require_lags(deltas, delta_zero)
    values, shares = _epi_require_ensemble(fields, weights)
    beta = _epi_require_scalar(beta, "beta", 0.0, True)

    def _kernel(argument):
        return beta / np.cosh(beta * argument) ** 2

    return _epi_pair_profile(lags, equal_time, values, shares, nodes, _kernel)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    digest = (
        "import numpy as np\n"
        "def _digest(values, rows, cols):\n"
        "    array = np.asarray(values, dtype=float)\n"
        "    if cols == 0:\n"
        "        if array.ndim != 1 or array.shape[0] != rows:\n"
        "            return -1.0\n"
        "    elif array.shape != (rows, cols):\n"
        "        return -1.0\n"
        "    flat = array.ravel()\n"
        "    phase = np.cos(np.arange(flat.size, dtype=float) + 1.0)\n"
        "    return float(flat.size) + float(np.sum(np.abs(flat)) + flat @ phase)\n"
    )
    ensemble = (
        "fields = np.array([-0.81857, -0.591251, 0.336601, -0.42859, 0.525256, -0.31618, 0.601964, 0.848877])\n"
        "shares = np.array([0.14, 0.11946, 0.10054, 0.13068, 0.13932, 0.09492, 0.11508, 0.16])\n"
    )
    status = (
        "import numpy as np\n"
        "def _status(action):\n"
        "    try:\n"
        "        action()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        # --- Normal: several interior lag values of a sharp-gain mixture ---
        {
            "setup": digest + ensemble,
            "call": "_digest(average_gain_overlap(np.array([0.1, 0.35, 0.6, 0.75]), 0.797, fields, shares, 4.0, 61), 4, 0)",
            "gold_call": "_digest(_oracle_average_gain_overlap(np.array([0.1, 0.35, 0.6, 0.75]), 0.797, fields, shares, 4.0, 61), 4, 0)",
        },
        # --- Boundary: lag equal to the equal-time correlation ---
        {
            "setup": digest + ensemble,
            "call": "_digest(average_gain_overlap(np.array([0.797]), 0.797, fields, shares, 4.0, 81), 1, 0)",
            "gold_call": "_digest(_oracle_average_gain_overlap(np.array([0.797]), 0.797, fields, shares, 4.0, 81), 1, 0)",
        },
        # --- Boundary: zero lag ---
        {
            "setup": digest + ensemble,
            "call": "_digest(average_gain_overlap(np.array([0.0]), 0.797, fields, shares, 4.0, 81), 1, 0)",
            "gold_call": "_digest(_oracle_average_gain_overlap(np.array([0.0]), 0.797, fields, shares, 4.0, 81), 1, 0)",
        },
        # --- Boundary: signed covariances with a non-centred ensemble ---
        {
            "setup": digest + ensemble,
            "call": "_digest(average_gain_overlap(np.array([-0.797, -0.31, 0.31]), 0.797, fields, shares, 4.0, 81), 3, 0)",
            "gold_call": "_digest(_oracle_average_gain_overlap(np.array([-0.797, -0.31, 0.31]), 0.797, fields, shares, 4.0, 81), 3, 0)",
        },
        # --- Edge: a single centred population without fluctuations, full slope squared ---
        {
            "setup": digest + "fields = np.array([0.0])\nshares = np.array([1.0])\n",
            "call": "_digest(average_gain_overlap(np.array([0.0]), 0.0, fields, shares, 4.0, 5), 1, 0)",
            "gold_call": "_digest(_oracle_average_gain_overlap(np.array([0.0]), 0.0, fields, shares, 4.0, 5), 1, 0)",
        },
        # --- Normal: weak gain, finer rule ---
        {
            "setup": digest + ensemble,
            "call": "_digest(average_gain_overlap(np.linspace(0.0, 0.6, 7), 0.6, fields, shares, 0.6, 121), 7, 0)",
            "gold_call": "_digest(_oracle_average_gain_overlap(np.linspace(0.0, 0.6, 7), 0.6, fields, shares, 0.6, 121), 7, 0)",
        },
        # --- Invalid: negative covariance below the positive-semidefinite limit ---
        {
            "setup": status + "fields = np.array([0.5, -0.5])\nshares = np.array([0.5, 0.5])\n",
            "call": "_status(lambda: average_gain_overlap(np.array([-0.91]), 0.9, fields, shares, 4.0, 21))",
            "gold_call": "_status(lambda: _oracle_average_gain_overlap(np.array([-0.91]), 0.9, fields, shares, 4.0, 21))",
        },
        # --- Invalid: mismatched ensemble lengths ---
        {
            "setup": status + "fields = np.array([0.5, -0.5, 1.0])\nshares = np.array([0.5, 0.5])\n",
            "call": "_status(lambda: average_gain_overlap(np.array([0.3]), 0.9, fields, shares, 4.0, 21))",
            "gold_call": "_status(lambda: _oracle_average_gain_overlap(np.array([0.3]), 0.9, fields, shares, 4.0, 21))",
        },
    ]
