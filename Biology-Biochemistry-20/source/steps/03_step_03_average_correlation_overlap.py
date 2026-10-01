"""
Evaluate, at a set of correlation lag values, the population-weighted three-Gaussian pair average of the saturating response for an ensemble of constant regulatory inputs.

In the mean-field reduction of a large randomly wired network, the Gaussian inputs at the two ends of a lag have equal variance and covariance equal to the lag correlation. A same-sign or opposite-sign shared component represents positive or negative covariance, while independent components carry the remaining variance, so every two-time average becomes a pair average over three independent standard Gaussians.

Returns
-------
np.ndarray: population-weighted three-Gaussian pair averages of tanh, one per lag value.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def average_correlation_overlap(deltas: "np.ndarray", delta_zero: float, fields: "np.ndarray",
                                weights: "np.ndarray", beta: float,
                                nodes: int = 241) -> "np.ndarray":
    """Return the population-weighted pair average of the saturating response at each lag value.

    For each entry ``d`` of ``deltas``, let ``a = sqrt(abs(d))``,
    ``b = sqrt(delta_zero - abs(d))`` and ``sigma = +1`` when ``d >= 0`` or
    ``-1`` otherwise. With ``z1``, ``z2`` and ``z3`` independent standard
    normal variables, entry ``i`` of the result is

    ``sum_j weights[j] * E[ tanh(beta * (b*z1 + a*z3 + fields[j]))
                          * tanh(beta * (b*z2 + sigma*a*z3 + fields[j])) ]``.

    Thus the two Gaussian arguments each have variance ``delta_zero`` and
    covariance ``d``, including anticorrelated lags. At ``d = 0`` the shared
    term vanishes, so the choice ``sigma = +1`` is immaterial.

    Every standard-normal expectation ``E[f(z)]`` is evaluated as
    ``sum_k q_k f(x_k)`` on the ``nodes`` equally spaced abscissae
    ``x_k = -12 + 24 k / (nodes - 1)`` with weights
    ``q_k = (24 / (nodes - 1)) * exp(-x_k**2 / 2) / sqrt(2 pi)``, in each of the
    three variables. Only lag values with ``abs(d) <= delta_zero`` are admissible.

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
        Float array of the same length as ``deltas``.

    Raises
    ------
    ValueError
        If ``deltas`` is not a non-empty one-dimensional array of finite numbers
        or holds a value outside ``[-delta_zero, delta_zero]``, if ``delta_zero`` is
        negative or not finite, if ``fields`` is not a non-empty one-dimensional
        array of finite numbers, if ``weights`` does not match ``fields`` in
        length or is negative or does not sum to one within ``1e-9``, if
        ``beta`` is not positive and finite, or if ``nodes`` is not an integer of
        at least 3.
    """
    return overlaps

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _epi_normal_nodes(nodes: object) -> tuple:
    """Return equally spaced standard-normal abscissae on [-12, 12] and their weights."""
    count = _epi_require_count(nodes, "nodes", 3)
    abscissae = np.linspace(-12.0, 12.0, count)
    step = 24.0 / (count - 1)
    return abscissae, step * np.exp(-0.5 * abscissae ** 2) / np.sqrt(2.0 * np.pi)

def _epi_require_ensemble(fields: object, weights: object) -> tuple:
    """Return the constant inputs and their population shares as validated float arrays."""
    values = _epi_require_vector(fields, "fields")
    shares = _epi_require_vector(weights, "weights")
    if shares.size != values.size:
        raise ValueError("fields and weights must have the same length")
    if bool(np.any(shares < 0.0)):
        raise ValueError("weights must be nonnegative")
    if abs(float(np.sum(shares)) - 1.0) > 1e-9:
        raise ValueError("weights must sum to one within 1e-9")
    return values, shares

def _epi_require_lags(deltas: object, delta_zero: object) -> tuple:
    """Return the lag values and the equal-time correlation after range checks."""
    lags = _epi_require_vector(deltas, "deltas")
    equal_time = _epi_require_scalar(delta_zero, "delta_zero", 0.0, False)
    if bool(np.any(np.abs(lags) > equal_time)):
        raise ValueError("every lag value must lie in [-delta_zero, delta_zero]")
    return lags, equal_time

def _epi_pair_profile(lags: "np.ndarray", equal_time: float, fields: "np.ndarray",
                      weights: "np.ndarray", nodes: object, kernel) -> "np.ndarray":
    """Return the population-weighted three-Gaussian pair average of ``kernel`` per lag."""
    abscissae, quad = _epi_normal_nodes(nodes)
    result = np.empty(lags.size)
    for start in range(0, lags.size, 8):
        lag = lags[start:start + 8]
        magnitude = np.abs(lag)
        spread = np.sqrt(equal_time - magnitude)
        shared = np.sqrt(magnitude)[:, None, None] * abscissae[None, :, None]
        centre_left = shared + fields[None, None, :]
        signs = np.where(lag < 0.0, -1.0, 1.0)[:, None, None]
        centre_right = signs * shared + fields[None, None, :]
        private = spread[:, None, None, None] * abscissae[None, :, None, None]
        values_left = kernel(private + centre_left[:, None, :, :])
        inner_left = np.tensordot(quad, values_left, axes=(0, 1))
        values_right = kernel(private + centre_right[:, None, :, :])
        inner_right = np.tensordot(quad, values_right, axes=(0, 1))
        result[start:start + 8] = np.einsum(
            "j,mjk,mjk,k->m", quad, inner_left, inner_right, weights)
    return result

def _oracle_average_correlation_overlap(deltas: "np.ndarray", delta_zero: float,
                                        fields: "np.ndarray", weights: "np.ndarray",
                                        beta: float, nodes: int = 241) -> "np.ndarray":
    """Reference implementation (trapezoidal Gaussian averages, vectorised over lags)."""
    import numpy as np

    lags, equal_time = _epi_require_lags(deltas, delta_zero)
    values, shares = _epi_require_ensemble(fields, weights)
    beta = _epi_require_scalar(beta, "beta", 0.0, True)

    def _kernel(argument):
        return np.tanh(beta * argument)

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
            "call": "_digest(average_correlation_overlap(np.array([0.1, 0.35, 0.6, 0.75]), 0.797, fields, shares, 4.0, 61), 4, 0)",
            "gold_call": "_digest(_oracle_average_correlation_overlap(np.array([0.1, 0.35, 0.6, 0.75]), 0.797, fields, shares, 4.0, 61), 4, 0)",
        },
        # --- Boundary: lag equal to the equal-time correlation, the private variables drop out ---
        {
            "setup": digest + ensemble,
            "call": "_digest(average_correlation_overlap(np.array([0.797]), 0.797, fields, shares, 4.0, 61), 1, 0)",
            "gold_call": "_digest(_oracle_average_correlation_overlap(np.array([0.797]), 0.797, fields, shares, 4.0, 61), 1, 0)",
        },
        # --- Boundary: zero lag, the shared variable drops out ---
        {
            "setup": digest + ensemble,
            "call": "_digest(average_correlation_overlap(np.array([0.0]), 0.797, fields, shares, 4.0, 61), 1, 0)",
            "gold_call": "_digest(_oracle_average_correlation_overlap(np.array([0.0]), 0.797, fields, shares, 4.0, 61), 1, 0)",
        },
        # --- Boundary: perfect anticorrelation and mixed signed covariances ---
        {
            "setup": digest + ensemble,
            "call": "_digest(average_correlation_overlap(np.array([-0.797, -0.41, 0.23]), 0.797, fields, shares, 4.0, 61), 3, 0)",
            "gold_call": "_digest(_oracle_average_correlation_overlap(np.array([-0.797, -0.41, 0.23]), 0.797, fields, shares, 4.0, 61), 3, 0)",
        },
        # --- Edge: no fluctuations at all, a deterministic product of saturations ---
        {
            "setup": digest + "fields = np.array([0.35, -0.2])\nshares = np.array([0.7, 0.3])\n",
            "call": "_digest(average_correlation_overlap(np.array([0.0]), 0.0, fields, shares, 4.0, 5), 1, 0)",
            "gold_call": "_digest(_oracle_average_correlation_overlap(np.array([0.0]), 0.0, fields, shares, 4.0, 5), 1, 0)",
        },
        # --- Normal: weak gain, a finer rule and a lag grid spanning the whole range ---
        {
            "setup": digest + ensemble,
            "call": "_digest(average_correlation_overlap(np.linspace(0.0, 0.6, 7), 0.6, fields, shares, 0.6, 121), 7, 0)",
            "gold_call": "_digest(_oracle_average_correlation_overlap(np.linspace(0.0, 0.6, 7), 0.6, fields, shares, 0.6, 121), 7, 0)",
        },
        # --- Invalid: a lag value above the equal-time correlation ---
        {
            "setup": status + "fields = np.array([0.5, -0.5])\nshares = np.array([0.5, 0.5])\n",
            "call": "_status(lambda: average_correlation_overlap(np.array([0.2, 0.95]), 0.9, fields, shares, 4.0, 21))",
            "gold_call": "_status(lambda: _oracle_average_correlation_overlap(np.array([0.2, 0.95]), 0.9, fields, shares, 4.0, 21))",
        },
        # --- Invalid: a negative covariance below the positive-semidefinite limit ---
        {
            "setup": status + "fields = np.array([0.5, -0.5])\nshares = np.array([0.5, 0.5])\n",
            "call": "_status(lambda: average_correlation_overlap(np.array([-0.91]), 0.9, fields, shares, 4.0, 21))",
            "gold_call": "_status(lambda: _oracle_average_correlation_overlap(np.array([-0.91]), 0.9, fields, shares, 4.0, 21))",
        },
        # --- Invalid: population shares that do not sum to one ---
        {
            "setup": status + "fields = np.array([0.5, -0.5])\nshares = np.array([0.5, 0.3])\n",
            "call": "_status(lambda: average_correlation_overlap(np.array([0.3]), 0.9, fields, shares, 4.0, 21))",
            "gold_call": "_status(lambda: _oracle_average_correlation_overlap(np.array([0.3]), 0.9, fields, shares, 4.0, 21))",
        },
        # --- Invalid: too few abscissae ---
        {
            "setup": status + "fields = np.array([0.5, -0.5])\nshares = np.array([0.5, 0.5])\n",
            "call": "_status(lambda: average_correlation_overlap(np.array([0.3]), 0.9, fields, shares, 4.0, 2))",
            "gold_call": "_status(lambda: _oracle_average_correlation_overlap(np.array([0.3]), 0.9, fields, shares, 4.0, 2))",
        },
    ]
