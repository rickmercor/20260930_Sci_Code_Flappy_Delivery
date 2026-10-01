"""
Evaluate the discretely sampled averaged-state contrast of the ensemble-mean record, its parameter gradient and its plug-in information matrix at one parameter value.

The contrast replaces the conditional drift in the Girsanov likelihood expression by the averaged-state drift, frozen over each sampling interval. Its plug-in information is the time-weighted Gram matrix of the drift sensitivities; the physical finite-mesh expected Hessian can also contain a discretization residual term.

Returns
-------
np.ndarray: shape (6,), [Phi, dPhi/dalpha, dPhi/dbeta, I_aa, I_ab, I_bb].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def evaluate_contrast_model(
    alpha: float,
    beta: float,
    mean_increments: 'np.ndarray',
    delta: float,
    drift_fn: "Callable[[float, float], np.ndarray]",
) -> 'np.ndarray':
    """Return the contrast, its gradient and the plug-in information at (alpha, beta).

    ``drift_fn(alpha, beta)`` follows the contract of
    ``propagate_averaged_drift`` with every other argument fixed: row ``j``
    holds ``[h_j, dh_j/dalpha, dh_j/dbeta]``, the averaged drift and its
    gradient at the start ``t_j = j * delta`` of the ``j``-th sampling
    interval. ``mean_increments[j]`` is the ensemble-mean record increment
    ``Ybar(t_{j+1}) - Ybar(t_j)`` over that interval. With ``g_j`` the
    gradient pair of row ``j``, return the length-6 array
    ``[Phi, dPhi/dalpha, dPhi/dbeta, I_aa, I_ab, I_bb]`` where
    ``Phi = sum_j h_j * mean_increments[j] - (delta / 2) * sum_j h_j**2``,
    ``(dPhi/dalpha, dPhi/dbeta) = sum_j g_j * (mean_increments[j] - h_j * delta)``
    and ``I = delta * sum_j g_j g_j^T``.

    Parameters
    ----------
    alpha, beta : float
        Parameter value passed to ``drift_fn``.
    mean_increments : np.ndarray
        Finite array of shape ``(n,)``.
    delta : float
        Positive sampling interval.
    drift_fn : callable
        Function of ``(alpha, beta)`` returning a finite ``(n, 3)`` array.

    Returns
    -------
    np.ndarray
        Float array of shape ``(6,)``.

    Raises
    ------
    ValueError
        If ``alpha``, ``beta`` or ``delta`` is not a finite real number,
        ``delta <= 0``, ``mean_increments`` is not a non-empty finite 1-D
        array, ``drift_fn`` is not callable, or ``drift_fn`` does not return
        a finite array of shape ``(n, 3)`` with ``n = len(mean_increments)``.
    """
    return model

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_evaluate_contrast_model(
    alpha: float,
    beta: float,
    mean_increments: 'np.ndarray',
    delta: float,
    drift_fn: "Callable[[float, float], np.ndarray]",
) -> 'np.ndarray':
    """Reference implementation (left-endpoint discretisation of the contrast)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_function(value):
        return callable(value)

    if not (_is_number(alpha) and _is_number(beta) and _is_number(delta)):
        raise ValueError("alpha, beta and delta must be finite real numbers")
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    if not _is_function(drift_fn):
        raise ValueError("drift_fn must be callable")
    try:
        increments = np.asarray(mean_increments, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("mean_increments must be a numeric array") from None
    if increments.ndim != 1 or increments.size == 0 or not np.all(np.isfinite(increments)):
        raise ValueError("mean_increments must be a non-empty finite 1-D array")
    rows = np.asarray(drift_fn(float(alpha), float(beta)), dtype=float)
    if rows.shape != (increments.size, 3) or not np.all(np.isfinite(rows)):
        raise ValueError("drift_fn must return a finite (n, 3) array")

    step = float(delta)
    drift = rows[:, 0]
    grads = rows[:, 1:]
    # The drift is frozen at the left end of each interval in both sums.
    value = float(drift @ increments - 0.5 * step * drift @ drift)
    residual = increments - drift * step
    gradient = grads.T @ residual
    information = step * grads.T @ grads
    return np.array([value, gradient[0], gradient[1],
                     information[0, 0], information[0, 1], information[1, 1]])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    helpers = (
        "import numpy as np\n"
        "def _fake(tau, n):\n"
        "    t = tau * np.arange(n, dtype=float)\n"
        "    def fn(a, b):\n"
        "        h = b * np.cos(a * t) + 0.2 * t\n"
        "        ha = -b * t * np.sin(a * t)\n"
        "        hb = np.cos(a * t) + 0.1 * a * t\n"
        "        return np.stack([h, ha, hb], axis=1)\n"
        "    return fn\n"
        "def _msig(v):\n"
        "    v = np.asarray(v, dtype=float)\n"
        "    if v.shape != (6,):\n"
        "        return -1.0\n"
        "    weights = np.cos(np.arange(1, 7, dtype=float))\n"
        "    return float(np.sum(np.abs(v)) + np.sum(v * weights))\n"
        "M7 = np.array([0.31, -0.12, 0.45, 0.08, -0.27, 0.19, 0.02])\n"
    )
    status = (
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        {
            "setup": helpers,
            "call": "_msig(evaluate_contrast_model(1.3, 0.8, M7, 0.3, _fake(0.3, 7)))",
            "gold_call": "_msig(_oracle_evaluate_contrast_model(1.3, 0.8, M7, 0.3, _fake(0.3, 7)))",
        },
        {
            "setup": helpers,
            "call": "float(evaluate_contrast_model(-0.7, 1.9, M7, 0.5, _fake(0.5, 7))[0])",
            "gold_call": "float(_oracle_evaluate_contrast_model(-0.7, 1.9, M7, 0.5, _fake(0.5, 7))[0])",
        },
        {
            "setup": helpers,
            "call": "float(evaluate_contrast_model(2.1, 0.4, M7, 0.25, _fake(0.25, 7))[2])",
            "gold_call": "float(_oracle_evaluate_contrast_model(2.1, 0.4, M7, 0.25, _fake(0.25, 7))[2])",
        },
        {
            "setup": helpers + "M1 = np.array([0.6])\n",
            "call": "_msig(evaluate_contrast_model(0.9, 1.1, M1, 0.8, _fake(0.8, 1)))",
            "gold_call": "_msig(_oracle_evaluate_contrast_model(0.9, 1.1, M1, 0.8, _fake(0.8, 1)))",
        },
        {
            "setup": helpers + "Z5 = np.zeros(5)\n",
            "call": "_msig(evaluate_contrast_model(0.4, 0.0, Z5, 0.6, _fake(0.6, 5)))",
            "gold_call": "_msig(_oracle_evaluate_contrast_model(0.4, 0.0, Z5, 0.6, _fake(0.6, 5)))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: evaluate_contrast_model(1.0, 0.5, M7, 0.3, _fake(0.3, 6)))",
            "gold_call": "_status(lambda: _oracle_evaluate_contrast_model(1.0, 0.5, M7, 0.3, _fake(0.3, 6)))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: evaluate_contrast_model(1.0, 0.5, M7, 0.0, _fake(0.3, 7)))",
            "gold_call": "_status(lambda: _oracle_evaluate_contrast_model(1.0, 0.5, M7, 0.0, _fake(0.3, 7)))",
        },
    ]
