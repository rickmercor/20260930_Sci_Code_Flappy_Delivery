"""
Estimate the covariance of the scaled estimation error from the per-record score contributions and the plug-in information at a parameter value.

The averaged-state contrast is not the likelihood of any single record, because each record's drift follows its own conditional state; the score variance therefore differs from the information, and the asymptotic covariance takes a sandwich form.

Returns
-------
np.ndarray: symmetric (2, 2) sandwich covariance I^{-1} S I^{-1}.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def estimate_sandwich_covariance(
    alpha: float,
    beta: float,
    increments: 'np.ndarray',
    delta: float,
    drift_fn: "Callable[[float, float], np.ndarray]",
) -> 'np.ndarray':
    """Return the empirical sandwich covariance of sqrt(N) (theta_hat - theta).

    ``increments[i, j]`` is record ``i``'s increment
    ``Y_i(t_{j+1}) - Y_i(t_j)`` over the ``j``-th sampling interval, and row
    ``j`` of ``drift_fn(alpha, beta)`` (contract of
    ``propagate_averaged_drift``) holds ``[h_j, g_j]`` at ``t_j = j * delta``,
    with ``g_j`` the gradient pair. Each record's score contribution is
    ``eta_i = sum_j g_j * (increments[i, j] - h_j * delta)``. Return
    ``V = I^{-1} S I^{-1}`` with ``I = delta * sum_j g_j g_j^T`` and
    ``S = (1 / N) sum_i (eta_i - eta_bar)(eta_i - eta_bar)^T``, where
    ``eta_bar`` is the mean of the ``eta_i`` over the ``N`` records.

    Parameters
    ----------
    alpha, beta : float
        Parameter value passed to ``drift_fn``.
    increments : np.ndarray
        Finite array of shape ``(N, n)`` with ``N >= 2``.
    delta : float
        Positive sampling interval.
    drift_fn : callable
        Function of ``(alpha, beta)`` returning a finite ``(n, 3)`` array.

    Returns
    -------
    np.ndarray
        Symmetric float array of shape ``(2, 2)``.

    Raises
    ------
    ValueError
        If ``alpha``, ``beta`` or ``delta`` is not a finite real number,
        ``delta <= 0``, ``increments`` is not a finite 2-D array with at
        least two rows and one column, ``drift_fn`` is not callable or does
        not return a finite ``(n, 3)`` array, or ``I`` is not positive
        definite (smallest eigenvalue at most ``1e-12`` times the largest).
    """
    return covariance

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_estimate_sandwich_covariance(
    alpha: float,
    beta: float,
    increments: 'np.ndarray',
    delta: float,
    drift_fn: "Callable[[float, float], np.ndarray]",
) -> 'np.ndarray':
    """Reference implementation (centred per-record scores, plug-in information)."""
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
        data = np.asarray(increments, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("increments must be a numeric array") from None
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 1 or not np.all(np.isfinite(data)):
        raise ValueError("increments must be a finite (N, n) array with N >= 2")
    rows = np.asarray(drift_fn(float(alpha), float(beta)), dtype=float)
    if rows.shape != (data.shape[1], 3) or not np.all(np.isfinite(rows)):
        raise ValueError("drift_fn must return a finite (n, 3) array")

    step = float(delta)
    drift, grads = rows[:, 0], rows[:, 1:]
    information = step * grads.T @ grads
    eigen = np.linalg.eigvalsh(information)
    if not eigen[-1] > 0.0 or eigen[0] <= 1e-12 * eigen[-1]:
        raise ValueError("the plug-in information is not positive definite")
    scores = (data - drift * step) @ grads                 # (N, 2) score contributions
    centred = scores - scores.mean(axis=0)
    meat = centred.T @ centred / data.shape[0]
    bread = np.linalg.inv(information)
    covariance = bread @ meat @ bread
    return 0.5 * (covariance + covariance.T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    helpers = (
        "import numpy as np\n"
        "def _fake(tau, n, flat=False):\n"
        "    t = tau * np.arange(n, dtype=float)\n"
        "    def fn(a, b):\n"
        "        h = 2.0 * b * np.cos(a * t) * np.exp(-0.3 * b * b * t)\n"
        "        ha = -2.0 * b * t * np.sin(a * t) * np.exp(-0.3 * b * b * t)\n"
        "        hb = h / b - 0.6 * b * t * h\n"
        "        if flat:\n"
        "            ha = 0.0 * t\n"
        "        return np.stack([h, ha, hb], axis=1)\n"
        "    return fn\n"
        "def _vsig(v):\n"
        "    v = np.asarray(v, dtype=float)\n"
        "    if v.shape != (2, 2):\n"
        "        return -1.0\n"
        "    return float(abs(v[0, 0]) + 2.0 * v[0, 1] + 3.0 * abs(v[1, 1]) + v[1, 0] * 0.5)\n"
        "rng = np.random.default_rng(20260921)\n"
        "X = rng.normal(0.0, np.sqrt(0.4), size=(12, 7)) + 0.5\n"
        "X2 = rng.normal(0.0, np.sqrt(0.25), size=(2, 9))\n"
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
            "call": "_vsig(estimate_sandwich_covariance(1.4, 0.7, X, 0.4, _fake(0.4, 7)))",
            "gold_call": "_vsig(_oracle_estimate_sandwich_covariance(1.4, 0.7, X, 0.4, _fake(0.4, 7)))",
        },
        {
            "setup": helpers,
            "call": "float(estimate_sandwich_covariance(-0.9, 1.1, X, 0.4, _fake(0.4, 7))[0, 1])",
            "gold_call": "float(_oracle_estimate_sandwich_covariance(-0.9, 1.1, X, 0.4, _fake(0.4, 7))[0, 1])",
        },
        {
            "setup": helpers,
            "call": "_vsig(estimate_sandwich_covariance(2.0, 0.5, X2, 0.25, _fake(0.25, 9)))",
            "gold_call": "_vsig(_oracle_estimate_sandwich_covariance(2.0, 0.5, X2, 0.25, _fake(0.25, 9)))",
        },
        {
            "setup": helpers + "Same = np.tile(X[0], (5, 1))\n",
            "call": "_vsig(estimate_sandwich_covariance(1.4, 0.7, Same, 0.4, _fake(0.4, 7)))",
            "gold_call": "_vsig(_oracle_estimate_sandwich_covariance(1.4, 0.7, Same, 0.4, _fake(0.4, 7)))",
        },
        {
            "setup": helpers,
            "call": "float(estimate_sandwich_covariance(0.6, 1.8, X[:, :5], 0.8, _fake(0.8, 5))[1, 1])",
            "gold_call": "float(_oracle_estimate_sandwich_covariance(0.6, 1.8, X[:, :5], 0.8, _fake(0.8, 5))[1, 1])",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: estimate_sandwich_covariance(1.4, 0.7, X[:1], 0.4, _fake(0.4, 7)))",
            "gold_call": "_status(lambda: _oracle_estimate_sandwich_covariance(1.4, 0.7, X[:1], 0.4, _fake(0.4, 7)))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: estimate_sandwich_covariance(1.4, 0.7, X, 0.4, _fake(0.4, 7, True)))",
            "gold_call": "_status(lambda: _oracle_estimate_sandwich_covariance(1.4, 0.7, X, 0.4, _fake(0.4, 7, True)))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: estimate_sandwich_covariance(1.4, 0.7, X, 0.4, _fake(0.4, 6)))",
            "gold_call": "_status(lambda: _oracle_estimate_sandwich_covariance(1.4, 0.7, X, 0.4, _fake(0.4, 6)))",
        },
    ]
