"""
Turn the enzyme amounts of a series of balanced-growth states into zero-growth response factors from straight-line fits and summarise them by their mass-weighted mean and spread.

Enzyme levels in a nutrient-limited proteome change nearly linearly with growth rate, so the intercept of each line relative to its rich-state value measures whether an enzyme is induced or repressed as growth slows.

Returns
-------
np.ndarray: [q_0, ..., q_{n-1}, weighted mean q, weighted rms spread], shape (n + 2,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def summarize_growth_response(growth_rates: "np.ndarray", amounts: "np.ndarray", growth_hi: float) -> "np.ndarray":
    """Return the enzymes' zero-growth response factors with their weighted mean and spread.

    ``amounts[s, i]`` is the proteome fraction of enzyme ``i`` in state ``s``,
    whose growth rate is ``growth_rates[s]``. For every enzyme, fit the
    ordinary least-squares straight line of its amount against the growth
    rate over all states and let ``h_i`` be the line's value at
    ``growth_hi``; the response factor ``q_i`` is the line's value at zero
    growth divided by ``h_i``. With weights ``w_i = h_i / sum(h)``, the
    weighted mean is ``qbar = sum(w * q)`` and the spread is the weighted
    root-mean-square deviation ``sqrt(sum(w * (q - qbar) ** 2))``.

    Parameters
    ----------
    growth_rates : np.ndarray
        Positive growth rates of the ``m`` states, shape ``(m,)``, with at
        least two distinct values.
    amounts : np.ndarray
        Enzyme proteome fractions, shape ``(m, n)`` with ``n >= 1``.
    growth_hi : float
        Positive growth rate at which the lines are normalised.

    Returns
    -------
    np.ndarray
        Float array of shape ``(n + 2,)``:
        ``[q_0, ..., q_{n-1}, qbar, spread]``.

    Raises
    ------
    ValueError
        If ``growth_rates`` is not a one-dimensional array of finite positive
        numbers with at least two distinct values, if ``amounts`` is not a
        finite array of shape ``(m, n)`` with ``n >= 1``, if ``growth_hi`` is
        not a finite positive number (booleans are rejected), or if any
        fitted amount ``h_i`` is not positive.
    """
    return summary

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_summarize_growth_response(growth_rates: "np.ndarray", amounts: "np.ndarray", growth_hi: float) -> "np.ndarray":
    """Reference implementation (closed-form least-squares lines)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    try:
        rates = np.array(growth_rates, dtype=float)
        table = np.array(amounts, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("growth_rates and amounts must be numeric arrays") from None
    if rates.ndim != 1 or rates.size < 2:
        raise ValueError("growth_rates must be a one-dimensional array of at least two states")
    if not (np.all(np.isfinite(rates)) and np.all(rates > 0.0)):
        raise ValueError("growth_rates must hold finite positive numbers")
    if table.ndim != 2 or table.shape[0] != rates.size or table.shape[1] == 0:
        raise ValueError("amounts must have shape (m, n) with n >= 1")
    if not np.all(np.isfinite(table)):
        raise ValueError("amounts must be finite")
    if not (_is_number(growth_hi) and growth_hi > 0.0):
        raise ValueError("growth_hi must be a finite positive number")
    if np.unique(rates).size < 2:
        raise ValueError("growth_rates must contain at least two distinct values")
    centred = rates - rates.mean()
    slope = centred @ (table - table.mean(axis=0)) / float(np.sum(centred * centred))
    intercept = table.mean(axis=0) - slope * rates.mean()
    at_hi = intercept + slope * float(growth_hi)
    if not np.all(at_hi > 0.0):
        raise ValueError("every fitted amount at growth_hi must be positive")
    response = intercept / at_hi
    weights = at_hi / at_hi.sum()
    mean = float(np.sum(weights * response))
    rms = float(np.sqrt(np.sum(weights * (response - mean) ** 2)))
    return np.concatenate([response, [mean, rms]])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    helpers = (
        "import numpy as np\n"
        "def _vsig(a, n):\n"
        "    a = np.asarray(a, dtype=float)\n"
        "    if a.shape != (n,):\n"
        "        return -1.0\n"
        "    w = np.cos(np.arange(1, n + 1, dtype=float))\n"
        "    return float(np.sum(np.abs(a)) + np.sum(a * w))\n"
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
    series = (
        "G = np.array([1.2, 1.0, 0.8, 0.6, 0.45])\n"
        "P = np.array([[0.100, 0.200, 0.050],\n"
        "              [0.090, 0.200, 0.060],\n"
        "              [0.081, 0.199, 0.071],\n"
        "              [0.070, 0.201, 0.080],\n"
        "              [0.063, 0.198, 0.088]])\n"
    )
    return [
        {
            "setup": helpers + series,
            "call": "_vsig(summarize_growth_response(G.copy(), P.copy(), 1.2), 5)",
            "gold_call": "_vsig(_oracle_summarize_growth_response(G.copy(), P.copy(), 1.2), 5)",
        },
        {
            "setup": helpers + series,
            "call": "float(10.0 * summarize_growth_response(G.copy(), P.copy(), 1.0)[-1])",
            "gold_call": "float(10.0 * _oracle_summarize_growth_response(G.copy(), P.copy(), 1.0)[-1])",
        },
        {
            "setup": helpers + (
                "G = np.array([0.9, 0.5])\n"
                "P = np.array([[0.30, 0.12, 0.08, 0.02], [0.18, 0.15, 0.05, 0.03]])\n"
            ),
            "call": "_vsig(summarize_growth_response(G.copy(), P.copy(), 0.9), 6)",
            "gold_call": "_vsig(_oracle_summarize_growth_response(G.copy(), P.copy(), 0.9), 6)",
        },
        {
            "setup": helpers + (
                "G = np.array([0.7, 0.6, 0.5, 0.4, 0.3, 0.2])\n"
                "P = np.outer(0.2 + 0.5 * G, np.array([0.1, 0.3, 0.2]))\n"
            ),
            "call": "_vsig(summarize_growth_response(G.copy(), P.copy(), 0.7), 5)",
            "gold_call": "_vsig(_oracle_summarize_growth_response(G.copy(), P.copy(), 0.7), 5)",
        },
        {
            "setup": helpers + "G = np.array([1.0, 0.8, 0.6]); P = np.array([[0.4], [0.35], [0.31]])\n",
            "call": "_vsig(summarize_growth_response(G.copy(), P.copy(), 1.0), 3)",
            "gold_call": "_vsig(_oracle_summarize_growth_response(G.copy(), P.copy(), 1.0), 3)",
        },
        {
            "setup": helpers + status + "G = np.array([0.8, 0.8, 0.8]); P = np.ones((3, 2)) * 0.1\n",
            "call": "_status(lambda: summarize_growth_response(G.copy(), P.copy(), 0.8))",
            "gold_call": "_status(lambda: _oracle_summarize_growth_response(G.copy(), P.copy(), 0.8))",
        },
        {
            "setup": helpers + status + (
                "G = np.array([1.0, 0.5]); P = np.array([[0.05, 0.2], [0.30, 0.1]])\n"
            ),
            "call": "_status(lambda: summarize_growth_response(G.copy(), P.copy(), 1.5))",
            "gold_call": "_status(lambda: _oracle_summarize_growth_response(G.copy(), P.copy(), 1.5))",
        },
    ]
