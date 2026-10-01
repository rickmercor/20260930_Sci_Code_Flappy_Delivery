"""
Transform an exact normalized partition polynomial into the Taylor coefficients of its log observable.

The nonlinear transform mixes substitution orders, so its finite-order coefficients must be obtained from the full normalized partition series rather than by taking logarithms term by term.

Returns
-------
np.ndarray: Taylor coefficients of the logarithm through max_order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_log_series_coefficients(
    normalized_coefficients: "np.ndarray",
    max_order: int,
) -> "np.ndarray":
    """Return the Taylor coefficients of ``log(A(mu))`` through ``max_order``.

    ``normalized_coefficients`` stores the increasing-order coefficients of
    a finite polynomial ``A(mu)`` with a strictly positive constant term.
    Return ``g[0:max_order+1]`` such that the formal power series of
    ``log(A(mu))`` is ``sum(g[r] * mu**r)`` through the requested order.
    The requested order may not exceed the supplied polynomial degree.

    Parameters
    ----------
    normalized_coefficients : np.ndarray
        Finite one-dimensional coefficient vector with positive first entry
        and at least two elements.
    max_order : int
        Integer in ``1..len(normalized_coefficients)-1``; booleans are
        rejected.

    Returns
    -------
    np.ndarray
        Float vector of shape ``(max_order + 1,)`` in increasing power order.

    Raises
    ------
    ValueError
        If the coefficient vector or order violates the conditions above.
    """
    return log_coefficients

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_log_series_coefficients(
    normalized_coefficients: "np.ndarray",
    max_order: int,
) -> "np.ndarray":
    """Reference formal-series logarithm from A' = (log A)' A."""
    import numpy as np

    try:
        coefficients = np.array(normalized_coefficients, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("normalized_coefficients must be numeric") from None
    if coefficients.ndim != 1 or coefficients.size < 2:
        raise ValueError("normalized_coefficients must be one-dimensional with at least two entries")
    if not np.all(np.isfinite(coefficients)) or coefficients[0] <= 0.0:
        raise ValueError("normalized_coefficients must be finite with a positive constant")
    if isinstance(max_order, bool) or not isinstance(max_order, (int, np.integer)):
        raise ValueError("max_order must be an integer")
    if max_order < 1 or max_order >= coefficients.size:
        raise ValueError("max_order is outside the supplied polynomial degree")

    unit = coefficients / coefficients[0]
    log_coefficients = np.zeros(max_order + 1, dtype=float)
    log_coefficients[0] = np.log(coefficients[0])
    for order in range(1, max_order + 1):
        correction = 0.0
        for lower in range(1, order):
            correction += lower * log_coefficients[lower] * unit[order - lower]
        log_coefficients[order] = unit[order] - correction / order
    return log_coefficients

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only normal, boundary, edge and validation cases."""
    base = (
        "import numpy as np\n"
        "def _sig(c):\n"
        "    c = np.asarray(c, dtype=float)\n"
        "    return float(np.dot(c, np.sin(np.arange(c.size) + .7)) + np.sum(c * c) / 101.0)\n"
    )
    status = (
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
    )
    return [
        {
            "setup": base + "C = np.array([1.0, 3.0, 3.0, 1.0])\n",
            "call": "_sig(compute_log_series_coefficients(C.copy(), 3))",
            "gold_call": "_sig(_oracle_compute_log_series_coefficients(C.copy(), 3))",
        },
        {
            "setup": base + "C = np.array([2.0, 1.0, -.25, .125])\n",
            "call": "_sig(compute_log_series_coefficients(C.copy(), 2))",
            "gold_call": "_sig(_oracle_compute_log_series_coefficients(C.copy(), 2))",
        },
        {
            "setup": base + "C = np.array([1.0, -4.0, 7.0, -2.0, .5, -.1])\n",
            "call": "_sig(compute_log_series_coefficients(C.copy(), 5))",
            "gold_call": "_sig(_oracle_compute_log_series_coefficients(C.copy(), 5))",
        },
        {
            "setup": base + "C = np.array([.5, 2.0])\n",
            "call": "float(compute_log_series_coefficients(C.copy(), 1)[0])",
            "gold_call": "float(_oracle_compute_log_series_coefficients(C.copy(), 1)[0])",
        },
        {
            "setup": base + status + "C = np.array([1.0, 2.0])\n",
            "call": "_status(lambda: compute_log_series_coefficients(C, True))",
            "gold_call": "_status(lambda: _oracle_compute_log_series_coefficients(C, True))",
        },
        {
            "setup": base + status + "C = np.array([0.0, 2.0, 1.0])\n",
            "call": "_status(lambda: compute_log_series_coefficients(C, 2))",
            "gold_call": "_status(lambda: _oracle_compute_log_series_coefficients(C, 2))",
        },
        {
            "setup": base + status + "C = np.array([1.0, 2.0, 1.0])\n",
            "call": "_status(lambda: compute_log_series_coefficients(C, 3))",
            "gold_call": "_status(lambda: _oracle_compute_log_series_coefficients(C, 3))",
        },
    ]
