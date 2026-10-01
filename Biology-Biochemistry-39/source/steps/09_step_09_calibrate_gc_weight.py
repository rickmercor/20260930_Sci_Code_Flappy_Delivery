"""
Bisect the GC Boltzmann weight until the expected G+C fraction of the weighted distinct designs meets a target.

The expected G+C content of an exponentially weighted ensemble rises with the weight, so a bracketed search over the weight reaches any composition between the ensemble's extremes.

Returns
-------
float: the bisected GC weight at which the expected G+C fraction meets the target.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def calibrate_gc_weight(
    distinct_counts: "np.ndarray",
    length: int,
    target_fraction: float,
    lower: float,
    upper: float,
    tolerance: float,
) -> float:
    """Return the GC weight whose expected G+C fraction equals the target.

    Let ``r(w)`` be the ``fraction`` entry of ``evaluate_gc_ensemble``
    called with ``distinct_counts``, a one-row table equal to
    ``distinct_counts``, ``length`` and weight ``w``, minus
    ``target_fraction``. Require ``r(lower) < 0 < r(upper)``. Then repeat:
    take the midpoint of the bracket, replace ``lower`` by it when ``r`` is
    negative there and ``upper`` otherwise, until
    ``upper - lower <= tolerance``; return the midpoint of the final bracket.

    Parameters
    ----------
    distinct_counts : np.ndarray
        Non-negative design counts indexed by the number of G-C and C-G
        pairs, as accepted by ``evaluate_gc_ensemble``.
    length : int
        Sequence length.
    target_fraction : float
        Target expected G+C fraction, strictly between 0 and 1.
    lower : float
        Lower end of the weight bracket, at least -50.
    upper : float
        Upper end of the weight bracket, above ``lower`` and at most 50.
    tolerance : float
        Final bracket width, from ``1e-14`` to ``1e-6`` inclusive.

    Returns
    -------
    float
        The calibrated weight.

    Raises
    ------
    ValueError
        If ``target_fraction``, ``lower``, ``upper`` or ``tolerance`` is not
        a finite real number (booleans are rejected), if ``target_fraction``
        is not strictly between 0 and 1, if the bracket does not satisfy
        ``-50 <= lower < upper <= 50``, if ``tolerance`` is not in
        ``[1e-14, 1e-6]``, if ``r(lower) < 0 < r(upper)`` fails, or if
        ``evaluate_gc_ensemble`` rejects its inputs.
    """
    return weight

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_calibrate_gc_weight(
    distinct_counts: "np.ndarray",
    length: int,
    target_fraction: float,
    lower: float,
    upper: float,
    tolerance: float,
) -> float:
    """Reference implementation (bracketed bisection on the expected fraction)."""
    import math

    import numpy as np

    def _is_real(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and math.isfinite(float(value)))

    arguments = (("target_fraction", target_fraction), ("lower", lower), ("upper", upper),
                 ("tolerance", tolerance))
    for name, value in arguments:
        if not _is_real(value):
            raise ValueError(f"{name} must be a finite real number")
    target = float(target_fraction)
    low, high = float(lower), float(upper)
    if not 0.0 < target < 1.0:
        raise ValueError("target_fraction must lie strictly between 0 and 1")
    if not -50.0 <= low < high <= 50.0:
        raise ValueError("the bracket must satisfy -50 <= lower < upper <= 50")
    if not 1e-14 <= float(tolerance) <= 1e-6:
        raise ValueError("tolerance must lie in [1e-14, 1e-6]")
    counts = np.asarray(distinct_counts)
    table = counts.reshape(1, -1) if counts.ndim == 1 else counts

    def _residual(value):
        return float(_oracle_evaluate_gc_ensemble(counts, table, length, value)[0]) - target

    if not (_residual(low) < 0.0 < _residual(high)):
        raise ValueError("the bracket does not enclose the target fraction")
    for _ in range(400):
        if high - low <= float(tolerance):
            break
        middle = 0.5 * (low + high)
        if middle <= low or middle >= high:
            break
        if _residual(middle) < 0.0:
            low = middle
        else:
            high = middle
    return float(0.5 * (low + high))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    data = (
        "import math\n"
        "import numpy as np\n"
        "small = np.array([0, 0, 4, 10, 6, 2])\n"
        "wide = np.array([math.comb(30, g) for g in range(31)])\n"
        "lumpy = np.array([0, 3, 0, 0, 40, 2, 0, 9])\n"
        "def _status(fn):\n"
        "    try:\n"
        "        return float(fn())\n"
        "    except ValueError:\n"
        "        return -100.0\n"
    )
    return [
        {
            "setup": data,
            "call": "calibrate_gc_weight(small.copy(), 14, 0.5, -50.0, 50.0, 1e-12)",
            "gold_call": "_oracle_calibrate_gc_weight(small.copy(), 14, 0.5, -50.0, 50.0, 1e-12)",
        },
        {
            "setup": data,
            "call": "calibrate_gc_weight(wide.copy(), 80, 0.3, -50.0, 50.0, 1e-12)",
            "gold_call": "_oracle_calibrate_gc_weight(wide.copy(), 80, 0.3, -50.0, 50.0, 1e-12)",
        },
        {
            "setup": data,
            "call": "calibrate_gc_weight(wide.copy(), 64, 0.9, -50.0, 50.0, 1e-12)",
            "gold_call": "_oracle_calibrate_gc_weight(wide.copy(), 64, 0.9, -50.0, 50.0, 1e-12)",
        },
        {
            "setup": data,
            "call": "calibrate_gc_weight(lumpy.copy(), 16, 0.6, -3.0, 7.5, 1e-7)",
            "gold_call": "_oracle_calibrate_gc_weight(lumpy.copy(), 16, 0.6, -3.0, 7.5, 1e-7)",
        },
        {
            "setup": data,
            "call": "_status(lambda: calibrate_gc_weight(small.copy(), 14, 0.5, 1.0, 2.0, 1e-12))",
            "gold_call": "_status(lambda: _oracle_calibrate_gc_weight(small.copy(), 14, 0.5, 1.0, 2.0, 1e-12))",
        },
        {
            "setup": data,
            "call": "_status(lambda: calibrate_gc_weight(small.copy(), 14, 0.5, -50.0, 50.0, 1e-3))",
            "gold_call": "_status(lambda: _oracle_calibrate_gc_weight(small.copy(), 14, 0.5, -50.0, 50.0, 1e-3))",
        },
        {
            "setup": data,
            "call": "_status(lambda: calibrate_gc_weight(small.copy(), 14, 0.5, -50.0, 50.0, 1e-15))",
            "gold_call": "_status(lambda: _oracle_calibrate_gc_weight(small.copy(), 14, 0.5, -50.0, 50.0, 1e-15))",
        },
    ]
