"""
Locate the polarity magnitude above which three adhering cells can no longer rest in a stable stacked arrangement.

Above this polarity magnitude a third cell cannot sit on two neighbours and must join their row, so it marks the few-cell estimate of the boundary between multilayer and monolayer growth.

Returns
-------
float: the threshold polarity magnitude p_c.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def locate_monolayer_threshold(
    triad_fn: "Callable[[float], np.ndarray]",
    magnitude_bracket: tuple = (0.0, 1.0),
    tolerance: float = 1e-12,
) -> float:
    """Return the upper end of the polarity-magnitude range with a stacked equilibrium.

    ``triad_fn(magnitude)`` follows the contract of ``solve_stacked_triad``
    with every other argument fixed: it returns the two pair separations of
    the stable non-collinear three-cell equilibrium at that polarity
    magnitude and raises ``ValueError`` when no such equilibrium exists.
    Return the magnitude ``p_c`` that separates the magnitudes at which
    ``triad_fn`` returns from those at which it raises ``ValueError``,
    within the bracket ``magnitude_bracket = (low, high)``, with absolute
    accuracy ``tolerance``. The equilibrium must exist at ``low`` and must
    not exist at ``high``.

    Parameters
    ----------
    triad_fn : callable
        Function of one magnitude returning a length-2 array or raising
        ``ValueError``.
    magnitude_bracket : tuple
        ``(low, high)`` with ``0 <= low < high``.
    tolerance : float
        Positive absolute accuracy of the returned magnitude.

    Returns
    -------
    float
        The threshold magnitude ``p_c``.

    Raises
    ------
    ValueError
        If ``triad_fn`` is not callable, if the bracket is not two finite
        numbers with ``0 <= low < high``, if ``tolerance`` is not a finite
        positive number, if the equilibrium does not exist at ``low`` or
        still exists at ``high``, or if ``triad_fn`` returns anything other
        than two finite numbers.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_locate_monolayer_threshold(
    triad_fn: "Callable[[float], np.ndarray]",
    magnitude_bracket: tuple = (0.0, 1.0),
    tolerance: float = 1e-12,
) -> float:
    """Reference implementation (bisection on the existence of the equilibrium)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_function(value):
        return callable(value)

    if not _is_function(triad_fn):
        raise ValueError("triad_fn must be callable")
    try:
        low, high = magnitude_bracket
    except (TypeError, ValueError):
        raise ValueError("magnitude_bracket must hold two numbers") from None
    if not (_is_number(low) and _is_number(high) and 0.0 <= low < high):
        raise ValueError("magnitude_bracket must satisfy 0 <= low < high")
    if not (_is_number(tolerance) and tolerance > 0.0):
        raise ValueError("tolerance must be a finite positive number")
    low, high = float(low), float(high)

    def _exists(strength):
        try:
            separations = triad_fn(strength)
        except ValueError:
            return False
        separations = np.asarray(separations, dtype=float)
        if separations.shape != (2,) or not np.all(np.isfinite(separations)):
            raise ValueError("triad_fn must return two finite separations")
        return True

    if not _exists(low):
        raise ValueError("the stacked equilibrium must exist at the lower bracket end")
    if _exists(high):
        raise ValueError("the stacked equilibrium must not exist at the upper bracket end")
    while high - low > tolerance:
        middle = 0.5 * (low + high)
        if middle <= low or middle >= high:
            break
        if _exists(middle):
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    stand_ins = (
        "import numpy as np\n"
        "def _edge(limit):\n"
        "    def fn(m):\n"
        "        if m < limit:\n"
        "            return np.array([2.0 + m, 1.9 - m])\n"
        "        raise ValueError('no stacked equilibrium')\n"
        "    return fn\n"
        "def _stretch(rest, reach, gain):\n"
        "    def fn(m):\n"
        "        base = rest * (1.0 + gain * m * m)\n"
        "        if base < reach:\n"
        "            return np.array([base, 0.9 * rest])\n"
        "        raise ValueError('base pair out of range')\n"
        "    return fn\n"
        "def _bad(m):\n"
        "    if m < 0.5:\n"
        "        return np.array([1.0, 2.0, 3.0])\n"
        "    raise ValueError('none')\n"
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
            "setup": stand_ins,
            "call": "locate_monolayer_threshold(_edge(0.2718281828459045), (0.0, 1.0), 1e-12)",
            "gold_call": "_oracle_locate_monolayer_threshold(_edge(0.2718281828459045), (0.0, 1.0), 1e-12)",
        },
        {
            "setup": stand_ins,
            "call": "locate_monolayer_threshold(_stretch(1.9, 2.45, 2.7), (0.05, 0.9), 1e-12)",
            "gold_call": "_oracle_locate_monolayer_threshold(_stretch(1.9, 2.45, 2.7), (0.05, 0.9), 1e-12)",
        },
        {
            "setup": stand_ins,
            "call": "locate_monolayer_threshold(_stretch(2.1, 2.6, 4.3))",
            "gold_call": "_oracle_locate_monolayer_threshold(_stretch(2.1, 2.6, 4.3))",
        },
        {
            "setup": stand_ins,
            "call": "locate_monolayer_threshold(_edge(0.61), (0.6, 0.75), 1e-13)",
            "gold_call": "_oracle_locate_monolayer_threshold(_edge(0.61), (0.6, 0.75), 1e-13)",
        },
        {
            "setup": stand_ins + status,
            "call": "_status(lambda: locate_monolayer_threshold(_edge(0.3), (0.35, 0.9), 1e-12))",
            "gold_call": "_status(lambda: _oracle_locate_monolayer_threshold(_edge(0.3), (0.35, 0.9), 1e-12))",
        },
        {
            "setup": stand_ins + status,
            "call": "_status(lambda: locate_monolayer_threshold(_edge(0.95), (0.1, 0.9), 1e-12))",
            "gold_call": "_status(lambda: _oracle_locate_monolayer_threshold(_edge(0.95), (0.1, 0.9), 1e-12))",
        },
        {
            "setup": stand_ins + status,
            "call": "_status(lambda: locate_monolayer_threshold(_bad, (0.0, 1.0), 1e-12))",
            "gold_call": "_status(lambda: _oracle_locate_monolayer_threshold(_bad, (0.0, 1.0), 1e-12))",
        },
        {
            "setup": stand_ins + status,
            "call": "_status(lambda: locate_monolayer_threshold(_edge(0.3), (0.5, 0.1), 1e-12))",
            "gold_call": "_status(lambda: _oracle_locate_monolayer_threshold(_edge(0.3), (0.5, 0.1), 1e-12))",
        },
    ]
