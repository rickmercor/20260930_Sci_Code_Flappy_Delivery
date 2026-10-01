"""
Select, by bisection on the level of the last metabolite, the member of the one-parameter family of growth-optimal states whose external nutrient level or growth rate equals a target.

Each environment picks one member of the family of optimal metabolite profiles, and both the nutrient level it belongs to and the growth it supports rise monotonically along the family.

Returns
-------
np.ndarray: internal levels of the selected family member, shape (n,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_family_member(
    target: float,
    component: int,
    kcat: "np.ndarray",
    km: "np.ndarray",
    propagate_fn: "Callable[..., np.ndarray]",
    close_fn: "Callable[[np.ndarray], np.ndarray]",
    bracket: tuple = (1e-12, 1.0),
    tolerance: float = 1e-13,
) -> "np.ndarray":
    """Return the levels of the family member at which a closed quantity equals a target.

    ``propagate_fn(x, kcat, km)`` follows the contract of
    ``propagate_optimal_levels`` and returns the internal levels of the
    family member whose last level is ``x``; ``close_fn(levels)`` returns
    ``[nu, phi_t, lam]`` for those levels, as ``close_optimal_transporter``
    or ``close_regulated_transporter`` does with every other argument fixed.
    Both ``nu`` and ``lam`` increase with ``x``. A point ``x`` at which either
    function raises ``ValueError`` lies beyond the upper end of the family
    and counts as above the target; any other point is above the target
    when ``close_fn(levels)[component] >= target``. Starting from
    ``[ln(bracket[0]), ln(bracket[1])]``, halve the interval in ``ln x``,
    keeping the half whose lower end is below and whose upper end is above
    the target, until its width is at most ``tolerance``, and return the
    levels at the midpoint of the final interval.

    Parameters
    ----------
    target : float
        Positive target value of the closed quantity.
    component : int
        ``0`` to match the external nutrient level, ``2`` to match the growth
        rate.
    kcat, km : np.ndarray
        Enzyme constants passed unchanged to ``propagate_fn``.
    propagate_fn, close_fn : callable
        Functions with the contracts named above.
    bracket : tuple
        ``(x_low, x_high)`` with ``0 < x_low < x_high``.
    tolerance : float
        Positive width, in ``ln x``, of the final interval.

    Returns
    -------
    np.ndarray
        Internal levels of the selected member, shape ``(n,)``.

    Raises
    ------
    ValueError
        If ``component`` is not ``0`` or ``2``, if ``target`` or
        ``tolerance`` is not a finite positive number (booleans are
        rejected), if the bracket is not two finite numbers with
        ``0 < x_low < x_high``, if a callable is missing, if ``x_low`` is not
        below the target or ``x_high`` is not above it, if ``close_fn``
        returns anything other than three finite numbers, or if the upper end
        of the final interval still lies beyond the family, which means the
        target is not reached.
    """
    return levels

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_solve_family_member(
    target: float,
    component: int,
    kcat: "np.ndarray",
    km: "np.ndarray",
    propagate_fn: "Callable[..., np.ndarray]",
    close_fn: "Callable[[np.ndarray], np.ndarray]",
    bracket: tuple = (1e-12, 1.0),
    tolerance: float = 1e-13,
) -> "np.ndarray":
    """Reference implementation (bisection in the logarithm of the last level)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_function(value):
        return callable(value)

    if isinstance(component, bool) or component not in (0, 2):
        raise ValueError("component must be 0 or 2")
    if not (_is_number(target) and target > 0.0):
        raise ValueError("target must be a finite positive number")
    if not (_is_number(tolerance) and tolerance > 0.0):
        raise ValueError("tolerance must be a finite positive number")
    try:
        low, high = bracket
    except (TypeError, ValueError):
        raise ValueError("bracket must hold two numbers") from None
    if not (_is_number(low) and _is_number(high) and 0.0 < low < high):
        raise ValueError("bracket must satisfy 0 < x_low < x_high")
    if not (_is_function(propagate_fn) and _is_function(close_fn)):
        raise ValueError("propagate_fn and close_fn must be callable")

    def _state(log_x):
        # Returns (levels, value) inside the family, (None, None) beyond it.
        try:
            levels = propagate_fn(float(np.exp(log_x)), kcat, km)
            closed = close_fn(np.asarray(levels, dtype=float).copy())
        except ValueError:
            return None, None
        closed = np.asarray(closed, dtype=float)
        if closed.shape != (3,) or not np.all(np.isfinite(closed)):
            raise ValueError("close_fn must return three finite numbers")
        return np.asarray(levels, dtype=float), float(closed[component])

    lo, hi = float(np.log(low)), float(np.log(high))
    _, value_lo = _state(lo)
    if value_lo is None or not value_lo < target:
        raise ValueError("the lower bracket end must lie in the family below the target")
    _, value_hi = _state(hi)
    if value_hi is not None and value_hi < target:
        raise ValueError("the upper bracket end must lie above the target")
    hi_inside = value_hi is not None
    while hi - lo > tolerance:
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        _, value = _state(mid)
        if value is None or value >= target:
            hi, hi_inside = mid, value is not None
        else:
            lo = mid
    if not hi_inside:
        raise ValueError("the target is not reached inside the family")
    levels, _ = _state(0.5 * (lo + hi))
    if levels is None:
        raise ValueError("the selected member lies beyond the family")
    return levels

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    stand_ins = (
        "import numpy as np\n"
        "def _geometric(x, kc, k):\n"
        "    if x > 0.02:\n"
        "        raise ValueError('beyond the family')\n"
        "    return x * np.asarray(kc, dtype=float) / np.asarray(k, dtype=float) * 1.0e-4\n"
        "def _close(levels):\n"
        "    s = float(np.sum(levels))\n"
        "    if 20.0 * s >= 1.0:\n"
        "        raise ValueError('no finite nutrient level')\n"
        "    return np.array([s * s / (1.0 - 20.0 * s), 0.05 + s, s / (1.0 + s)])\n"
        "def _lsig(a, n):\n"
        "    a = np.asarray(a, dtype=float)\n"
        "    if a.shape != (n,) or not np.all(a > 0.0):\n"
        "        return -1.0\n"
        "    w = np.cos(np.arange(1, n + 1, dtype=float))\n"
        "    return float(np.sum(np.log(a) * w) / 10.0)\n"
        "KC = np.array([10.0, 20.0, 5.0, 8.0])\n"
        "KM = np.array([1.0e-3, 5.0e-4, 2.0e-3, 1.0e-3])\n"
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
            "call": "_lsig(solve_family_member(0.02, 2, KC.copy(), KM.copy(), _geometric, _close), 4)",
            "gold_call": "_lsig(_oracle_solve_family_member(0.02, 2, KC.copy(), KM.copy(), _geometric, _close), 4)",
        },
        {
            "setup": stand_ins,
            "call": "_lsig(solve_family_member(1.0e-4, 0, KC.copy(), KM.copy(), _geometric, _close), 4)",
            "gold_call": "_lsig(_oracle_solve_family_member(1.0e-4, 0, KC.copy(), KM.copy(), _geometric, _close), 4)",
        },
        {
            "setup": stand_ins,
            "call": "float(np.log(np.sum(solve_family_member(0.003, 2, KC.copy(), KM.copy(), _geometric, _close, (1e-9, 0.05), 1e-10))))",
            "gold_call": "float(np.log(np.sum(_oracle_solve_family_member(0.003, 2, KC.copy(), KM.copy(), _geometric, _close, (1e-9, 0.05), 1e-10))))",
        },
        {
            "setup": stand_ins,
            "call": "_lsig(solve_family_member(5.0, 0, KC.copy(), KM.copy(), _geometric, _close), 4)",
            "gold_call": "_lsig(_oracle_solve_family_member(5.0, 0, KC.copy(), KM.copy(), _geometric, _close), 4)",
        },
        {
            "setup": stand_ins + status,
            "call": "_status(lambda: solve_family_member(0.2, 2, KC.copy(), KM.copy(), _geometric, _close))",
            "gold_call": "_status(lambda: _oracle_solve_family_member(0.2, 2, KC.copy(), KM.copy(), _geometric, _close))",
        },
        {
            "setup": stand_ins + status,
            "call": "_status(lambda: solve_family_member(0.02, 1, KC.copy(), KM.copy(), _geometric, _close))",
            "gold_call": "_status(lambda: _oracle_solve_family_member(0.02, 1, KC.copy(), KM.copy(), _geometric, _close))",
        },
        {
            "setup": stand_ins + status,
            "call": "_status(lambda: solve_family_member(1.0e-9, 2, KC.copy(), KM.copy(), _geometric, _close, (1e-3, 1.0)))",
            "gold_call": "_status(lambda: _oracle_solve_family_member(1.0e-9, 2, KC.copy(), KM.copy(), _geometric, _close, (1e-3, 1.0)))",
        },
    ]
