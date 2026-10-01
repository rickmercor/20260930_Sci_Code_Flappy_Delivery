"""
Construct the metabolite levels of a linear pathway at which its total enzyme demand is stationary, indexing the one-parameter family either by the last metabolite level or by the total metabolite pool.

In a growth-optimal pathway no reshuffling of metabolite between steps can save enzyme, which ties every metabolite level to the levels downstream of it and leaves a one-parameter family of optimal states. Recovering a member from its total pool additionally requires inverting that family rather than only evaluating its upstream recurrence.

Returns
-------
np.ndarray: levels, or rows [levels, dlevels/danchor] when requested.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def propagate_optimal_levels(
    anchor: float,
    kcat: "np.ndarray",
    km: "np.ndarray",
    coordinate: str = "last",
    return_tangent: bool = False,
) -> "np.ndarray":
    """Return one stationary metabolite profile from its last level or total pool.

    Use the pathway, units and balanced-growth fluxes of
    ``compute_enzyme_demand``, and let ``D(levels)`` be the sum over the ``n``
    enzymes of their proteome fractions per unit growth rate. Return the
    positive levels, with ``levels[n - 1] = last_level``, at which the
    partial derivatives of ``D`` with respect to all ``n`` levels are equal.
    These levels need the least enzyme among all positive levels with the
    same total. If ``coordinate == "last"``, ``anchor`` is the prescribed
    value of ``levels[n - 1]``. If ``coordinate == "total"``, ``anchor`` is
    the prescribed value of ``sum(levels)``. There is at most one positive
    profile in either case. Levels, and the requested total in ``"total"``
    mode, must be accurate to a relative ``1e-12``. If ``return_tangent`` is
    true, also return the tangent of every level with respect to the chosen
    anchor along the same stationary branch. Tangent entries must have
    relative error at most ``1e-10`` where nonzero and absolute error at most
    ``1e-12`` near zero.

    Parameters
    ----------
    anchor : float
        Positive last-metabolite level when ``coordinate == "last"``, or
        positive total internal metabolite level when
        ``coordinate == "total"``.
    kcat : np.ndarray
        Positive catalytic constants per unit enzyme mass fraction, shape
        ``(n,)`` with ``n >= 1``.
    km : np.ndarray
        Positive Michaelis constants in the units of the levels, shape
        ``(n,)``.
    coordinate : str
        Either ``"last"`` or ``"total"``.
    return_tangent : bool
        If false, return only the profile. If true, return the profile and its
        derivative with respect to ``anchor`` as two rows.

    Returns
    -------
    np.ndarray
        Float array of shape ``(n,)`` with the levels in pathway order, or
        shape ``(2, n)`` with rows ``[levels, dlevels/danchor]`` when
        ``return_tangent`` is true.

    Raises
    ------
    ValueError
        If ``anchor`` is not a finite positive number (booleans are rejected),
        if ``coordinate`` is not ``"last"`` or ``"total"``, if
        ``return_tangent`` is not boolean, if ``kcat`` or ``km`` is not a
        non-empty one-dimensional array of finite positive numbers, if they
        differ in length, or if no positive profile with the requested
        coordinate satisfies the conditions.
    """
    return levels

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_propagate_optimal_levels(
    anchor: float,
    kcat: np.ndarray,
    km: np.ndarray,
    coordinate: str = "last",
    return_tangent: bool = False,
) -> np.ndarray:
    """Reference implementation (upstream recursion and family inversion)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _positive_vector(value, name):
        try:
            array = np.array(value, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a numeric array") from None
        if array.ndim != 1 or array.size == 0:
            raise ValueError(f"{name} must be a non-empty one-dimensional array")
        if not (np.all(np.isfinite(array)) and np.all(array > 0.0)):
            raise ValueError(f"{name} must hold finite positive numbers")
        return array

    if not (_is_number(anchor) and anchor > 0.0):
        raise ValueError("anchor must be a finite positive number")
    if coordinate not in ("last", "total") or not isinstance(coordinate, str):
        raise ValueError("coordinate must be 'last' or 'total'")
    if not isinstance(return_tangent, (bool, np.bool_)):
        raise ValueError("return_tangent must be boolean")
    rate = _positive_vector(kcat, "kcat")
    affinity = _positive_vector(km, "km")
    if rate.shape != affinity.shape:
        raise ValueError("kcat and km must have the same length")
    count = rate.size

    def _from_last(last_level):
        rho = np.empty(count)
        tangent = np.empty(count)
        rho[-1] = float(last_level)
        tangent[-1] = 1.0
        below = 0.0        # total level of metabolites downstream of rho[i]
        below_tangent = 0.0
        for i in range(count - 1, 0, -1):
            # Equal derivatives of D with respect to rho[i - 1] and rho[i]
            # leave one positive root of a quadratic for rho[i - 1].
            cost = affinity[i] * (1.0 + below) / (rate[i] * rho[i] ** 2)
            lead = cost - 1.0 / rate[i - 1]
            if not (np.isfinite(lead) and lead > 0.0):
                raise ValueError("no positive optimal levels for this last level")
            lead_tangent = cost * (
                below_tangent / (1.0 + below) - 2.0 * tangent[i] / rho[i]
            )
            below += rho[i]
            below_tangent += tangent[i]
            linear = affinity[i - 1] / rate[i - 1]
            constant = affinity[i - 1] * (1.0 + below) / rate[i - 1]
            constant_tangent = linear * below_tangent
            discriminant = linear * linear + 4.0 * lead * constant
            rho[i - 1] = (linear + np.sqrt(discriminant)) / (2.0 * lead)
            if not (np.isfinite(rho[i - 1]) and rho[i - 1] > 0.0):
                raise ValueError("no finite positive optimal levels for this last level")
            tangent[i - 1] = (
                constant_tangent - lead_tangent * rho[i - 1] ** 2
            ) / (2.0 * lead * rho[i - 1] - linear)
            if not np.isfinite(tangent[i - 1]):
                raise ValueError("stationary-family tangent is not finite")
        return rho, tangent

    if coordinate == "last":
        profile, tangent = _from_last(float(anchor))
        return np.vstack([profile, tangent]) if return_tangent else profile
    if count == 1:
        profile = np.array([float(anchor)])
        return np.vstack([profile, np.ones(1)]) if return_tangent else profile

    # The total pool grows strictly along the feasible branch. The requested
    # last level cannot exceed the total, while an infeasible upper point is
    # also known to lie above the requested member. Find a feasible lower
    # point, then invert the family in log(last_level).
    log_high = float(np.log(anchor))
    log_low = log_high
    lower = None
    log_two = float(np.log(2.0))
    log_tiny = float(np.log(np.nextafter(0.0, 1.0)))
    for _ in range(2048):
        log_low -= log_two
        if log_low <= log_tiny:
            break
        try:
            candidate, _ = _from_last(float(np.exp(log_low)))
        except ValueError:
            continue
        if float(candidate.sum()) < float(anchor):
            lower = candidate
            break
    if lower is None:
        raise ValueError("no positive optimal profile has this total level")

    for _ in range(256):
        if log_high - log_low <= 5.0e-14:
            break
        log_mid = 0.5 * (log_low + log_high)
        try:
            candidate, _ = _from_last(float(np.exp(log_mid)))
        except ValueError:
            log_high = log_mid
            continue
        if float(candidate.sum()) < float(anchor):
            log_low = log_mid
            lower = candidate
        else:
            log_high = log_mid
    log_mid = 0.5 * (log_low + log_high)
    try:
        result, tangent = _from_last(float(np.exp(log_mid)))
    except ValueError:
        result = lower
        _, tangent = _from_last(float(result[-1]))
    if abs(float(result.sum()) / float(anchor) - 1.0) > 1.0e-12:
        raise ValueError("requested total level was not reached")
    total_tangent = float(tangent.sum())
    if not (np.isfinite(total_tangent) and total_tangent > 0.0):
        raise ValueError("stationary-family total tangent is not positive")
    tangent = tangent / total_tangent
    return np.vstack([result, tangent]) if return_tangent else result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    helpers = (
        "import numpy as np\n"
        "def _lsig(a, n):\n"
        "    a = np.asarray(a, dtype=float)\n"
        "    if a.shape != (n,) or not np.all(a > 0.0):\n"
        "        return -1.0\n"
        "    w = np.cos(np.arange(1, n + 1, dtype=float))\n"
        "    return float(np.sum(np.log(a) * w) / 10.0)\n"
        "def _stationary(a, kc, k):\n"
        "    a = np.asarray(a, dtype=float)\n"
        "    if a.shape != kc.shape or not np.all(a > 0.0):\n"
        "        return 0.0\n"
        "    n = a.size\n"
        "    grads = []\n"
        "    for m in range(n):\n"
        "        g = -k[m] * (1.0 + a[m + 1:].sum()) / (kc[m] * a[m] ** 2)\n"
        "        g += sum((1.0 + k[i] / a[i]) / kc[i] for i in range(m))\n"
        "        grads.append(g)\n"
        "    grads = np.array(grads)\n"
        "    spread = np.max(np.abs(grads - grads.mean())) / np.max(np.abs(grads))\n"
        "    return 1.0 if spread < 1e-9 else 0.0\n"
        "def _total_sig(a, n, total):\n"
        "    a = np.asarray(a, dtype=float)\n"
        "    if a.shape != (n,) or not np.all(a > 0.0):\n"
        "        return -1.0\n"
        "    w = np.sin(np.arange(1, n + 1, dtype=float))\n"
        "    shape = np.sum(np.log(a) * w) / 10.0\n"
        "    closure = 100.0 * (np.sum(a) / total - 1.0)\n"
        "    return float(shape + closure)\n"
        "def _tangent_sig(a, n, total_mode):\n"
        "    a = np.asarray(a, dtype=float)\n"
        "    if a.shape != (2, n) or not np.all(np.isfinite(a)) or not np.all(a[0] > 0.0):\n"
        "        return -1.0\n"
        "    w = np.cos(np.arange(1, n + 1, dtype=float))\n"
        "    out = np.sum(np.log(a[0]) * w) / 10.0 + np.sum(a[1] * w) / 10.0\n"
        "    if total_mode:\n"
        "        out += 100.0 * (np.sum(a[1]) - 1.0)\n"
        "    return float(out)\n"
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
    chain = (
        "KC = np.array([14.0, 26.0, 8.0, 33.0, 11.0])\n"
        "KM = np.array([3.0e-4, 1.6e-3, 1.2e-4, 9.0e-4, 5.0e-4])\n"
    )
    long_chain = (
        "KC = np.array([31.0, 9.0, 17.0, 12.5, 40.0, 6.8, 22.0, 15.0])\n"
        "KM = np.array([2.2e-4, 8.0e-4, 1.4e-3, 3.5e-4, 6.0e-5, 1.0e-3, 4.8e-4, 2.6e-3])\n"
    )
    return [
        {
            "setup": helpers + chain,
            "call": "_lsig(propagate_optimal_levels(2.0e-3, KC.copy(), KM.copy()), 5)",
            "gold_call": "_lsig(_oracle_propagate_optimal_levels(2.0e-3, KC.copy(), KM.copy()), 5)",
        },
        {
            "setup": helpers + chain,
            "call": "float(np.log(propagate_optimal_levels(6.0e-4, KC.copy(), KM.copy())[0]))",
            "gold_call": "float(np.log(_oracle_propagate_optimal_levels(6.0e-4, KC.copy(), KM.copy())[0]))",
        },
        {
            "setup": helpers + long_chain,
            "call": "_lsig(propagate_optimal_levels(9.0e-4, KC.copy(), KM.copy()), 8)",
            "gold_call": "_lsig(_oracle_propagate_optimal_levels(9.0e-4, KC.copy(), KM.copy()), 8)",
        },
        {
            "setup": helpers + long_chain,
            "call": "_stationary(propagate_optimal_levels(3.0e-4, KC.copy(), KM.copy()), KC, KM)",
            "gold_call": "_stationary(_oracle_propagate_optimal_levels(3.0e-4, KC.copy(), KM.copy()), KC, KM)",
        },
        {
            "setup": helpers + "KC = np.array([19.0]); KM = np.array([7.0e-4])\n",
            "call": "_lsig(propagate_optimal_levels(4.0e-3, KC.copy(), KM.copy()), 1)",
            "gold_call": "_lsig(_oracle_propagate_optimal_levels(4.0e-3, KC.copy(), KM.copy()), 1)",
        },
        {
            "setup": helpers + (
                "KC = np.array([7.0, 7.0, 7.0])\n"
                "KM = np.array([2.0e-3, 2.0e-3, 2.0e-3])\n"
            ),
            "call": "_lsig(propagate_optimal_levels(1.0e-2, KC.copy(), KM.copy()), 3)",
            "gold_call": "_lsig(_oracle_propagate_optimal_levels(1.0e-2, KC.copy(), KM.copy()), 3)",
        },
        {
            "setup": helpers + chain + "TOTAL = 2.5e-2\n",
            "call": "_tangent_sig(propagate_optimal_levels(TOTAL, KC.copy(), KM.copy(), 'total', True), 5, True)",
            "gold_call": "_tangent_sig(_oracle_propagate_optimal_levels(TOTAL, KC.copy(), KM.copy(), 'total', True), 5, True)",
        },
        {
            "setup": helpers + long_chain + "TOTAL = 1.7e-1\n",
            "call": "_total_sig(propagate_optimal_levels(TOTAL, KC.copy(), KM.copy(), coordinate='total'), 8, TOTAL)",
            "gold_call": "_total_sig(_oracle_propagate_optimal_levels(TOTAL, KC.copy(), KM.copy(), coordinate='total'), 8, TOTAL)",
        },
        {
            "setup": helpers + long_chain,
            "call": "_tangent_sig(propagate_optimal_levels(9.0e-4, KC.copy(), KM.copy(), 'last', True), 8, False)",
            "gold_call": "_tangent_sig(_oracle_propagate_optimal_levels(9.0e-4, KC.copy(), KM.copy(), 'last', True), 8, False)",
        },
        {
            "setup": helpers + "KC = np.array([19.0]); KM = np.array([7.0e-4]); TOTAL = 7.0e-3\n",
            "call": "_total_sig(propagate_optimal_levels(TOTAL, KC.copy(), KM.copy(), 'total'), 1, TOTAL)",
            "gold_call": "_total_sig(_oracle_propagate_optimal_levels(TOTAL, KC.copy(), KM.copy(), 'total'), 1, TOTAL)",
        },
        {
            "setup": helpers + status + chain,
            "call": "_status(lambda: propagate_optimal_levels(5.0e-2, KC.copy(), KM.copy()))",
            "gold_call": "_status(lambda: _oracle_propagate_optimal_levels(5.0e-2, KC.copy(), KM.copy()))",
        },
        {
            "setup": helpers + status + chain,
            "call": "_status(lambda: propagate_optimal_levels(-1.0e-3, KC.copy(), KM.copy()))",
            "gold_call": "_status(lambda: _oracle_propagate_optimal_levels(-1.0e-3, KC.copy(), KM.copy()))",
        },
        {
            "setup": helpers + status + chain,
            "call": "_status(lambda: propagate_optimal_levels(2.5e-2, KC.copy(), KM.copy(), 'pool'))",
            "gold_call": "_status(lambda: _oracle_propagate_optimal_levels(2.5e-2, KC.copy(), KM.copy(), 'pool'))",
        },
        {
            "setup": helpers + status + chain,
            "call": "_status(lambda: propagate_optimal_levels(2.5e-2, KC.copy(), KM.copy(), 'total', 1))",
            "gold_call": "_status(lambda: _oracle_propagate_optimal_levels(2.5e-2, KC.copy(), KM.copy(), 'total', 1))",
        },
    ]
