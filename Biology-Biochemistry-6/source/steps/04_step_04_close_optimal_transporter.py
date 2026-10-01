"""
Close a growth-optimal set of internal metabolite levels at the uptake step when the transporter amount is itself growth-maximising, returning the external nutrient level, the transporter amount and the growth rate.

A freely allocated transporter is just the first enzyme of the pathway, so optimality extends to it and fixes which external nutrient level a given set of internal levels belongs to.

Returns
-------
np.ndarray: [external nutrient level, transporter amount, growth rate].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def close_optimal_transporter(
    levels: "np.ndarray",
    kcat: "np.ndarray",
    km: "np.ndarray",
    kcat_t: float,
    km_t: float,
    demand_fn: "Callable[..., np.ndarray]",
    cost_fn: "Callable[..., np.ndarray]",
) -> "np.ndarray":
    """Return the nutrient level, transporter amount and growth rate of an optimal state.

    The pathway, units and balanced-growth fluxes are those of
    ``compute_enzyme_demand``. A transporter of amount ``phi_t`` converts an
    external nutrient, held at level ``nu`` and not diluted, into metabolite
    ``0`` at the rate ``kcat_t * phi_t * nu / (km_t + nu)``, so that in
    balanced growth it supplies metabolite ``0`` faster than enzyme ``0``
    consumes it by exactly that metabolite's dilution; the transporter and
    the ``n`` enzymes make up the whole proteome, ``phi_t + sum(phi) = 1``,
    and metabolites are outside this budget. ``levels`` are positive internal
    levels satisfying the conditions of ``propagate_optimal_levels``. Return
    the external level ``nu`` at which the growth rate maximised over all
    ``n + 1`` amounts is attained at exactly these internal levels, together
    with the transporter amount and the growth rate of that state.
    ``demand_fn(levels, kcat, km)`` and ``cost_fn(levels, kcat, km)`` follow
    the contracts of ``compute_enzyme_demand`` and
    ``compute_metabolite_costs``; use them for those quantities.

    Parameters
    ----------
    levels : np.ndarray
        Positive internal metabolite levels, shape ``(n,)`` with ``n >= 1``.
    kcat, km : np.ndarray
        Positive enzyme constants, each of shape ``(n,)``.
    kcat_t : float
        Positive catalytic constant of the transporter.
    km_t : float
        Positive Michaelis constant of the transporter for the external
        nutrient.
    demand_fn, cost_fn : callable
        Functions with the contracts named above.

    Returns
    -------
    np.ndarray
        Float array ``[nu, phi_t, lam]``.

    Raises
    ------
    ValueError
        If ``levels``, ``kcat`` or ``km`` is not a non-empty one-dimensional
        array of finite positive numbers or they differ in length, if
        ``kcat_t`` or ``km_t`` is not a finite positive number (booleans are
        rejected), if a callable is missing or returns anything other than
        ``n`` finite positive numbers, or if no finite positive ``nu`` makes
        these levels optimal.
    """
    return state

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_close_optimal_transporter(
    levels: "np.ndarray",
    kcat: "np.ndarray",
    km: "np.ndarray",
    kcat_t: float,
    km_t: float,
    demand_fn: "Callable[..., np.ndarray]",
    cost_fn: "Callable[..., np.ndarray]",
) -> "np.ndarray":
    """Reference implementation (the transporter's cost per unit flux closes the costs)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_function(value):
        return callable(value)

    def _positive_vector(value, name, size=None):
        try:
            array = np.array(value, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a numeric array") from None
        if array.ndim != 1 or array.size == 0 or (size is not None and array.size != size):
            raise ValueError(f"{name} must be a one-dimensional array of the pathway length")
        if not (np.all(np.isfinite(array)) and np.all(array > 0.0)):
            raise ValueError(f"{name} must hold finite positive numbers")
        return array

    rho = _positive_vector(levels, "levels")
    rate = _positive_vector(kcat, "kcat", rho.size)
    affinity = _positive_vector(km, "km", rho.size)
    if not (_is_number(kcat_t) and kcat_t > 0.0 and _is_number(km_t) and km_t > 0.0):
        raise ValueError("kcat_t and km_t must be finite positive numbers")
    if not (_is_function(demand_fn) and _is_function(cost_fn)):
        raise ValueError("demand_fn and cost_fn must be callable")
    demand = _positive_vector(demand_fn(rho.copy(), rate.copy(), affinity.copy()), "demand", rho.size)
    cost = _positive_vector(cost_fn(rho.copy(), rate.copy(), affinity.copy()), "cost", rho.size)
    # Metabolite 0 is worth exactly the transporter protein that one more unit
    # of supply flux needs: cost[0] = (1 + km_t / nu) / kcat_t.
    excess = float(kcat_t) * cost[0] - 1.0
    if not excess > 0.0:
        raise ValueError("these levels are not optimal at any finite nutrient level")
    nu = float(km_t) / excess
    supply = 1.0 + rho.sum()                  # transporter flux per unit growth rate
    transporter = supply * cost[0]            # transporter amount per unit growth rate
    lam = 1.0 / (demand.sum() + transporter)
    return np.array([nu, lam * transporter, lam])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    model = (
        "import numpy as np\n"
        "def _demand(r, kc, k):\n"
        "    r = np.asarray(r, dtype=float); out = np.empty(r.size)\n"
        "    for i in range(r.size):\n"
        "        out[i] = (1.0 + r[i + 1:].sum()) * (1.0 + k[i] / r[i]) / kc[i]\n"
        "    return out\n"
        "def _cost(r, kc, k):\n"
        "    r = np.asarray(r, dtype=float); out = np.empty(r.size)\n"
        "    for i in range(r.size):\n"
        "        out[i] = k[i] * (1.0 + r[i + 1:].sum()) / (kc[i] * r[i] ** 2)\n"
        "    return out\n"
        "def _chain(x, kc, k):\n"
        "    r = [x]\n"
        "    for i in range(len(kc) - 1, 0, -1):\n"
        "        below = sum(r[1:])\n"
        "        a = k[i] * (1.0 + below) / (kc[i] * r[0] ** 2) - 1.0 / kc[i - 1]\n"
        "        b = k[i - 1] / kc[i - 1]; c = k[i - 1] * (1.0 + below + r[0]) / kc[i - 1]\n"
        "        r.insert(0, (b + np.sqrt(b * b + 4.0 * a * c)) / (2.0 * a))\n"
        "    return np.array(r)\n"
        "def _tsig(s):\n"
        "    s = np.asarray(s, dtype=float)\n"
        "    if s.shape != (3,) or not np.all(s > 0.0):\n"
        "        return -1.0\n"
        "    return float(np.log(s[0]) / 10.0 + 10.0 * s[1] + s[2])\n"
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
    return [
        {
            "setup": model + chain + "L = _chain(2.0e-3, KC, KM)\n",
            "call": "_tsig(close_optimal_transporter(L.copy(), KC.copy(), KM.copy(), 40.0, 1.0e-3, _demand, _cost))",
            "gold_call": "_tsig(_oracle_close_optimal_transporter(L.copy(), KC.copy(), KM.copy(), 40.0, 1.0e-3, _demand, _cost))",
        },
        {
            "setup": model + chain + "L = _chain(6.0e-4, KC, KM)\n",
            "call": "float(10.0 * close_optimal_transporter(L.copy(), KC.copy(), KM.copy(), 40.0, 1.0e-3, _demand, _cost)[1])",
            "gold_call": "float(10.0 * _oracle_close_optimal_transporter(L.copy(), KC.copy(), KM.copy(), 40.0, 1.0e-3, _demand, _cost)[1])",
        },
        {
            "setup": model + (
                "KC = np.array([31.0, 9.0, 17.0, 12.5, 40.0, 6.8, 22.0, 15.0])\n"
                "KM = np.array([2.2e-4, 8.0e-4, 1.4e-3, 3.5e-4, 6.0e-5, 1.0e-3, 4.8e-4, 2.6e-3])\n"
                "L = _chain(9.0e-4, KC, KM)\n"
            ),
            "call": "_tsig(close_optimal_transporter(L.copy(), KC.copy(), KM.copy(), 18.0, 2.5e-3, _demand, _cost))",
            "gold_call": "_tsig(_oracle_close_optimal_transporter(L.copy(), KC.copy(), KM.copy(), 18.0, 2.5e-3, _demand, _cost))",
        },
        {
            "setup": model + "KC = np.array([12.0]); KM = np.array([6.0e-4]); L = np.array([3.0e-3])\n",
            "call": "_tsig(close_optimal_transporter(L.copy(), KC.copy(), KM.copy(), 25.0, 8.0e-4, _demand, _cost))",
            "gold_call": "_tsig(_oracle_close_optimal_transporter(L.copy(), KC.copy(), KM.copy(), 25.0, 8.0e-4, _demand, _cost))",
        },
        {
            "setup": model + chain + (
                "L = _chain(2.0e-3, KC, KM)\n"
                "heavy = lambda r, kc, k: 1.5 * _demand(r, kc, k)\n"
            ),
            "call": "_tsig(close_optimal_transporter(L.copy(), KC.copy(), KM.copy(), 40.0, 1.0e-3, heavy, _cost))",
            "gold_call": "_tsig(_oracle_close_optimal_transporter(L.copy(), KC.copy(), KM.copy(), 40.0, 1.0e-3, heavy, _cost))",
        },
        {
            "setup": model + status + chain + "L = _chain(9.0e-3, KC, KM)\n",
            "call": "_status(lambda: close_optimal_transporter(L.copy(), KC.copy(), KM.copy(), 2.0, 1.0e-3, _demand, _cost))",
            "gold_call": "_status(lambda: _oracle_close_optimal_transporter(L.copy(), KC.copy(), KM.copy(), 2.0, 1.0e-3, _demand, _cost))",
        },
        {
            "setup": model + status + chain + "L = _chain(2.0e-3, KC, KM)\n",
            "call": "_status(lambda: close_optimal_transporter(L.copy(), KC.copy(), KM.copy(), 40.0, 0.0, _demand, _cost))",
            "gold_call": "_status(lambda: _oracle_close_optimal_transporter(L.copy(), KC.copy(), KM.copy(), 40.0, 0.0, _demand, _cost))",
        },
    ]
