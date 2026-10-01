"""
Close a growth-optimal set of internal metabolite levels at the uptake step when the transporter amount is set by nutrient-dependent regulation, returning the external nutrient level, the transporter amount and the growth rate.

Cells answer a falling nutrient level by making more transporter, but only up to a bounded fold change, so under limitation the uptake step is no longer free to take its growth-optimal share of the proteome.

Returns
-------
np.ndarray: [external nutrient level, transporter amount, growth rate].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def close_regulated_transporter(
    levels: "np.ndarray",
    kcat: "np.ndarray",
    km: "np.ndarray",
    kcat_t: float,
    km_t: float,
    basal_fraction: float,
    max_fold: float,
    demand_fn: "Callable[..., np.ndarray]",
) -> "np.ndarray":
    """Return the nutrient level, transporter amount and growth rate under regulated uptake.

    The pathway, units and balanced-growth fluxes are those of
    ``compute_enzyme_demand``, and the transporter, its rate law, the supply
    balance of metabolite ``0`` and the proteome budget
    ``phi_t + sum(phi) = 1`` are those of ``close_optimal_transporter``. The
    transporter amount is not chosen for growth but follows the external
    nutrient level ``nu`` as
    ``phi_t(nu) = basal_fraction * (1 + (max_fold - 1) / (1 + max_fold * nu / km_t))``.
    Given the internal levels, return the positive external level ``nu`` at
    which balanced growth holds with this transporter amount, together with
    ``phi_t(nu)`` and the growth rate. There is at most one such ``nu``.
    ``demand_fn(levels, kcat, km)`` follows the contract of
    ``compute_enzyme_demand``; use it for the enzyme demand.

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
    basal_fraction : float
        Transporter amount at saturating nutrient, in ``(0, 1)``.
    max_fold : float
        Largest fold increase of the transporter amount, at least 1, with
        ``basal_fraction * max_fold < 1``.
    demand_fn : callable
        Function with the contract of ``compute_enzyme_demand``.

    Returns
    -------
    np.ndarray
        Float array ``[nu, phi_t, lam]``.

    Raises
    ------
    ValueError
        If ``levels``, ``kcat`` or ``km`` is not a non-empty one-dimensional
        array of finite positive numbers or they differ in length, if a
        scalar argument is outside its stated domain (booleans are rejected),
        if ``demand_fn`` is not callable or returns anything other than ``n``
        finite positive numbers, or if no positive ``nu`` sustains these
        levels.
    """
    return state

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_close_regulated_transporter(
    levels: "np.ndarray",
    kcat: "np.ndarray",
    km: "np.ndarray",
    kcat_t: float,
    km_t: float,
    basal_fraction: float,
    max_fold: float,
    demand_fn: "Callable[..., np.ndarray]",
) -> "np.ndarray":
    """Reference implementation (closed-form solution of budget and supply balance)."""
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
    if not (_is_number(basal_fraction) and 0.0 < basal_fraction < 1.0):
        raise ValueError("basal_fraction must lie in (0, 1)")
    if not (_is_number(max_fold) and max_fold >= 1.0 and basal_fraction * max_fold < 1.0):
        raise ValueError("max_fold must be at least 1 with basal_fraction * max_fold < 1")
    if not _is_function(demand_fn):
        raise ValueError("demand_fn must be callable")
    demand = _positive_vector(demand_fn(rho.copy(), rate.copy(), affinity.copy()), "demand", rho.size)
    # Budget: phi_t = 1 - lam * D. Supply: j_t = lam * (1 + S). With
    # u = max_fold * nu / km_t the regulated uptake is kcat_t * basal * u / (1 + u)
    # and phi_t = basal * (u + max_fold) / (1 + u), so the balance is linear in u.
    ratio = (1.0 + rho.sum()) / demand.sum()
    numerator = ratio * (1.0 - basal_fraction * max_fold)
    denominator = float(kcat_t) * basal_fraction - ratio * (1.0 - basal_fraction)
    if not denominator > 0.0:
        raise ValueError("the regulated transporter cannot supply these levels")
    u = numerator / denominator
    nu = u * float(km_t) / float(max_fold)
    phi_t = basal_fraction * (u + max_fold) / (1.0 + u)
    lam = (1.0 - phi_t) / demand.sum()
    return np.array([nu, phi_t, lam])

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
            "setup": model + chain + "L = _chain(6.0e-4, KC, KM)\n",
            "call": "_tsig(close_regulated_transporter(L.copy(), KC.copy(), KM.copy(), 40.0, 1.0e-3, 0.05, 3.0, _demand))",
            "gold_call": "_tsig(_oracle_close_regulated_transporter(L.copy(), KC.copy(), KM.copy(), 40.0, 1.0e-3, 0.05, 3.0, _demand))",
        },
        {
            "setup": model + chain + "L = _chain(2.0e-4, KC, KM)\n",
            "call": "float(10.0 * close_regulated_transporter(L.copy(), KC.copy(), KM.copy(), 40.0, 1.0e-3, 0.05, 3.0, _demand)[1])",
            "gold_call": "float(10.0 * _oracle_close_regulated_transporter(L.copy(), KC.copy(), KM.copy(), 40.0, 1.0e-3, 0.05, 3.0, _demand)[1])",
        },
        {
            "setup": model + (
                "KC = np.array([31.0, 9.0, 17.0, 12.5, 40.0, 6.8, 22.0, 15.0])\n"
                "KM = np.array([2.2e-4, 8.0e-4, 1.4e-3, 3.5e-4, 6.0e-5, 1.0e-3, 4.8e-4, 2.6e-3])\n"
                "L = _chain(4.0e-4, KC, KM)\n"
            ),
            "call": "_tsig(close_regulated_transporter(L.copy(), KC.copy(), KM.copy(), 18.0, 2.5e-3, 0.08, 4.5, _demand))",
            "gold_call": "_tsig(_oracle_close_regulated_transporter(L.copy(), KC.copy(), KM.copy(), 18.0, 2.5e-3, 0.08, 4.5, _demand))",
        },
        {
            "setup": model + chain + "L = _chain(6.0e-4, KC, KM)\n",
            "call": "_tsig(close_regulated_transporter(L.copy(), KC.copy(), KM.copy(), 40.0, 1.0e-3, 0.05, 1.0, _demand))",
            "gold_call": "_tsig(_oracle_close_regulated_transporter(L.copy(), KC.copy(), KM.copy(), 40.0, 1.0e-3, 0.05, 1.0, _demand))",
        },
        {
            "setup": model + "KC = np.array([12.0]); KM = np.array([6.0e-4]); L = np.array([1.0e-3])\n",
            "call": "_tsig(close_regulated_transporter(L.copy(), KC.copy(), KM.copy(), 120.0, 8.0e-4, 0.1, 2.0, _demand))",
            "gold_call": "_tsig(_oracle_close_regulated_transporter(L.copy(), KC.copy(), KM.copy(), 120.0, 8.0e-4, 0.1, 2.0, _demand))",
        },
        {
            "setup": model + chain + (
                "L = _chain(6.0e-4, KC, KM)\n"
                "heavy = lambda r, kc, k: 1.5 * _demand(r, kc, k)\n"
            ),
            "call": "_tsig(close_regulated_transporter(L.copy(), KC.copy(), KM.copy(), 40.0, 1.0e-3, 0.05, 3.0, heavy))",
            "gold_call": "_tsig(_oracle_close_regulated_transporter(L.copy(), KC.copy(), KM.copy(), 40.0, 1.0e-3, 0.05, 3.0, heavy))",
        },
        {
            "setup": model + status + chain + "L = _chain(6.0e-3, KC, KM)\n",
            "call": "_status(lambda: close_regulated_transporter(L.copy(), KC.copy(), KM.copy(), 4.0, 1.0e-3, 0.05, 3.0, _demand))",
            "gold_call": "_status(lambda: _oracle_close_regulated_transporter(L.copy(), KC.copy(), KM.copy(), 4.0, 1.0e-3, 0.05, 3.0, _demand))",
        },
        {
            "setup": model + status + chain + "L = _chain(6.0e-4, KC, KM)\n",
            "call": "_status(lambda: close_regulated_transporter(L.copy(), KC.copy(), KM.copy(), 40.0, 1.0e-3, 0.4, 3.0, _demand))",
            "gold_call": "_status(lambda: _oracle_close_regulated_transporter(L.copy(), KC.copy(), KM.copy(), 40.0, 1.0e-3, 0.4, 3.0, _demand))",
        },
    ]
