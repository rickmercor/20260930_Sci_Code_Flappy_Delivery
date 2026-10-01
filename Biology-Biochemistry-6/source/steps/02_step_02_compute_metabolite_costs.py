"""
Evaluate, for every metabolite of a linear pathway, the proteome cost per unit metabolite level at which its current level would be the cheapest way to carry the enzyme's flux.

A higher substrate level saturates the consuming enzyme and saves enzyme, but a metabolite is diluted by growth and so has a proteome price of its own; each level is optimal only for one price.

Returns
-------
np.ndarray: per-growth-rate cost per unit level at which each level is optimal, shape (n,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_metabolite_costs(levels: "np.ndarray", kcat: "np.ndarray", km: "np.ndarray") -> "np.ndarray":
    """Return the cost per unit level at which each metabolite level is optimal.

    Use the pathway, units and balanced-growth fluxes of
    ``compute_enzyme_demand``: enzyme ``i`` consumes metabolite ``i`` at level
    ``levels[i]`` and its flux per unit growth rate is fixed by the levels of
    the metabolites downstream of it. For metabolite ``i``, hold that flux
    fixed and let ``g_i(x)`` be the proteome fraction per unit growth rate
    that enzyme ``i`` would then need if its substrate were at level ``x``.
    Return, for every ``i``, the cost ``c_i`` (proteome fraction per unit
    metabolite level, per unit growth rate) for which ``x = levels[i]``
    minimises ``c_i * x + g_i(x)`` over ``x > 0``.

    Parameters
    ----------
    levels : np.ndarray
        Positive internal metabolite levels, shape ``(n,)`` with ``n >= 1``.
    kcat : np.ndarray
        Positive catalytic constants per unit enzyme mass fraction, shape
        ``(n,)``.
    km : np.ndarray
        Positive Michaelis constants in the units of ``levels``, shape
        ``(n,)``.

    Returns
    -------
    np.ndarray
        Float array of shape ``(n,)``: entry ``i`` is ``c_i``.

    Raises
    ------
    ValueError
        If any argument is not a non-empty one-dimensional array of finite
        positive numbers, or if the three arrays differ in length.
    """
    return costs

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_metabolite_costs(levels: "np.ndarray", kcat: "np.ndarray", km: "np.ndarray") -> "np.ndarray":
    """Reference implementation (stationarity of c * x + g_i(x) at the given level)."""
    import numpy as np

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

    rho = _positive_vector(levels, "levels")
    rate = _positive_vector(kcat, "kcat")
    affinity = _positive_vector(km, "km")
    if not (rho.shape == rate.shape == affinity.shape):
        raise ValueError("levels, kcat and km must have the same length")
    # g_i(x) = flux_i (1 + km_i / x) / kcat_i with flux_i fixed; the minimiser of
    # c x + g_i(x) is x = sqrt(flux_i km_i / (kcat_i c)), inverted for c here.
    downstream = np.concatenate([np.cumsum(rho[::-1])[::-1][1:], [0.0]])
    flux = 1.0 + downstream
    return affinity * flux / (rate * rho * rho)

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
        "R = np.array([2.5e-3, 9.0e-3, 7.0e-4, 1.1e-2, 4.0e-3])\n"
        "KC = np.array([14.0, 26.0, 8.0, 33.0, 11.0])\n"
        "KM = np.array([3.0e-4, 1.6e-3, 1.2e-4, 9.0e-4, 5.0e-4])\n"
    )
    return [
        {
            "setup": helpers + chain,
            "call": "_lsig(compute_metabolite_costs(R.copy(), KC.copy(), KM.copy()), 5)",
            "gold_call": "_lsig(_oracle_compute_metabolite_costs(R.copy(), KC.copy(), KM.copy()), 5)",
        },
        {
            "setup": helpers + chain,
            "call": "float(np.log(compute_metabolite_costs(R.copy(), KC.copy(), KM.copy())[0]))",
            "gold_call": "float(np.log(_oracle_compute_metabolite_costs(R.copy(), KC.copy(), KM.copy())[0]))",
        },
        {
            "setup": helpers + (
                "R = np.array([6.0e-3]); KC = np.array([21.0]); KM = np.array([8.0e-4])\n"
            ),
            "call": "_lsig(compute_metabolite_costs(R.copy(), KC.copy(), KM.copy()), 1)",
            "gold_call": "_lsig(_oracle_compute_metabolite_costs(R.copy(), KC.copy(), KM.copy()), 1)",
        },
        {
            "setup": helpers + (
                "R = np.array([0.25, 0.4, 0.15])\n"
                "KC = np.array([6.0, 9.5, 12.0])\n"
                "KM = np.array([2.0e-3, 5.0e-3, 1.0e-3])\n"
            ),
            "call": "_lsig(compute_metabolite_costs(R.copy(), KC.copy(), KM.copy()), 3)",
            "gold_call": "_lsig(_oracle_compute_metabolite_costs(R.copy(), KC.copy(), KM.copy()), 3)",
        },
        {
            "setup": helpers + (
                "R = np.array([1.0e-5, 3.0e-3, 2.0e-2, 5.0e-4, 8.0e-3, 1.5e-3, 4.0e-2])\n"
                "KC = np.array([10.0, 17.0, 45.0, 7.0, 25.0, 13.0, 30.0])\n"
                "KM = np.array([2.0e-4, 6.0e-4, 3.0e-3, 1.0e-4, 1.0e-3, 4.0e-4, 2.5e-3])\n"
            ),
            "call": "_lsig(compute_metabolite_costs(R.copy(), KC.copy(), KM.copy()), 7)",
            "gold_call": "_lsig(_oracle_compute_metabolite_costs(R.copy(), KC.copy(), KM.copy()), 7)",
        },
        {
            "setup": helpers + status + chain,
            "call": "_status(lambda: compute_metabolite_costs(R.copy(), KC.copy(), np.zeros(5)))",
            "gold_call": "_status(lambda: _oracle_compute_metabolite_costs(R.copy(), KC.copy(), np.zeros(5)))",
        },
        {
            "setup": helpers + status + chain,
            "call": "_status(lambda: compute_metabolite_costs(R.reshape(1, 5), KC.copy(), KM.copy()))",
            "gold_call": "_status(lambda: _oracle_compute_metabolite_costs(R.reshape(1, 5), KC.copy(), KM.copy()))",
        },
    ]
