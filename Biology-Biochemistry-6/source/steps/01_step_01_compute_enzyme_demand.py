"""
Evaluate the proteome fraction that each enzyme of a linear pathway needs, per unit growth rate, to hold its metabolites at given levels in balanced exponential growth.

Growth dilutes every intracellular metabolite, so an enzyme must also make up for the losses of everything downstream of it, and how much of it is needed depends on how saturated it is.

Returns
-------
np.ndarray: proteome fraction per unit growth rate of every enzyme, shape (n,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_enzyme_demand(levels: "np.ndarray", kcat: "np.ndarray", km: "np.ndarray") -> "np.ndarray":
    """Return each pathway enzyme's proteome fraction per unit growth rate.

    Enzyme ``i`` (``i = 0, ..., n - 1`` in pathway order) consumes internal
    metabolite ``i``, held at level ``levels[i]``, and produces metabolite
    ``i + 1``; enzyme ``n - 1`` produces protein. Levels and enzyme amounts
    are mass fractions of total protein and fluxes are per unit total
    protein. In balanced growth at rate ``lam``, enzyme ``n - 1`` makes
    protein at the flux ``lam``, each metabolite ``k >= 1`` is produced by
    enzyme ``k - 1`` faster than enzyme ``k`` consumes it by exactly its
    dilution ``lam * levels[k]``, and enzyme ``i`` of amount ``phi_i``
    carries its flux at the irreversible Michaelis-Menten rate
    ``kcat[i] * phi_i * levels[i] / (km[i] + levels[i])``. Return
    ``phi_i / lam`` for every enzyme.

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
        Float array of shape ``(n,)``: entry ``i`` is ``phi_i / lam``.

    Raises
    ------
    ValueError
        If any argument is not a non-empty one-dimensional array of finite
        positive numbers, or if the three arrays differ in length.
    """
    return demand

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_enzyme_demand(levels: "np.ndarray", kcat: "np.ndarray", km: "np.ndarray") -> "np.ndarray":
    """Reference implementation (backward accumulation of the diluted flux)."""
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
    # Flux of enzyme i per unit growth rate: the protein flux (1) plus the
    # dilution of every metabolite downstream of its product, i + 1 .. n - 1.
    downstream = np.concatenate([np.cumsum(rho[::-1])[::-1][1:], [0.0]])
    flux = 1.0 + downstream
    return flux * (1.0 + affinity / rho) / rate

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
    chain = (
        "R = np.array([3.0e-3, 1.2e-2, 5.0e-4, 8.0e-3])\n"
        "KC = np.array([12.0, 30.0, 7.5, 20.0])\n"
        "KM = np.array([4.0e-4, 2.0e-3, 1.0e-4, 6.0e-4])\n"
    )
    return [
        {
            "setup": helpers + chain,
            "call": "_vsig(compute_enzyme_demand(R.copy(), KC.copy(), KM.copy()), 4)",
            "gold_call": "_vsig(_oracle_compute_enzyme_demand(R.copy(), KC.copy(), KM.copy()), 4)",
        },
        {
            "setup": helpers + chain,
            "call": "float(100.0 * compute_enzyme_demand(R.copy(), KC.copy(), KM.copy())[1])",
            "gold_call": "float(100.0 * _oracle_compute_enzyme_demand(R.copy(), KC.copy(), KM.copy())[1])",
        },
        {
            "setup": helpers + (
                "R = np.array([2.0e-3]); KC = np.array([15.0]); KM = np.array([5.0e-4])\n"
            ),
            "call": "_vsig(compute_enzyme_demand(R.copy(), KC.copy(), KM.copy()), 1)",
            "gold_call": "_vsig(_oracle_compute_enzyme_demand(R.copy(), KC.copy(), KM.copy()), 1)",
        },
        {
            "setup": helpers + (
                "R = np.array([4.0e-3, 1.0e-5, 2.0e-2, 6.0e-3, 9.0e-4, 3.0e-2])\n"
                "KC = np.array([9.0, 22.0, 14.0, 41.0, 6.5, 18.0])\n"
                "KM = np.array([3.0e-4, 1.0e-3, 7.0e-4, 2.0e-4, 1.5e-3, 5.0e-4])\n"
            ),
            "call": "_vsig(compute_enzyme_demand(R.copy(), KC.copy(), KM.copy()), 6) / 10.0",
            "gold_call": "_vsig(_oracle_compute_enzyme_demand(R.copy(), KC.copy(), KM.copy()), 6) / 10.0",
        },
        {
            "setup": helpers + (
                "R = np.array([0.3, 0.5, 0.2])\n"
                "KC = np.array([5.0, 8.0, 11.0])\n"
                "KM = np.array([1.0e-3, 4.0e-3, 2.0e-3])\n"
            ),
            "call": "_vsig(compute_enzyme_demand(R.copy(), KC.copy(), KM.copy()), 3)",
            "gold_call": "_vsig(_oracle_compute_enzyme_demand(R.copy(), KC.copy(), KM.copy()), 3)",
        },
        {
            "setup": helpers + status + chain,
            "call": "_status(lambda: compute_enzyme_demand(-R, KC.copy(), KM.copy()))",
            "gold_call": "_status(lambda: _oracle_compute_enzyme_demand(-R, KC.copy(), KM.copy()))",
        },
        {
            "setup": helpers + status + chain,
            "call": "_status(lambda: compute_enzyme_demand(R[:3].copy(), KC.copy(), KM.copy()))",
            "gold_call": "_status(lambda: _oracle_compute_enzyme_demand(R[:3].copy(), KC.copy(), KM.copy()))",
        },
    ]
