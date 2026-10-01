"""
Convert each bond's tabulated potential of mean force into its bond-resolved transition-state rupture rate, with the dividing surface at the bond's rupture threshold and the intact basin below it.

In transition-state theory, a rate is the equilibrium one-sided flux through a dividing surface divided by the population of the reactant basin. For a single bond-length coordinate, both are determined by its potential of mean force, so any constant offset in that profile cancels.

Returns
-------
np.ndarray: float array of shape (n_bonds,) with the bond-resolved transition-state rupture rates in s^-1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_bond_scission_rates(
    pmf_table: "np.ndarray",
    barrier_tops: "np.ndarray",
    prefactors: "np.ndarray",
) -> "np.ndarray":
    """Return the transition-state rupture rate of every bond in s^-1.

    Row ``i`` of ``pmf_table`` holds ``beta W_i`` in units of k_B T, up to an
    arbitrary additive constant per row: column 0 at the reduced rupture
    threshold ``barrier_tops[i]``, and columns ``1..n_length`` at the
    ``n_length`` Gauss-Legendre nodes of ``[0.5, barrier_tops[i]]`` in
    ``leggauss`` order. The intact basin of bond ``i`` is ``0.5 <= x <=
    barrier_tops[i]`` and its population integral uses that same
    Gauss-Legendre rule. ``prefactors[i]`` is the kinetic frequency prefactor
    of bond ``i`` for the reduced length coordinate, in s^-1.

    Parameters
    ----------
    pmf_table : np.ndarray
        Finite float array of shape ``(n_bonds, n_length + 1)``, ``n_length >= 2``.
    barrier_tops : np.ndarray
        Finite float array of shape ``(n_bonds,)`` with every entry above 0.5.
    prefactors : np.ndarray
        Finite positive float array of shape ``(n_bonds,)``.

    Returns
    -------
    np.ndarray
        Float array of shape ``(n_bonds,)`` in s^-1.

    Raises
    ------
    ValueError
        If ``pmf_table`` is not a finite two-dimensional array with at least
        one row and three columns, if ``barrier_tops`` is not a finite
        one-dimensional array of matching length with entries above 0.5, or if
        ``prefactors`` is not a finite positive one-dimensional array of
        matching length.
    """
    return rates

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_bond_scission_rates(
    pmf_table: "np.ndarray",
    barrier_tops: "np.ndarray",
    prefactors: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation of the full (non-harmonic) transition-state rate."""
    import numpy as np
    from scipy.special import logsumexp

    table = np.asarray(pmf_table, dtype=float)
    tops = np.asarray(barrier_tops, dtype=float)
    nu = np.asarray(prefactors, dtype=float)
    if table.ndim != 2 or table.shape[0] < 1 or table.shape[1] < 3 or not np.all(np.isfinite(table)):
        raise ValueError("pmf_table must be a finite 2D array with at least one row and three columns")
    n_bonds, n_length = table.shape[0], table.shape[1] - 1
    if tops.shape != (n_bonds,) or not np.all(np.isfinite(tops)) or np.any(tops <= 0.5):
        raise ValueError("barrier_tops must match pmf_table and lie above 0.5")
    if nu.shape != (n_bonds,) or not np.all(np.isfinite(nu)) or np.any(nu <= 0):
        raise ValueError("prefactors must match pmf_table and be finite and positive")
    _, node_weights = np.polynomial.legendre.leggauss(n_length)
    rates = np.empty(n_bonds)
    for i in range(n_bonds):
        half = 0.5 * (tops[i] - 0.5)
        # Normalized Boltzmann density of the PMF at the threshold, times the mean positive velocity.
        log_population = logsumexp(np.log(half * node_weights) - (table[i, 1:] - table[i, 0]))
        rates[i] = nu[i] * np.exp(-log_population)
    return rates

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    digest = (
        "import numpy as np\n"
        "def _sig(values):\n"
        "    arr = np.asarray(values, dtype=float)\n"
        "    logs = np.log(np.abs(arr.ravel()) + 1.0e-300)\n"
        "    weight = 1.0 + 0.5 * np.cos(0.7 * np.arange(logs.size))\n"
        "    return float(arr.shape[0] + np.sum(weight * logs))\n"
        "def table_for(tops, n, curvature, cubic, shift):\n"
        "    t, _ = np.polynomial.legendre.leggauss(n)\n"
        "    rows = []\n"
        "    for top, c2, c3, c0 in zip(tops, curvature, cubic, shift):\n"
        "        half = 0.5 * (top - 0.5)\n"
        "        x = np.concatenate(([top], half * t + 0.5 + half))\n"
        "        rows.append(c2 * (x - 1.1) ** 2 - c3 * (x - 1.1) ** 3 + c0)\n"
        "    return np.array(rows)\n"
    )
    raises = (
        "import numpy as np\n"
        "def _candidate():\n    try:\n        compute_bond_scission_rates({args})\n        return 0\n"
        "    except ValueError:\n        return 1\n    except Exception:\n        return 2\n"
        "def _reference():\n    try:\n        _oracle_compute_bond_scission_rates({args})\n        return 0\n"
        "    except ValueError:\n        return 1\n    except Exception:\n        return 2\n"
    )
    invalid = [
        "np.zeros((2, 5)), np.array([1.6, 0.4]), np.array([1.0e12, 1.0e12])",
        "np.zeros((2, 5)), np.array([1.6, 1.7]), np.array([1.0e12, -1.0e12])",
        "np.zeros((2, 5)), np.array([1.6, 1.7, 1.8]), np.ones(2)",
        "np.zeros((1, 2)), np.array([1.6]), np.ones(1)",
    ]
    return [
        {
            "setup": digest + "tab = table_for([1.6, 1.7], 40, [600.0, 500.0], [700.0, 450.0], [0.0, 0.0])\n",
            "call": "_sig(compute_bond_scission_rates(tab.copy(), np.array([1.6, 1.7]), np.array([1.18e12, 1.67e12])))",
            "gold_call": "_sig(_oracle_compute_bond_scission_rates(tab.copy(), np.array([1.6, 1.7]), np.array([1.18e12, 1.67e12])))",
        },
        {
            "setup": digest + "tab = table_for([1.6, 1.7], 40, [600.0, 500.0], [700.0, 450.0], [55.0, -30.0])\n",
            "call": "_sig(compute_bond_scission_rates(tab.copy(), np.array([1.6, 1.7]), np.array([1.18e12, 1.67e12])))",
            "gold_call": "_sig(_oracle_compute_bond_scission_rates(tab.copy(), np.array([1.6, 1.7]), np.array([1.18e12, 1.67e12])))",
        },
        {
            "setup": digest + "tab = table_for([2.3], 2, [20.0], [1.0], [0.0])\n",
            "call": "_sig(compute_bond_scission_rates(tab.copy(), np.array([2.3]), np.array([3.0e11])))",
            "gold_call": "_sig(_oracle_compute_bond_scission_rates(tab.copy(), np.array([2.3]), np.array([3.0e11])))",
        },
        {
            "setup": digest + "tab = table_for([1.5, 1.9, 1.8], 120, [900.0, 300.0, 400.0], [1200.0, 150.0, 300.0], [1.0, 2.0, 3.0])\n",
            "call": "_sig(compute_bond_scission_rates(tab.copy(), np.array([1.5, 1.9, 1.8]), np.array([1.0, 2.0, 3.0])))",
            "gold_call": "_sig(_oracle_compute_bond_scission_rates(tab.copy(), np.array([1.5, 1.9, 1.8]), np.array([1.0, 2.0, 3.0])))",
        },
        {
            "setup": digest + "tab = table_for([0.50000001], 3, [2.0], [0.5], [700.0])\n",
            "call": "_sig(compute_bond_scission_rates(tab.copy(), np.array([0.50000001]), np.array([1.0e12])))",
            "gold_call": "_sig(_oracle_compute_bond_scission_rates(tab.copy(), np.array([0.50000001]), np.array([1.0e12])))",
        },
    ] + [
        {"setup": raises.replace("{args}", args), "call": "_candidate()", "gold_call": "_reference()"}
        for args in invalid
    ]
