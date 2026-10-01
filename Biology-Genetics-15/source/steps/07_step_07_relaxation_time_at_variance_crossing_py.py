"""
E-folding time of the mean trait's autocorrelation at the first grid strength where the equilibrium genetic variance falls below its neutral value.

This step returns the deliverable: the relaxation time of the mean trait at the first

grid strength of stabilizing selection at which the equilibrium genetic variance of

the trait falls strictly below the variance it would keep under mutation and drift

alone. It traces the variance along the grid, applies the crossing rule, and evaluates

the equilibrium summaries of the previous step at the selected strength.

Returns
-------
float, the e-folding time of the autocorrelation of the mean trait's deviation at the selected strength, in generations, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def relaxation_time_at_variance_crossing(n_diploid: int, mu_plus: "np.ndarray", mu_minus: "np.ndarray",
                                         alpha: "np.ndarray", dom: "np.ndarray", counts: "np.ndarray", eta: float,
                                         omega_inv2_grid: "np.ndarray", n_nodes: int, tol: float) -> float:
    '''E-folding time of the mean trait's autocorrelation at the first grid strength where the equilibrium genetic variance falls below its neutral value.

    The population and the architecture are as in the earlier steps. Compute the ratio
    of the equilibrium total genetic variance to its neutral value at every entry of
    omega_inv2_grid, in the order given, and select the first entry whose ratio is
    strictly below 1. Return the e-folding time, in generations, of the autocorrelation
    of the deviation of the mean trait from its equilibrium value at that selection
    strength, as given by the previous step.

    Parameters
    ----------
    n_diploid : int
        Number of diploid individuals, >= 1.
    mu_plus : np.ndarray
        Shape (K,): mutation rates per locus per generation towards the
        trait-increasing allele, each > 0.
    mu_minus : np.ndarray
        Shape (K,): mutation rates per locus per generation towards the
        trait-decreasing allele, each > 0.
    alpha : np.ndarray
        Shape (K,): additive effects of the K classes, in trait units, each > 0.
    dom : np.ndarray
        Shape (K,): dominance deviations of the heterozygotes, in trait units.
    counts : np.ndarray
        Shape (K,): numbers of loci in the classes, positive integers.
    eta : float
        Selection optimum, in trait units, finite.
    omega_inv2_grid : np.ndarray
        Shape (G,): the grid of selection strengths, inverse squared widths of the
        Gaussian fitness function in inverse squared trait units, each > 0.
    n_nodes : int
        Number of quadrature nodes passed to the earlier steps, >= 2.
    tol : float
        Absolute accuracy of each equilibrium deviation, in trait units, > 0.

    Returns
    -------
    relaxation_time : float
        The e-folding time of the autocorrelation of the mean trait's deviation at
        the selected strength, in generations, as a native Python float.

    Raises
    ------
    ValueError
        If omega_inv2_grid is not a one-dimensional array of finite numbers > 0 with
        at least one entry, if no entry of the grid gives a variance ratio strictly
        below 1, or on any condition raised by the earlier steps for the other inputs.
    '''
    return relaxation_time  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_relaxation_time_at_variance_crossing(n_diploid: int, mu_plus: "np.ndarray", mu_minus: "np.ndarray",
                                                 alpha: "np.ndarray", dom: "np.ndarray", counts: "np.ndarray", eta: float,
                                                 omega_inv2_grid: "np.ndarray", n_nodes: int, tol: float) -> float:
    if isinstance(omega_inv2_grid, (str, bytes)):
        raise ValueError("omega_inv2_grid must be a one-dimensional array of finite numbers > 0")
    g = np.asarray(omega_inv2_grid, dtype=float)
    if g.ndim != 1 or g.size < 1 or not np.all(np.isfinite(g)) or np.any(g <= 0.0):
        raise ValueError("omega_inv2_grid must be a one-dimensional array of finite numbers > 0")
    ratio = _oracle_variance_ratio_curve(n_diploid, mu_plus, mu_minus, alpha, dom, counts, eta, g, n_nodes, tol)
    below = np.flatnonzero(ratio < 1.0)
    if below.size == 0:
        raise ValueError("no grid strength gives a variance ratio strictly below 1")
    state = _oracle_equilibrium_state_at_strength(n_diploid, mu_plus, mu_minus, alpha, dom, counts, eta,
                                                  float(g[below[0]]), n_nodes, tol)
    return float(state[4])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the shipped configuration and grid; this is the reported answer ---
        {
            "setup": "import numpy as np\nmup = np.array([3e-06, 5e-06, 1e-05])\nmum = np.array([2e-05, 1.5e-05, 1e-05])\n"
                     "alpha = np.array([0.012, 0.008, 0.010])\ndom = np.array([0.0, 0.007, -0.008])\ncounts = np.array([60, 40, 20])\n"
                     "grid = 10.0 ** (np.arange(-20, 11) / 10.0)\n",
            "call": "relaxation_time_at_variance_crossing(10000, mup, mum, alpha, dom, counts, 1.15, grid, 400, 1e-13)",
            "gold_call": "_oracle_relaxation_time_at_variance_crossing(10000, mup, mum, alpha, dom, counts, 1.15, grid, 400, 1e-13)",
            "tol": 1e-07,
        },
        # --- boundary: a coarser grid whose last entry is the first below neutral ---
        {
            "setup": "import numpy as np\nmup = np.array([3e-06, 5e-06, 1e-05])\nmum = np.array([2e-05, 1.5e-05, 1e-05])\n"
                     "alpha = np.array([0.012, 0.008, 0.010])\ndom = np.array([0.0, 0.007, -0.008])\ncounts = np.array([60, 40, 20])\n"
                     "grid = 10.0 ** (np.arange(-4, 2) / 5.0)\n",
            "call": "relaxation_time_at_variance_crossing(10000, mup, mum, alpha, dom, counts, 1.15, grid, 400, 1e-13)",
            "gold_call": "_oracle_relaxation_time_at_variance_crossing(10000, mup, mum, alpha, dom, counts, 1.15, grid, 400, 1e-13)",
            "tol": 1e-07,
        },
        # --- edge: a small population with upward mutational bias, an optimum below the mutational one, a coarse grid ---
        {
            "setup": "import numpy as np\nmup = np.array([1e-04, 5e-05])\nmum = np.array([2.5e-05, 5e-05])\nalpha = np.array([0.015, 0.01])\n"
                     "dom = np.array([0.012, -0.009])\ncounts = np.array([30, 70])\ngrid = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])\n",
            "call": "relaxation_time_at_variance_crossing(1000, mup, mum, alpha, dom, counts, 0.7, grid, 400, 1e-13)",
            "gold_call": "_oracle_relaxation_time_at_variance_crossing(1000, mup, mum, alpha, dom, counts, 0.7, grid, 400, 1e-13)",
            "tol": 1e-07,
        },
    ]
