"""
Equilibrium genetic variance relative to its neutral value along a grid of selection strengths.

How much genetic variance a trait keeps at equilibrium does not change monotonically

with the strength of stabilizing selection. When mutation is biased away from the

optimum, selection at the trait level turns into directional selection at the loci that

counteracts the bias and can recentre allele frequencies, which raises the variance;

stronger selection also acts directly against heterozygotes, which depletes it. This

step traces the variance along a grid of selection strengths given in natural units:

the population size in diploid individuals, the mutation rates per locus per generation

and the width of the fitness function, which it converts to the source's scaling before

solving the equilibrium at each strength.

Returns
-------
np.ndarray, shape (G,): the equilibrium total genetic variance divided by the neutral total genetic variance at each grid strength
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def variance_ratio_curve(n_diploid: int, mu_plus: "np.ndarray", mu_minus: "np.ndarray", alpha: "np.ndarray",
                         dom: "np.ndarray", counts: "np.ndarray", eta: float, omega_inv2: "np.ndarray",
                         n_nodes: int, tol: float) -> "np.ndarray":
    '''Equilibrium genetic variance relative to its neutral value along a grid of selection strengths.

    The population has n_diploid diploid individuals; class k has counts[k] loci with
    additive effect alpha[k], dominance deviation dom[k] and per-generation mutation
    rates mu_plus[k] (towards the trait-increasing allele) and mu_minus[k]; log fitness
    is Gaussian in the trait with optimum eta and inverse squared width omega_inv2. For
    each entry of omega_inv2, convert the population size, the mutation rates and the
    selection strength to the source's scaled parameters, solve the equilibrium
    deviation of the mean trait as in the earlier steps, and return the total genetic
    variance (additive plus dominance) at that equilibrium divided by the total genetic
    variance under mutation and drift alone.

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
    omega_inv2 : np.ndarray
        Shape (G,): selection strengths, the inverse squared widths of the Gaussian
        fitness function in inverse squared trait units, each >= 0.
    n_nodes : int
        Number of quadrature nodes passed to the earlier steps, >= 2.
    tol : float
        Absolute accuracy of each equilibrium deviation, in trait units, > 0.

    Returns
    -------
    ratio : np.ndarray
        Shape (G,): the equilibrium total genetic variance divided by the neutral
        total genetic variance, one entry per selection strength.

    Raises
    ------
    ValueError
        If n_diploid is not an integer >= 1, if mu_plus or mu_minus is not a
        one-dimensional array of finite numbers > 0 of the same length as the other
        class arrays, if omega_inv2 is not a one-dimensional array of finite numbers
        >= 0 with at least one entry, or on any condition raised by the earlier steps.
    '''
    return ratio  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _class_arrays(alpha, dom, theta_plus, theta_minus, counts):
    arrs = []
    for name, v in (("alpha", alpha), ("dom", dom), ("theta_plus", theta_plus), ("theta_minus", theta_minus)):
        if isinstance(v, (str, bytes)):
            raise ValueError(f"{name} must be a one-dimensional array of finite numbers")
        a = np.asarray(v, dtype=float)
        if a.ndim != 1 or a.size < 1 or not np.all(np.isfinite(a)):
            raise ValueError(f"{name} must be a one-dimensional array of finite numbers")
        arrs.append(a)
    if isinstance(counts, (str, bytes)):
        raise ValueError("counts must be a one-dimensional array of positive integers")
    c = np.asarray(counts)
    if c.ndim != 1 or c.size < 1 or c.dtype.kind not in "iu" or np.any(c < 1):
        raise ValueError("counts must be a one-dimensional array of positive integers")
    if len({a.size for a in arrs} | {c.size}) != 1:
        raise ValueError("the class arrays must all have the same length")
    a, d, tp, tm = arrs
    if np.any(a <= 0.0) or np.any(tp <= 0.0) or np.any(tm <= 0.0):
        raise ValueError("alpha, theta_plus and theta_minus entries must be > 0")
    return a, d, tp, tm, c.astype(int)


def _scaled_architecture(n_diploid, mu_plus, mu_minus, alpha, dom, counts):
    """Natural units to the source's scaling: theta = 2 N mu, with N the number of diploid individuals."""
    if isinstance(n_diploid, bool) or not isinstance(n_diploid, (int, np.integer)) or int(n_diploid) < 1:
        raise ValueError("n_diploid must be an integer >= 1")
    for name, v in (("mu_plus", mu_plus), ("mu_minus", mu_minus)):
        if isinstance(v, (str, bytes)):
            raise ValueError(f"{name} must be a one-dimensional array of finite numbers > 0")
        m = np.asarray(v, dtype=float)
        if m.ndim != 1 or m.size < 1 or not np.all(np.isfinite(m)) or np.any(m <= 0.0):
            raise ValueError(f"{name} must be a one-dimensional array of finite numbers > 0")
    two_n = 2.0 * int(n_diploid)
    tp = two_n * np.asarray(mu_plus, dtype=float)
    tm = two_n * np.asarray(mu_minus, dtype=float)
    return _class_arrays(alpha, dom, tp, tm, counts)


def _oracle_variance_ratio_curve(n_diploid: int, mu_plus: "np.ndarray", mu_minus: "np.ndarray", alpha: "np.ndarray",
                                 dom: "np.ndarray", counts: "np.ndarray", eta: float, omega_inv2: "np.ndarray",
                                 n_nodes: int, tol: float) -> "np.ndarray":
    a, d, tp, tm, c = _scaled_architecture(n_diploid, mu_plus, mu_minus, alpha, dom, counts)
    if isinstance(omega_inv2, (str, bytes)):
        raise ValueError("omega_inv2 must be a one-dimensional array of finite numbers >= 0")
    g = np.asarray(omega_inv2, dtype=float)
    if g.ndim != 1 or g.size < 1 or not np.all(np.isfinite(g)) or np.any(g < 0.0):
        raise ValueError("omega_inv2 must be a one-dimensional array of finite numbers >= 0")
    neutral = _oracle_neutral_variance_components(a, d, tp, tm, c)
    s0 = float(neutral[0] + neutral[1])
    two_n = 2.0 * int(n_diploid)
    out = np.empty(g.size)
    for i, w in enumerate(g):
        sel_ratio = two_n * float(w)                    # the source's selection-drift ratio, 2 N / omega^2
        delta = _oracle_equilibrium_trait_deviation(sel_ratio, eta, a, d, tp, tm, c, n_nodes, tol)
        mv = _oracle_architecture_mean_and_variances(delta, sel_ratio, a, d, tp, tm, c, n_nodes)
        out[i] = (mv[1] + mv[2]) / s0
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the shipped configuration on the part of its grid around the crossing ---
        {
            "setup": "import numpy as np\nmup = np.array([3e-06, 5e-06, 1e-05])\nmum = np.array([2e-05, 1.5e-05, 1e-05])\n"
                     "alpha = np.array([0.012, 0.008, 0.010])\ndom = np.array([0.0, 0.007, -0.008])\ncounts = np.array([60, 40, 20])\n"
                     "grid = 10.0 ** (np.arange(-4, 5) / 10.0)\n",
            "call": "variance_ratio_curve(10000, mup, mum, alpha, dom, counts, 1.15, grid, 400, 1e-13)",
            "gold_call": "_oracle_variance_ratio_curve(10000, mup, mum, alpha, dom, counts, 1.15, grid, 400, 1e-13)",
            "tol": 1e-07,
        },
        # --- boundary: a single strength of zero (mutation and drift alone) next to a weak one, one additive class ---
        {
            "setup": "import numpy as np\nmup = np.array([2e-05])\nmum = np.array([6e-05])\nalpha = np.array([0.02])\ndom = np.array([0.0])\n"
                     "counts = np.array([50])\ngrid = np.array([0.0, 0.05])\n",
            "call": "variance_ratio_curve(2500, mup, mum, alpha, dom, counts, 1.1, grid, 400, 1e-13)",
            "gold_call": "_oracle_variance_ratio_curve(2500, mup, mum, alpha, dom, counts, 1.1, grid, 400, 1e-13)",
            "tol": 1e-07,
        },
        # --- edge: a small population with upward mutational bias and an optimum below the mutational one, coarse grid ---
        {
            "setup": "import numpy as np\nmup = np.array([1e-04, 5e-05])\nmum = np.array([2.5e-05, 5e-05])\nalpha = np.array([0.015, 0.01])\n"
                     "dom = np.array([0.012, -0.009])\ncounts = np.array([30, 70])\ngrid = np.array([0.02, 0.2, 2.0, 20.0])\n",
            "call": "variance_ratio_curve(1000, mup, mum, alpha, dom, counts, 0.7, grid, 400, 1e-13)",
            "gold_call": "_oracle_variance_ratio_curve(1000, mup, mum, alpha, dom, counts, 0.7, grid, 400, 1e-13)",
            "tol": 1e-07,
        },
    ]
