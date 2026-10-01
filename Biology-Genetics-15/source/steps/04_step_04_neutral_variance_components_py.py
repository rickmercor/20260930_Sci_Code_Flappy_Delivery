"""
Additive and dominance components of the genetic variance under mutation and drift alone, in closed form.

The reference for everything that selection does to the genetic variance is the

variance the same architecture would keep under mutation and drift alone. In a finite

population that reference is not the variance at the deterministic mutation

equilibrium: drift spreads the allele frequencies around it, and the stationary

distribution under reversible mutation is a Beta law whose parameters follow the

source's scaling. Its moments give the additive and dominance components of the

neutral genetic variance in closed form.

Returns
-------
np.ndarray, shape (2,): the additive and the dominance genetic variance under mutation and drift alone
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def neutral_variance_components(alpha: "np.ndarray", dom: "np.ndarray", theta_plus: "np.ndarray",
                                theta_minus: "np.ndarray", counts: "np.ndarray") -> "np.ndarray":
    '''Additive and dominance components of the genetic variance under mutation and drift alone, in closed form.

    Each locus of class k has additive effect alpha[k], dominance deviation dom[k] and
    scaled mutation rates theta_plus[k], theta_minus[k] in the source's scaling; with
    no selection its frequency has the Beta stationary distribution of reversible
    mutation and drift. Return the sums over all loci of the expected additive and
    dominance variances of a locus under Hardy-Weinberg proportions, using the exact
    moments of that distribution.

    Parameters
    ----------
    alpha : np.ndarray
        Shape (K,): additive effects of the K classes, in trait units, each > 0.
    dom : np.ndarray
        Shape (K,): dominance deviations of the heterozygotes, in trait units.
    theta_plus : np.ndarray
        Shape (K,): scaled mutation rates towards the trait-increasing allele, each > 0.
    theta_minus : np.ndarray
        Shape (K,): scaled mutation rates towards the trait-decreasing allele, each > 0.
    counts : np.ndarray
        Shape (K,): numbers of loci in the classes, positive integers.

    Returns
    -------
    out : np.ndarray
        Shape (2,): the additive and the dominance genetic variance under mutation and
        drift alone, in squared trait units.

    Raises
    ------
    ValueError
        If the five class arrays are not one-dimensional of the same length K >= 1
        with finite entries, if any alpha, theta_plus or theta_minus entry is not > 0,
        or if any count is not a positive integer.
    '''
    return out  # placeholder

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


def _oracle_neutral_variance_components(alpha: "np.ndarray", dom: "np.ndarray", theta_plus: "np.ndarray",
                                        theta_minus: "np.ndarray", counts: "np.ndarray") -> "np.ndarray":
    a, d, tp, tm, c = _class_arrays(alpha, dom, theta_plus, theta_minus, counts)
    # Beta(2 theta_plus, 2 theta_minus) moments: E[p^i (1-p)^j] = prod_{r<i}(A+r) prod_{r<j}(B+r) / prod_{r<i+j}(A+B+r)
    A, B = 2.0 * tp, 2.0 * tm

    def _mom(i, j):
        num = np.ones_like(A)
        for r in range(i):
            num = num * (A + r)
        for r in range(j):
            num = num * (B + r)
        den = np.ones_like(A)
        for r in range(i + j):
            den = den * (A + B + r)
        return num / den

    e_pq, e_p2q, e_p3q, e_p2q2 = _mom(1, 1), _mom(2, 1), _mom(3, 1), _mom(2, 2)
    # beta(p) = (a + D) - 2 D p:  E[beta^2 p q] = (a+D)^2 E[pq] - 4 D (a+D) E[p^2 q] + 4 D^2 E[p^3 q]
    e_b2pq = (a + d) ** 2 * e_pq - 4.0 * d * (a + d) * e_p2q + 4.0 * d * d * e_p3q
    va = float(np.sum(c * 2.0 * e_b2pq))
    vd = float(np.sum(c * 4.0 * d * d * e_p2q2))
    return np.array([va, vd])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the shipped three-class architecture ---
        {
            "setup": "import numpy as np\nalpha = np.array([0.012, 0.008, 0.010])\ndom = np.array([0.0, 0.007, -0.008])\n"
                     "tp = np.array([0.06, 0.1, 0.2])\ntm = np.array([0.4, 0.3, 0.2])\ncounts = np.array([60, 40, 20])\n",
            "call": "neutral_variance_components(alpha, dom, tp, tm, counts)",
            "gold_call": "_oracle_neutral_variance_components(alpha, dom, tp, tm, counts)",
            "tol": 1e-07,
        },
        # --- boundary: one class with symmetric mutation and complete dominance ---
        {
            "setup": "import numpy as np\nalpha = np.array([0.01])\ndom = np.array([0.01])\ntp = np.array([0.25])\ntm = np.array([0.25])\n"
                     "counts = np.array([100])\n",
            "call": "neutral_variance_components(alpha, dom, tp, tm, counts)",
            "gold_call": "_oracle_neutral_variance_components(alpha, dom, tp, tm, counts)",
            "tol": 1e-07,
        },
        # --- edge: very low mutation rates (frequencies piled near 0 and 1) with strong bias, two classes ---
        {
            "setup": "import numpy as np\nalpha = np.array([0.03, 0.005])\ndom = np.array([-0.02, 0.004])\ntp = np.array([0.01, 0.02])\n"
                     "tm = np.array([0.05, 0.01])\ncounts = np.array([10, 200])\n",
            "call": "neutral_variance_components(alpha, dom, tp, tm, counts)",
            "gold_call": "_oracle_neutral_variance_components(alpha, dom, tp, tm, counts)",
            "tol": 1e-07,
        },
    ]
