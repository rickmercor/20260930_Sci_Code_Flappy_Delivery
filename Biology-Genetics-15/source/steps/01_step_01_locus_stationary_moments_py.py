"""
Stationary moments of the trait-increasing allele frequency at one locus under stabilizing selection with dominance.

A quantitative trait under stabilizing selection is shaped by many loci at once, but

when linkage disequilibrium is negligible each locus can be followed on its own: its

allele frequency drifts, mutates in both directions, and responds to selection through a

coefficient that depends on the frequency itself and on how far the population's mean

trait sits from the optimum. At stationarity the frequency has the classical

distribution of a diffusion with reversible mutation and frequency-dependent selection,

and a few moments of that distribution are all the later steps need. The source scales

mutation rates, selection and time by the population size in a convention of its own,

and this step takes its inputs in that scaling.

Returns
-------
np.ndarray, shape (4,): the stationary expectations of p, p(1 - p), beta(p)^2 p(1 - p) and (p(1 - p))^2 for the locus, beta(p) the average effect of an allele substitution
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def locus_stationary_moments(delta: float, alpha: float, dom: float, theta_plus: float, theta_minus: float,
                             sel_ratio: float, n_nodes: int) -> "np.ndarray":
    '''Stationary moments of the trait-increasing allele frequency at one locus under stabilizing selection with dominance.

    The locus has additive effect alpha and dominance deviation dom, so that the trait
    values of the genotypes carrying 0, 1 and 2 copies of the trait-increasing allele
    are 0, alpha + dom and 2 alpha; mutation is reversible at scaled rates theta_plus
    (towards the trait-increasing allele) and theta_minus; the trait is under Gaussian
    stabilizing selection with scaled selection-drift ratio sel_ratio, and the mean
    trait of the population sits delta above the selection optimum. theta_plus,
    theta_minus and sel_ratio are in the source's scaling of mutation, selection and
    time. The selection coefficient at the locus is the regression coefficient of log
    fitness on the count of the trait-increasing allele under Hardy-Weinberg and
    linkage equilibrium, the other loci entering only through delta, and the allele
    frequency follows the diffusion with that frequency-dependent coefficient. Return
    moments of Wright's stationary distribution of that diffusion, evaluated by
    quadrature after the endpoint singularities of the density have been removed by
    substitution.

    Parameters
    ----------
    delta : float
        Deviation of the mean trait value from the selection optimum, in trait units.
    alpha : float
        Additive effect of the trait-increasing allele, in trait units, > 0.
    dom : float
        Dominance deviation of the heterozygote, in trait units.
    theta_plus : float
        Scaled mutation rate towards the trait-increasing allele, > 0.
    theta_minus : float
        Scaled mutation rate towards the trait-decreasing allele, > 0.
    sel_ratio : float
        Scaled selection-drift ratio, >= 0; 0 is mutation and drift alone.
    n_nodes : int
        Number of quadrature nodes per sub-interval, >= 2; with 400 nodes every
        entry is accurate to a relative error below 1e-10 over the parameter ranges
        of the tests, which compare at a relative tolerance of 1e-7.

    Returns
    -------
    moments : np.ndarray
        Shape (4,): the expectations of p, p(1 - p), beta(p)^2 p(1 - p) and
        (p(1 - p))^2 under the stationary distribution of the frequency p, where
        beta(p) is the average effect of an allele substitution at the locus.

    Raises
    ------
    ValueError
        If delta or dom is not a finite number, if alpha is not a finite number > 0,
        if theta_plus or theta_minus is not a finite number > 0, if sel_ratio is not
        a finite number >= 0, or if n_nodes is not an integer >= 2.
    '''
    return moments  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from functools import lru_cache
from numpy.polynomial.legendre import leggauss


@lru_cache(maxsize=None)
def _cached_leggauss_nodes_weights(n_nodes: int) -> "tuple[np.ndarray, np.ndarray]":
    """Return cached Gauss-Legendre nodes and weights for the requested order."""
    x, w = leggauss(int(n_nodes))
    x.setflags(write=False)
    w.setflags(write=False)
    return x, w


def _oracle_locus_stationary_moments(delta: float, alpha: float, dom: float, theta_plus: float, theta_minus: float,
                                     sel_ratio: float, n_nodes: int) -> "np.ndarray":
    def _num(v):
        return (not isinstance(v, bool)) and isinstance(v, (int, float, np.integer, np.floating)) and bool(np.isfinite(float(v)))

    if not _num(delta) or not _num(dom):
        raise ValueError("delta and dom must be finite numbers")
    if not _num(alpha) or float(alpha) <= 0.0:
        raise ValueError("alpha must be a finite number > 0")
    if not _num(theta_plus) or float(theta_plus) <= 0.0 or not _num(theta_minus) or float(theta_minus) <= 0.0:
        raise ValueError("theta_plus and theta_minus must be finite numbers > 0")
    if not _num(sel_ratio) or float(sel_ratio) < 0.0:
        raise ValueError("sel_ratio must be a finite number >= 0")
    if isinstance(n_nodes, bool) or not isinstance(n_nodes, (int, np.integer)) or int(n_nodes) < 2:
        raise ValueError("n_nodes must be an integer >= 2")
    d, a, D, tp, tm, s = float(delta), float(alpha), float(dom), float(theta_plus), float(theta_minus), float(sel_ratio)
    # Wright's stationary density in the source's scaling (theta = 2 N mu, time in 2N generations, sel_ratio = 2 N / omega^2):
    #   p^(2 theta_plus - 1) (1 - p)^(2 theta_minus - 1) exp(2 int_0^p xi(u) du),
    #   xi(u) = sel_ratio [ -beta(u) delta + beta(u)^2 (u - 1/2) + 2 alpha D u (1 - u) ],  beta(u) = alpha + D (1 - 2u).
    # 2 int_0^p xi = sel_ratio * S(p), S a quartic with the coefficients below.
    c4 = 2.0 * D * D
    c3 = -4.0 * a * D - 4.0 * D * D
    c2 = a * a + 6.0 * a * D + 3.0 * D * D + 2.0 * D * d
    c1 = -a * a - 2.0 * a * D - D * D - 2.0 * a * d - 2.0 * D * d
    # Quadrature: split at 1/2; on [0, 1/2] substitute p = u^(1/A) / 2, on [1/2, 1] substitute 1 - p = v^(1/B) / 2,
    # with A = 2 theta_plus and B = 2 theta_minus, which turns each endpoint weight into a constant; Gauss-Legendre
    # on u and v in (0, 1) with n_nodes nodes each. Gauss-Jacobi rules on the raw weight lose accuracy at these exponents.
    A, B = 2.0 * tp, 2.0 * tm
    x, w = _cached_leggauss_nodes_weights(int(n_nodes))
    u = 0.5 * (x + 1.0)
    w = 0.5 * w
    p_left = 0.5 * u ** (1.0 / A)
    p_right = 1.0 - 0.5 * u ** (1.0 / B)
    log_w_left = A * np.log(0.5) - np.log(A) + (B - 1.0) * np.log1p(-p_left) + np.log(w)
    log_w_right = B * np.log(0.5) - np.log(B) + (A - 1.0) * np.log(p_right) + np.log(w)
    p = np.concatenate([p_left, p_right])
    log_w = np.concatenate([log_w_left, log_w_right])
    expo = log_w + s * (((c4 * p + c3) * p + c2) * p + c1) * p
    f = np.exp(expo - expo.max())
    z = f.sum()
    pq = p * (1.0 - p)
    beta = a + D * (1.0 - 2.0 * p)
    return np.array([(f * p).sum() / z, (f * pq).sum() / z, (f * beta * beta * pq).sum() / z, (f * pq * pq).sum() / z])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary and edge comparisons with precision checks."""
    return [{'setup': 'import numpy as np\n',
      'call': 'np.asarray(locus_stationary_moments(-0.0051, 0.012, 0.0, 0.06, 0.4, 20000.0, 400)) / '
              'np.array([0.30000000000000004, 0.07, 9.999999999999999e-06, 0.01])',
      'gold_call': 'np.asarray(_oracle_locus_stationary_moments(-0.0051, 0.012, 0.0, 0.06, 0.4, 20000.0, 400)) / '
                   'np.array([0.30000000000000004, 0.07, 9.999999999999999e-06, 0.01])',
      'tol': 6e-09},
     {'setup': 'import numpy as np\n',
      'call': 'np.asarray(locus_stationary_moments(0.2, 0.008, 0.007, 0.1, 0.3, 0.0, 400)) / np.array([0.2, 0.08, '
              '8e-06, 0.02])',
      'gold_call': 'np.asarray(_oracle_locus_stationary_moments(0.2, 0.008, 0.007, 0.1, 0.3, 0.0, 400)) / '
                   'np.array([0.2, 0.08, 8e-06, 0.02])',
      'tol': 6e-09},
     {'setup': 'import numpy as np\n',
      'call': 'np.asarray(locus_stationary_moments(-0.0007, 0.01, -0.008, 0.2, 0.2, 200000.0, 400)) / '
              'np.array([0.9, 0.01, 2e-06, 0.0008])',
      'gold_call': 'np.asarray(_oracle_locus_stationary_moments(-0.0007, 0.01, -0.008, 0.2, 0.2, 200000.0, 400)) / '
                   'np.array([0.9, 0.01, 2e-06, 0.0008])',
      'tol': 6e-09},
     {'setup': 'import numpy as np\n',
      'call': 'np.asarray(locus_stationary_moments(0.05, 0.02, 0.015, 0.5, 0.5, 2000.0, 400)) / np.array([0.2, '
              '0.1, 7.999999999999999e-05, 0.02])',
      'gold_call': 'np.asarray(_oracle_locus_stationary_moments(0.05, 0.02, 0.015, 0.5, 0.5, 2000.0, 400)) / '
                   'np.array([0.2, 0.1, 7.999999999999999e-05, 0.02])',
      'tol': 6e-09}]
