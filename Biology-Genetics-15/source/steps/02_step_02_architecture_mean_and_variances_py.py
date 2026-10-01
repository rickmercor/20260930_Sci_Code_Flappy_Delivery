"""
Mean trait value and the additive and dominance components of the genetic variance of a locus-class architecture at a given mean-trait deviation.

The genetic architecture of the trait is a set of locus classes, each with its own

additive effect, dominance deviation, mutation rates and number of loci. Loci of one

class share the stationary distribution of the previous step, so the mean trait value

of the population and the components of its genetic variance are sums over loci of

expectations under those distributions: the mean is the expected trait value of a

random genotype, and the variance splits into the additive and dominance components

that Hardy-Weinberg proportions assign to each locus. All inputs are in the source's

scaling.

Returns
-------
np.ndarray, shape (3,): the mean trait value, the additive genetic variance and the dominance genetic variance of the architecture at the given deviation
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def architecture_mean_and_variances(delta: float, sel_ratio: float, alpha: "np.ndarray", dom: "np.ndarray",
                                    theta_plus: "np.ndarray", theta_minus: "np.ndarray", counts: "np.ndarray",
                                    n_nodes: int) -> "np.ndarray":
    '''Mean trait value and the additive and dominance components of the genetic variance of a locus-class architecture at a given mean-trait deviation.

    Class k consists of counts[k] loci with additive effect alpha[k], dominance
    deviation dom[k] and scaled mutation rates theta_plus[k], theta_minus[k]; every
    locus of the class has the stationary frequency distribution of the previous step
    at the deviation delta and the scaled selection-drift ratio sel_ratio. The mean
    trait value is the sum over all loci of the expected trait contribution of a
    locus, and the additive and dominance components of the genetic variance are the
    sums over all loci of the expected additive and dominance variances of a locus
    under Hardy-Weinberg proportions, every expectation taken under the stationary
    distribution of the locus.

    Parameters
    ----------
    delta : float
        Deviation of the mean trait value from the selection optimum, in trait units.
    sel_ratio : float
        Scaled selection-drift ratio, >= 0.
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
    n_nodes : int
        Number of quadrature nodes passed to the previous step, >= 2.

    Returns
    -------
    out : np.ndarray
        Shape (3,): the mean trait value, the additive genetic variance and the
        dominance genetic variance of the population, in trait units and squared
        trait units.

    Raises
    ------
    ValueError
        If the five class arrays are not one-dimensional of the same length K >= 1
        with finite entries, if any alpha, theta_plus or theta_minus entry is not > 0,
        if any count is not a positive integer, or on any condition raised by the
        previous step for delta, sel_ratio or n_nodes.
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


def _oracle_architecture_mean_and_variances(delta: float, sel_ratio: float, alpha: "np.ndarray", dom: "np.ndarray",
                                            theta_plus: "np.ndarray", theta_minus: "np.ndarray", counts: "np.ndarray",
                                            n_nodes: int) -> "np.ndarray":
    a, d, tp, tm, c = _class_arrays(alpha, dom, theta_plus, theta_minus, counts)
    mean = va = vd = 0.0
    for k in range(a.size):
        m = _oracle_locus_stationary_moments(delta, float(a[k]), float(d[k]), float(tp[k]), float(tm[k]), sel_ratio, n_nodes)
        mean += c[k] * 2.0 * (a[k] * m[0] + d[k] * m[1])       # expected genotypic value 2 alpha p + 2 D p(1-p)
        va += c[k] * 2.0 * m[2]                                # 2 p(1-p) beta(p)^2
        vd += c[k] * 4.0 * d[k] * d[k] * m[3]                  # (2 p(1-p) D)^2
    return np.array([float(mean), float(va), float(vd)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary and edge comparisons with precision checks."""
    return [{'setup': 'import numpy as np\n'
               'alpha = np.array([0.012, 0.008, 0.010])\n'
               'dom = np.array([0.0, 0.007, -0.008])\n'
               'tp = np.array([0.06, 0.1, 0.2])\n'
               'tm = np.array([0.4, 0.3, 0.2])\n'
               'counts = np.array([60, 40, 20])\n',
      'call': 'np.asarray(architecture_mean_and_variances(-0.00512, 20000.0, alpha, dom, tp, tm, counts, 400)) / '
              'np.array([1.0, 0.002, 0.0002])',
      'gold_call': 'np.asarray(_oracle_architecture_mean_and_variances(-0.00512, 20000.0, alpha, dom, tp, tm, '
                   'counts, 400)) / np.array([1.0, 0.002, 0.0002])',
      'tol': 6e-09},
     {'setup': 'import numpy as np\n'
               'alpha = np.array([0.02])\n'
               'dom = np.array([0.003])\n'
               'tp = np.array([0.15])\n'
               'tm = np.array([0.45])\n'
               'counts = np.array([50])\n',
      'call': 'np.asarray(architecture_mean_and_variances(0.3, 0.0, alpha, dom, tp, tm, counts, 400)) / '
              'np.array([0.5, 0.004, 2.9999999999999997e-05])',
      'gold_call': 'np.asarray(_oracle_architecture_mean_and_variances(0.3, 0.0, alpha, dom, tp, tm, counts, 400)) '
                   '/ np.array([0.5, 0.004, 2.9999999999999997e-05])',
      'tol': 6e-09},
     {'setup': 'import numpy as np\n'
               'alpha = np.array([0.015, 0.01])\n'
               'dom = np.array([0.012, -0.009])\n'
               'tp = np.array([0.05, 0.25])\n'
               'tm = np.array([0.35, 0.25])\n'
               'counts = np.array([30, 70])\n',
      'call': 'np.asarray(architecture_mean_and_variances(-0.35, 300.0, alpha, dom, tp, tm, counts, 400)) / '
              'np.array([1.0, 0.004, 0.0008])',
      'gold_call': 'np.asarray(_oracle_architecture_mean_and_variances(-0.35, 300.0, alpha, dom, tp, tm, counts, '
                   '400)) / np.array([1.0, 0.004, 0.0008])',
      'tol': 6e-09}]
