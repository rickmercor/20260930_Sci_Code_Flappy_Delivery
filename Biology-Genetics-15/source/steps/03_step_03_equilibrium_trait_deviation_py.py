"""
Self-consistent equilibrium deviation of the mean trait value from the selection optimum.

Each locus feels the rest of the genome only through the deviation of the mean trait

from the optimum, and that deviation is itself the outcome of what every locus does.

At stationarity the two views must agree: the mean trait value that the loci produce

under their stationary distributions at a given deviation must be the value that

defines the deviation. This closes the description of the polygenic system from the

perspective of its loci, and it is what makes a mutational bias leave a lasting signature at

the locus level even when the trait mean sits close to the optimum. All inputs are in

the source's scaling.

Returns
-------
float, the equilibrium deviation of the mean trait value from the optimum, in trait units, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def equilibrium_trait_deviation(sel_ratio: float, eta: float, alpha: "np.ndarray", dom: "np.ndarray", theta_plus: "np.ndarray", theta_minus: "np.ndarray", counts: "np.ndarray", n_nodes: int, tol: float) -> float:
    '''Return the self-consistent equilibrium deviation of the mean trait from the optimum. Use the minimum and maximum of the three genotype values 0, alpha + dom, and 2 alpha in each class to bracket the unique root. Raise ValueError for invalid eta or tol, or for invalid inputs reported by earlier steps.'''
    return delta_star  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq


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


def _oracle_equilibrium_trait_deviation(sel_ratio: float, eta: float, alpha: "np.ndarray", dom: "np.ndarray", theta_plus: "np.ndarray", theta_minus: "np.ndarray", counts: "np.ndarray", n_nodes: int, tol: float) -> float:
    def _num(v):
        return (not isinstance(v, bool)) and isinstance(v, (int, float, np.integer, np.floating)) and bool(np.isfinite(float(v)))
    if not _num(eta): raise ValueError("eta must be a finite number")
    if not _num(tol) or float(tol) <= 0.0: raise ValueError("tol must be a finite number > 0")
    a, d, tp, tm, c = _class_arrays(alpha, dom, theta_plus, theta_minus, counts)
    e = float(eta)
    genotype_values = np.stack([np.zeros_like(a), a + d, 2.0 * a], axis=0)
    z_min = float(np.sum(c * np.min(genotype_values, axis=0)))
    z_max = float(np.sum(c * np.max(genotype_values, axis=0)))
    def _gap(delta):
        mean = _oracle_architecture_mean_and_variances(delta, sel_ratio, a, d, tp, tm, c, n_nodes)[0]
        return mean - e - delta
    return float(brentq(_gap, z_min - e, z_max - e, xtol=float(tol), rtol=4.0 * np.finfo(float).eps, maxiter=500))

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
      'call': 'equilibrium_trait_deviation(20000.0, 1.15, alpha, dom, tp, tm, counts, 400, 1e-13)',
      'gold_call': '_oracle_equilibrium_trait_deviation(20000.0, 1.15, alpha, dom, tp, tm, counts, 400, 1e-13)',
      'tol': 5e-11},
     {'setup': 'import numpy as np\n'
               'alpha = np.array([0.012, 0.008, 0.010])\n'
               'dom = np.array([0.0, 0.007, -0.008])\n'
               'tp = np.array([0.06, 0.1, 0.2])\n'
               'tm = np.array([0.4, 0.3, 0.2])\n'
               'counts = np.array([60, 40, 20])\n',
      'call': 'equilibrium_trait_deviation(200.0, 1.15, alpha, dom, tp, tm, counts, 400, 1e-13)',
      'gold_call': '_oracle_equilibrium_trait_deviation(200.0, 1.15, alpha, dom, tp, tm, counts, 400, 1e-13)',
      'tol': 5e-11},
     {'setup': 'import numpy as np\n'
               'alpha = np.array([0.02, 0.01])\n'
               'dom = np.array([0.01, -0.005])\n'
               'tp = np.array([0.3, 0.4])\n'
               'tm = np.array([0.1, 0.2])\n'
               'counts = np.array([40, 40])\n',
      'call': 'equilibrium_trait_deviation(5000.0, 1.6, alpha, dom, tp, tm, counts, 400, 1e-13)',
      'gold_call': '_oracle_equilibrium_trait_deviation(5000.0, 1.6, alpha, dom, tp, tm, counts, 400, 1e-13)',
      'tol': 5e-11},
     {'setup': 'import numpy as np\n'
               'alpha = np.array([0.01])\n'
               'dom = np.array([0.05])\n'
               'tp = np.array([0.2])\n'
               'tm = np.array([0.2])\n'
               'counts = np.array([10])\n',
      'call': 'equilibrium_trait_deviation(10000.0, 0.5, alpha, dom, tp, tm, counts, 400, 1e-13)',
      'gold_call': '_oracle_equilibrium_trait_deviation(10000.0, 0.5, alpha, dom, tp, tm, counts, 400, 1e-13)',
      'tol': 5e-11}]
