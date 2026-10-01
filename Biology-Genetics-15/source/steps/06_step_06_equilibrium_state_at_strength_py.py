"""
Equilibrium summaries of the polygenic system at one selection strength, including the relaxation time of the mean trait in generations.

At a given strength of stabilizing selection the equilibrium of the polygenic system

is summarised by a few numbers: how far the mean trait sits from the optimum, how much

additive and dominance variance the trait carries, where the allele frequencies of a

locus class sit, and how quickly the mean trait relaxes after a random excursion. The

source describes the stationary fluctuations of the mean trait as an

Ornstein-Uhlenbeck process, and the relaxation rate of that process, expressed per

generation, is the last of these summaries. Inputs are in natural units and are

converted to the source's scaling.

Returns
-------
np.ndarray, shape (5,): the equilibrium deviation, the additive variance, the dominance share, the class-1 mean frequency of the trait-increasing allele, and the e-folding time in generations
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def equilibrium_state_at_strength(n_diploid: int, mu_plus: "np.ndarray", mu_minus: "np.ndarray", alpha: "np.ndarray",
                                  dom: "np.ndarray", counts: "np.ndarray", eta: float, omega_inv2: float,
                                  n_nodes: int, tol: float) -> "np.ndarray":
    '''Equilibrium summaries of the polygenic system at one selection strength, including the relaxation time of the mean trait in generations.

    The population and the architecture are as in the previous step, and omega_inv2 is
    one selection strength. Convert to the source's scaled parameters, solve the
    equilibrium deviation of the mean trait, and return the deviation, the additive
    genetic variance, the share of the total genetic variance that is dominance
    variance, the mean frequency of the trait-increasing allele at a locus of the first
    class, and the e-folding time, in generations, of the autocorrelation of the
    deviation of the mean trait from its equilibrium value under the source's
    Ornstein-Uhlenbeck description of the stationary fluctuations of the mean trait.

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
    omega_inv2 : float
        Selection strength, the inverse squared width of the Gaussian fitness
        function in inverse squared trait units, > 0.
    n_nodes : int
        Number of quadrature nodes passed to the earlier steps, >= 2.
    tol : float
        Absolute accuracy of the equilibrium deviation, in trait units, > 0.

    Returns
    -------
    state : np.ndarray
        Shape (5,): the equilibrium deviation of the mean trait from the optimum (trait
        units), the additive genetic variance (squared trait units), the dominance
        share of the total genetic variance (a fraction), the mean frequency of the
        trait-increasing allele in the first class, and the e-folding time of the
        autocorrelation of the mean trait's deviation, in generations.

    Raises
    ------
    ValueError
        If omega_inv2 is not a finite number > 0, or on any condition raised by the
        earlier steps for the other inputs.
    '''
    return state  # placeholder

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


def _oracle_equilibrium_state_at_strength(n_diploid: int, mu_plus: "np.ndarray", mu_minus: "np.ndarray", alpha: "np.ndarray",
                                          dom: "np.ndarray", counts: "np.ndarray", eta: float, omega_inv2: float,
                                          n_nodes: int, tol: float) -> "np.ndarray":
    def _num(v):
        return (not isinstance(v, bool)) and isinstance(v, (int, float, np.integer, np.floating)) and bool(np.isfinite(float(v)))

    if not _num(omega_inv2) or float(omega_inv2) <= 0.0:
        raise ValueError("omega_inv2 must be a finite number > 0")
    a, d, tp, tm, c = _scaled_architecture(n_diploid, mu_plus, mu_minus, alpha, dom, counts)
    sel_ratio = 2.0 * int(n_diploid) * float(omega_inv2)
    delta = _oracle_equilibrium_trait_deviation(sel_ratio, eta, a, d, tp, tm, c, n_nodes, tol)
    mv = _oracle_architecture_mean_and_variances(delta, sel_ratio, a, d, tp, tm, c, n_nodes)
    p_first = _oracle_locus_stationary_moments(delta, float(a[0]), float(d[0]), float(tp[0]), float(tm[0]), sel_ratio, n_nodes)[0]
    # Ornstein-Uhlenbeck relaxation of the mean trait's deviation: the rate is the additive variance times the
    # selection strength per generation (the source's rate in units of 2N generations, divided by 2N)
    rate_per_generation = mv[1] * float(omega_inv2)
    return np.array([float(delta), float(mv[1]), float(mv[2] / (mv[1] + mv[2])), float(p_first), float(1.0 / rate_per_generation)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary and edge comparisons with precision checks."""
    return [{'setup': 'import numpy as np\n'
               'mup = np.array([3e-06, 5e-06, 1e-05])\n'
               'mum = np.array([2e-05, 1.5e-05, 1e-05])\n'
               'alpha = np.array([0.012, 0.008, 0.010])\n'
               'dom = np.array([0.0, 0.007, -0.008])\n'
               'counts = np.array([60, 40, 20])\n',
      'call': 'np.asarray(equilibrium_state_at_strength(10000, mup, mum, alpha, dom, counts, 1.15, 1.0, 400, '
              '1e-13)) / np.array([0.005, 0.002, 0.08, 0.30000000000000004, 500.0])',
      'gold_call': 'np.asarray(_oracle_equilibrium_state_at_strength(10000, mup, mum, alpha, dom, counts, 1.15, '
                   '1.0, 400, 1e-13)) / np.array([0.005, 0.002, 0.08, 0.30000000000000004, 500.0])',
      'tol': 6e-09},
     {'setup': 'import numpy as np\n'
               'mup = np.array([2e-05])\n'
               'mum = np.array([6e-05])\n'
               'alpha = np.array([0.02])\n'
               'dom = np.array([0.004])\n'
               'counts = np.array([50])\n',
      'call': 'np.asarray(equilibrium_state_at_strength(2500, mup, mum, alpha, dom, counts, 1.1, 0.002, 400, '
              '1e-13)) / np.array([0.5, 0.004, 0.01, 0.30000000000000004, 100000.0])',
      'gold_call': 'np.asarray(_oracle_equilibrium_state_at_strength(2500, mup, mum, alpha, dom, counts, 1.1, '
                   '0.002, 400, 1e-13)) / np.array([0.5, 0.004, 0.01, 0.30000000000000004, 100000.0])',
      'tol': 6e-09},
     {'setup': 'import numpy as np\n'
               'mup = np.array([1e-04, 5e-05])\n'
               'mum = np.array([2.5e-05, 5e-05])\n'
               'alpha = np.array([0.015, 0.01])\n'
               'dom = np.array([0.012, -0.009])\n'
               'counts = np.array([30, 70])\n',
      'call': 'np.asarray(equilibrium_state_at_strength(1000, mup, mum, alpha, dom, counts, 0.7, 30.0, 400, '
              '1e-13)) / np.array([0.001, 0.0004, 0.30000000000000004, 0.6000000000000001, 80.0])',
      'gold_call': 'np.asarray(_oracle_equilibrium_state_at_strength(1000, mup, mum, alpha, dom, counts, 0.7, '
                   '30.0, 400, 1e-13)) / np.array([0.001, 0.0004, 0.30000000000000004, 0.6000000000000001, 80.0])',
      'tol': 6e-09},
     {'setup': 'import numpy as np\n'
               'mup = np.array([3e-06, 5e-06, 1e-05])\n'
               'mum = np.array([2e-05, 1.5e-05, 1e-05])\n'
               'alpha = np.array([0.012, 0.008, 0.010])\n'
               'dom = np.array([0.0, 0.007, -0.008])\n'
               'counts = np.array([60, 40, 20])\n',
      'call': '(equilibrium_state_at_strength(10000, mup, mum, alpha, dom, counts, 1.15, 1.0, 400, 1e-13))[0]',
      'gold_call': '(_oracle_equilibrium_state_at_strength(10000, mup, mum, alpha, dom, counts, 1.15, 1.0, 400, '
                   '1e-13))[0]',
      'tol': 5e-11},
     {'setup': 'import numpy as np\n'
               'mup = np.array([2e-05])\n'
               'mum = np.array([6e-05])\n'
               'alpha = np.array([0.02])\n'
               'dom = np.array([0.004])\n'
               'counts = np.array([50])\n',
      'call': '(equilibrium_state_at_strength(2500, mup, mum, alpha, dom, counts, 1.1, 0.002, 400, 1e-13))[0]',
      'gold_call': '(_oracle_equilibrium_state_at_strength(2500, mup, mum, alpha, dom, counts, 1.1, 0.002, 400, '
                   '1e-13))[0]',
      'tol': 5e-11},
     {'setup': 'import numpy as np\n'
               'mup = np.array([1e-04, 5e-05])\n'
               'mum = np.array([2.5e-05, 5e-05])\n'
               'alpha = np.array([0.015, 0.01])\n'
               'dom = np.array([0.012, -0.009])\n'
               'counts = np.array([30, 70])\n',
      'call': '(equilibrium_state_at_strength(1000, mup, mum, alpha, dom, counts, 0.7, 30.0, 400, 1e-13))[0]',
      'gold_call': '(_oracle_equilibrium_state_at_strength(1000, mup, mum, alpha, dom, counts, 0.7, 30.0, 400, '
                   '1e-13))[0]',
      'tol': 5e-11}]
