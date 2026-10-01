"""
Implement compute_fission_moments which computes the first and raw second moments of a fission population increment.

Fission consumes one neutron and creates prompt neutrons and delayed-neutron
precursors. Their joint fluctuations affect correlations between populations.
A six-group precursor description retains differences in decay times while
allowing a single fission to contribute to several groups.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_fission_moments(probabilities: "np.ndarray", beta: float, alpha: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
    """Return moments of the increment in neutron and six core-precursor populations.
    
    Parameters
    ----------
    probabilities : np.ndarray
        Shape (K,), probabilities of prompt yields 0 through K-1; sum one
        and have positive mean. The prompt yield is independent of delayed yields.
    beta : float
        Total delayed fraction, 0 <= beta < 1. Set the delayed-yield mean
        by consistency with the source model's mean precursor-production rate.
    alpha : np.ndarray
        Shape (6,), nonnegative relative delayed fractions summing to one.
        The Poisson total delayed yield is split multinomially with these weights.
    
    Returns
    -------
    mean_increment : np.ndarray
        Shape (7,), ordered neutron increment then groups 1 through 6.
    raw_second : np.ndarray
        Shape (7, 7), E[delta X delta X^T], including neutron consumption.
        This is a raw second moment, before subtracting the mean outer product.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_fission_moments(probabilities: "np.ndarray", beta: float, alpha: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
    """Implement compute_fission_moments which computes the first and raw second moments of a fission population increment."""
    probabilities = np.asarray(probabilities, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    yields = np.arange(probabilities.size, dtype=float)
    prompt_mean = float(probabilities @ yields)
    delayed = alpha * beta * prompt_mean / (1.0 - beta)
    mean_increment = np.concatenate(([prompt_mean - 1.0], delayed))
    raw_second = np.outer(mean_increment, mean_increment)
    raw_second[0, 0] = probabilities @ ((yields - 1.0) ** 2)
    raw_second[np.arange(1, 7), np.arange(1, 7)] += delayed
    return mean_increment, raw_second

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent candidate/oracle comparisons."""
    return [
        {
            'setup': 'import numpy as np\np = np.array([.027, .158, .339, .305, .133, .038])\nalpha = np.array([.033, .219, .196, .395, .115, .042])\ndecay = np.array([.0124, .0305, .111, .301, 1.14, 3.01])\n\ndef pack(fn, p, beta, alpha):\n    mean, second = fn(p.copy(), beta, alpha.copy())\n    assert np.shape(mean) == (7,) and np.shape(second) == (7, 7)\n    return np.concatenate((np.asarray(mean), np.asarray(second).ravel()))\n',
            'call': 'pack(compute_fission_moments, p, .0065, alpha)',
            'gold_call': 'pack(_oracle_compute_fission_moments, p, .0065, alpha)',
            'tol': 1e-08,
        },
        {
            'setup': 'import numpy as np\np = np.array([.027, .158, .339, .305, .133, .038])\nalpha = np.array([.033, .219, .196, .395, .115, .042])\ndecay = np.array([.0124, .0305, .111, .301, 1.14, 3.01])\n\ndef pack(fn, p, beta, alpha):\n    mean, second = fn(p.copy(), beta, alpha.copy())\n    assert np.shape(mean) == (7,) and np.shape(second) == (7, 7)\n    return np.concatenate((np.asarray(mean), np.asarray(second).ravel()))\n\np = np.array([0., 0., 1.])\n',
            'call': 'pack(compute_fission_moments, p, 0., alpha)',
            'gold_call': 'pack(_oracle_compute_fission_moments, p, 0., alpha)',
            'tol': 1e-08,
        },
        {
            'setup': 'import numpy as np\np = np.array([.027, .158, .339, .305, .133, .038])\nalpha = np.array([.033, .219, .196, .395, .115, .042])\ndecay = np.array([.0124, .0305, .111, .301, 1.14, 3.01])\n\ndef pack(fn, p, beta, alpha):\n    mean, second = fn(p.copy(), beta, alpha.copy())\n    assert np.shape(mean) == (7,) and np.shape(second) == (7, 7)\n    return np.concatenate((np.asarray(mean), np.asarray(second).ravel()))\n\np = np.array([.25, 0., .75])\nalpha = np.array([1., 0., 0., 0., 0., 0.])\n',
            'call': 'pack(compute_fission_moments, p, .1, alpha)',
            'gold_call': 'pack(_oracle_compute_fission_moments, p, .1, alpha)',
            'tol': 1e-08,
        },
    ]
