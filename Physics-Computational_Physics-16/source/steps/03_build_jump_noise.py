"""
Implement build_jump_noise which computes the instantaneous covariance-production matrix of the discrete-event model.

Individual reactions and transfers change several populations at once.
These shared events generate correlations even when the populations start
from a deterministic state. Covariance production records the fluctuations
contributed by the model's complete event set.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_jump_noise(mean: "np.ndarray", phi: float, gamma: float, source: float, tau_core: float, tau_excore: float, decay: "np.ndarray", fission_second: "np.ndarray") -> "np.ndarray":
    """Compute the noise term Q in dC/dt = A C + C A^T + Q for the jump model.
    
    Parameters
    ----------
    mean : np.ndarray
        Shape (13,), nonnegative expected populations in neutron/core/ex-core order.
    phi, gamma : float
        Nonnegative per-neutron fission and loss rates in inverse seconds.
    source : float
        Nonnegative Poisson neutron-source intensity in inverse seconds.
    tau_core, tau_excore : float
        Positive residence times in seconds; positive infinity disables that transfer.
    decay : np.ndarray
        Shape (6,), nonnegative precursor decay constants in inverse seconds.
    fission_second : np.ndarray
        Shape (7, 7), raw second moment of a fission increment, ordered
        neutron followed by the six core groups, including neutron consumption.
    
    Returns
    -------
    noise : np.ndarray
        Shape (13, 13), expected instantaneous quadratic increment per second,
        in neutron, core groups 1-6, ex-core groups 1-6 order.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_build_jump_noise(mean: "np.ndarray", phi: float, gamma: float, source: float, tau_core: float, tau_excore: float, decay: "np.ndarray", fission_second: "np.ndarray") -> "np.ndarray":
    """Implement build_jump_noise which computes the instantaneous covariance-production matrix of the discrete-event model."""
    mean = np.asarray(mean, dtype=float)
    decay = np.asarray(decay, dtype=float)
    noise = np.zeros((13, 13))
    noise[:7, :7] = phi * mean[0] * np.asarray(fission_second, dtype=float)
    noise[0, 0] += gamma * mean[0] + source
    for i in range(6):
        c, e = i + 1, i + 7
        core_decay = decay[i] * mean[c]
        noise[0, 0] += core_decay
        noise[c, c] += core_decay
        noise[0, c] -= core_decay
        noise[c, 0] -= core_decay
        noise[e, e] += decay[i] * mean[e]
        transfer = mean[c] / tau_core + mean[e] / tau_excore
        noise[c, c] += transfer
        noise[e, e] += transfer
        noise[c, e] -= transfer
        noise[e, c] -= transfer
    return noise

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent candidate/oracle comparisons."""
    return [
        {
            'setup': 'import numpy as np\np = np.array([.027, .158, .339, .305, .133, .038])\nalpha = np.array([.033, .219, .196, .395, .115, .042])\ndecay = np.array([.0124, .0305, .111, .301, 1.14, 3.01])\n\nmean = np.arange(1., 14.)\nfission_mean = np.r_[1.473, alpha * .0065 * 2.473 / .9935]\nfission_second = np.outer(fission_mean, fission_mean)\nfission_second[0, 0] = p @ (np.arange(6) - 1.) ** 2\nfission_second[np.arange(1, 7), np.arange(1, 7)] += fission_mean[1:]\n',
            'call': 'build_jump_noise(mean.copy(), 401.7, 608.3, 8800., 10., 15., decay.copy(), fission_second.copy())',
            'gold_call': '_oracle_build_jump_noise(mean.copy(), 401.7, 608.3, 8800., 10., 15., decay.copy(), fission_second.copy())',
        },
        {
            'setup': 'import numpy as np\np = np.array([.027, .158, .339, .305, .133, .038])\nalpha = np.array([.033, .219, .196, .395, .115, .042])\ndecay = np.array([.0124, .0305, .111, .301, 1.14, 3.01])\n\nmean = np.arange(1., 14.)\nfission_mean = np.r_[1.473, alpha * .0065 * 2.473 / .9935]\nfission_second = np.outer(fission_mean, fission_mean)\nfission_second[0, 0] = p @ (np.arange(6) - 1.) ** 2\nfission_second[np.arange(1, 7), np.arange(1, 7)] += fission_mean[1:]\n\nmean = np.zeros(13)\n',
            'call': 'build_jump_noise(mean.copy(), 401.7, 608.3, 8800., 10., 15., decay.copy(), fission_second.copy())',
            'gold_call': '_oracle_build_jump_noise(mean.copy(), 401.7, 608.3, 8800., 10., 15., decay.copy(), fission_second.copy())',
            'tol': 1e-08,
        },
        {
            'setup': 'import numpy as np\np = np.array([.027, .158, .339, .305, .133, .038])\nalpha = np.array([.033, .219, .196, .395, .115, .042])\ndecay = np.array([.0124, .0305, .111, .301, 1.14, 3.01])\n\nmean = np.arange(1., 14.)\nfission_mean = np.r_[1.473, alpha * .0065 * 2.473 / .9935]\nfission_second = np.outer(fission_mean, fission_mean)\nfission_second[0, 0] = p @ (np.arange(6) - 1.) ** 2\nfission_second[np.arange(1, 7), np.arange(1, 7)] += fission_mean[1:]\n\nmean[0] = 0.\ndecay = np.zeros(6)\n',
            'call': 'build_jump_noise(mean.copy(), 0., 0., 0., 2., 4., decay.copy(), fission_second.copy())',
            'gold_call': '_oracle_build_jump_noise(mean.copy(), 0., 0., 0., 2., 4., decay.copy(), fission_second.copy())',
            'tol': 1e-08,
        },
    ]
