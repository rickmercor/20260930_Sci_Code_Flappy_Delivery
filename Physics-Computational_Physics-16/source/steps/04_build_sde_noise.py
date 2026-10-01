"""
Implement build_sde_noise which computes covariance production for the source paper’s continuous stochastic differential equation model.

A diffusion approximation represents fluctuations through stochastic
forcing added to population drift. Agreement of the drift with a discrete
model does not establish agreement of their covariance evolution.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_sde_noise(mean: "np.ndarray", phi: float, gamma: float, prompt_increment_second: float) -> "np.ndarray":
    """Return the noise matrix of the paper's continuous Itô approximation.
    
    Parameters
    ----------
    mean : np.ndarray
        Shape (13,), nonnegative expected populations, ordered neutron,
        core groups 1-6, ex-core groups 1-6. Use the nonnegative-state regime
        guaranteed by the benchmark's source-positivity condition.
    phi, gamma : float
        Nonnegative per-neutron fission and loss rates in inverse seconds.
    prompt_increment_second : float
        E[(nu_p - 1)^2], the raw second moment of the prompt neutron increment.
    
    Returns
    -------
    noise : np.ndarray
        Shape (13, 13), Q in the continuous model's covariance equation.
        Use the stochastic terms specified in the source's Itô model;
        finite-time-step discretization effects are outside this function.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_build_sde_noise(mean: "np.ndarray", phi: float, gamma: float, prompt_increment_second: float) -> "np.ndarray":
    """Implement build_sde_noise which computes covariance production for the source paper’s continuous stochastic differential equation model."""
    noise = np.zeros((13, 13))
    noise[0, 0] = (phi * prompt_increment_second + gamma) * mean[0]
    return noise

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent candidate/oracle comparisons."""
    return [
        {
            'setup': 'import numpy as np\nmean = np.arange(1., 14.)\n',
            'call': 'build_sde_noise(mean.copy(), 401.7, 608.3, 3.393)',
            'gold_call': '_oracle_build_sde_noise(mean.copy(), 401.7, 608.3, 3.393)',
            'tol': 1e-08,
        },
        {
            'setup': 'import numpy as np\nmean = np.r_[0., np.ones(12)]\n',
            'call': 'build_sde_noise(mean.copy(), 401.7, 608.3, 3.393)',
            'gold_call': '_oracle_build_sde_noise(mean.copy(), 401.7, 608.3, 3.393)',
            'tol': 1e-08,
        },
        {
            'setup': 'import numpy as np\nmean = np.r_[10., np.zeros(12)]\n',
            'call': 'build_sde_noise(mean.copy(), 0., 2., 0.)',
            'gold_call': '_oracle_build_sde_noise(mean.copy(), 0., 2., 0.)',
            'tol': 1e-08,
        },
    ]
