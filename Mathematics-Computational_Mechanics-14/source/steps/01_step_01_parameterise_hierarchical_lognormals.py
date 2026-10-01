"""
The output contains the Gaussian parameters of their logarithms in the column order $(m_{r,j}, s_{r,j}, m_{K,j}, s_{K,j})$.

For each replicate j, association rate and carrying capacity are positive random variables with arithmetic moments $(\mu_{r,j}, \sigma_r)$ and $(\mu_{K,j}, \sigma_K)$.

Returns
-------
np.ndarray, an M-by-4 array of Gaussian log-scale parameters.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def parameterise_hierarchical_lognormals(
    mu_r: np.ndarray,
    mu_k: np.ndarray,
    sigma_r: float,
    sigma_k: float,
) -> np.ndarray:
    """Convert hierarchical arithmetic moments to log-normal parameters.

    Parameters
    ----------
    mu_r : np.ndarray
        Positive replicate-specific arithmetic means of association rate.
    mu_k : np.ndarray
        Positive replicate-specific arithmetic means of carrying capacity.
    sigma_r : float
        Shared non-negative arithmetic standard deviation of association rate.
    sigma_k : float
        Shared non-negative arithmetic standard deviation of carrying capacity.

    Returns
    -------
    result : np.ndarray
        Array of shape (M, 4) with columns $(m_r, s_r, m_K, s_K)$.

    Raises
    ------
    ValueError
        If the mean arrays are not non-empty one-dimensional arrays of equal
        length, or any mean is non-positive, or any standard deviation is
        negative or non-finite.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_parameterise_hierarchical_lognormals(
    mu_r: np.ndarray,
    mu_k: np.ndarray,
    sigma_r: float,
    sigma_k: float,
) -> np.ndarray:
    def _arithmetic_to_log_parameters(mean, sd):
        mean = np.asarray(mean, dtype=float)
        sd = np.asarray(sd, dtype=float)
        if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(sd)):
            raise ValueError("mean and sd must be finite")
        if np.any(mean <= 0.0) or np.any(sd < 0.0):
            raise ValueError("mean must be positive and sd must be non-negative")
        log_sd = np.sqrt(np.log1p((sd / mean) ** 2))
        log_mean = np.log(mean) - 0.5 * log_sd**2
        return log_mean, log_sd

    mu_r = np.asarray(mu_r, dtype=float)
    mu_k = np.asarray(mu_k, dtype=float)
    if mu_r.ndim != 1 or mu_k.ndim != 1 or mu_r.size == 0 or mu_r.shape != mu_k.shape:
        raise ValueError("mu_r and mu_k must be non-empty one-dimensional arrays of equal length")

    mr, sr = _arithmetic_to_log_parameters(mu_r, float(sigma_r))
    mk, sk = _arithmetic_to_log_parameters(mu_k, float(sigma_k))
    return np.column_stack((mr, sr, mk, sk)).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic test cases for the moment transformation."""
    return [
        {
            "setup": """import numpy as np
mu_r = np.array([9.0e-7, 3.86125e-7, 3.86125e-7])
mu_k = np.array([10.0, 20.0, 10.0])
sigma_r = 4.56962e-7
sigma_k = 2.0
""",
            "call": "parameterise_hierarchical_lognormals(mu_r, mu_k, sigma_r, sigma_k)",
            "gold_call": "_oracle_parameterise_hierarchical_lognormals(mu_r, mu_k, sigma_r, sigma_k)",
        },
        {
            "setup": """import numpy as np
mu_r = np.array([2.0e-7, 5.0e-7])
mu_k = np.array([4.0, 11.0])
sigma_r = 0.0
sigma_k = 0.0
""",
            "call": "parameterise_hierarchical_lognormals(mu_r, mu_k, sigma_r, sigma_k)",
            "gold_call": "_oracle_parameterise_hierarchical_lognormals(mu_r, mu_k, sigma_r, sigma_k)",
        },
        {
            "setup": """import numpy as np
mu_r = np.array([1.0e-9, 2.0e-6])
mu_k = np.array([0.25, 80.0])
sigma_r = 7.5e-7
sigma_k = 12.0
""",
            "call": "parameterise_hierarchical_lognormals(mu_r, mu_k, sigma_r, sigma_k)",
            "gold_call": "_oracle_parameterise_hierarchical_lognormals(mu_r, mu_k, sigma_r, sigma_k)",
        },
        {
            "setup": """import numpy as np
def value_error_code(fn):
    try:
        fn()
    except ValueError:
        return 1.0
    return 0.0
mu_r = np.array([-1.0e-7, 2.0e-7])
mu_k = np.array([4.0, 8.0])
sigma_r = 1.0e-7
sigma_k = 1.0
""",
            "call": "value_error_code(lambda: parameterise_hierarchical_lognormals(mu_r, mu_k, sigma_r, sigma_k))",
            "gold_call": "value_error_code(lambda: _oracle_parameterise_hierarchical_lognormals(mu_r, mu_k, sigma_r, sigma_k))",
        },
    ]
