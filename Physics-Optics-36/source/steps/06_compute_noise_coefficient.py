"""
Compute the exact RMS normalized-profile-noise coefficient of a finite packet population.

The reference profile is the population expectation at the same launch count. Outcomes are exclusive within each launch, and launches are independent.

Returns
-------
float, the dimensionless RMS noise coefficient against the exact population reference, with the NRMSD normalization specified in the function docstring.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_noise_coefficient(responses: "np.ndarray", probabilities: "np.ndarray") -> float:
    """Parameters
    ----------
    responses : np.ndarray
        Shape (K, M), K,M >= 1, nonnegative per-launch outcome intensities.
    probabilities : np.ndarray
        Shape (K,), nonnegative unconditional outcome probabilities with
        sum at most one. Remaining probability contributes a zero vector.
        The population mean profile must have positive total intensity.

    Returns
    -------
    coefficient : float
        Dimensionless sqrt(N times expected squared NRMSD), where NRMSD
        is the root mean squared bin residual relative to the exact
        N-launch population profile divided by that profile's mean
        intensity across the M bins. No finite-reference noise is included.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_noise_coefficient(responses: "np.ndarray", probabilities: "np.ndarray") -> float:
    responses = np.asarray(responses, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    means = probabilities @ responses
    second = probabilities @ (responses**2)
    variance = np.maximum(second - means**2, 0.0)
    return float(np.sqrt(responses.shape[1] * np.sum(variance)) / np.sum(means))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent normal, boundary and edge cases."""
    return [{'setup': 'import numpy as np\ny=np.array([[.001,0,0],[.002,0,0],[0,.003,0],[0,0,.002]])\np=np.array([.1,.2,.3,.15])', 'call': 'compute_noise_coefficient(y.copy(), p.copy())', 'gold_call': '_oracle_compute_noise_coefficient(y.copy(), p.copy())', 'tol': 1e-10}, {'setup': 'import numpy as np\ny=np.array([[.001,.003]])\np=np.array([1.])', 'call': 'compute_noise_coefficient(y.copy(), p.copy())', 'gold_call': '_oracle_compute_noise_coefficient(y.copy(), p.copy())', 'tol': 1e-10}, {'setup': 'import numpy as np\ny=np.array([[.001,0,0],[0,0,0]])\np=np.array([.01,.49])', 'call': 'compute_noise_coefficient(y.copy(), p.copy())', 'gold_call': '_oracle_compute_noise_coefficient(y.copy(), p.copy())', 'tol': 1e-10}]
