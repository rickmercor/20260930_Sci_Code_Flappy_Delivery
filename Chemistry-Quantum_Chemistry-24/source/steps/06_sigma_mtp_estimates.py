"""
Evaluate the source nonlinear estimator on a complete sample block and its two consecutive halves.

The Sigma-MTP construction is a nonlinear estimator formed from sample-major transition-amplitude data. Evaluate it independently on the full block and on each consecutive half, preserving their required output order. The exact estimator convention must follow the cited source.

Returns
-------
Return a length-three floating-point array containing the full-block estimate, the first-half estimate, and the second-half estimate, in that order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def sigma_mtp_estimates(amplitude_samples: np.ndarray) -> np.ndarray:
    """Return the source nonlinear estimates for the full block and its two consecutive halves, in that order."""
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_sigma_mtp_estimates(amplitude_samples: np.ndarray) -> np.ndarray:
    import numpy as np

    samples = np.asarray(amplitude_samples)
    if samples.ndim != 3 or samples.shape[1] != samples.shape[2]:
        raise ValueError("samples must have shape (M, dimension, dimension)")
    n_samples = len(samples)
    if n_samples < 2 or n_samples % 2:
        raise ValueError("an even positive sample count is required")
    if np.any(~np.isfinite(samples)):
        raise ValueError("amplitudes must be finite")

    def estimate(block):
        mean_amplitude = np.mean(block, axis=0)
        return float(np.sum(np.abs(mean_amplitude) ** 2))

    midpoint = n_samples // 2
    return np.asarray(
        [
            estimate(samples),
            estimate(samples[:midpoint]),
            estimate(samples[midpoint:]),
        ],
        dtype=float,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\na=np.array([[[1.,0.],[0.,2.]],[[3.,0.],[0.,0.]],[[1.,-1.],[0.,2.]],[[1.,1.],[0.,-2.]]])",
            "call": "sigma_mtp_estimates(a)",
            "gold_call": "_oracle_sigma_mtp_estimates(a)",
        },
        {
            "setup": "import numpy as np\na=np.array([[[1.]],[[3.]]])",
            "call": "sigma_mtp_estimates(a)",
            "gold_call": "_oracle_sigma_mtp_estimates(a)",
        },
        {
            "setup": "import numpy as np\na=np.array([[[1+1j,0],[0,2-1j]],[[1-1j,0],[0,2+1j]],[[2+0j,1j],[-1j,0]],[[0,0],[0,4]]])",
            "call": "sigma_mtp_estimates(a)",
            "gold_call": "_oracle_sigma_mtp_estimates(a)",
        },
    ]
