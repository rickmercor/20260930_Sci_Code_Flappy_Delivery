"""
Implement density_estimate, which estimates the on-the-fly well-tempered probability density at a collective-variable point as the average of the isotropic Gaussian kernel over all previously visited reference points. This kernel density estimate is later compared, through a normalization constant, against the density at the reference points themselves to build the bias potential.

The kernel density estimate approximates the probability of having previously visited the neighborhood of a given point in CV space. Averaging (rather than summing) over the reference kernel centers keeps the density on a consistent scale regardless of how many reference points have been collected, which matters for the ratio computed in the final bias-potential step.

Returns
-------
float, the estimated density p_n^WT(s) as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def density_estimate(s: np.ndarray, S_cv: np.ndarray, sigma: float) -> float:
    '''Estimate the well-tempered probability density at a CV point.

    Parameters
    ----------
    s : np.ndarray
        Array of shape (k,), the collective-variable point to evaluate the
        density at.
    S_cv : np.ndarray
        Array of shape (N_ref, k), the reference collective-variable points
        (kernel centers).
    sigma : float
        Positive kernel bandwidth, shared by all kernels.

    Returns
    -------
    density : float
        The estimated density p_n^WT(s), as a native Python float.

    Raises
    ------
    ValueError
        If s is not 1D, if S_cv is not 2D, if S_cv has zero rows, if
        s.shape[0] != S_cv.shape[1], or if sigma is not a finite number > 0.
    '''
    return density  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_density_estimate(s: np.ndarray, S_cv: np.ndarray, sigma: float) -> float:
    """Reference implementation."""
    s = np.asarray(s, dtype=float)
    S_cv = np.asarray(S_cv, dtype=float)
    if s.ndim != 1:
        raise ValueError("s must be a 1D array")
    if S_cv.ndim != 2:
        raise ValueError("S_cv must be a 2D array of shape (N_ref, k)")
    if S_cv.shape[0] == 0:
        raise ValueError("S_cv must contain at least one reference point")
    if s.shape[0] != S_cv.shape[1]:
        raise ValueError("s and S_cv must have matching CV dimension k")
    if not (isinstance(sigma, (int, float)) and np.isfinite(sigma) and sigma > 0.0):
        raise ValueError("sigma must be a finite number > 0")

    kernels = [_oracle_gaussian_kernel(s, s_j, sigma) for s_j in S_cv]
    return float(np.mean(kernels))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            # Normal case: task's projected reference set and current CV point
            "setup": (
                "import numpy as np\n"
                "rng_ref = np.random.default_rng(7)\n"
                "S_ref = rng_ref.standard_normal((20, 12))\n"
                "mu = S_ref.mean(axis=0)\n"
                "Shat = S_ref - mu\n"
                "_, _, Vt = np.linalg.svd(Shat, full_matrices=False)\n"
                "V_k = Vt.T[:, :3]\n"
                "S_cv = (S_ref - mu) @ V_k\n"
                "s = np.array([0.70577083, -0.6775732, 0.14877657])\n"
                "sigma = 0.3"
            ),
            "call": "density_estimate(s, S_cv, sigma)",
            "gold_call": "_oracle_density_estimate(s, S_cv, sigma)",
        },
        {
            # Boundary case: single reference point (N_ref=1)
            "setup": (
                "import numpy as np\n"
                "s = np.array([0.3])\n"
                "S_cv = np.array([[0.3]])\n"
                "sigma = 0.5"
            ),
            "call": "density_estimate(s, S_cv, sigma)",
            "gold_call": "_oracle_density_estimate(s, S_cv, sigma)",
        },
        {
            # Edge case: reference points symmetric around s, far apart
            "setup": (
                "import numpy as np\n"
                "s = np.array([0.0])\n"
                "S_cv = np.array([[-1.0], [1.0]])\n"
                "sigma = 0.5"
            ),
            "call": "density_estimate(s, S_cv, sigma)",
            "gold_call": "_oracle_density_estimate(s, S_cv, sigma)",
        },
        {
            # Invalid-input case: empty S_cv should raise ValueError
            "setup": (
        "import numpy as np\n"
        "s = np.array([0.0, 0.0])\n"
        "S_cv = np.empty((0, 2))\n"
        "sigma = 0.3\n"
        "def run_model():\n"
        "    try:\n"
        "        density_estimate(s, S_cv, sigma)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "def run_gold():\n"
        "    try:\n"
        "        _oracle_density_estimate(s, S_cv, sigma)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2"
    ),
    "call": "run_model()",
    "gold_call": "run_gold()",
},
    ]
