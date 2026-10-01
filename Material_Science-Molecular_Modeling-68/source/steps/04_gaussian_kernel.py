"""
Implement gaussian_kernel, which evaluates an isotropic Gaussian kernel between two points in collective-variable space. ERBS builds its on-the-fly probability density from Gaussian kernels deposited at previously visited collective-variable points, each with equal, uncorrelated variance along every CV diemension.

Using an isotropic kernel (a single scalar bandwidth sigma rather than a full covariance matrix) keeps the density estimate's only free hyperparameter simple and interpretable, while still allowing the kernel to be deposited at any point discovered during exploration of the reduced collective-variable space.

Returns
-------
float, the kernel value K(s, s_j) as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def gaussian_kernel(s: np.ndarray, s_j: np.ndarray, sigma: float) -> float:
    '''Evaluate an isotropic Gaussian kernel between two points in CV space.

    Parameters
    ----------
    s : np.ndarray
        Array of shape (k,), a point in collective-variable space.
    s_j : np.ndarray
        Array of shape (k,), a kernel center in collective-variable space.
    sigma : float
        Positive kernel bandwidth (standard deviation along each CV dimension).

    Returns
    -------
    value : float
        The kernel value K(s, s_j), as a native Python float.

    Raises
    ------
    ValueError
        If s or s_j is not a 1D array, if s.shape != s_j.shape, if sigma is
        not a finite number > 0, or if s or s_j contains non-finite values.
    '''
    return value  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_gaussian_kernel(s: np.ndarray, s_j: np.ndarray, sigma: float) -> float:
    """Reference implementation."""
    s = np.asarray(s, dtype=float)
    s_j = np.asarray(s_j, dtype=float)
    if s.ndim != 1 or s_j.ndim != 1:
        raise ValueError("s and s_j must be 1D arrays")
    if s.shape != s_j.shape:
        raise ValueError("s and s_j must have the same shape")
    if not np.all(np.isfinite(s)) or not np.all(np.isfinite(s_j)):
        raise ValueError("s and s_j must contain only finite values")
    if not (isinstance(sigma, (int, float)) and np.isfinite(sigma) and sigma > 0.0):
        raise ValueError("sigma must be a finite number > 0")

    k = s.shape[0]
    sigma2 = float(sigma) ** 2
    diff = s - s_j
    quad = np.dot(diff, diff) / sigma2
    norm_const = 1.0 / np.sqrt((sigma2 ** k) * (2 * np.pi) ** k)
    return float(norm_const * np.exp(-0.5 * quad))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            # Normal case: two distinct 3D CV points, task's sigma=0.3
            "setup": (
                "import numpy as np\n"
                "s = np.array([0.70577083, -0.6775732, 0.14877657])\n"
                "s_j = np.array([0.1, -0.2, 0.05])\n"
                "sigma = 0.3"
            ),
            "call": "gaussian_kernel(s, s_j, sigma)",
            "gold_call": "_oracle_gaussian_kernel(s, s_j, sigma)",
        },
        {
            # Boundary case: s equals s_j exactly (kernel at its own center)
            "setup": (
                "import numpy as np\n"
                "s = np.array([0.3])\n"
                "s_j = np.array([0.3])\n"
                "sigma = 0.5"
            ),
            "call": "gaussian_kernel(s, s_j, sigma)",
            "gold_call": "_oracle_gaussian_kernel(s, s_j, sigma)",
        },
        {
            # Edge case: 1D CV space, points far apart (near-zero kernel value)
            "setup": (
                "import numpy as np\n"
                "s = np.array([-1.0])\n"
                "s_j = np.array([1.0])\n"
                "sigma = 0.5"
            ),
            "call": "gaussian_kernel(s, s_j, sigma)",
            "gold_call": "_oracle_gaussian_kernel(s, s_j, sigma)",
        },
        {
            # Invalid-input case: non-positive sigma should raise ValueError
           "setup": (
        "import numpy as np\n"
        "s = np.array([0.1, 0.2])\n"
        "s_j = np.array([0.0, 0.0])\n"
        "sigma = 0.0\n"
        "def run_model():\n"
        "    try:\n"
        "        gaussian_kernel(s, s_j, sigma)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "def run_gold():\n"
        "    try:\n"
        "        _oracle_gaussian_kernel(s, s_j, sigma)\n"
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
