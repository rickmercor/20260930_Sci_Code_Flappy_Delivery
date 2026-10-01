"""
Implement normalization_constant using the task's leave-one-out convention.

For N reference CV points, compute



    Z_n = [1 / (N * (N - 1))] * sum_j sum_{i != j} K(s_j, s_i).



Each inner density uses the other N - 1 centers, and the outer mean uses all N

centers. K is the same isotropic Gaussian density kernel used by gaussian_kernel.

This leave-one-out quadrature is an explicit convention of the present task.

ERBS motivates using kernel centers as integration points; the self-exclusion

and N - 1 denominator here are the task-specific choices. The bias compares

the current-point density with this reference normalization.

Returns
-------
float, the normalization constant Z_n as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def normalization_constant(S_cv: np.ndarray, sigma: float) -> float:

    '''Compute the mean leave-one-out density over the reference CV points.


    Parameters

    ----------

    S_cv : np.ndarray

        Array of shape (N_ref, k), the reference collective-variable points

        (kernel centers).

    sigma : float

        Positive kernel bandwidth, shared by all kernels.


    Returns

    -------

    Z_n : float

        The mean of the N leave-one-out densities. Each density averages

        exactly N - 1 non-self kernels. Returned as a native Python float.


    Raises

    ------

    ValueError

         If S_cv is not a 2D array, if S_cv has fewer than 2 rows (leave-one-out

         requires at least one other point to average against), or if sigma is

         not a finite number > 0.

    '''

    return Z_n  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_normalization_constant(S_cv: np.ndarray, sigma: float) -> float:
    """Reference implementation. Leave-one-out: each center's density
    contribution excludes its own self-kernel term."""
    S_cv = np.asarray(S_cv, dtype=float)
    if S_cv.ndim != 2:
        raise ValueError("S_cv must be a 2D array of shape (N_ref, k)")
    if S_cv.shape[0] < 2:
        raise ValueError("S_cv must contain at least 2 reference points for leave-one-out")
    if not (isinstance(sigma, (int, float)) and np.isfinite(sigma) and sigma > 0.0):
        raise ValueError("sigma must be a finite number > 0")

    N = S_cv.shape[0]
    loo_vals = []
    for j in range(N):
        others = np.delete(S_cv, j, axis=0)
        d_j = np.mean([_oracle_gaussian_kernel(S_cv[j], s_k, sigma) for s_k in others])
        loo_vals.append(d_j)
    return float(np.mean(loo_vals))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            # Normal case: task's projected reference set, k=3, sigma=0.3
            "setup": (
                "import numpy as np\n"
                "rng_ref = np.random.default_rng(7)\n"
                "S_ref = rng_ref.standard_normal((20, 12))\n"
                "mu = S_ref.mean(axis=0)\n"
                "Shat = S_ref - mu\n"
                "_, _, Vt = np.linalg.svd(Shat, full_matrices=False)\n"
                "V_k = Vt.T[:, :3]\n"
                "S_cv = (S_ref - mu) @ V_k\n"
                "sigma = 0.3"
            ),
            "call": "normalization_constant(S_cv, sigma)",
            "gold_call": "_oracle_normalization_constant(S_cv, sigma)",
        },
        {
            # Boundary case: exactly 2 reference points (minimum for leave-one-out)
            "setup": (
                "import numpy as np\n"
                "S_cv = np.array([[0.2, -0.1], [0.5, 0.3]])\n"
                "sigma = 0.4"
            ),
            "call": "normalization_constant(S_cv, sigma)",
            "gold_call": "_oracle_normalization_constant(S_cv, sigma)",
        },
        {
            # Edge case: 4 symmetric points, 1D CV space
            "setup": (
                "import numpy as np\n"
                "S_cv = np.array([[-1.0], [0.0], [0.0], [1.0]])\n"
                "sigma = 0.5"
            ),
            "call": "normalization_constant(S_cv, sigma)",
            "gold_call": "_oracle_normalization_constant(S_cv, sigma)",
        },
        {
            # Invalid-input case: N=1 cannot support leave-one-out -> ValueError
            "setup": (
                "import numpy as np\n"
                "S_cv = np.array([[0.1, 0.2]])\n"
                "sigma = 0.3\n"
                "def run_model():\n"
                "    try:\n"
                "        normalization_constant(S_cv, sigma)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_normalization_constant(S_cv, sigma)\n"
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
