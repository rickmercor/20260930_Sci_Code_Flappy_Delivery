"""
Build the paper's noise-aware row-sampling distribution and the matching per-row weights for a system whose right-hand side is observed under heterogeneous, fresh, zero-mean noise.

The paper's method queries rows of A at random and averages weighted row-action directions. It ties the sampling probabilities p_i to the weights w_i through a single coupling identity involving the row norms, the Frobenius norm of A and a relaxation parameter alpha (eq. (3)), so that fixing either the probabilities or the weights determines the other. Among all couplings, the paper derives one noise-aware choice that minimizes the noise prefactor of its convergence bound: the sampling probabilities depend on both each row's norm and its noise level, and the weights follow from the coupling (Section 2.4, Corollary 2.3). Consult the paper for the exact probability rule, the exact coupling identity, and therefore the exact normalization of the weights.

Returns
-------
tuple (np.ndarray of shape (m,), np.ndarray of shape (m,)) — (p, w), both dtype float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def noise_aware_coupling(A: "np.ndarray", sigma: "np.ndarray",
                         alpha: float) -> tuple:
    '''Compute the paper's optimal noise-aware sampling probabilities and
    the coupled per-row weights.

    Parameters
    ----------
    A : np.ndarray
        (m, n) coefficient matrix with no zero row.
    sigma : np.ndarray
        (m,) per-row noise standard deviations, all strictly positive.
    alpha : float
        Relaxation parameter of the paper's coupling identity, > 0.

    Returns
    -------
    result : tuple of (np.ndarray, np.ndarray)
        (p, w): p is the (m,) sampling distribution over rows (nonnegative,
        summing to 1) and w the (m,) per-row weights, both as defined by the
        paper's optimal noise-aware scheme under its coupling identity with
        relaxation parameter alpha (not restated here).

    Raises
    ------
    ValueError
        If A is not a 2D array, if sigma does not have shape (m,) matching
        A's rows, if any entry of sigma is not strictly positive, if alpha
        is not strictly positive, or if any row of A is exactly the zero
        vector.
    '''
    return p, w  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_noise_aware_coupling(A: "np.ndarray", sigma: "np.ndarray",
                                 alpha: float) -> tuple:
    A = np.asarray(A, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be a 2D array")
    m, n = A.shape
    if sigma.shape != (m,):
        raise ValueError("sigma must have shape (m,) matching A's rows")
    if np.any(sigma <= 0):
        raise ValueError("all entries of sigma must be strictly positive")
    if not (alpha > 0):
        raise ValueError("alpha must be strictly positive")
    row_norms = np.linalg.norm(A, axis=1)
    if np.any(row_norms == 0):
        raise ValueError("A must not contain a zero row")
    frob2 = float(np.sum(A ** 2))
    # Corollary 2.3(ii): p_i proportional to sigma_i * ||a_i||; the weights
    # then follow from the coupling (3): p_i w_i / ||a_i||^2 = alpha / ||A||_F^2.
    p = sigma * row_norms
    p = p / np.sum(p)
    w = alpha * row_norms ** 2 / (p * frob2)
    return p, w

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the exact problem instance (heterogeneous noise) ---
        {
            "setup": """import numpy as np
A = np.array([[2.0, 1.0, 0.0, 1.0], [1.0, 3.0, 1.0, 0.0], [0.0, 1.0, 4.0, 1.0],
              [3.0, 0.0, 1.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, -1.0, 2.0, 0.0]])
sigma = np.array([0.5, 0.5, 3.0, 0.5, 3.0, 0.5])
alpha = 1.0
""",
            "call": "np.concatenate(noise_aware_coupling(A, sigma, alpha))",
            "gold_call": "np.concatenate(_oracle_noise_aware_coupling(A, sigma, alpha))",
        },
        # --- Boundary: homogeneous noise, so the sampling law reduces to a
        #     purely geometric one while the weights stay coupled to it. ---
        {
            "setup": """import numpy as np
A = np.array([[3.0, 0.0], [0.0, 4.0], [1.0, 1.0]])
sigma = np.array([0.2, 0.2, 0.2])
alpha = 1.0
""",
            "call": "np.concatenate(noise_aware_coupling(A, sigma, alpha))",
            "gold_call": "np.concatenate(_oracle_noise_aware_coupling(A, sigma, alpha))",
        },
        # --- Edge: noise exactly proportional to the row norms (the paper's
        #     equality case), with a relaxation parameter different from 1. ---
        {
            "setup": """import numpy as np
A = np.array([[3.0, 0.0], [0.0, 4.0], [1.0, 1.0]])
sigma = 0.1 * np.linalg.norm(A, axis=1)
alpha = 0.5
""",
            "call": "np.concatenate(noise_aware_coupling(A, sigma, alpha))",
            "gold_call": "np.concatenate(_oracle_noise_aware_coupling(A, sigma, alpha))",
        },
        # --- Edge: one row far noisier than the rest and rows of very
        #     different norms. ---
        {
            "setup": """import numpy as np
A = np.array([[10.0, 0.0, 0.0], [0.0, 0.1, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 2.0]])
sigma = np.array([0.01, 5.0, 0.1, 0.1])
alpha = 2.0
""",
            "call": "np.concatenate(noise_aware_coupling(A, sigma, alpha))",
            "gold_call": "np.concatenate(_oracle_noise_aware_coupling(A, sigma, alpha))",
        },
        # --- Invalid: a zero row in A -> ValueError ---
        {
            "setup": """import numpy as np
A = np.array([[1.0, 2.0], [0.0, 0.0], [3.0, 1.0]])
sigma = np.array([0.1, 0.1, 0.1])
alpha = 1.0
def run_model():
    try:
        noise_aware_coupling(A, sigma, alpha)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_noise_aware_coupling(A, sigma, alpha)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a non-positive noise level -> ValueError ---
        {
            "setup": """import numpy as np
A = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 1.0]])
sigma = np.array([0.1, 0.0, 0.1])
alpha = 1.0
def run_model():
    try:
        noise_aware_coupling(A, sigma, alpha)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_noise_aware_coupling(A, sigma, alpha)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: sigma shape does not match A's rows -> ValueError ---
        {
            "setup": """import numpy as np
A = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 1.0]])
sigma = np.array([0.1, 0.1])
alpha = 1.0
def run_model():
    try:
        noise_aware_coupling(A, sigma, alpha)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_noise_aware_coupling(A, sigma, alpha)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
