"""
Compute the exact initial value of the paper's auxiliary step-size sequence from the initial Bregman distance to the exact solution and the paper's per-step noise prefactor.

The paper's adaptive step size is driven by a deterministic auxiliary sequence beta_k that tracks an upper bound on the expected Bregman error. Its exact initial value beta_0 (the value the paper's experiments call the "exact beta_0", computed with knowledge of the ground-truth solution) combines three quantities: the batch size, the initial Bregman distance from the starting point to the exact solution, and the paper's noise prefactor, a trace of a product of the sampling, weight, noise and row-norm diagonal matrices that quantifies the noise injected per step (Theorem 2.1 and Section 2.4). Consult Theorem 2.1 for the exact definition of beta_0 and of the noise prefactor.

Returns
-------
float — beta_0, the exact initial value of the paper's auxiliary step-size sequence, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def aabk_beta_init(A: "np.ndarray", sigma: "np.ndarray", p: "np.ndarray",
                   w: "np.ndarray", tau: int, breg0: float) -> float:
    '''Compute the exact initial auxiliary value beta_0 of the paper's
    adaptive step-size rule (Theorem 2.1).

    Parameters
    ----------
    A : np.ndarray
        (m, n) coefficient matrix with no zero row.
    sigma : np.ndarray
        (m,) per-row noise standard deviations, all strictly positive.
    p : np.ndarray
        (m,) sampling distribution over rows, from noise_aware_coupling.
    w : np.ndarray
        (m,) per-row weights, from noise_aware_coupling.
    tau : int
        Batch size, >= 1.
    breg0 : float
        The Bregman distance D_f^{x*_0}(x_0, x_hat) from the starting point
        to the exact solution (as produced by sparse_bregman_distance),
        >= 0.

    Returns
    -------
    beta0 : float
        The exact initial value beta_0 of the paper's auxiliary sequence,
        as defined in Theorem 2.1 (not restated here), as a native Python
        float.

    Raises
    ------
    ValueError
        If A is not a 2D array, if sigma, p or w do not have shape (m,)
        matching A's rows, if any entry of sigma is not strictly positive,
        if any row of A is exactly the zero vector, if tau is not a
        positive integer, or if breg0 is negative.
    '''
    return beta0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_aabk_beta_init(A: "np.ndarray", sigma: "np.ndarray", p: "np.ndarray",
                           w: "np.ndarray", tau: int, breg0: float) -> float:
    A = np.asarray(A, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    p = np.asarray(p, dtype=float)
    w = np.asarray(w, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be a 2D array")
    m, n = A.shape
    if sigma.shape != (m,) or p.shape != (m,) or w.shape != (m,):
        raise ValueError("sigma, p and w must have shape (m,) matching A's rows")
    if np.any(sigma <= 0):
        raise ValueError("all entries of sigma must be strictly positive")
    row_norms2 = np.sum(A ** 2, axis=1)
    if np.any(row_norms2 == 0):
        raise ValueError("A must not contain a zero row")
    if not (isinstance(tau, (int, np.integer)) and tau >= 1):
        raise ValueError("tau must be a positive integer")
    if breg0 < 0:
        raise ValueError("breg0 must be nonnegative")
    # Theorem 2.1: beta_0 = tau * D_f^{x*_0}(x_0, x_hat) / Tr(P W^2 Sigma D^{-2}),
    # with D = Diag(||a_i||) and Sigma = Diag(sigma_i^2).
    c_noise = float(np.sum(p * w ** 2 * sigma ** 2 / row_norms2))
    return float(tau * breg0 / c_noise)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the exact problem instance (tau = 3, breg0 = 2.8) ---
        {
            "setup": """import numpy as np
A = np.array([[2.0, 1.0, 0.0, 1.0], [1.0, 3.0, 1.0, 0.0], [0.0, 1.0, 4.0, 1.0],
              [3.0, 0.0, 1.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, -1.0, 2.0, 0.0]])
sigma = np.array([0.5, 0.5, 3.0, 0.5, 3.0, 0.5])
p = np.array([0.0490254697, 0.0663807997, 0.5094876260,
              0.0748876419, 0.2401747702, 0.0600436925])
w = np.array([1.9739575001, 2.6727510900, 0.5698324470,
              3.0152698877, 0.2686215916, 2.4175943246])
tau = 3
breg0 = 2.8
""",
            "call": "aabk_beta_init(A, sigma, p, w, tau, breg0)",
            "gold_call": "_oracle_aabk_beta_init(A, sigma, p, w, tau, breg0)",
        },
        # --- Boundary: batch size one on the same instance (beta_0 scales
        #     with the batch size). ---
        {
            "setup": """import numpy as np
A = np.array([[2.0, 1.0, 0.0, 1.0], [1.0, 3.0, 1.0, 0.0], [0.0, 1.0, 4.0, 1.0],
              [3.0, 0.0, 1.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, -1.0, 2.0, 0.0]])
sigma = np.array([0.5, 0.5, 3.0, 0.5, 3.0, 0.5])
p = np.array([0.0490254697, 0.0663807997, 0.5094876260,
              0.0748876419, 0.2401747702, 0.0600436925])
w = np.array([1.9739575001, 2.6727510900, 0.5698324470,
              3.0152698877, 0.2686215916, 2.4175943246])
tau = 1
breg0 = 2.8
""",
            "call": "aabk_beta_init(A, sigma, p, w, tau, breg0)",
            "gold_call": "_oracle_aabk_beta_init(A, sigma, p, w, tau, breg0)",
        },
        # --- Edge: uniform weights and geometric sampling (the paper's
        #     uniform scheme) on rows of unequal norm, where the noise
        #     prefactor takes its non-optimal form. ---
        {
            "setup": """import numpy as np
A = np.array([[3.0, 0.0], [0.0, 4.0], [1.0, 1.0]])
sigma = np.array([0.1, 1.0, 0.1])
alpha = 1.0
p = np.sum(A ** 2, axis=1) / np.sum(A ** 2)
w = alpha * np.ones(3)
tau = 2
breg0 = 0.75
""",
            "call": "aabk_beta_init(A, sigma, p, w, tau, breg0)",
            "gold_call": "_oracle_aabk_beta_init(A, sigma, p, w, tau, breg0)",
        },
        # --- Edge: zero initial Bregman distance (start already at the
        #     solution) gives beta_0 = 0 regardless of the noise. ---
        {
            "setup": """import numpy as np
A = np.array([[3.0, 0.0], [0.0, 4.0], [1.0, 1.0]])
sigma = np.array([0.1, 1.0, 0.1])
p = np.array([0.2, 0.5, 0.3])
w = np.array([1.5, 1.0, 0.5])
tau = 2
breg0 = 0.0
""",
            "call": "aabk_beta_init(A, sigma, p, w, tau, breg0)",
            "gold_call": "_oracle_aabk_beta_init(A, sigma, p, w, tau, breg0)",
        },
        # --- Invalid: negative breg0 -> ValueError ---
        {
            "setup": """import numpy as np
A = np.array([[3.0, 0.0], [0.0, 4.0], [1.0, 1.0]])
sigma = np.array([0.1, 1.0, 0.1])
p = np.array([0.2, 0.5, 0.3])
w = np.array([1.5, 1.0, 0.5])
tau = 2
breg0 = -1.0
def run_model():
    try:
        aabk_beta_init(A, sigma, p, w, tau, breg0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_aabk_beta_init(A, sigma, p, w, tau, breg0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: p shape mismatch -> ValueError ---
        {
            "setup": """import numpy as np
A = np.array([[3.0, 0.0], [0.0, 4.0], [1.0, 1.0]])
sigma = np.array([0.1, 1.0, 0.1])
p = np.array([0.5, 0.5])
w = np.array([1.5, 1.0, 0.5])
tau = 2
breg0 = 1.0
def run_model():
    try:
        aabk_beta_init(A, sigma, p, w, tau, breg0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_aabk_beta_init(A, sigma, p, w, tau, breg0)
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
