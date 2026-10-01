"""
Compute the largest eigenvalue of the paper's symmetric positive semidefinite matrix that governs the convergence of its block-averaged, weighted row-action iteration.

The paper's convergence analysis of its averaged, weighted iteration collapses onto a single m x m symmetric positive semidefinite matrix T (eq. (6)). T is assembled from the diagonal weight matrix W, the Gram matrix A A^T, the Frobenius norm of A, the relaxation parameter alpha of the coupling identity, and the batch size tau; the batch size enters through two different factors, one of which vanishes for a batch of size one. Its largest eigenvalue sigma_max(T) is the spectral quantity that appears in the paper's adaptive step-size rule and that shrinks monotonically as the batch grows (Proposition 2.2). Consult eq. (6) of the paper for the exact definition of T.

Returns
-------
float — sigma_max(T), the largest eigenvalue of the paper's matrix T, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def aabk_spectral_bound(A: "np.ndarray", w: "np.ndarray", alpha: float,
                        tau: int) -> float:
    '''Compute sigma_max(T), the largest eigenvalue of the paper's
    convergence matrix T (eq. (6)).

    Parameters
    ----------
    A : np.ndarray
        (m, n) coefficient matrix.
    w : np.ndarray
        (m,) per-row weights produced by noise_aware_coupling (the diagonal
        of the weight matrix W).
    alpha : float
        Relaxation parameter of the paper's coupling identity, > 0.
    tau : int
        Batch size (number of rows drawn per iteration), >= 1.

    Returns
    -------
    sigma_max_T : float
        The largest eigenvalue of the matrix T defined in eq. (6) of the
        paper (not restated here), as a native Python float.

    Raises
    ------
    ValueError
        If A is not a 2D array, if w does not have shape (m,) matching A's
        rows, if alpha is not strictly positive, or if tau is not a
        positive integer.
    '''
    return sigma_max_T  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_aabk_spectral_bound(A: "np.ndarray", w: "np.ndarray", alpha: float,
                                tau: int) -> float:
    A = np.asarray(A, dtype=float)
    w = np.asarray(w, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be a 2D array")
    m, n = A.shape
    if w.shape != (m,):
        raise ValueError("w must have shape (m,) matching A's rows")
    if not (alpha > 0):
        raise ValueError("alpha must be strictly positive")
    if not (isinstance(tau, (int, np.integer)) and tau >= 1):
        raise ValueError("tau must be a positive integer")
    frob2 = float(np.sum(A ** 2))
    # eq. (6): T = W/(2 tau) + alpha/(2 ||A||_F^2) (1 - 1/tau) A A^T
    T = np.diag(w) / (2.0 * tau) + alpha / (2.0 * frob2) * (1.0 - 1.0 / tau) * (A @ A.T)
    return float(np.linalg.eigvalsh(T).max())

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the exact problem instance (tau = 3, noise-aware weights) ---
        {
            "setup": """import numpy as np
A = np.array([[2.0, 1.0, 0.0, 1.0], [1.0, 3.0, 1.0, 0.0], [0.0, 1.0, 4.0, 1.0],
              [3.0, 0.0, 1.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, -1.0, 2.0, 0.0]])
w = np.array([1.9739575001, 2.6727510900, 0.5698324470,
              3.0152698877, 0.2686215916, 2.4175943246])
alpha = 1.0
tau = 3
""",
            "call": "aabk_spectral_bound(A, w, alpha, tau)",
            "gold_call": "_oracle_aabk_spectral_bound(A, w, alpha, tau)",
        },
        # --- Boundary: batch size one, where the Gram-matrix term drops out
        #     and only the weights matter. ---
        {
            "setup": """import numpy as np
A = np.array([[2.0, 1.0, 0.0, 1.0], [1.0, 3.0, 1.0, 0.0], [0.0, 1.0, 4.0, 1.0],
              [3.0, 0.0, 1.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, -1.0, 2.0, 0.0]])
w = np.array([1.9739575001, 2.6727510900, 0.5698324470,
              3.0152698877, 0.2686215916, 2.4175943246])
alpha = 1.0
tau = 1
""",
            "call": "aabk_spectral_bound(A, w, alpha, tau)",
            "gold_call": "_oracle_aabk_spectral_bound(A, w, alpha, tau)",
        },
        # --- Edge: uniform weights W = alpha I with alpha != 1 and a
        #     rank-one matrix (stable rank exactly 1), so the batch size
        #     should change nothing about the largest eigenvalue. ---
        {
            "setup": """import numpy as np
A = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
alpha = 0.5
w = alpha * np.ones(3)
tau = 4
""",
            "call": "aabk_spectral_bound(A, w, alpha, tau)",
            "gold_call": "_oracle_aabk_spectral_bound(A, w, alpha, tau)",
        },
        # --- Edge: a large batch on an orthogonal-row matrix (flat spectrum),
        #     where averaging reduces the spectral bound substantially. ---
        {
            "setup": """import numpy as np
A = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])
w = np.array([1.0, 1.0, 1.0])
alpha = 1.0
tau = 50
""",
            "call": "aabk_spectral_bound(A, w, alpha, tau)",
            "gold_call": "_oracle_aabk_spectral_bound(A, w, alpha, tau)",
        },
        # --- Invalid: tau = 0 -> ValueError ---
        {
            "setup": """import numpy as np
A = np.array([[1.0, 0.0], [0.0, 1.0]])
w = np.array([1.0, 1.0])
alpha = 1.0
tau = 0
def run_model():
    try:
        aabk_spectral_bound(A, w, alpha, tau)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_aabk_spectral_bound(A, w, alpha, tau)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: w shape mismatch -> ValueError ---
        {
            "setup": """import numpy as np
A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
w = np.array([1.0, 1.0])
alpha = 1.0
tau = 2
def run_model():
    try:
        aabk_spectral_bound(A, w, alpha, tau)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_aabk_spectral_bound(A, w, alpha, tau)
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
