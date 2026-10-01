"""
Evaluate the margin by which a candidate certificate pair satisfies the closed-loop contraction inequality for one realized system.

A gain and matrix certify a decay rate when the squared-rate-scaled matrix minus its closed-loop pullback is positive definite. The margin is the smallest eigenvalue of that difference: positive exactly when the certificate holds. This lets a supervisor test validity from the trajectory alone, without seeing the true system.

Returns
-------
float, native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def lyapunov_certificate_margin(A: np.ndarray, B: np.ndarray, K: np.ndarray,
                                P: np.ndarray, alpha: float) -> float:
    """Smallest eigenvalue of the closed-loop contraction difference.

    Parameters
    ----------
    A : np.ndarray
        (n, n) realized state matrix.
    B : np.ndarray
        (n, m) realized input matrix.
    K : np.ndarray
        (m, n) static state-feedback gain.
    P : np.ndarray
        (n, n) symmetric positive definite certificate matrix.
    alpha : float
        Decay rate in (0, 1].

    Returns
    -------
    margin : float
        Smallest eigenvalue of alpha**2 * P - (A + B K)^T P (A + B K), as a
        native Python float. Positive exactly when the certificate holds for
        this realization.

    Raises
    ------
    ValueError
        If A is not a square 2-D array; if B is not 2-D with the same number of
        rows as A; if K does not have shape (m, n) with m the number of columns
        of B; if P is not (n, n) or is not symmetric within atol 1e-10; if alpha
        is not a finite real number in (0, 1]; or if any input contains
        non-finite entries.
    """
    return margin

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_lyapunov_certificate_margin(A: np.ndarray, B: np.ndarray, K: np.ndarray, P: np.ndarray, alpha: float) -> float:
    """Reference implementation."""
    import numpy as np
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    K = np.asarray(K, dtype=float)
    P = np.asarray(P, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] < 1:
        raise ValueError("A must be a square 2-D array")
    n = A.shape[0]
    if B.ndim != 2 or B.shape[0] != n:
        raise ValueError("B must be 2-D with the same number of rows as A")
    m = B.shape[1]
    if K.shape != (m, n):
        raise ValueError("K must have shape (m, n)")
    if P.shape != (n, n):
        raise ValueError("P must have shape (n, n)")
    if not np.allclose(P, P.T, rtol=0.0, atol=1e-10):
        raise ValueError("P must be symmetric within atol 1e-10")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.floating, np.integer)):
        raise ValueError("alpha must be a real number in (0, 1]")
    a = float(alpha)
    if not np.isfinite(a) or a <= 0.0 or a > 1.0:
        raise ValueError("alpha must be a finite real number in (0, 1]")
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(B))
            and np.all(np.isfinite(K)) and np.all(np.isfinite(P))):
        raise ValueError("inputs must be finite")
    cl = A + B @ K
    M = a * a * P - cl.T @ P @ cl
    M = 0.5 * (M + M.T)
    return float(np.min(np.linalg.eigvalsh(M)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup = """import numpy as np
alpha = 0.9
A = np.array([[1.10, 0.00, 0.00],
              [0.00, 0.95, 0.30],
              [-0.10, 0.00, 1.05]], dtype=float)
B = np.array([[1.0, 0.0],
              [0.0, 0.8],
              [0.3, 0.5]], dtype=float)
K1 = np.array([[-0.981340, -0.214471,  0.313087],
               [ 0.171413, -0.576659, -0.644444]], dtype=float)
P1 = np.array([[ 4.109287,  1.923951, -3.168718],
               [ 1.923951,  4.508113, -2.980617],
               [-3.168718, -2.980617,  7.039886]], dtype=float)
K4 = np.array([[-0.599742,  0.020224, -0.514584],
               [-0.107856, -0.745123, -0.040444]], dtype=float)
P4 = np.array([[ 2.580081,  1.236551, -1.007780],
               [ 1.236551,  3.831790, -2.254034],
               [-1.007780, -2.254034,  4.587889]], dtype=float)
"""
    return [
        # --- Normal: a certificate that holds for this realization ---
        {
            "setup": setup,
            "call": "lyapunov_certificate_margin(A.copy(), B.copy(), K1.copy(), P1.copy(), alpha)",
            "gold_call": "_oracle_lyapunov_certificate_margin(A.copy(), B.copy(), K1.copy(), P1.copy(), alpha)",
        },
        # --- Boundary: a certificate that fails for the same realization ---
        {
            "setup": setup,
            "call": "lyapunov_certificate_margin(A.copy(), B.copy(), K4.copy(), P4.copy(), alpha)",
            "gold_call": "_oracle_lyapunov_certificate_margin(A.copy(), B.copy(), K4.copy(), P4.copy(), alpha)",
        },
        # --- Edge: identity certificate with zero gain and unit decay rate ---
        {
            "setup": """import numpy as np
A = np.array([[0.5, 0.0], [0.0, -0.25]], dtype=float)
B = np.zeros((2, 1), dtype=float)
K = np.zeros((1, 2), dtype=float)
P = np.eye(2, dtype=float)
alpha = 1.0
""",
            "call": "lyapunov_certificate_margin(A.copy(), B.copy(), K.copy(), P.copy(), alpha)",
            "gold_call": "_oracle_lyapunov_certificate_margin(A.copy(), B.copy(), K.copy(), P.copy(), alpha)",
        },
        # --- Invalid: alpha outside (0, 1] ---
        {
            "setup": setup + """def run_model():
    try:
        lyapunov_certificate_margin(A.copy(), B.copy(), K1.copy(), P1.copy(), 1.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_lyapunov_certificate_margin(A.copy(), B.copy(), K1.copy(), P1.copy(), 1.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-symmetric certificate matrix ---
        {
            "setup": setup + """P_bad = P1.copy(); P_bad[0, 1] = P_bad[0, 1] + 0.5
def run_model():
    try:
        lyapunov_certificate_margin(A.copy(), B.copy(), K1.copy(), P_bad.copy(), alpha)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_lyapunov_certificate_margin(A.copy(), B.copy(), K1.copy(), P_bad.copy(), alpha)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: gain with the wrong shape ---
        {
            "setup": setup + """K_bad = K1.T.copy()
def run_model():
    try:
        lyapunov_certificate_margin(A.copy(), B.copy(), K_bad.copy(), P1.copy(), alpha)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_lyapunov_certificate_margin(A.copy(), B.copy(), K_bad.copy(), P1.copy(), alpha)
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
