"""
Implement schrodinger_rhs for a discrete nonlinear Schrödinger lattice.

Given the complex state matrix A, a same-sized square coupling matrix B, and a nonnegative real coefficient alpha, return the complex velocity field F(A) = dA/dt. Reject incompatible matrix shapes or a negative coefficient with ValueError.

The discrete nonlinear Schrödinger equation evolves a complex lattice state under nearest-neighbor coupling and a local cubic self-interaction. In matrix form, B couples adjacent coordinates, alpha controls the nonlinear strength, and the output is the n-by-n complex velocity evaluated at A.

Returns
-------
np.ndarray as specified by the function Returns section.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def schrodinger_rhs(A: np.ndarray, B: np.ndarray, alpha: float) -> np.ndarray:
    """Evaluate the velocity field F(A) = dA/dt for the DNLS lattice equation.

    Parameters
    ----------
    A : np.ndarray, shape (n, n)
        Current matrix state (complex).
    B : np.ndarray, shape (n, n)
        Tridiagonal coupling matrix.
    alpha : float
        Nonlinearity parameter (must be >= 0).

    Returns
    -------
    F_A : np.ndarray, shape (n, n)
        Velocity field dA/dt.
    """
    F_A = np.zeros_like(A, dtype=complex)
    return F_A

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_schrodinger_rhs(A: np.ndarray, B: np.ndarray, alpha: float) -> np.ndarray:
    """Reference implementation."""
    A = np.asarray(A, dtype=complex)
    B = np.asarray(B, dtype=float)
    n = A.shape[0]
    if A.shape != (n, n) or B.shape != (n, n):
        raise ValueError('A and B must be square matrices of the same size')
    if not (isinstance(alpha, (int, float)) and float(alpha) >= 0):
        raise ValueError('alpha must be >= 0')
    return 0.5j * (B @ A + A @ B) + 1j * float(alpha) * (A * A * A)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Case 1
        {
            "setup": """import numpy as np
n = 4
A = np.zeros((n, n), dtype=complex)
B = np.diag(np.ones(n-1), 1) + np.diag(np.ones(n-1), -1)
alpha = 0.5
""",
            "call": "float(np.linalg.norm(schrodinger_rhs(A, B, alpha)))",
            "gold_call": "float(np.linalg.norm(_oracle_schrodinger_rhs(A, B, alpha)))",
        },
        # Case 2
        {
            "setup": """import numpy as np
n = 5
A = np.eye(n, dtype=complex) * 3.0
B = np.diag(np.ones(n-1), 1) + np.diag(np.ones(n-1), -1)
alpha = 0.0
result_model = schrodinger_rhs(A, B, alpha)
result_gold = _oracle_schrodinger_rhs(A, B, alpha)
def check_imag(F_A):
    return float(np.max(np.abs(F_A.real)))
""",
            "call": "check_imag(result_model)",
            "gold_call": "check_imag(result_gold)",
        },
        # Case 3
        {
            "setup": """import numpy as np
np.random.seed(17)
n = 6
A = (np.random.randn(n, n) + 1j * np.random.randn(n, n))
B = np.diag(np.ones(n-1), 1) + np.diag(np.ones(n-1), -1)
alpha = 0.5
""",
            "call": "np.round(schrodinger_rhs(A, B, alpha), 8).tolist()",
            "gold_call": "np.round(_oracle_schrodinger_rhs(A, B, alpha), 8).tolist()",
        },
        # Case 4
        {
            "setup": """import numpy as np
n = 4
A = np.ones((n, n), dtype=complex) * (1.0 + 0.5j)
B = np.diag(np.ones(n-1), 1) + np.diag(np.ones(n-1), -1)
alpha = 2.0
""",
            "call": "np.round(schrodinger_rhs(A, B, alpha), 8).tolist()",
            "gold_call": "np.round(_oracle_schrodinger_rhs(A, B, alpha), 8).tolist()",
        },
        # Case 5
        {
            "setup": """import numpy as np
n = 8
sigma = 0.15 * n
mu1, mu2 = 0.7*n, 0.4*n
nu1, nu2 = 0.6*n, 0.3*n
j = np.arange(1, n+1, dtype=float)
k = np.arange(1, n+1, dtype=float)
A0 = (np.exp(-((j[:,None]-mu1)**2 + (k[None,:]-nu1)**2)/sigma**2)
    + np.exp(-((j[:,None]-mu2)**2 + (k[None,:]-nu2)**2)/sigma**2)).astype(complex)
B = np.diag(np.ones(n-1), 1) + np.diag(np.ones(n-1), -1)
alpha = 0.5
""",
            "call": "round(float(np.linalg.norm(schrodinger_rhs(A0, B, alpha))), 8)",
            "gold_call": "round(float(np.linalg.norm(_oracle_schrodinger_rhs(A0, B, alpha))), 8)",
        },
        # Case 6
        {
            "setup": """import numpy as np
n = 5
A = np.array([[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7],
              [4,5,6,7,8],[5,6,7,8,9]], dtype=complex)
B = np.diag(np.ones(n-1), 1) + np.diag(np.ones(n-1), -1)
alpha = 0.5
def is_symmetric_imaginary(F_A):
    return (float(np.max(np.abs(F_A.real))) < 1e-12 and
            float(np.max(np.abs(F_A - F_A.T))) < 1e-12)
""",
            "call": "is_symmetric_imaginary(schrodinger_rhs(A, B, alpha))",
            "gold_call": "is_symmetric_imaginary(_oracle_schrodinger_rhs(A, B, alpha))",
        },
        # Case 7
        {
            "setup": """import numpy as np
A = np.zeros((3, 3), dtype=complex)
B = np.zeros((4, 4))
def run_model():
    try:
        schrodinger_rhs(A, B, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_schrodinger_rhs(A, B, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # Case 8
        {
            "setup": """import numpy as np
A = np.ones((3, 3), dtype=complex)
B = np.diag(np.ones(2), 1) + np.diag(np.ones(2), -1)
def run_model():
    try:
        schrodinger_rhs(A, B, -0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_schrodinger_rhs(A, B, -0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # Case 9
        {
            "setup": """import numpy as np
n = 4
A = np.eye(n, dtype=complex)
B = np.diag(np.ones(n-1), 1) + np.diag(np.ones(n-1), -1)
alpha = 0.0
def sign_check(F_A):
    return float(F_A[0, 1].imag) > 0
""",
            "call": "sign_check(schrodinger_rhs(A, B, alpha))",
            "gold_call": "sign_check(_oracle_schrodinger_rhs(A, B, alpha))",
        },
        # Case 10
        {
            "setup": """import numpy as np
np.random.seed(303)
n = 5
A = (np.random.randn(n, n) + 1j * np.random.randn(n, n))
B = np.diag(np.ones(n-1), 1) + np.diag(np.ones(n-1), -1)
alpha = 1.0
""",
            "call": "np.round(schrodinger_rhs(A, B, alpha), 8).tolist()",
            "gold_call": "np.round(_oracle_schrodinger_rhs(A, B, alpha), 8).tolist()",
        },
        # Case 11
        {
            "setup": """import numpy as np
A = np.array([[1+2j, 3+4j], [5+6j, 7+8j]])
B = np.array([[0, 1], [1, 0]], dtype=float)
alpha = 1.0
def cubic_check(F_A):
    return tuple(np.round(F_A.ravel(), 8).tolist())
""",
            "call": "cubic_check(schrodinger_rhs(A, B, alpha))",
            "gold_call": "cubic_check(_oracle_schrodinger_rhs(A, B, alpha))",
        },
    ]
