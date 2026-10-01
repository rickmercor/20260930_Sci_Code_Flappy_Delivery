"""
Initialize the DPCG state from the supplied problem data and previously constructed coarse-space representation. The result must preserve the interface required by the next solver stage: the packed state contains the current iterate, residual, preconditioned residual, search direction, and coarse coefficients in the order $$ [u_0,r_0,z_0,p_0,\mu_0], $$ with the vector blocks retaining their natural dimensions.

The initialization establishes the starting state of the deflated Krylov process from the supplied approximation, linear system, preconditioner, and coarse-space representation. The downstream iteration consumes this packed state directly, so its component ordering and dimensions must be preserved.

Returns
-------
return np.concatenate([     u0,     r0,     z0,     p0,     mu0, ])
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def initialize_dpcg(
    A: np.ndarray,
    f: np.ndarray,
    u00: np.ndarray,
    M: np.ndarray,
    P: np.ndarray,
    coarse_data: np.ndarray,
) -> np.ndarray:
    """Initialize the finite iterative solver state and return its packed representation.

    Parameters
    ----------
    A : np.ndarray
        Two-dimensional square finite floating-point system matrix with shape
        ``(n, n)``.
    f : np.ndarray
        One-dimensional finite floating-point right-hand-side vector with
        shape ``(n,)``.
    u00 : np.ndarray
        One-dimensional finite floating-point starting vector with shape
        ``(n,)``.
    M : np.ndarray
        Two-dimensional finite floating-point preconditioner with shape
        ``(n, n)``.
    P : np.ndarray
        Two-dimensional finite floating-point global auxiliary basis with
        shape ``(n, k)``.
    coarse_data : np.ndarray
        One-dimensional packed representation produced by the preceding
        coarse-operator stage. Its contents must satisfy the interface
        expected by this initialization stage.

    Returns
    -------
    np.ndarray
        One-dimensional ``float64`` packed solver state of length
        ``4*n + k``. The packed state contains four full-length vector
        segments of length ``n`` followed by one reduced-space segment of
        length ``k``. The segment ordering is part of the solver interface
        and must remain unchanged for the next iteration stage.

    Raises
    ------
    ValueError
        If ``A`` is not two-dimensional or square.
        If ``f`` or ``u00`` does not have shape ``(n,)``.
        If ``M`` does not have shape ``(n, n)``.
        If ``P`` does not have shape ``(n, k)`` for a valid reduced dimension
        ``k``.
        If ``coarse_data`` is not one-dimensional.
        If the dimensions of the supplied arrays are mutually inconsistent.
        If any required input has a zero-sized dimension.
        If any input contains a non-finite value.
        If the supplied coarse representation is incompatible with the
        dimensions implied by ``A`` and ``P``.
    TypeError
        If any argument cannot be interpreted as a numerical NumPy array.

    Notes
    -----
    The mathematical construction of the initial state and the interpretation
    of its individual vector segments are defined by the source method and
    are intentionally not specified here. This function is responsible for
    returning the packed interface consumed by the subsequent iteration
    stage.
    """
    n = A.shape[0]
    k = P.shape[1]
    return np.empty(4 * n + k, dtype=np.float64)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_initialize_dpcg(
    A: np.ndarray,
    f: np.ndarray,
    u00: np.ndarray,
    M: np.ndarray,
    P: np.ndarray,
    coarse_data: np.ndarray,
) -> np.ndarray:
    """Reference implementation of DPCG initialization."""
    A = np.asarray(A, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    u00 = np.asarray(u00, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    coarse_data = np.asarray(coarse_data, dtype=np.float64)

    n = A.shape[0]
    k = P.shape[1]

    if A.ndim != 2 or A.shape != (n, n):
        raise ValueError("A must be square.")

    if f.shape != (n,) or u00.shape != (n,):
        raise ValueError("f and u00 must have shape (n,).")

    if M.shape != (n, n):
        raise ValueError("M must have shape (n, n).")

    if P.ndim != 2 or P.shape[0] != n:
        raise ValueError("P has incompatible dimensions.")

    r_size = k * n
    ac_size = k * k
    c_size = n * n

    expected_coarse_size = r_size + ac_size + c_size

    if coarse_data.size != expected_coarse_size:
        raise ValueError("Invalid packed coarse-data size.")

    offset = 0

    R = coarse_data[offset:offset + r_size].reshape(k, n)
    offset += r_size

    Ac = coarse_data[offset:offset + ac_size].reshape(k, k)
    offset += ac_size

    C = coarse_data[offset:offset + c_size].reshape(n, n)

    u0 = u00 + C @ (f - A @ u00)
    r0 = f - A @ u0
    z0 = M @ r0
    mu0 = np.linalg.solve(Ac, R @ A @ z0)
    p0 = z0 - P @ mu0

    return np.concatenate([
        u0,
        r0,
        z0,
        p0,
        mu0,
    ])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
import numpy as np

A = np.array([
    [4.0, 1.0],
    [1.0, 4.0],
], dtype=np.float64)

f = np.array([1.0, 2.0], dtype=np.float64)
u00 = np.zeros(2, dtype=np.float64)
M = np.diag(1.0 / np.diag(A))

P = np.array([
    [1.0],
    [1.0],
], dtype=np.float64) / np.sqrt(2.0)

coarse_data = _oracle_build_coarse_operators(A, P)
""",
            "call": "initialize_dpcg(A, f, u00, M, P, coarse_data)",
            "gold_call": "_oracle_initialize_dpcg(A, f, u00, M, P, coarse_data)",
        },
        {
            "setup": """
import numpy as np

A = np.diag(
    np.array([2.0, 3.0, 5.0], dtype=np.float64)
)

f = np.array([1.0, 2.0, 3.0], dtype=np.float64)
u00 = np.array([0.1, -0.2, 0.3], dtype=np.float64)
M = np.diag(1.0 / np.diag(A))

P = np.eye(3, 1, dtype=np.float64)

coarse_data = _oracle_build_coarse_operators(A, P)
""",
            "call": "initialize_dpcg(A, f, u00, M, P, coarse_data)",
            "gold_call": "_oracle_initialize_dpcg(A, f, u00, M, P, coarse_data)",
        },
        {
            "setup": """
import numpy as np

A = np.array([
    [6.0, 1.0, 0.0, 0.0],
    [1.0, 6.0, 1.0, 0.0],
    [0.0, 1.0, 6.0, 1.0],
    [0.0, 0.0, 1.0, 6.0],
], dtype=np.float64)

f = np.ones(4, dtype=np.float64)
u00 = np.zeros(4, dtype=np.float64)
M = np.diag(1.0 / np.diag(A))

P = np.eye(4, 2, dtype=np.float64)

coarse_data = _oracle_build_coarse_operators(A, P)
""",
            "call": "initialize_dpcg(A, f, u00, M, P, coarse_data)",
            "gold_call": "_oracle_initialize_dpcg(A, f, u00, M, P, coarse_data)",
        },
    ]
