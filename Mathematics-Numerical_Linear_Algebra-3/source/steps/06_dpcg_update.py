"""
Advance the packed DPCG state by one complete iteration and return the updated state together with the two scalar quantities required to continue the recurrence. The output consists of the updated state in the Step 5 layout, followed by alpha and then beta.

The iteration advances the finite solver trajectory while retaining the auxiliary quantities needed for the following update. The scalar recurrence data are returned explicitly so the next stage can continue from the resulting state.

Returns
-------
return np.concatenate([     u_new,     r_new,     z_new,     p_new,     mu_new,     np.array([alpha, beta], dtype=np.float64), ])
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def dpcg_update(
    A: np.ndarray,
    M: np.ndarray,
    P: np.ndarray,
    coarse_data: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
    """Advance the packed iterative solver state by one complete update.

    Parameters
    ----------
    A : np.ndarray
        Two-dimensional square finite floating-point system matrix with shape
        ``(n, n)``.
    M : np.ndarray
        Two-dimensional finite floating-point preconditioner with shape
        ``(n, n)`` compatible with ``A``.
    P : np.ndarray
        Two-dimensional finite floating-point auxiliary basis with shape
        ``(n, k)`` compatible with the solver state and coarse representation.
    coarse_data : np.ndarray
        One-dimensional packed reduced/full-space operator representation
        produced by the preceding coarse-operator stage.
    state : np.ndarray
        One-dimensional packed solver state produced by the initialization
        stage or by an earlier call to this function. Its length must be
        consistent with the dimensions of ``A`` and ``P``.

    Returns
    -------
    np.ndarray
        One-dimensional ``float64`` array containing the updated packed
        solver state followed by two scalar continuation values. If
        ``state.size`` is ``4*n + k``, the returned array has length
        ``state.size + 2``. The first ``state.size`` entries retain the
        prescribed packed-state layout, and the final two entries contain the
        scalar recurrence quantities required by the subsequent update.

    Raises
    ------
    ValueError
        If ``A`` is not two-dimensional or square.
        If ``M`` is not shape ``(n, n)``.
        If ``P`` does not have ``n`` rows.
        If ``coarse_data`` is not one-dimensional.
        If ``state`` is not one-dimensional.
        If ``state.size`` is inconsistent with the dimensions implied by
        ``A`` and ``P``.
        If any input has a zero-sized dimension.
        If any input contains a non-finite value.
        If the supplied coarse representation is incompatible with the
        supplied matrix and basis dimensions.
    TypeError
        If any argument cannot be interpreted as a numerical NumPy array.
    """
    return np.empty(
        state.size + 2,
        dtype=np.float64,
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_dpcg_update(
    A: np.ndarray,
    M: np.ndarray,
    P: np.ndarray,
    coarse_data: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
    """Reference implementation of one complete DPCG update."""
    A = np.asarray(A, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    coarse_data = np.asarray(coarse_data, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)

    n = A.shape[0]
    k = P.shape[1]

    # Unpack coarse operators.
    r_size = k * n
    ac_size = k * k
    c_size = n * n

    offset = 0
    R = coarse_data[offset:offset + r_size].reshape(k, n)
    offset += r_size

    Ac = coarse_data[offset:offset + ac_size].reshape(k, k)

    # Unpack current state.
    offset = 0

    u = state[offset:offset + n]
    offset += n

    r = state[offset:offset + n]
    offset += n

    z = state[offset:offset + n]
    offset += n

    p = state[offset:offset + n]
    offset += n

    mu = state[offset:offset + k]

    alpha = np.dot(r, z) / np.dot(p, A @ p)

    u_new = u + alpha * p
    r_new = r - alpha * (A @ p)
    z_new = M @ r_new

    mu_new = np.linalg.solve(
        Ac,
        R @ A @ z_new,
    )

    beta = np.dot(r_new, z_new) / np.dot(r, z)

    p_new = (
        beta * p
        + z_new
        - P @ mu_new
    )

    return np.concatenate([
        u_new,
        r_new,
        z_new,
        p_new,
        mu_new,
        np.array([alpha, beta], dtype=np.float64),
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
state = _oracle_initialize_dpcg(
    A, f, u00, M, P, coarse_data
)
""",
            "call": "dpcg_update(A, M, P, coarse_data, state)",
            "gold_call": "_oracle_dpcg_update(A, M, P, coarse_data, state)",
        },
        {
            "setup": """
import numpy as np

A = np.diag(
    np.array([2.0, 3.0, 5.0], dtype=np.float64)
)

f = np.array([1.0, 2.0, 3.0], dtype=np.float64)
u00 = np.zeros(3, dtype=np.float64)
M = np.diag(1.0 / np.diag(A))

P = np.eye(3, 1, dtype=np.float64)

coarse_data = _oracle_build_coarse_operators(A, P)
state = _oracle_initialize_dpcg(
    A, f, u00, M, P, coarse_data
)
""",
            "call": "dpcg_update(A, M, P, coarse_data, state)",
            "gold_call": "_oracle_dpcg_update(A, M, P, coarse_data, state)",
        },
        {
            "setup": """
import numpy as np

A = np.array([
    [5.0, 1.0, 0.0],
    [1.0, 5.0, 1.0],
    [0.0, 1.0, 5.0],
], dtype=np.float64)

f = np.array([1.0, 0.0, 2.0], dtype=np.float64)
u00 = np.zeros(3, dtype=np.float64)
M = np.diag(1.0 / np.diag(A))

P = np.eye(3, 2, dtype=np.float64)

coarse_data = _oracle_build_coarse_operators(A, P)
state = _oracle_initialize_dpcg(
    A, f, u00, M, P, coarse_data
)
""",
            "call": "dpcg_update(A, M, P, coarse_data, state)",
            "gold_call": "_oracle_dpcg_update(A, M, P, coarse_data, state)",
        },
    ]
