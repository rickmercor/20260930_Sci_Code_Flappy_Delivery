"""
Propagate the initialized DPCG state through exactly three complete updates and return only the final iterate u3 of length n, after the third update.

This stage advances the finite DPCG trajectory through three completed updates. The returned object is the final solution iterate u3 itself, not the packed auxiliary solver state and not the fully converged solution of the linear system.

Returns
-------
return u3
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def run_three_dpcg_updates(
    A: np.ndarray,
    M: np.ndarray,
    P: np.ndarray,
    coarse_data: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
    """Advance the solver state through the required finite-update sequence.

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
        ``(n, k)`` compatible with the supplied solver state and coarse data.
    coarse_data : np.ndarray
        One-dimensional packed reduced/full-space operator representation
        produced by the preceding coarse-operator stage.
    state : np.ndarray
        One-dimensional packed solver state obtained from the initialization
        stage. Its dimensions must be consistent with ``A`` and ``P``.

    Returns
    -------
    np.ndarray
        One-dimensional ``float64`` vector of shape ``(n,)`` containing the
        final full-space iterate produced by this finite-update stage.

    Raises
    ------
    ValueError
        If ``A`` is not two-dimensional or square.
        If ``M`` does not have shape ``(n, n)``.
        If ``P`` does not have ``n`` rows.
        If ``coarse_data`` or ``state`` is not one-dimensional.
        If the dimensions of the supplied arguments are mutually
        inconsistent.
        If any required input has a zero-sized dimension.
        If any input contains a non-finite value.
        If the supplied packed state is incompatible with the dimensions
        implied by ``A`` and ``P``.
    TypeError
        If any argument cannot be interpreted as a numerical NumPy array.
    """
    return np.empty(A.shape[0], dtype=np.float64)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_run_three_dpcg_updates(
    A: np.ndarray,
    M: np.ndarray,
    P: np.ndarray,
    coarse_data: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
    """Reference execution of exactly three DPCG updates."""
    current = np.asarray(state, dtype=np.float64)

    for _ in range(3):
        current = _oracle_dpcg_update(
            A,
            M,
            P,
            coarse_data,
            current,
        )

        current = current[:-2]

    n = A.shape[0]
    return current[:n].copy()

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
import numpy as np

A = np.array([
    [4.5, 0.35, 0.00, 0.00, 0.00, 0.00],
    [0.35, 5.0, 0.45, 0.00, 0.00, 0.00],
    [0.00, 0.45, 4.2, 0.30, 0.00, 0.00],
    [0.00, 0.00, 0.30, 5.4, 0.40, 0.00],
    [0.00, 0.00, 0.00, 0.40, 4.8, 0.50],
    [0.00, 0.00, 0.00, 0.00, 0.50, 5.2],
], dtype=np.float64)

f = np.array(
    [1.0, -0.8, 1.7, 0.6, -1.3, 0.9],
    dtype=np.float64,
)

u00 = np.array(
    [0.1, -0.05, 0.08, -0.12, 0.04, 0.07],
    dtype=np.float64,
)

M = np.diag(
    1.0 / np.diag(A)
).astype(np.float64)

P = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.5],
    [0.5, 1.0],
    [1.0, -0.5],
    [-0.5, 1.0],
], dtype=np.float64)

P, _ = np.linalg.qr(
    P,
    mode="reduced",
)

coarse_data = _oracle_build_coarse_operators(
    A,
    P,
)

state = _oracle_initialize_dpcg(
    A,
    f,
    u00,
    M,
    P,
    coarse_data,
)
""",
            "call": (
                "run_three_dpcg_updates("
                "A, M, P, coarse_data, state)"
            ),
            "gold_call": (
                "_oracle_run_three_dpcg_updates("
                "A, M, P, coarse_data, state)"
            ),
        },
        {
            "setup": """
import numpy as np

A = np.array([
    [5., 0.5, 0., 0., 0., 0.],
    [0.5, 4., 0.4, 0., 0., 0.],
    [0., 0.4, 6., 0.3, 0., 0.],
    [0., 0., 0.3, 4.5, 0.6, 0.],
    [0., 0., 0., 0.6, 5.5, 0.7],
    [0., 0., 0., 0., 0.7, 6.5],
], dtype=np.float64)

f = np.array(
    [1., 2., -1., 0.5, -2., 1.5],
    dtype=np.float64,
)

u00 = np.zeros(6, dtype=np.float64)

M = np.diag(
    1.0 / np.diag(A)
).astype(np.float64)

P = np.array([
    [1., 0.],
    [0., 1.],
    [1., 1.],
    [0., 1.],
    [1., 0.],
    [0., 1.],
], dtype=np.float64)

P, _ = np.linalg.qr(
    P,
    mode="reduced",
)

coarse_data = _oracle_build_coarse_operators(
    A,
    P,
)

state = _oracle_initialize_dpcg(
    A,
    f,
    u00,
    M,
    P,
    coarse_data,
)
""",
            "call": (
                "run_three_dpcg_updates("
                "A, M, P, coarse_data, state)"
            ),
            "gold_call": (
                "_oracle_run_three_dpcg_updates("
                "A, M, P, coarse_data, state)"
            ),
        },
        {
            "setup": """
import numpy as np

A = np.array([
    [3., 0.8, 0.2, 0., 0., 0.],
    [0.8, 4., -0.4, 0.1, 0., 0.],
    [0.2, -0.4, 3.5, 0.7, 0.2, 0.],
    [0., 0.1, 0.7, 4.2, -0.5, 0.3],
    [0., 0., 0.2, -0.5, 5., 0.6],
    [0., 0., 0., 0.3, 0.6, 3.8],
], dtype=np.float64)

f = np.array(
    [-1., 0.5, 2., -0.8, 1.2, -1.5],
    dtype=np.float64,
)

u00 = np.array(
    [0.1, -0.1, 0.05, 0., -0.05, 0.1],
    dtype=np.float64,
)

M = np.diag(
    1.0 / np.diag(A)
).astype(np.float64)

P = np.eye(
    6,
    2,
    dtype=np.float64,
)

coarse_data = _oracle_build_coarse_operators(
    A,
    P,
)

state = _oracle_initialize_dpcg(
    A,
    f,
    u00,
    M,
    P,
    coarse_data,
)
""",
            "call": (
                "run_three_dpcg_updates("
                "A, M, P, coarse_data, state)"
            ),
            "gold_call": (
                "_oracle_run_three_dpcg_updates("
                "A, M, P, coarse_data, state)"
            ),
        },
    ]
