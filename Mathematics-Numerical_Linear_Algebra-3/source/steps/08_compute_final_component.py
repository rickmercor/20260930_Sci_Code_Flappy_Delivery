"""
Combine the outputs of the preceding numerical stages and extract the scalar required by the task specification.

The final stage represents the composition of the preceding numerical transformations into one deterministic result. Its purpose is to expose only the requested scalar from the completed computational pipeline.

Returns
-------
return float(u3[5])
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_final_component(
    A: np.ndarray,
    f: np.ndarray,
    u00: np.ndarray,
    M: np.ndarray,
    B: np.ndarray,
    T: np.ndarray,
    groups: tuple[np.ndarray, ...],
) -> float:
    """Compute the requested scalar from the supplied solver instance.

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
        ``(n, n)`` compatible with ``A``.
    B : np.ndarray
        Two-dimensional finite floating-point operator-learning data matrix.
        Its dimensions must be compatible with the supplied trunk data and
        the source-defined numerical construction.
    T : np.ndarray
        Two-dimensional finite floating-point operator-learning data matrix.
        Its dimensions must be compatible with the supplied branch data and
        the source-defined numerical construction.
    groups : tuple[np.ndarray, ...]
        Tuple of non-empty one-dimensional integer index arrays specifying
        the prescribed partition of the system degrees of freedom.

    Returns
    -------
    float
        The finite scalar requested by the task for the supplied deterministic
        instance. The result is returned as a Python ``float`` and is computed
        using the numerical convention required by the solver pipeline.

    Raises
    ------
    ValueError
        If ``A`` is not two-dimensional or square.
        If ``f`` or ``u00`` does not have shape ``(n,)``.
        If ``M`` does not have shape ``(n, n)``.
        If ``B`` or ``T`` is not two-dimensional.
        If the dimensions of ``B`` and ``T`` are incompatible with the
        source-defined construction.
        If ``groups`` is empty.
        If a group is not one-dimensional or is empty.
        If any group contains an invalid, repeated, or out-of-range index.
        If the groups do not form a valid partition of the system degrees of
        freedom.
        If the supplied inputs have mutually inconsistent dimensions.
        If any required input contains a non-finite value.
        If the requested scalar cannot be produced from the supplied
        deterministic instance.
    TypeError
        If an argument cannot be interpreted as the required numerical NumPy
        array or integer index-array representation.

    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_final_component(
    A: np.ndarray,
    f: np.ndarray,
    u00: np.ndarray,
    M: np.ndarray,
    B: np.ndarray,
    T: np.ndarray,
    groups: tuple[np.ndarray, ...],
) -> float:
    """Reference end-to-end computation."""
    P_tilde = _oracle_construct_rs_deflation(B, T)

    grouped = _oracle_split_deflation_blocks(
        P_tilde,
        groups,
    )

    P = _oracle_build_block_deflation_operator(
        grouped,
        groups,
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

    u3 = _oracle_run_three_dpcg_updates(
        A,
        M,
        P,
        coarse_data,
        state,
    )

    return float(u3[5])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
import numpy as np

A = np.array([
    [4.8, 0.6, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.6, 5.1, -0.4, 0.3, 0.0, 0.0, 0.0, 0.0],
    [0.2, -0.4, 4.6, 0.7, 0.2, 0.0, 0.0, 0.0],
    [0.0, 0.3, 0.7, 5.3, -0.5, 0.1, 0.0, 0.0],
    [0.0, 0.0, 0.2, -0.5, 4.9, 0.8, 0.3, 0.0],
    [0.0, 0.0, 0.0, 0.1, 0.8, 5.2, -0.6, 0.2],
    [0.0, 0.0, 0.0, 0.0, 0.3, -0.6, 4.7, 0.5],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.5, 5.4],
], dtype=np.float64)

f = np.array(
    [1.1, -0.8, 2.0, -1.4, 0.7, 1.6, -1.1, 0.9],
    dtype=np.float64,
)

u00 = np.array(
    [0.2, -0.1, 0.15, -0.05, 0.08, -0.12, 0.04, 0.1],
    dtype=np.float64,
)

M = np.diag(
    1.0 / np.diag(A)
).astype(np.float64)

B = np.array([
    [0.9, -0.5, 1.1, 0.3],
    [-0.7, 1.2, 0.4, -1.0],
], dtype=np.float64)

T = np.array([
    [0.2, 0.8, -0.3, 1.1],
    [0.6, -0.4, 0.9, 0.2],
    [1.0, 0.3, -0.7, 0.5],
    [-0.5, 1.1, 0.2, -0.8],
    [0.7, -0.9, 0.6, 0.4],
    [1.2, 0.1, -0.5, 0.8],
    [-0.3, 0.7, 1.0, -0.2],
    [0.5, -1.1, 0.4, 0.9],
], dtype=np.float64)

groups = (
    np.array([0, 2, 4, 6], dtype=int),
    np.array([1, 3, 5, 7], dtype=int),
)
""",
            "call": (
                "compute_final_component("
                "A, f, u00, M, B, T, groups)"
            ),
            "gold_call": (
                "_oracle_compute_final_component("
                "A, f, u00, M, B, T, groups)"
            ),
        },

        {
            "setup": """
import numpy as np

A = np.array([
    [6.0, 0.4, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.4, 4.9, 0.5, -0.2, 0.0, 0.0, 0.0, 0.0],
    [0.1, 0.5, 5.7, 0.6, 0.1, 0.0, 0.0, 0.0],
    [0.0, -0.2, 0.6, 4.8, -0.3, 0.2, 0.0, 0.0],
    [0.0, 0.0, 0.1, -0.3, 5.5, 0.7, -0.1, 0.0],
    [0.0, 0.0, 0.0, 0.2, 0.7, 5.0, 0.4, -0.2],
    [0.0, 0.0, 0.0, 0.0, -0.1, 0.4, 5.9, 0.6],
    [0.0, 0.0, 0.0, 0.0, 0.0, -0.2, 0.6, 4.6],
], dtype=np.float64)

f = np.array(
    [-0.8, 1.4, 0.6, -1.2, 1.7, -0.5, 2.1, 0.3],
    dtype=np.float64,
)

u00 = np.array(
    [-0.05, 0.12, -0.08, 0.1, 0.04, -0.15, 0.09, -0.03],
    dtype=np.float64,
)

M = np.diag(
    1.0 / np.diag(A)
).astype(np.float64)

B = np.array([
    [1.1, 0.2, -0.8, 0.6],
    [-0.3, 1.4, 0.5, -0.9],
], dtype=np.float64)

T = np.array([
    [0.4, -0.2, 0.7, 1.0],
    [0.9, 0.3, -0.5, 0.1],
    [-0.6, 1.0, 0.2, 0.8],
    [0.5, -0.7, 1.1, -0.3],
    [1.2, 0.4, -0.1, 0.6],
    [-0.2, 0.8, 0.9, -0.5],
    [0.7, -1.0, 0.3, 0.4],
    [0.1, 0.5, -0.9, 1.2],
], dtype=np.float64)

groups = (
    np.array([0, 2, 4, 6], dtype=int),
    np.array([1, 3, 5, 7], dtype=int),
)
""",
            "call": (
                "compute_final_component("
                "A, f, u00, M, B, T, groups)"
            ),
            "gold_call": (
                "_oracle_compute_final_component("
                "A, f, u00, M, B, T, groups)"
            ),
        },

        {
            "setup": """
import numpy as np

A = np.array([
    [4.4, 0.7, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.7, 5.6, 0.3, -0.4, 0.0, 0.0, 0.0, 0.0],
    [-0.1, 0.3, 4.9, 0.5, 0.2, 0.0, 0.0, 0.0],
    [0.0, -0.4, 0.5, 5.2, -0.6, 0.1, 0.0, 0.0],
    [0.0, 0.0, 0.2, -0.6, 5.7, 0.4, -0.2, 0.0],
    [0.0, 0.0, 0.0, 0.1, 0.4, 4.6, 0.5, -0.3],
    [0.0, 0.0, 0.0, 0.0, -0.2, 0.5, 5.3, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.0, -0.3, 0.4, 4.8],
], dtype=np.float64)

f = np.array(
    [0.6, -1.3, 1.8, 0.9, -0.7, 1.1, -1.9, 0.4],
    dtype=np.float64,
)

u00 = np.array(
    [0.05, -0.08, 0.11, -0.06, 0.09, 0.03, -0.07, 0.12],
    dtype=np.float64,
)

M = np.diag(
    1.0 / np.diag(A)
).astype(np.float64)

B = np.array([
    [0.5, 1.0, -0.6, 0.8],
    [1.2, -0.3, 0.9, -0.7],
], dtype=np.float64)

T = np.array([
    [0.7, -0.1, 0.5, 0.9],
    [-0.4, 0.8, 0.2, -0.6],
    [1.1, 0.3, -0.9, 0.4],
    [0.2, -0.7, 1.0, 0.5],
    [-0.8, 0.6, 0.3, 1.1],
    [0.9, -0.5, 0.7, -0.2],
    [0.4, 1.2, -0.1, 0.6],
    [-0.3, 0.5, 0.8, -0.9],
], dtype=np.float64)

groups = (
    np.array([0, 2, 4, 6], dtype=int),
    np.array([1, 3, 5, 7], dtype=int),
)
""",
            "call": (
                "compute_final_component("
                "A, f, u00, M, B, T, groups)"
            ),
            "gold_call": (
                "_oracle_compute_final_component("
                "A, f, u00, M, B, T, groups)"
            ),
        },
    ]
