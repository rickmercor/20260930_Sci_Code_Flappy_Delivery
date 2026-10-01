"""
Generate the global structured deflation basis by applying a reduced QR factorization independently to the restricted tentative deflation block for each supplied degree-of-freedom group, using numpy.linalg.qr with mode="reduced", and place each resulting orthonormal block on its original global degree-of-freedom indices.

The structured deflation method constructs local orthonormal bases for the supplied degree-of-freedom groups and embeds them into the global basis at their original row locations. The implementation uses NumPy's reduced QR factorization, so the returned column signs follow NumPy's convention without additional sign normalization, pivoting, or post-processing.

Returns
-------
return np.empty( ( grouped_p_tilde.shape[0], grouped_p_tilde.shape[1] * len(groups), ), dtype=np.float64, )
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def build_block_deflation_operator(
    grouped_p_tilde: np.ndarray,
    groups: tuple[np.ndarray, ...],
) -> np.ndarray:
    """Build the global structured representation from the grouped input.

    Parameters
    ----------
    grouped_p_tilde : np.ndarray
        Two-dimensional floating-point array with shape ``(n, k)``
        representing the grouped intermediate data produced by the preceding
        stage. The row dimension must be consistent with the supplied global
        degree-of-freedom indexing, and the array must contain finite values.
    groups : tuple[np.ndarray, ...]
        Tuple of non-empty one-dimensional integer index arrays describing the
        prescribed degree-of-freedom groups. The groups must be mutually
        disjoint, collectively cover the valid row indices, and be consistent
        with ``grouped_p_tilde``.

    Returns
    -------
    np.ndarray
        A two-dimensional ``float64`` array with shape
        ``(n, k * len(groups))`` representing the global structured basis
        required by the next solver stage.

    Raises
    ------
    ValueError
        If ``grouped_p_tilde`` is not two-dimensional.
        If either dimension of ``grouped_p_tilde`` is zero.
        If ``groups`` is empty.
        If any group is not one-dimensional or is empty.
        If a group contains an invalid, repeated, or out-of-range index.
        If the supplied groups do not form a valid disjoint partition of the
        rows of ``grouped_p_tilde``.
        If the dimensions of the supplied groups are inconsistent with the
        input representation.
        If the resulting structured representation cannot be formed with the
        required dimensions.
    TypeError
        If ``grouped_p_tilde`` cannot be interpreted as a numerical NumPy
        array or a group cannot be interpreted as an integer NumPy index
        array.
    """
    n = grouped_p_tilde.shape[0]
    k = grouped_p_tilde.shape[1]

    return np.empty(
        (n, k * len(groups)),
        dtype=np.float64,
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_build_block_deflation_operator(
    grouped_p_tilde: np.ndarray,
    groups: tuple[np.ndarray, ...],
) -> np.ndarray:
    """Reference Eq. (25): place each local Q block at its original rows."""
    grouped_p_tilde = np.asarray(
        grouped_p_tilde,
        dtype=np.float64,
    )

    n = grouped_p_tilde.shape[0]
    k = grouped_p_tilde.shape[1]
    S = len(groups)

    P = np.zeros(
        (n, k * S),
        dtype=np.float64,
    )

    seen = set()

    for s, group in enumerate(groups):
        idx = np.asarray(group, dtype=int)

        if idx.ndim != 1 or idx.size == 0:
            raise ValueError("Each group must be a non-empty 1D index array.")

        if idx.size < k:
            raise ValueError(
                "Each block must have at least k rows."
            )

        if any(int(i) in seen for i in idx):
            raise ValueError("Groups must be disjoint.")

        for i in idx:
            seen.add(int(i))

        block = grouped_p_tilde[idx, :]

        Q, _ = np.linalg.qr(
            block,
            mode="reduced",
        )

        P[idx, s * k:(s + 1) * k] = Q

    if len(seen) != n:
        raise ValueError("Groups must cover all degrees of freedom.")

    return P

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
import numpy as np

grouped_p_tilde = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [1.0, 2.0],
    [2.0, 1.0],
    [1.0, 2.0],
    [3.0, 1.0],
    [1.0, 3.0],
], dtype=np.float64)

groups = (
    np.array([0, 1, 2, 3], dtype=int),
    np.array([4, 5, 6, 7], dtype=int),
)
""",
            "call": (
                "build_block_deflation_operator("
                "grouped_p_tilde, groups)"
            ),
            "gold_call": (
                "_oracle_build_block_deflation_operator("
                "grouped_p_tilde, groups)"
            ),
        },
        {
            "setup": """
import numpy as np

grouped_p_tilde = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [2.0, 1.0],
    [1.0, 2.0],
    [3.0, 1.0],
], dtype=np.float64)

groups = (
    np.array([0, 1, 2], dtype=int),
    np.array([3, 4, 5], dtype=int),
)
""",
            "call": (
                "build_block_deflation_operator("
                "grouped_p_tilde, groups)"
            ),
            "gold_call": (
                "_oracle_build_block_deflation_operator("
                "grouped_p_tilde, groups)"
            ),
        },
        {
            "setup": """
import numpy as np

grouped_p_tilde = np.array([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0],
], dtype=np.float64)

groups = (
    np.array([0, 1], dtype=int),
    np.array([2, 3, 4], dtype=int),
)
""",
            "call": (
                "build_block_deflation_operator("
                "grouped_p_tilde, groups)"
            ),
            "gold_call": (
                "_oracle_build_block_deflation_operator("
                "grouped_p_tilde, groups)"
            ),
        },
    ]
