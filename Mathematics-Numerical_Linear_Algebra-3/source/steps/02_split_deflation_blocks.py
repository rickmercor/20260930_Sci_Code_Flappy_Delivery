"""
Validate the supplied degree-of-freedom groups while preserving the tentative deflation representation in its original global row ordering and numerical values.

The tentative deflation representation is associated with the original degrees of freedom of the linear system. The supplied groups identify which rows belong to each structural block used by the subsequent deflation-basis construction. This stage preserves the numerical representation and its original global row locations; it does not permute, mask, zero, or otherwise modify the entries. The groups are validated as a disjoint cover of the available degrees of freedom before the next stage uses them to construct the structured basis.

Returns
-------
return np.empty_like(P_tilde, dtype=np.float64)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def split_deflation_blocks(
    P_tilde: np.ndarray,
    groups: tuple[np.ndarray, ...],
) -> np.ndarray:
    """Validate and prepare the supplied grouped representation for the next solver stage.

    Parameters
    ----------
    P_tilde : np.ndarray
        Two-dimensional tentative representation with shape ``(n, k)``.
        The array must contain finite numerical values. Its row dimension
        determines the number of degrees of freedom passed to this stage.
    groups : tuple[np.ndarray, ...]
        Tuple of one-dimensional integer index arrays describing the
        prescribed partition of the degree-of-freedom rows. Each group must
        be non-empty, contain valid integer indices, and collectively define
        the row partition expected by the solver.

    Returns
    -------
    np.ndarray
        A two-dimensional ``float64`` array with the same shape as
        ``P_tilde``. The result is expressed in the original global row
        indexing used by ``P_tilde`` and is suitable for consumption by the
        following construction stage.

    Raises
    ------
    ValueError
        If ``P_tilde`` is not two-dimensional.
        If ``P_tilde`` has an invalid or empty shape.
        If ``groups`` is empty.
        If any group is not one-dimensional or is empty.
        If any group contains an invalid, repeated, or out-of-range index.
        If the supplied groups do not form a valid disjoint partition of
        the rows of ``P_tilde``.
        If the number of groups or their indices are inconsistent with the
        supplied representation.
    TypeError
        If ``P_tilde`` or the group entries cannot be interpreted as
        numerical/integer NumPy arrays.
    """
    return np.empty_like(P_tilde, dtype=np.float64)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_split_deflation_blocks(
    P_tilde: np.ndarray,
    groups: tuple[np.ndarray, ...],
) -> np.ndarray:
    """Validate groups while preserving values and original global row order."""
    P_tilde = np.asarray(P_tilde, dtype=np.float64)

    if P_tilde.ndim != 2:
        raise ValueError("P_tilde must be 2D.")

    n = P_tilde.shape[0]
    preserved = np.empty_like(P_tilde, dtype=np.float64)

    seen = set()

    for group in groups:
        idx = np.asarray(group, dtype=int)

        if idx.ndim != 1 or idx.size == 0:
            raise ValueError("Each group must be a non-empty 1D index array.")

        if np.any(idx < 0) or np.any(idx >= n):
            raise ValueError("Group index out of range.")

        for i in idx:
            if int(i) in seen:
                raise ValueError("Groups must be disjoint.")
            seen.add(int(i))

        preserved[idx, :] = P_tilde[idx, :]

    if len(seen) != n:
        raise ValueError("Groups must cover all degrees of freedom.")

    return preserved

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
import numpy as np

P_tilde = np.array([
    [0., 1.],
    [2., 3.],
    [4., 5.],
    [6., 7.],
    [8., 9.],
    [10., 11.],
    [12., 13.],
    [14., 15.],
], dtype=np.float64)

groups = (
    np.array([0, 1, 2, 3], dtype=int),
    np.array([4, 5, 6, 7], dtype=int),
)
""",
            "call": (
                "split_deflation_blocks("
                "P_tilde, groups)"
            ),
            "gold_call": (
                "_oracle_split_deflation_blocks("
                "P_tilde, groups)"
            ),
        },
        {
            "setup": """
import numpy as np

P_tilde = np.array([
    [0., 1.],
    [2., 3.],
    [4., 5.],
    [6., 7.],
    [8., 9.],
    [10., 11.],
], dtype=np.float64)

groups = (
    np.array([0, 1, 2], dtype=int),
    np.array([3, 4, 5], dtype=int),
)
""",
            "call": (
                "split_deflation_blocks("
                "P_tilde, groups)"
            ),
            "gold_call": (
                "_oracle_split_deflation_blocks("
                "P_tilde, groups)"
            ),
        },
        {
            "setup": """
import numpy as np

P_tilde = np.array([
    [1.],
    [2.],
    [3.],
    [4.],
    [5.],
], dtype=np.float64)

groups = (
    np.array([0, 1], dtype=int),
    np.array([2, 3, 4], dtype=int),
)
""",
            "call": (
                "split_deflation_blocks("
                "P_tilde, groups)"
            ),
            "gold_call": (
                "_oracle_split_deflation_blocks("
                "P_tilde, groups)"
            ),
        },
    ]
