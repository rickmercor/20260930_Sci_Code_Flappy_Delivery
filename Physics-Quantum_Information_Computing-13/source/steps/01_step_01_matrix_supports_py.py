"""
Identify the active row and column indices of a binary syndrome-like matrix.

The decoder begins with row and column support rather than individual nonzero entries. For a binary matrix, row support contains every row with at least one nonzero entry and column support is defined analogously.

Returns
-------
tuple[list[int], list[int]], containing ascending row support followed by ascending column support
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def matrix_supports(M: "np.ndarray") -> tuple[list[int], list[int]]:
    """Return ascending row-support and column-support indices of a binary matrix.

    Parameters
    ----------
    M : np.ndarray
        Nonempty two-dimensional binary matrix.

    Returns
    -------
    result : tuple[list[int], list[int]]
        Row-support indices followed by column-support indices.
    """
    return [], []

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_matrix_supports(M: np.ndarray) -> tuple[list[int], list[int]]:
    import numpy as np

    M = np.asarray(M)
    if M.ndim != 2 or M.size == 0:
        raise ValueError("M must be a nonempty 2D matrix")
    if not np.all((M == 0) | (M == 1)):
        raise ValueError("M must be binary")

    row_support = np.flatnonzero(np.any(M, axis=1)).astype(int).tolist()
    col_support = np.flatnonzero(np.any(M, axis=0)).astype(int).tolist()
    return row_support, col_support

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
def _pack(result):
    rows, cols = result
    return np.asarray([len(rows), *rows, len(cols), *cols], dtype=int)
M_candidate = np.array([[1,0,1],[0,0,0],[0,1,0]], dtype=int)
M_oracle = M_candidate.copy()
""",
            "call": "_pack(matrix_supports(M_candidate))",
            "gold_call": "_pack(_oracle_matrix_supports(M_oracle))",
        },
        {
            "setup": """import numpy as np
def _pack(result):
    rows, cols = result
    return np.asarray([len(rows), *rows, len(cols), *cols], dtype=int)
M_candidate = np.zeros((1,1), dtype=int)
M_oracle = M_candidate.copy()
""",
            "call": "_pack(matrix_supports(M_candidate))",
            "gold_call": "_pack(_oracle_matrix_supports(M_oracle))",
        },
        {
            "setup": """import numpy as np
def _pack(result):
    rows, cols = result
    return np.asarray([len(rows), *rows, len(cols), *cols], dtype=int)
M_candidate = np.zeros((2,3), dtype=int)
M_oracle = M_candidate.copy()
""",
            "call": "_pack(matrix_supports(M_candidate))",
            "gold_call": "_pack(_oracle_matrix_supports(M_oracle))",
        },
    ]
