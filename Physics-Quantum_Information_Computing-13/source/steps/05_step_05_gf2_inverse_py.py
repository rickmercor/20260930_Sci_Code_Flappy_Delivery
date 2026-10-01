"""
Invert a square binary matrix exactly over $\mathbb F_2$.

Ordinary floating-point inversion is wrong for this decoder. Pivoting, elimination, addition, and multiplication must all be interpreted in the two-element field.

Returns
-------
np.ndarray, uint8 square inverse over GF(2)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def gf2_inverse(M: "np.ndarray") -> "np.ndarray":
    """Return the inverse of a square binary matrix over GF(2)."""
    return M

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_gf2_inverse(M: np.ndarray) -> np.ndarray:
    import numpy as np

    M = np.asarray(M)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M must be square")
    if not np.all((M == 0) | (M == 1)):
        raise ValueError("M must be binary")

    n = M.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    aug = np.concatenate(
        [M.astype(np.uint8), np.eye(n, dtype=np.uint8)],
        axis=1,
    )

    row = 0
    for col in range(n):
        pivot = next(
            (r for r in range(row, n) if aug[r, col] == 1),
            None,
        )
        if pivot is None:
            raise ValueError("matrix is singular over GF(2)")

        if pivot != row:
            aug[[row, pivot]] = aug[[pivot, row]]

        for r in range(n):
            if r != row and aug[r, col] == 1:
                aug[r] ^= aug[row]

        row += 1

    return aug[:, n:].astype(np.uint8)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
M_candidate = np.array([[1,1],[1,0]], dtype=int)
M_oracle = M_candidate.copy()
""",
            "call": "gf2_inverse(M_candidate)",
            "gold_call": "_oracle_gf2_inverse(M_oracle)",
        },
        {
            "setup": """import numpy as np
M_candidate = np.array([[1]], dtype=int)
M_oracle = M_candidate.copy()
""",
            "call": "gf2_inverse(M_candidate)",
            "gold_call": "_oracle_gf2_inverse(M_oracle)",
        },
        {
            "setup": """import numpy as np
M_candidate = np.zeros((0,0), dtype=int)
M_oracle = M_candidate.copy()
""",
            "call": "gf2_inverse(M_candidate)",
            "gold_call": "_oracle_gf2_inverse(M_oracle)",
        },
    ]
