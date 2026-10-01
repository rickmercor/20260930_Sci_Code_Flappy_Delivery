"""
Invert a selected square submatrix of $H$ and embed the inverse back into the full $A\times B$ index space.

The decoder's inverse is not a full inverse of $H$. It acts only on the square submatrix selected by a peeling, with zeros everywhere outside the selected row and column indices.

Returns
-------
np.ndarray, shape (|A|, |B|), containing the GF(2) inverse in the selected coordinates
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def embedded_inverse(
    H: "np.ndarray",
    a_indices: list[int],
    b_indices: list[int],
) -> "np.ndarray":
    """Embed the GF(2) inverse of H[b_indices, a_indices] into A-by-B coordinates."""
    return H

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_embedded_inverse(
    H: np.ndarray,
    a_indices: list[int],
    b_indices: list[int],
) -> np.ndarray:
    import numpy as np

    H = np.asarray(H)
    if H.ndim != 2 or H.size == 0:
        raise ValueError("H must be a nonempty 2D matrix")
    if not np.all((H == 0) | (H == 1)):
        raise ValueError("H must be binary")

    a_indices = sorted(int(v) for v in a_indices)
    b_indices = sorted(int(v) for v in b_indices)

    if len(a_indices) != len(set(a_indices)):
        raise ValueError("a_indices must be unique")
    if len(b_indices) != len(set(b_indices)):
        raise ValueError("b_indices must be unique")
    if len(a_indices) != len(b_indices):
        raise ValueError("selected submatrix must be square")
    if any(v < 0 or v >= H.shape[1] for v in a_indices):
        raise ValueError("A index out of range")
    if any(v < 0 or v >= H.shape[0] for v in b_indices):
        raise ValueError("B index out of range")

    embedded = np.zeros((H.shape[1], H.shape[0]), dtype=np.uint8)

    if a_indices:
        submatrix = H[np.ix_(b_indices, a_indices)]
        inverse = _oracle_gf2_inverse(submatrix)
        embedded[np.ix_(a_indices, b_indices)] = inverse

    return embedded

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
H_candidate = np.array([[1,1,0],[0,1,1]], dtype=int)
H_oracle = H_candidate.copy()
a_candidate = [0,2]
a_oracle = a_candidate.copy()
b_candidate = [0,1]
b_oracle = b_candidate.copy()
""",
            "call": "embedded_inverse(H_candidate, a_candidate, b_candidate)",
            "gold_call": "_oracle_embedded_inverse(H_oracle, a_oracle, b_oracle)",
        },
        {
            "setup": """import numpy as np
H_candidate = np.array([[1]], dtype=int)
H_oracle = H_candidate.copy()
a_candidate = [0]
a_oracle = [0]
b_candidate = [0]
b_oracle = [0]
""",
            "call": "embedded_inverse(H_candidate, a_candidate, b_candidate)",
            "gold_call": "_oracle_embedded_inverse(H_oracle, a_oracle, b_oracle)",
        },
        {
            "setup": """import numpy as np
H_candidate = np.eye(2, dtype=int)
H_oracle = H_candidate.copy()
a_candidate = []
a_oracle = []
b_candidate = []
b_oracle = []
""",
            "call": "embedded_inverse(H_candidate, a_candidate, b_candidate)",
            "gold_call": "_oracle_embedded_inverse(H_oracle, a_oracle, b_oracle)",
        },
    ]
