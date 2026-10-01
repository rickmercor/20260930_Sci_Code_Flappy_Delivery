"""
Assemble the previous graph-growth, peeling, and embedded-inverse components into the Pauli-$X$ half of the decoder.

For an $X$ syndrome $W\in\mathbb F_2^{B\times A}$, the row support drives an $A$-side region and the column support drives a $B$-side region. The first correction removes syndrome rows associated with the first peeling; the second correction removes the remaining syndrome through the opposite peeling.

Returns
-------
tuple[np.ndarray, np.ndarray], uint8 corrections X_A of shape A×A and X_B of shape B×B
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def decode_x_sector(
    H: "np.ndarray",
    W: "np.ndarray",
    beta: float,
    rounds: int,
) -> tuple["np.ndarray", "np.ndarray"]:
    """Decode the X sector and return the A-by-A and B-by-B corrections."""
    return W, W

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_decode_x_sector(
    H: np.ndarray,
    W: np.ndarray,
    beta: float,
    rounds: int,
) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np

    H = np.asarray(H)
    W = np.asarray(W)

    if H.ndim != 2 or H.size == 0:
        raise ValueError("H must be a nonempty 2D matrix")
    if W.shape != H.shape:
        raise ValueError("W must have the same shape as H")
    if not np.all((H == 0) | (H == 1)):
        raise ValueError("H must be binary")
    if not np.all((W == 0) | (W == 1)):
        raise ValueError("W must be binary")

    H = H.astype(np.uint8)
    W = W.astype(np.uint8)

    row_support, col_support = _oracle_matrix_supports(W)

    S, _ = _oracle_grow_beta_region(
        H, row_support, "B", beta, rounds
    )
    _, T = _oracle_grow_beta_region(
        H, col_support, "A", beta, rounds
    )

    _, Bp = _oracle_deterministic_peeling(H, S, "A")
    _, Ap = _oracle_deterministic_peeling(H, T, "B")

    left_inverse = _oracle_embedded_inverse(H, S, Bp)
    X_A = (left_inverse @ W) % 2

    remaining = (H @ X_A + W) % 2

    right_inverse = _oracle_embedded_inverse(H, Ap, T)
    X_B = (remaining @ right_inverse) % 2

    return X_A.astype(np.uint8), X_B.astype(np.uint8)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
def _pack(result):
    pieces = [np.asarray([len(result)], dtype=int)]
    for M in result:
        M = np.asarray(M, dtype=int)
        pieces.append(np.asarray([M.ndim, *M.shape], dtype=int))
        pieces.append(M.ravel())
    return np.concatenate(pieces)
H_candidate = np.eye(2, dtype=int)
H_oracle = H_candidate.copy()
W_candidate = np.array([[1,0],[0,0]], dtype=int)
W_oracle = W_candidate.copy()
""",
            "call": "_pack(decode_x_sector(H_candidate, W_candidate, 0.5, 1))",
            "gold_call": "_pack(_oracle_decode_x_sector(H_oracle, W_oracle, 0.5, 1))",
        },
        {
            "setup": """import numpy as np
def _pack(result):
    pieces = [np.asarray([len(result)], dtype=int)]
    for M in result:
        M = np.asarray(M, dtype=int)
        pieces.append(np.asarray([M.ndim, *M.shape], dtype=int))
        pieces.append(M.ravel())
    return np.concatenate(pieces)
H_candidate = np.eye(2, dtype=int)
H_oracle = H_candidate.copy()
W_candidate = np.zeros((2,2), dtype=int)
W_oracle = W_candidate.copy()
""",
            "call": "_pack(decode_x_sector(H_candidate, W_candidate, 0.5, 1))",
            "gold_call": "_pack(_oracle_decode_x_sector(H_oracle, W_oracle, 0.5, 1))",
        },
        {
            "setup": """import numpy as np
def _pack(result):
    pieces = [np.asarray([len(result)], dtype=int)]
    for M in result:
        M = np.asarray(M, dtype=int)
        pieces.append(np.asarray([M.ndim, *M.shape], dtype=int))
        pieces.append(M.ravel())
    return np.concatenate(pieces)
H_candidate = np.array([[1,1,0],[0,1,1]], dtype=int)
H_oracle = H_candidate.copy()
W_candidate = np.array([[1,0,0],[0,0,0]], dtype=int)
W_oracle = W_candidate.copy()
""",
            "call": "_pack(decode_x_sector(H_candidate, W_candidate, 0.5, 1))",
            "gold_call": "_pack(_oracle_decode_x_sector(H_oracle, W_oracle, 0.5, 1))",
        },
        {
            "setup": "import numpy as np\ndef _pack(result):\n    pieces = [np.asarray([len(result)], dtype=int)]\n    for M in result:\n        M = np.asarray(M, dtype=int)\n        pieces.append(np.asarray([M.ndim, *M.shape], dtype=int))\n        pieces.append(M.ravel())\n    return np.concatenate(pieces)\ndef _from_masks(values, width):\n    return np.asarray([[(m >> j) & 1 for j in range(width)] for m in values], dtype=int)\nH_candidate = _from_masks([192,3,289,52,384,14,28,72], 9)\nH_oracle = H_candidate.copy()\nW_candidate = _from_masks([192, 0, 0, 6, 0, 2, 6, 384], 9)\nW_oracle = W_candidate.copy()\n",
            "call": "_pack(decode_x_sector(H_candidate, W_candidate, 0.5, 2))",
            "gold_call": "_pack(_oracle_decode_x_sector(H_oracle, W_oracle, 0.5, 2))",
        },
    ]
