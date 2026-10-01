"""
Implement the transposed Pauli-$Z$ half of the decoder.

The $Z$ syndrome lives in $A\times B$, so row and column roles swap relative to the $X$ sector. The same peeling-selected inverses reappear transposed and on opposite sides of the matrix products.

Returns
-------
tuple[np.ndarray, np.ndarray], uint8 corrections Z_A of shape A×A and Z_B of shape B×B
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def decode_z_sector(
    H: "np.ndarray",
    U: "np.ndarray",
    beta: float,
    rounds: int,
) -> tuple["np.ndarray", "np.ndarray"]:
    """Decode the Z sector and return the A-by-A and B-by-B corrections."""
    return U, U

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_decode_z_sector(
    H: np.ndarray,
    U: np.ndarray,
    beta: float,
    rounds: int,
) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np

    H = np.asarray(H)
    U = np.asarray(U)

    if H.ndim != 2 or H.size == 0:
        raise ValueError("H must be a nonempty 2D matrix")
    if U.shape != (H.shape[1], H.shape[0]):
        raise ValueError("U must have shape (A, B)")
    if not np.all((H == 0) | (H == 1)):
        raise ValueError("H must be binary")
    if not np.all((U == 0) | (U == 1)):
        raise ValueError("U must be binary")

    H = H.astype(np.uint8)
    U = U.astype(np.uint8)

    row_support, col_support = _oracle_matrix_supports(U)

    S, _ = _oracle_grow_beta_region(
        H, col_support, "B", beta, rounds
    )
    _, T = _oracle_grow_beta_region(
        H, row_support, "A", beta, rounds
    )

    _, Bp = _oracle_deterministic_peeling(H, S, "A")
    _, Ap = _oracle_deterministic_peeling(H, T, "B")

    left_inverse = _oracle_embedded_inverse(H, S, Bp)
    Z_A = (U @ left_inverse.T) % 2

    remaining = (U + Z_A @ H.T) % 2

    right_inverse = _oracle_embedded_inverse(H, Ap, T)
    Z_B = (right_inverse.T @ remaining) % 2

    return Z_A.astype(np.uint8), Z_B.astype(np.uint8)

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
U_candidate = np.array([[0,1],[0,0]], dtype=int)
U_oracle = U_candidate.copy()
""",
            "call": "_pack(decode_z_sector(H_candidate, U_candidate, 0.5, 1))",
            "gold_call": "_pack(_oracle_decode_z_sector(H_oracle, U_oracle, 0.5, 1))",
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
U_candidate = np.zeros((2,2), dtype=int)
U_oracle = U_candidate.copy()
""",
            "call": "_pack(decode_z_sector(H_candidate, U_candidate, 0.5, 1))",
            "gold_call": "_pack(_oracle_decode_z_sector(H_oracle, U_oracle, 0.5, 1))",
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
U_candidate = np.array([[1,0],[0,0],[0,0]], dtype=int)
U_oracle = U_candidate.copy()
""",
            "call": "_pack(decode_z_sector(H_candidate, U_candidate, 0.5, 1))",
            "gold_call": "_pack(_oracle_decode_z_sector(H_oracle, U_oracle, 0.5, 1))",
        },
        {
            "setup": "import numpy as np\ndef _pack(result):\n    pieces = [np.asarray([len(result)], dtype=int)]\n    for M in result:\n        M = np.asarray(M, dtype=int)\n        pieces.append(np.asarray([M.ndim, *M.shape], dtype=int))\n        pieces.append(M.ravel())\n    return np.concatenate(pieces)\ndef _from_masks(values, width):\n    return np.asarray([[(m >> j) & 1 for j in range(width)] for m in values], dtype=int)\nH_candidate = _from_masks([192,3,289,52,384,14,28,72], 9)\nH_oracle = H_candidate.copy()\nU_candidate = _from_masks([0, 0, 36, 1, 36, 36, 1, 129, 0], 8)\nU_oracle = U_candidate.copy()\n",
            "call": "_pack(decode_z_sector(H_candidate, U_candidate, 0.5, 2))",
            "gold_call": "_pack(_oracle_decode_z_sector(H_oracle, U_oracle, 0.5, 2))",
        },
    ]
