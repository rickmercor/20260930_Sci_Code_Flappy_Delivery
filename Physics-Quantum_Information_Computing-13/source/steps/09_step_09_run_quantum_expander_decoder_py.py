"""
Run the two previously implemented sector decoders and reduce their four corrections to the single benchmark scalar.

The physical hypergraph-product blocklength is $|A|^2+|B|^2$. The requested statistic averages the total $X$- and $Z$-correction Hamming weight over both Pauli sectors. This step must compose decode_x_sector and decode_z_sector; it must not reimplement their internal pipeline.

Returns
-------
float, (wt(X_A)+wt(X_B)+wt(Z_A)+wt(Z_B)) / (2*(|A|^2+|B|^2)) as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_quantum_expander_decoder(
    H: "np.ndarray",
    W: "np.ndarray",
    U: "np.ndarray",
    beta: float,
    rounds: int,
) -> float:
    """Run both Pauli-sector decoders and return the total correction density."""
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_run_quantum_expander_decoder(
    H: np.ndarray,
    W: np.ndarray,
    U: np.ndarray,
    beta: float,
    rounds: int,
) -> float:
    import numpy as np

    H = np.asarray(H)

    X_A, X_B = _oracle_decode_x_sector(H, W, beta, rounds)
    Z_A, Z_B = _oracle_decode_z_sector(H, U, beta, rounds)

    total_weight = (
        int(np.sum(X_A))
        + int(np.sum(X_B))
        + int(np.sum(Z_A))
        + int(np.sum(Z_B))
    )

    blocklength = H.shape[1] ** 2 + H.shape[0] ** 2
    return float(total_weight / (2 * blocklength))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
def _from_masks(values, width):
    return np.asarray([[(m >> j) & 1 for j in range(width)] for m in values], dtype=int)
H_candidate = _from_masks([192,3,289,52,384,14,28,72], 9)
H_oracle = H_candidate.copy()
W_candidate = _from_masks([192,0,0,6,0,2,6,384], 9)
W_oracle = W_candidate.copy()
U_candidate = _from_masks([0,0,36,1,36,36,1,129,0], 8)
U_oracle = U_candidate.copy()
""",
            "call": "run_quantum_expander_decoder(H_candidate, W_candidate, U_candidate, 0.5, 2)",
            "gold_call": "_oracle_run_quantum_expander_decoder(H_oracle, W_oracle, U_oracle, 0.5, 2)",
        },
        {
            "setup": """import numpy as np
H_candidate = np.eye(2, dtype=int)
H_oracle = H_candidate.copy()
W_candidate = np.array([[1,0],[0,0]], dtype=int)
W_oracle = W_candidate.copy()
U_candidate = np.array([[0,1],[0,0]], dtype=int)
U_oracle = U_candidate.copy()
""",
            "call": "run_quantum_expander_decoder(H_candidate, W_candidate, U_candidate, 0.5, 1)",
            "gold_call": "_oracle_run_quantum_expander_decoder(H_oracle, W_oracle, U_oracle, 0.5, 1)",
        },
        {
            "setup": """import numpy as np
H_candidate = np.array([[1,1,0],[0,1,1]], dtype=int)
H_oracle = H_candidate.copy()
W_candidate = np.array([[1,0,0],[0,0,0]], dtype=int)
W_oracle = W_candidate.copy()
U_candidate = np.array([[1,0],[0,0],[0,0]], dtype=int)
U_oracle = U_candidate.copy()
""",
            "call": "run_quantum_expander_decoder(H_candidate, W_candidate, U_candidate, 0.5, 1)",
            "gold_call": "_oracle_run_quantum_expander_decoder(H_oracle, W_oracle, U_oracle, 0.5, 1)",
        },
    ]
