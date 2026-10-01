"""
Assemble the low-rank correction matrix required by the subsequent randomized preconditioning subproblem from the previously computed basis and core.

The randomized construction uses the computed basis and core to form the final matrix state for the approximation. For this benchmark, the correction is assembled deterministically from the outputs of the preceding subproblems.

Returns
-------
np.ndarray, shape $(m,m)$ — symmetric correction matrix $\widehat{\Delta} = VHV^T$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def assemble_low_rank_correction(
    V: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    r"""
    Assemble the correction matrix required by the subsequent
    randomized preconditioning subproblems.

    Raises
    ------
    ValueError
        If $V$ is not a 2D array, if $H$ is not a square 2D array, or if
        the column dimension of $V$ does not match the dimension of $H$.
    """
    return Delta_hat

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_assemble_low_rank_correction(
    V: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    V = np.asarray(V, dtype=float)
    H = np.asarray(H, dtype=float)

    if V.ndim != 2:
        raise ValueError("V must be 2D")

    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("H must be square")

    if V.shape[1] != H.shape[0]:
        raise ValueError("V and H have incompatible dimensions")

    Delta_hat = V @ H @ V.T

    Delta_hat = 0.5 * (Delta_hat + Delta_hat.T)

    return Delta_hat.astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: ordinary case ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(51)
m, k = 12, 4
W = rng.standard_normal((m, k))
V, R = np.linalg.qr(W, mode="reduced")
S = rng.standard_normal((k, k))
H = S + S.T
""",
            "call": "assemble_low_rank_correction(V, H)",
            "gold_call": "_oracle_assemble_low_rank_correction(V, H)",
        },

        # --- Valid: zero core ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(52)
m, k = 9, 3
W = rng.standard_normal((m, k))
V, R = np.linalg.qr(W, mode="reduced")
H = np.zeros((k, k))
""",
            "call": "assemble_low_rank_correction(V, H)",
            "gold_call": "_oracle_assemble_low_rank_correction(V, H)",
        },

        # --- Valid: rank-1 correction ---
        {
            "setup": """import numpy as np

m = 8
v = np.random.default_rng(53).standard_normal((m, 1))
v = v / np.linalg.norm(v)

V = v
H = np.array([[3.7]])
""",
            "call": "assemble_low_rank_correction(V, H)",
            "gold_call": "_oracle_assemble_low_rank_correction(V, H)",
        },

        # --- Invalid: dimension mismatch ---
        {
            "setup": """import numpy as np
V = np.random.default_rng(54).standard_normal((10, 4))
H = np.random.default_rng(55).standard_normal((3, 3))

def run_model():
    try:
        assemble_low_rank_correction(V, H)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_assemble_low_rank_correction(V, H)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
