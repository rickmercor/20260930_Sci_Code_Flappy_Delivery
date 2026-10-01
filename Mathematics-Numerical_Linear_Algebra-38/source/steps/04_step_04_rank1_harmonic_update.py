"""
Apply the correction required by the reduced representation after a truncated Krylov cycle.

A truncated Krylov basis need not satisfy the orthogonality property required by the subsequent reduced matrix-function calculation. A low-rank correction can modify the final basis vector and the associated reduced representation while preserving the underlying decomposition.

Returns
-------
float, Frobenius norm of the reduced-matrix correction
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def rank1_harmonic_update(
    state: dict,
) -> float:
    """Apply the rank-1 correction to the current Krylov cycle.

    Parameters
    ----------
    state : dict
        Mutable pipeline state containing the current decomposition.

    Returns
    -------
    float
        Norm of the reduced-matrix correction.
    """
    result = 0.0
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_rank1_harmonic_update(state: dict) -> float:
    import numpy as np

    if not isinstance(state, dict):
        raise ValueError("state must be a dict")

    required = (
        "A",
        "S",
        "cycle_Bm",
        "cycle_Hm",
        "cycle_bmp1",
        "cycle_hmp1_m",
    )
    if any(key not in state for key in required):
        raise ValueError("state is missing rank-1 update data")

    A = np.asarray(state["A"], dtype=float)
    S = np.asarray(state["S"], dtype=float)
    Bm = np.asarray(state["cycle_Bm"], dtype=float)
    Hm = np.asarray(state["cycle_Hm"], dtype=float)
    bmp1 = np.asarray(state["cycle_bmp1"], dtype=float)
    hmp1_m = float(state["cycle_hmp1_m"])

    if not np.isclose(np.linalg.norm(bmp1), 1.0, atol=1e-10):
        raise ValueError("bmp1 must be unit norm")

    ABm = A @ Bm
    SABm = S @ ABm
    SBm = S @ Bm
    Sbmp1 = S @ bmp1

    G = SABm.T @ SBm
    rhs = SABm.T @ Sbmp1

    cm = np.linalg.solve(G, rhs)

    residual = bmp1 - Bm @ cm
    alpha = float(np.linalg.norm(residual))

    if alpha <= np.finfo(float).eps:
        raise ValueError("rank-1 correction is singular")

    ebmp1 = residual / alpha

    Hf = Hm.copy()
    Hf[:, -1] += cm * hmp1_m

    htilde = alpha * hmp1_m

    state["cm"] = cm
    state["alpha"] = alpha
    state["corrected_bmp1"] = ebmp1
    state["corrected_Hm"] = Hf
    state["corrected_hmp1_m"] = htilde

    return float(np.linalg.norm(Hf - Hm, ord="fro"))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
N = 8
m = 3
A = np.diag(np.arange(1.0, N + 1))
Bm = np.zeros((N, m))
Bm[0, 0] = 1.0
Bm[1, 1] = 1.0
Bm[2, 2] = 1.0
Hm = np.eye(m)
bmp1 = np.zeros(N)
bmp1[3] = 1.0
S = np.eye(N)
state = {
    "A": A,
    "S": S,
    "cycle_Bm": Bm,
    "cycle_Hm": Hm,
    "cycle_bmp1": bmp1,
    "cycle_hmp1_m": 0.5,
}
""",
            "call": "rank1_harmonic_update(state)",
            "gold_call": "_oracle_rank1_harmonic_update(state)",
        },
        {
            "setup": """import numpy as np
N = 4
A = np.eye(N)
Bm = np.array([[1.0], [0.0], [0.0], [0.0]])
Hm = np.array([[2.0]])
bmp1 = np.array([0.0, 1.0, 0.0, 0.0])
S = np.eye(N)
state = {
    "A": A,
    "S": S,
    "cycle_Bm": Bm,
    "cycle_Hm": Hm,
    "cycle_bmp1": bmp1,
    "cycle_hmp1_m": 1.0,
}
""",
            "call": "rank1_harmonic_update(state)",
            "gold_call": "_oracle_rank1_harmonic_update(state)",
        },
        {
            "setup": """import numpy as np
N = 5
A = np.diag(np.arange(1.0, 6.0))
Bm = np.zeros((N, 1))
Bm[0, 0] = 1.0
Hm = np.array([[3.0]])
bmp1 = np.zeros(N)
bmp1[4] = 1.0
S = np.eye(N)
state = {
    "A": A,
    "S": S,
    "cycle_Bm": Bm,
    "cycle_Hm": Hm,
    "cycle_bmp1": bmp1,
    "cycle_hmp1_m": 0.25,
}
""",
            "call": "rank1_harmonic_update(state)",
            "gold_call": "_oracle_rank1_harmonic_update(state)",
        },
    ]
