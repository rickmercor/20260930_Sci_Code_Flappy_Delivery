"""
Evaluate the reduced inverse-square-root approximation associated with the corrected cycle.

Krylov-based matrix-function methods replace a large matrix-function evaluation with a corresponding computation on a much smaller reduced matrix. The resulting reduced quantity is then mapped back to the original vector space through the generated basis.

Returns
-------
float, Euclidean norm of f^[0]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def arnoldi_like_approx(
    state: dict,
) -> float:
    """Evaluate the reduced matrix-function approximation.

    Parameters
    ----------
    state : dict
        Mutable pipeline state containing the corrected reduced matrix.

    Returns
    -------
    float
        Euclidean norm of the approximation.
    """
    result = 0.0
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_arnoldi_like_approx(state: dict) -> float:
    import numpy as np

    if not isinstance(state, dict):
        raise ValueError("state must be a dict")

    required = ("cycle_Bm", "corrected_Hm", "beta")
    if any(key not in state for key in required):
        raise ValueError("state is missing reduced approximation data")

    Bm = np.asarray(state["cycle_Bm"], dtype=float)
    Hm = np.asarray(state["corrected_Hm"])

    beta = float(state["beta"])
    m = Bm.shape[1]

    if Hm.shape != (m, m):
        raise ValueError("invalid reduced matrix")

    eigvals, V = np.linalg.eig(Hm)
    if np.any(np.real(eigvals) <= 0.0):
        raise ValueError("principal inverse square root requires positive-real spectrum")

    Vinv = np.linalg.inv(V)
    fHm = V @ np.diag(eigvals ** (-0.5)) @ Vinv

    e1 = np.zeros(m, dtype=complex)
    e1[0] = 1.0

    f0 = Bm @ (fHm @ (beta * e1))

    state["f0"] = np.real_if_close(f0)

    return float(np.linalg.norm(f0))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
state = {
    "cycle_Bm": np.eye(3),
    "corrected_Hm": np.diag([4.0, 9.0, 16.0]),
    "beta": 2.0,
}
""",
            "call": "arnoldi_like_approx(state)",
            "gold_call": "_oracle_arnoldi_like_approx(state)",
        },
        {
            "setup": """import numpy as np
state = {
    "cycle_Bm": np.array([[1.0], [0.0], [0.0]]),
    "corrected_Hm": np.array([[25.0]]),
    "beta": 1.0,
}
""",
            "call": "arnoldi_like_approx(state)",
            "gold_call": "_oracle_arnoldi_like_approx(state)",
        },
        {
            "setup": """import numpy as np
state = {
    "cycle_Bm": np.eye(2),
    "corrected_Hm": np.array([[2.0, -1.0], [1.0, 2.0]]),
    "beta": 1.5,
}
""",
            "call": "arnoldi_like_approx(state)",
            "gold_call": "_oracle_arnoldi_like_approx(state)",
        },
    ]
