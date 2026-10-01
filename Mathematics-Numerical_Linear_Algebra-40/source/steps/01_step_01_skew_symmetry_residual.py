"""
Skew-block magnitude of a valid dissipative symmetric/skew splitting

A dissipative discretization of a stationary advection-diffusion-reaction operator on n interior nodes of the unit interval separates into a coercive self-adjoint part carrying diffusion and reaction, and a non-self-adjoint part carrying transport. Establish that separation for the second-order discretization implied by the configuration, verify that each part has the algebraic symmetry the dissipative theory requires, and report the size of the transport part in the Frobenius sense. A configuration that violates the required symmetry is not admissible and must be rejected rather than silently accepted. The reported quantity is a magnitude, not a validation residual, so it is generally far from zero.

Returns
-------
float, Frobenius magnitude of the skew block for a valid dissipative split
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def skew_symmetry_residual(
    n: int, nu: float, b_adv: float, c: float
) -> float:
    """Return ||S||_F for a valid dissipative FD splitting.

    

    Returns
    -------
    float
        Frobenius magnitude of the skew block.

    Raises
    ------
    ValueError
        If n < 2, nu <= 0, c < 0, or the split fails its symmetry checks.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_skew_symmetry_residual(
    n: int, nu: float, b_adv: float, c: float
) -> float:
    if n < 2:
        raise ValueError("n must be at least 2")
    if nu <= 0:
        raise ValueError("nu must be positive")
    if c < 0:
        raise ValueError("c must be nonnegative")
    h = 1.0 / (n + 1)
    main = -2.0 / h**2
    off = 1.0 / h**2
    lap = np.diag(main * np.ones(n)) + np.diag(off * np.ones(n - 1), 1)
    lap = lap + np.diag(off * np.ones(n - 1), -1)
    H = -nu * lap + c * np.eye(n)
    S = np.zeros((n, n))
    for i in range(n):
        if i > 0:
            S[i, i - 1] = -b_adv / (2.0 * h)
        if i < n - 1:
            S[i, i + 1] = b_adv / (2.0 * h)
    struct = float(
        np.linalg.norm(H - H.T, ord="fro") + np.linalg.norm(S + S.T, ord="fro")
    )
    if struct > 1e-10:
        raise ValueError("dissipative split fails symmetry structure checks")
    
    return float(np.linalg.norm(S, ord="fro"))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np",
            "call": "skew_symmetry_residual(8, 0.37, 1.85, 0.18)",
            "gold_call": "_oracle_skew_symmetry_residual(8, 0.37, 1.85, 0.18)",
        },
        {
            "setup": "import numpy as np",
            "call": "skew_symmetry_residual(2, 1e-3, 5.0, 0.0)",
            "gold_call": "_oracle_skew_symmetry_residual(2, 1e-3, 5.0, 0.0)",
        },
        {
            "setup": "import numpy as np",
            "call": "skew_symmetry_residual(9, 2.5, -1.75, 1.0)",
            "gold_call": "_oracle_skew_symmetry_residual(9, 2.5, -1.75, 1.0)",
        },
        {
            "setup": "import numpy as np",
            "call": "skew_symmetry_residual(12, 1e-4, 8.0, 1e-6)",
            "gold_call": "_oracle_skew_symmetry_residual(12, 1e-4, 8.0, 1e-6)",
        },
        {
            "setup": "import numpy as np",
            "call": "skew_symmetry_residual(3, 10.0, 0.0, 0.0)",
            "gold_call": "_oracle_skew_symmetry_residual(3, 10.0, 0.0, 0.0)",
        },
        {
            "setup": "import numpy as np",
            "call": "skew_symmetry_residual(5, 0.2, 3.25, 0.5)",
            "gold_call": "_oracle_skew_symmetry_residual(5, 0.2, 3.25, 0.5)",
        },
    ]
