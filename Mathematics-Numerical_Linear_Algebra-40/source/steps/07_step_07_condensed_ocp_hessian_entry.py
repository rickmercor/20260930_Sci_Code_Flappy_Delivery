"""
(0,0) entry of the condensed optimal-control Hessian for the split system.

Eliminating the state from a regularized PDE-constrained control problem with full observation and full control leaves a symmetric positive definite reduced operator acting on the control, assembled from the solution operator of the dissipative system together with the regularization term. Report its leading diagonal entry for the given configuration. Note that the reduced operator involves the solution operator composed with its adjoint, not the system operator composed with its transpose, and that the regularization enters additively; each of these confusions produces a plausible positive number

Returns
-------
float, (0, 0) entry of the condensed control Hessian
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def condensed_ocp_hessian_entry(
    n: int, nu: float, b_adv: float, c: float, mu: float
) -> float:
    """Return the (0, 0) entry of the condensed OCP Hessian.

    Returns
    -------
    float
        Hessian entry K[0, 0].

    Raises
    ------
    ValueError
        If mu <= 0 or the assembled split disagrees with the prior skew fingerprint.

    Assembled by calling the earlier sub-problem functions.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_condensed_ocp_hessian_entry(
    n: int, nu: float, b_adv: float, c: float, mu: float
) -> float:
    if mu <= 0:
        raise ValueError("mu must be positive")
    s_mag = _oracle_skew_symmetry_residual(n, nu, b_adv, c)
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
    A = H + S
    # Cross-step check: the split used here must carry the step-01 skew block.
    if abs(float(np.linalg.norm(A - H, ord="fro")) - s_mag) > 1e-9 * (1.0 + abs(s_mag)):
        raise ValueError("split operator disagrees with the preceding step")
    e0 = np.zeros(n)
    e0[0] = 1.0
    # Two-solve form e0^T A^{-T} A^{-1} e0 + mu  (paper Sec. 4.2 with B=C=I).
    z = np.linalg.solve(A, e0)
    w = np.linalg.solve(A.T, z)
    return float(e0 @ w + mu)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np",
            "call": "condensed_ocp_hessian_entry(8, 0.37, 1.85, 0.18, 0.07)",
            "gold_call": "_oracle_condensed_ocp_hessian_entry(8, 0.37, 1.85, 0.18, 0.07)",
        },
        {
            "setup": "import numpy as np",
            "call": "condensed_ocp_hessian_entry(4, 1.0, 0.5, 0.0, 0.25)",
            "gold_call": "_oracle_condensed_ocp_hessian_entry(4, 1.0, 0.5, 0.0, 0.25)",
        },
        {
            "setup": "import numpy as np",
            "call": "condensed_ocp_hessian_entry(7, 0.03, 2.2, 0.2, 1e-3)",
            "gold_call": "_oracle_condensed_ocp_hessian_entry(7, 0.03, 2.2, 0.2, 1e-3)",
        },
        {
            "setup": "import numpy as np",
            "call": "condensed_ocp_hessian_entry(9, 0.01, -3.0, 0.0, 1e-6)",
            "gold_call": "_oracle_condensed_ocp_hessian_entry(9, 0.01, -3.0, 0.0, 1e-6)",
        },
        {
            "setup": "import numpy as np",
            "call": "condensed_ocp_hessian_entry(2, 5.0, 0.0, 1.0, 10.0)",
            "gold_call": "_oracle_condensed_ocp_hessian_entry(2, 5.0, 0.0, 1.0, 10.0)",
        },
        {
            "setup": "import numpy as np",
            "call": "condensed_ocp_hessian_entry(5, 0.2, 1.5, 0.3, 0.05)",
            "gold_call": "_oracle_condensed_ocp_hessian_entry(5, 0.2, 1.5, 0.3, 0.05)",
        },
    ]
