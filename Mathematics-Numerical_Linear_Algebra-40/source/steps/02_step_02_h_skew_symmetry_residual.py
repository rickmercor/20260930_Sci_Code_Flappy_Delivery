"""
Congruence magnitude of the H-skew preconditioned advection block

The transport part is not skew in the Euclidean sense once it is preconditioned by the coercive part, yet the dissipative theory still applies because the relevant adjoint is taken in the energy geometry induced by that coercive part. Move the transport part into coordinates in which that energy geometry becomes Euclidean, certify that the resulting operator is genuinely skew there, and report the Frobenius size of that transformed operator. Reject the configuration if the certification fails. The reported quantity is again a magnitude rather than a validation residual, and it is invariant to how the change of coordinates is realized. Depends on $skew_symmetry_residual$.

Returns
-------
float, Frobenius magnitude of the H-congruence of the skew block
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def h_skew_symmetry_residual(
    n: int, nu: float, b_adv: float, c: float
) -> float:
    """Return ||K||_F, where K = L^{-1} S L^{-T} and H = L L^T.

   

    

    Returns
    -------
    float
        ||K||_F, generally far from zero when b_adv is nonzero.

    Raises
    ------
    ValueError
        If the prior split is invalid or the H-congruence is not skew.

    Assembled by calling the earlier sub-problem functions.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_h_skew_symmetry_residual(
    n: int, nu: float, b_adv: float, c: float
) -> float:
    # Structural validation via prior step (raises if invalid).
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
    # Cross-step check: the skew block must match the step-01 fingerprint.
    if abs(float(np.linalg.norm(S, ord="fro")) - s_mag) > 1e-9 * (1.0 + abs(s_mag)):
        raise ValueError("skew block disagrees with the preceding step")
    L = np.linalg.cholesky(H)
    Y = np.linalg.solve(L, S)
    K = np.linalg.solve(L, Y.T).T
    skew_res = float(np.linalg.norm(K + K.T, ord="fro"))
    if skew_res > 1e-8:
        raise ValueError("H-congruence is not skew within tolerance")
    # Nonzero fingerprint: stubs returning 0.0 fail whenever b_adv != 0.
    return float(np.linalg.norm(K, ord="fro"))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np",
            "call": "h_skew_symmetry_residual(8, 0.37, 1.85, 0.18)",
            "gold_call": "_oracle_h_skew_symmetry_residual(8, 0.37, 1.85, 0.18)",
        },
        {
            "setup": "import numpy as np",
            "call": "h_skew_symmetry_residual(3, 1.0, 0.0, 0.5)",
            "gold_call": "_oracle_h_skew_symmetry_residual(3, 1.0, 0.0, 0.5)",
        },
        {
            "setup": "import numpy as np",
            "call": "h_skew_symmetry_residual(8, 0.05, 3.0, 0.01)",
            "gold_call": "_oracle_h_skew_symmetry_residual(8, 0.05, 3.0, 0.01)",
        },
        {
            "setup": "import numpy as np",
            "call": "h_skew_symmetry_residual(10, 1e-3, -4.5, 0.0)",
            "gold_call": "_oracle_h_skew_symmetry_residual(10, 1e-3, -4.5, 0.0)",
        },
        {
            "setup": "import numpy as np",
            "call": "h_skew_symmetry_residual(2, 0.25, 7.0, 2.0)",
            "gold_call": "_oracle_h_skew_symmetry_residual(2, 0.25, 7.0, 2.0)",
        },
        {
            "setup": "import numpy as np",
            "call": "h_skew_symmetry_residual(7, 0.4, 2.0, 0.15)",
            "gold_call": "_oracle_h_skew_symmetry_residual(7, 0.4, 2.0, 0.15)",
        },
    ]
