"""
Spectral width of the preconditioned skew operator.

Because the preconditioned transport operator is skew in the energy geometry, its spectrum is confined to a symmetric segment of the imaginary axis. Every convergence estimate for the short-recurrence methods used later in this pipeline is expressed through the half-length of that segment. Determine that half-length for the given configuration.

Returns
-------
float, spectral width lambda >= 0
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def spectral_width_lambda(n: int, nu: float, b_adv: float, c: float) -> float:
    """Return the spectral width of the preconditioned skew operator.

    
    Returns
    -------
    float
        Spectral width lambda.

    Raises
    ------
    ValueError
        If the spectral width exceeds the congruence magnitude from the prior step.

    Assembled by calling the earlier sub-problem functions.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_spectral_width_lambda(
    n: int, nu: float, b_adv: float, c: float
) -> float:
    k_mag = _oracle_h_skew_symmetry_residual(n, nu, b_adv, c)
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
    # Generalized pencil S v = mu H v  <=>  eigenvalues of H^{-1}S.
    eigvals = np.linalg.eigvals(np.linalg.solve(H, S))
    imag_parts = []
    for mu in eigvals:
        if abs(np.real(mu)) <= 1e-8 * (1.0 + abs(mu)):
            imag_parts.append(abs(np.imag(mu)))
    if not imag_parts:
        return 0.0
    lam = float(max(imag_parts))
    # Cross-step check: H^{-1}S and K are similar, so their spectra coincide and
    # the spectral width cannot exceed the step-02 congruence magnitude.
    if lam > k_mag + 1e-8 * (1.0 + abs(k_mag)):
        raise ValueError("spectral width exceeds the congruence magnitude")
    return lam

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np",
            "call": "spectral_width_lambda(8, 0.37, 1.85, 0.18)",
            "gold_call": "_oracle_spectral_width_lambda(8, 0.37, 1.85, 0.18)",
        },
        {
            "setup": "import numpy as np",
            "call": "spectral_width_lambda(4, 1.0, 0.0, 0.25)",
            "gold_call": "_oracle_spectral_width_lambda(4, 1.0, 0.0, 0.25)",
        },
        {
            "setup": "import numpy as np",
            "call": "spectral_width_lambda(7, 0.02, 2.5, 0.05)",
            "gold_call": "_oracle_spectral_width_lambda(7, 0.02, 2.5, 0.05)",
        },
        {
            "setup": "import numpy as np",
            "call": "spectral_width_lambda(11, 5e-3, 6.0, 1e-4)",
            "gold_call": "_oracle_spectral_width_lambda(11, 5e-3, 6.0, 1e-4)",
        },
        {
            "setup": "import numpy as np",
            "call": "spectral_width_lambda(5, 0.75, -2.0, 0.0)",
            "gold_call": "_oracle_spectral_width_lambda(5, 0.75, -2.0, 0.0)",
        },
        {
            "setup": "import numpy as np",
            "call": "spectral_width_lambda(8, 0.01, 4.5, 0.0)",
            "gold_call": "_oracle_spectral_width_lambda(8, 0.01, 4.5, 0.0)",
        },
    ]
