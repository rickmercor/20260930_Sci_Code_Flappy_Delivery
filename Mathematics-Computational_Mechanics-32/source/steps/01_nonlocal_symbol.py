"""
Return the eigenvalue of the nonlocal operator L_delta of the source, Eq (2.1), acting on the plane wave exp(i xi x), for the radial kernel family of Eq (5.1) with exponent alpha and horizon delta. The kernel is non-integrable for alpha >= 1, so evaluate the defining integral with a quadrature that resolves the algebraic behaviour of the integrand at s = 0 to at least ten significant digits for every 0 < alpha < 3. Raise ValueError if delta is not positive or alpha is not in (0, 3).

A periodic nonlocal operator with a radial kernel is diagonal in the Fourier basis: every plane wave is an eigenfunction, and the eigenvalue plays the role that xi^2 plays for -d^2/dx^2. It fixes the exact dynamics of every Fourier mode of the nonlocal wave equation and reduces to xi^2 as the horizon shrinks.

Returns
-------
float, the eigenvalue of L_delta on exp(i xi x).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def nonlocal_symbol(xi, alpha, delta):
    """float, the eigenvalue of L_delta on exp(i xi x)."""
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _gauss_jacobi(n, a, b):
    """Golub-Welsch nodes/weights for the weight (1-x)^a (1+x)^b on [-1, 1], numpy only."""
    import math
    import numpy as np
    n = int(n)
    if n < 1 or a <= -1.0 or b <= -1.0:
        raise ValueError("need n >= 1 and a, b > -1")
    k = np.arange(n, dtype=np.float64)
    ab = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        diag = (b * b - a * a) / ((2.0 * k + ab) * (2.0 * k + ab + 2.0))
    diag[0] = (b - a) / (ab + 2.0)
    kk = np.arange(1, n, dtype=np.float64)
    num = 4.0 * kk * (kk + a) * (kk + b) * (kk + ab)
    den = (2.0 * kk + ab) ** 2 * (2.0 * kk + ab + 1.0) * (2.0 * kk + ab - 1.0)
    off2 = num / den
    if n > 1 and abs(ab) < 1e-300:            # k = 1 with a + b = 0 is the 0/0 case
        off2[0] = 4.0 * (1.0 + a) * (1.0 + b) / 12.0
    J = np.diag(diag)
    if n > 1:
        off = np.sqrt(off2)
        J += np.diag(off, 1) + np.diag(off, -1)
    x, V = np.linalg.eigh(J)
    mu0 = 2.0 ** (ab + 1.0) * math.gamma(a + 1.0) * math.gamma(b + 1.0) / math.gamma(ab + 2.0)
    w = mu0 * V[0, :] ** 2
    return x, w


def _oracle_nonlocal_symbol(xi, alpha, delta):
    """Eigenvalue of L_delta on the plane wave e^{i xi x} for the kernel (5.1).

    lambda_d(xi) = 2 int_{-d}^{d} gamma_d(s) (1 - cos(xi s)) ds
                 = 4 c int_0^d s^(2-a) [2 sin^2(xi s/2) / s^2] ds,   c = (3-a)/(2 d^(3-a)),
    computed with a 200-point Gauss-Jacobi rule for the weight s^(2-a) (exact to round-off
    for the smooth bracket), which handles 0 < alpha < 3 uniformly.  lambda_d(xi) -> xi^2 as
    delta -> 0 because int_{-d}^{d} s^2 gamma_d = 1.
    """
    import numpy as np
    xi = float(xi)
    alpha = float(alpha)
    delta = float(delta)
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    if not (0.0 < alpha < 3.0):
        raise ValueError("alpha must lie in (0, 3)")
    beta = 2.0 - alpha
    c = (3.0 - alpha) / (2.0 * delta ** (3.0 - alpha))
    x, w = _gauss_jacobi(200, 0.0, beta)
    s = 0.5 * delta * (x + 1.0)
    ws = c * w * (0.5 * delta) ** (1.0 + beta)          # includes c s^(2-a) ds
    bracket = 2.0 * np.sin(0.5 * xi * s) ** 2 / s ** 2   # (1 - cos(xi s)) / s^2, smooth
    return float(4.0 * np.sum(ws * bracket))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "xi, alpha, delta = 2.0 * np.pi, 1.5, 0.25",
            "call": "nonlocal_symbol(xi, alpha, delta)",
            "gold_call": "_oracle_nonlocal_symbol(xi, alpha, delta)",
        },
        {
            "setup": "xi, alpha, delta = 4.0 * np.pi, 0.5, 0.125",
            "call": "nonlocal_symbol(xi, alpha, delta)",
            "gold_call": "_oracle_nonlocal_symbol(xi, alpha, delta)",
        },
        {
            "setup": "xi, alpha, delta = 2.0 * np.pi, 2.5, 0.001",
            "call": "nonlocal_symbol(xi, alpha, delta)",
            "gold_call": "_oracle_nonlocal_symbol(xi, alpha, delta)",
        },
        {
            "setup": "xi, alpha, delta = 0.0, 0.75, 0.375",
            "call": "nonlocal_symbol(xi, alpha, delta)",
            "gold_call": "_oracle_nonlocal_symbol(xi, alpha, delta)",
        },
    ]
