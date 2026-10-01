"""
Evaluate the coefficients through which the stochastic market-liquidity factor enters the characteristic function, and the part of the constant term they generate.

The liquidity factor alpha enters the variance of the log price through (beta1^2 + beta2^2) alpha^2, so the exponent of the affine characteristic function contains C(delta, tau) alpha^2 + B(delta, tau) alpha. Under the measure in which the characteristic function is taken, alpha has drift a b + m alpha, volatility eta, and instantaneous covariance with the log price of c_x eta alpha dt. C and B then solve dC/dtau = 2 eta^2 C^2 + 2 (m + i delta c_x eta) C - (1/2)(beta1^2 + beta2^2)(i delta + delta^2) and dB/dtau = (2 eta^2 C + m + i delta c_x eta) B + 2 a b C, both starting from 0. The constant term collects int_0^tau (a b B + (1/2) eta^2 B^2 + eta^2 C) ds. C, B and the integral of eta^2 C are evaluated from their exact solutions. The remaining integral of a b B + (1/2) eta^2 B^2 is computed with n_quad-point Gauss-Legendre quadrature on [0, tau] applied to the exact B.

Returns
-------
np.ndarray, float, shape (6, n): real and imaginary parts of C (rows 0, 1), of B (rows 2, 3) and of the constant-term contribution (rows 4, 5).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def liquidity_coefficients(delta: np.ndarray, tau: float, a: float, b: float, eta: float,
                           m: float, c_x: float, beta_sq: float, n_quad: int = 64) -> np.ndarray:
    '''Liquidity coefficients C, B and their constant-term contribution.

    Parameters
    ----------
    delta : np.ndarray
        One-dimensional array of n >= 1 finite transform values (real or complex).
    tau : float
        Remaining time, tau >= 0.
    a : float
        Mean-reversion rate a > 0 in the constant drift a b of alpha.
    b : float
        Long-run level b in the constant drift a b of alpha.
    eta : float
        Volatility of alpha, eta > 0.
    m : float
        Coefficient of alpha in its drift under the measure used.
    c_x : float
        Instantaneous covariance of alpha with the log price per unit
        eta * alpha.
    beta_sq : float
        beta1^2 + beta2^2 >= 0.
    n_quad : int
        Number of Gauss-Legendre nodes for the remaining integral, >= 2.

    Returns
    -------
    coeffs : np.ndarray
        Shape (6, n) float array; see the module description.

    Raises
    ------
    ValueError
        If any argument is outside its domain.

    Notes
    -----
    Include every import your implementation needs inside the function body.
    '''
    return np.zeros((6, np.atleast_1d(delta).size), dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================


def _oracle_liquidity_coefficients(delta: np.ndarray, tau: float, a: float, b: float, eta: float,
                                   m: float, c_x: float, beta_sq: float, n_quad: int = 64) -> np.ndarray:
    import numpy as np

    d = np.atleast_1d(np.asarray(delta, dtype=complex))
    if d.ndim != 1 or d.size < 1 or not np.all(np.isfinite(d)):
        raise ValueError("delta must be a finite one-dimensional array")
    for name, value in (("tau", tau), ("a", a), ("b", b), ("eta", eta), ("m", m),
                        ("c_x", c_x), ("beta_sq", beta_sq)):
        if isinstance(value, bool) or not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if float(tau) < 0.0 or float(a) <= 0.0 or float(eta) <= 0.0 or float(beta_sq) < 0.0:
        raise ValueError("need tau >= 0, a > 0, eta > 0 and beta_sq >= 0")
    if isinstance(n_quad, bool) or not isinstance(n_quad, (int, np.integer)) or int(n_quad) < 2:
        raise ValueError("n_quad must be an integer >= 2")
    t, a_, b_, e_ = float(tau), float(a), float(b), float(eta)

    q = 1j * d + d * d
    lin = float(m) + 1j * d * float(c_x) * e_
    disc = np.sqrt(lin * lin + e_ * e_ * float(beta_sq) * q)

    def c_coef(s):
        e2 = np.exp(-2.0 * disc * s)
        return -(disc ** 2 - lin ** 2) * (1.0 - e2) / (2.0 * e_ ** 2 * (2.0 * disc - (disc + lin) * (1.0 - e2)))

    def b_coef(s):
        e1 = np.exp(-disc * s)
        e2 = e1 * e1
        return (-a_ * b_ * (disc ** 2 - lin ** 2) / (disc * e_ ** 2)
                * (1.0 - e1) ** 2 / (2.0 * disc - (disc + lin) * (1.0 - e2)))

    C = c_coef(t)
    B = b_coef(t)
    e2 = np.exp(-2.0 * disc * t)
    int_eta2_c = 0.5 * (-(disc + lin) * t - np.log((2.0 * disc - (lin + disc) * (1.0 - e2)) / (2.0 * disc)))
    nodes, weights = np.polynomial.legendre.leggauss(int(n_quad))
    s = 0.5 * t * (nodes + 1.0)
    w = 0.5 * t * weights
    Bs = b_coef(s[:, None])
    rest = np.sum(w[:, None] * (a_ * b_ * Bs + 0.5 * e_ ** 2 * Bs ** 2), axis=0)
    const = int_eta2_c + rest
    return np.vstack([C.real, C.imag, B.real, B.imag, const.real, const.imag]).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned: an independent fine RK4 integration of the C, B and
        # constant-term equations at the task's liquidity parameters.
        {
            "setup": """import numpy as np
d = np.array([0.3, 1.7, 5.0, 12.0])
tau, a, b, eta, m, cx, bsq = 1.0, 0.2, 0.3, 0.9, -0.515, 0.0, 0.5
def rk4(n=20000):
    q = 1j * d + d * d; lin = m + 1j * d * cx * eta
    def f(y):
        C, B, K = y
        return np.array([2 * eta**2 * C**2 + 2 * lin * C - 0.5 * bsq * q,
                         (2 * eta**2 * C + lin) * B + 2 * a * b * C,
                         a * b * B + 0.5 * eta**2 * B**2 + eta**2 * C])
    y = np.zeros((3, d.size), complex); h = tau / n
    for _ in range(n):
        k1 = f(y); k2 = f(y + h/2*k1); k3 = f(y + h/2*k2); k4 = f(y + h*k3)
        y = y + h/6*(k1 + 2*k2 + 2*k3 + k4)
    return np.vstack([y[0].real, y[0].imag, y[1].real, y[1].imag, y[2].real, y[2].imag])
EXPECTED = rk4()
""",
            "call": "liquidity_coefficients(d, tau, a, b, eta, m, cx, bsq, 64)",
            "gold_call": "EXPECTED",
        },
        # --- Pinned: at delta = -i all three vanish.
        {
            "setup": """import numpy as np
EXPECTED = np.zeros((6, 2))
""",
            "call": "liquidity_coefficients(np.array([-1j, -1j]), 1.5, 0.3, 0.2, 0.8, -0.4, 0.1, 0.5, 32)",
            "gold_call": "EXPECTED",
        },
        # --- Normal: nonzero covariance with the log price, real grid.
        {
            "setup": """import numpy as np
d = np.linspace(0.05, 40.0, 40)
""",
            "call": "liquidity_coefficients(d, 1.0, 0.2, 0.3, 0.9, -0.515, 0.2, 0.5, 64)",
            "gold_call": "_oracle_liquidity_coefficients(d, 1.0, 0.2, 0.3, 0.9, -0.515, 0.2, 0.5, 64)",
        },
        # --- Boundary: shifted transform values and a coarse quadrature.
        {
            "setup": """import numpy as np
d = np.linspace(0.05, 20.0, 15) - 1j
""",
            "call": "liquidity_coefficients(d, 0.5, 1.0, 0.1, 0.4, -1.2, -0.3, 0.2, 8)",
            "gold_call": "_oracle_liquidity_coefficients(d, 0.5, 1.0, 0.1, 0.4, -1.2, -0.3, 0.2, 8)",
        },
        # --- Edge: no liquidity loading, so C and B vanish while a b B is zero.
        {
            "setup": """import numpy as np
d = np.array([0.5, 2.0])
""",
            "call": "liquidity_coefficients(d, 2.0, 0.2, 0.3, 0.9, -0.2, 0.0, 0.0, 16)",
            "gold_call": "_oracle_liquidity_coefficients(d, 2.0, 0.2, 0.3, 0.9, -0.2, 0.0, 0.0, 16)",
        },
        # --- Invalid: non-positive eta ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        liquidity_coefficients(np.array([1.0]), 1.0, 0.2, 0.3, 0.0, -0.5, 0.0, 0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_liquidity_coefficients(np.array([1.0]), 1.0, 0.2, 0.3, 0.0, -0.5, 0.0, 0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: too few quadrature nodes ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        liquidity_coefficients(np.array([1.0]), 1.0, 0.2, 0.3, 0.9, -0.5, 0.0, 0.5, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_liquidity_coefficients(np.array([1.0]), 1.0, 0.2, 0.3, 0.9, -0.5, 0.0, 0.5, 1)
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
