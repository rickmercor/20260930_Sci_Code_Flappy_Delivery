"""
Evaluate, in closed form, the coefficient that multiplies a Heston variance in the exponent of the log-price characteristic function.

For a variance factor v with volatility of variance xi, the affine characteristic function of a log price x contains a term Y(delta, tau) v, where tau is the remaining time and delta the transform variable. Y solves the scalar Riccati equation dY/dtau = (1/2) xi^2 Y^2 + (c0 + i delta c1) Y - (1/2)(i delta + delta^2) with Y(delta, 0) = 0. The real coefficient c0 collects the mean reversion of the variance, together with any drift shift the pricing measure adds to it, and c1 comes from the correlation between the variance and the log price. Y must be returned from the exact solution, evaluated so that it stays bounded for every real delta and for delta shifted by -i.

Returns
-------
np.ndarray, float, shape (2, n): row 0 the real part and row 1 the imaginary part of Y at the n transform values.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def variance_riccati(delta: np.ndarray, tau: float, xi: float, c0: float, c1: float) -> np.ndarray:
    '''Closed-form variance coefficient of the affine characteristic function.

    Parameters
    ----------
    delta : np.ndarray
        One-dimensional array of n >= 1 transform values; real or complex
        (for example delta - 1j), all finite.
    tau : float
        Remaining time, tau >= 0.
    xi : float
        Volatility of the variance factor, xi > 0.
    c0 : float
        Real part of the linear coefficient of the Riccati equation.
    c1 : float
        Coefficient of i * delta in the linear coefficient.

    Returns
    -------
    y : np.ndarray
        Shape (2, n) float array: real and imaginary parts of Y(delta, tau).

    Raises
    ------
    ValueError
        If delta is not a finite one-dimensional array with at least one
        entry, if tau is negative, if xi is not positive, or if c0 or c1 is
        not finite.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs inside the function body.
    '''
    return np.zeros((2, np.atleast_1d(delta).size), dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================


def _oracle_variance_riccati(delta: np.ndarray, tau: float, xi: float, c0: float, c1: float) -> np.ndarray:
    import numpy as np

    d = np.atleast_1d(np.asarray(delta, dtype=complex))
    if d.ndim != 1 or d.size < 1 or not np.all(np.isfinite(d)):
        raise ValueError("delta must be a finite one-dimensional array")
    for name, value in (("tau", tau), ("xi", xi), ("c0", c0), ("c1", c1)):
        if isinstance(value, bool) or not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if float(tau) < 0.0:
        raise ValueError("tau must be non-negative")
    if float(xi) <= 0.0:
        raise ValueError("xi must be positive")
    t, x = float(tau), float(xi)

    q = 1j * d + d * d
    c = float(c0) + 1j * d * float(c1)
    # The solution is invariant under disc -> -disc; the principal root keeps
    # exp(-disc * tau) bounded.
    disc = np.sqrt(c * c + x * x * q)
    decay = np.exp(-disc * t)
    y = -q * (1.0 - decay) / (2.0 * disc - (c + disc) * (1.0 - decay))
    return np.vstack([y.real, y.imag]).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned: at delta = -i the forcing i delta + delta^2 vanishes, so
        # the coefficient is identically zero; at tau = 0 it is zero for any
        # delta.
        {
            "setup": """import numpy as np
EXPECTED = np.zeros((2, 3))
def at_zero(fn):
    a = fn(np.array([-1j, -1j, -1j]), 1.3, 0.4, -2.0, -0.2)
    b = fn(np.array([0.5, 3.0, 9.0]), 0.0, 0.4, -2.0, -0.2)
    return a + b
""",
            "call": "at_zero(variance_riccati)",
            "gold_call": "EXPECTED",
        },
        # --- Pinned: an independent fine RK4 integration of the Riccati
        # equation.
        {
            "setup": """import numpy as np
d = np.array([0.3, 1.7, 5.0, 12.0])
tau, xi, c0, c1 = 1.0, 0.1, -2.0, -0.05
def rk4(n=20000):
    q = 1j * d + d * d; c = c0 + 1j * d * c1
    f = lambda y: 0.5 * xi ** 2 * y ** 2 + c * y - 0.5 * q
    y = np.zeros(d.size, complex); h = tau / n
    for _ in range(n):
        k1 = f(y); k2 = f(y + h / 2 * k1); k3 = f(y + h / 2 * k2); k4 = f(y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return np.vstack([y.real, y.imag])
EXPECTED = rk4()
""",
            "call": "variance_riccati(d, tau, xi, c0, c1)",
            "gold_call": "EXPECTED",
        },
        # --- Normal: the second asset's coefficient under a shifted measure,
        # on a grid of real transform values.
        {
            "setup": """import numpy as np
d = np.linspace(0.05, 40.0, 50)
""",
            "call": "variance_riccati(d, 1.0, 0.1, -1.95, 0.05)",
            "gold_call": "_oracle_variance_riccati(d, 1.0, 0.1, -1.95, 0.05)",
        },
        # --- Boundary: transform values shifted by -i, as needed for the
        # share-measure probability.
        {
            "setup": """import numpy as np
d = np.linspace(0.05, 30.0, 20) - 1j
""",
            "call": "variance_riccati(d, 2.0, 0.6, -1.5, -0.3)",
            "gold_call": "_oracle_variance_riccati(d, 2.0, 0.6, -1.5, -0.3)",
        },
        # --- Edge: large volatility of variance and a long horizon.
        {
            "setup": """import numpy as np
d = np.array([0.01, 0.7, 25.0])
""",
            "call": "variance_riccati(d, 10.0, 1.5, -0.5, -1.2)",
            "gold_call": "_oracle_variance_riccati(d, 10.0, 1.5, -0.5, -1.2)",
        },
        # --- Invalid: non-positive volatility of variance ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        variance_riccati(np.array([1.0]), 1.0, 0.0, -2.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_variance_riccati(np.array([1.0]), 1.0, 0.0, -2.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: negative remaining time ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        variance_riccati(np.array([1.0]), -0.5, 0.1, -2.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_variance_riccati(np.array([1.0]), -0.5, 0.1, -2.0, 0.0)
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
