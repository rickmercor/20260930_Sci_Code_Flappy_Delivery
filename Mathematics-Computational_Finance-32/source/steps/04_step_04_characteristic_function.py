"""
Assemble the characteristic function of the log price ratio ln(S1(T)/S2(T)) under the measure that takes the second asset as numeraire, at the initial state and regime.

Taking S2 as numeraire turns the exchange option into a call with unit strike on Z = S1/S2, and removes the interest rate. Under the numeraire measure every Brownian motion correlated with S2 acquires a drift, so the second variance factor and the liquidity factor change their drifts. The characteristic function is exp(i delta x + K + B alpha + C alpha^2 + D nu1 + E nu2) times the regime factor, with x = ln(S1/S2), K the liquidity part of the constant term, and D, E, C, B the coefficients of steps 01-03. The model is supplied as a dictionary whose keys are listed in the signature. Regime-dependent entries have one value per regime, and the chain generator has row i holding the rates out of regime i.

Returns
-------
np.ndarray, float, shape (2, n): real and imaginary parts of the characteristic function at the n transform values.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def characteristic_function(delta: np.ndarray, tau: float, model: dict, n_quad: int = 64,
                            n_steps: int = 200) -> np.ndarray:
    '''Characteristic function of ln(S1(T)/S2(T)) under the S2-numeraire measure.

    Parameters
    ----------
    delta : np.ndarray
        One-dimensional array of n >= 1 finite transform values (real or complex).
    tau : float
        Time to maturity, tau >= 0.
    model : dict
        Keys: s1, s2, nu1, nu2, alpha (initial values, s1, s2 > 0, nu1, nu2 >= 0);
        kappa1, kappa2 > 0; xi1, xi2 > 0; rho1, rho2 (each asset with its own
        variance); rho (between the two diffusive Brownian motions);
        rho_tilde1, rho_tilde2 (each asset's liquidity noise with the market
        liquidity noise); beta1, beta2 >= 0; a > 0, b, eta > 0 (market
        liquidity under the pricing measure); sigma1, sigma2, theta1, theta2
        (sequences, one value per regime); generator (k x k); initial_regime
        (1 to k).
    n_quad : int
        Gauss-Legendre nodes for the liquidity constant term.
    n_steps : int
        Exponential midpoint steps for the regime factor.

    Returns
    -------
    phi : np.ndarray
        Shape (2, n) float array: real and imaginary parts.

    Raises
    ------
    ValueError
        If a key is missing or any entry is outside its domain.

    Notes
    -----
    Use ``variance_riccati``, ``liquidity_coefficients`` and
    ``regime_factor``. Include every import your implementation needs inside
    the function body.
    '''
    return np.zeros((2, np.atleast_1d(delta).size), dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================


def _oracle_characteristic_function(delta: np.ndarray, tau: float, model: dict, n_quad: int = 64,
                                    n_steps: int = 200) -> np.ndarray:
    import numpy as np

    keys = ("s1", "s2", "nu1", "nu2", "alpha", "kappa1", "kappa2", "xi1", "xi2", "rho1", "rho2",
            "rho", "rho_tilde1", "rho_tilde2", "beta1", "beta2", "a", "b", "eta",
            "sigma1", "sigma2", "theta1", "theta2", "generator", "initial_regime")
    if not isinstance(model, dict) or any(k not in model for k in keys):
        raise ValueError("model must be a dict with keys " + ", ".join(keys))
    M = model
    for name in ("s1", "s2", "kappa1", "kappa2", "xi1", "xi2", "a", "eta"):
        if not np.isfinite(float(M[name])) or float(M[name]) <= 0.0:
            raise ValueError(f"{name} must be positive")
    for name in ("nu1", "nu2", "beta1", "beta2"):
        if not np.isfinite(float(M[name])) or float(M[name]) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    for name in ("rho", "rho1", "rho2", "rho_tilde1", "rho_tilde2"):
        if not (-1.0 <= float(M[name]) <= 1.0):
            raise ValueError(f"{name} must lie in [-1, 1]")
    d = np.atleast_1d(np.asarray(delta, dtype=complex))

    sig1 = np.asarray(M["sigma1"], dtype=float)
    sig2 = np.asarray(M["sigma2"], dtype=float)
    rho = float(M["rho"])
    # Total constant variance rate of ln(S1/S2) in each regime.
    sig2_ratio = sig1 ** 2 + sig2 ** 2 - 2.0 * rho * sig1 * sig2
    k1, k2 = float(M["kappa1"]), float(M["kappa2"])
    x1, x2 = float(M["xi1"]), float(M["xi2"])
    r1, r2 = float(M["rho1"]), float(M["rho2"])
    eta = float(M["eta"])
    b1, b2 = float(M["beta1"]), float(M["beta2"])
    rt1, rt2 = float(M["rho_tilde1"]), float(M["rho_tilde2"])

    # Variance of asset 1: unchanged drift, covariance +rho1 xi1 nu1 with x.
    d_c0, d_c1 = -k1, r1 * x1
    # Variance of asset 2: the numeraire adds rho2 xi2 nu2 to its drift and
    # its covariance with x is -rho2 xi2 nu2.
    e_c0, e_c1 = r2 * x2 - k2, -r2 * x2
    # Liquidity: the numeraire adds rho_tilde2 beta2 eta alpha to the drift;
    # covariance with x is (rho_tilde1 beta1 - rho_tilde2 beta2) eta alpha.
    m_alpha = rt2 * b2 * eta - float(M["a"])
    c_x = rt1 * b1 - rt2 * b2

    Dr = _oracle_variance_riccati(d, tau, x1, d_c0, d_c1)
    Er = _oracle_variance_riccati(d, tau, x2, e_c0, e_c1)
    Lr = _oracle_liquidity_coefficients(d, tau, float(M["a"]), float(M["b"]), eta, m_alpha, c_x,
                                        b1 * b1 + b2 * b2, n_quad)
    Hr = _oracle_regime_factor(d, tau, np.asarray(M["generator"], dtype=float), sig2_ratio,
                               k1, np.asarray(M["theta1"], dtype=float), x1, d_c0, d_c1,
                               k2, np.asarray(M["theta2"], dtype=float), x2, e_c0, e_c1,
                               int(M["initial_regime"]), n_steps)
    D = Dr[0] + 1j * Dr[1]
    E = Er[0] + 1j * Er[1]
    C = Lr[0] + 1j * Lr[1]
    B = Lr[2] + 1j * Lr[3]
    K = Lr[4] + 1j * Lr[5]
    H = Hr[0] + 1j * Hr[1]
    x = np.log(float(M["s1"]) / float(M["s2"]))
    al = float(M["alpha"])
    phi = np.exp(1j * d * x + K + B * al + C * al * al + D * float(M["nu1"]) + E * float(M["nu2"])) * H
    return np.vstack([phi.real, phi.imag]).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned: phi(0) = 1, and phi(-i) = E[Z_T] = S1/S2 because the
        # price ratio is a martingale under the numeraire measure.
        {
            "setup": """import numpy as np
model = dict(s1=100.0, s2=95.0, nu1=0.1, nu2=0.1, alpha=0.3, kappa1=2.0, kappa2=2.0, xi1=0.1, xi2=0.1,
             rho1=-0.5, rho2=-0.5, rho=-0.25, rho_tilde1=-0.7, rho_tilde2=-0.7, beta1=0.5, beta2=0.5,
             a=0.2, b=0.3, eta=0.9, sigma1=[0.1, 0.1], sigma2=[0.2, 0.2], theta1=[0.1, 0.3], theta2=[0.1, 0.3],
             generator=[[-0.5, 0.5], [0.45, -0.45]], initial_regime=2)
EXPECTED = np.array([[1.0, 100.0 / 95.0], [0.0, 0.0]])
""",
            "call": "characteristic_function(np.array([0.0, -1j]), 1.0, model)",
            "gold_call": "EXPECTED",
        },
        # --- Normal: the task's model on a grid of real transform values.
        {
            "setup": """import numpy as np
model = dict(s1=100.0, s2=95.0, nu1=0.1, nu2=0.1, alpha=0.3, kappa1=2.0, kappa2=2.0, xi1=0.1, xi2=0.1,
             rho1=-0.5, rho2=-0.5, rho=-0.25, rho_tilde1=-0.7, rho_tilde2=-0.7, beta1=0.5, beta2=0.5,
             a=0.2, b=0.3, eta=0.9, sigma1=[0.1, 0.1], sigma2=[0.2, 0.2], theta1=[0.1, 0.3], theta2=[0.1, 0.3],
             generator=[[-0.5, 0.5], [0.45, -0.45]], initial_regime=2)
d = np.linspace(0.05, 40.0, 40)
""",
            "call": "characteristic_function(d, 1.0, model)",
            "gold_call": "_oracle_characteristic_function(d, 1.0, model)",
        },
        # --- Boundary: shifted transform values, starting in regime 1.
        {
            "setup": """import numpy as np
model = dict(s1=90.0, s2=100.0, nu1=0.05, nu2=0.2, alpha=0.1, kappa1=1.5, kappa2=3.0, xi1=0.3, xi2=0.2,
             rho1=-0.3, rho2=-0.6, rho=0.4, rho_tilde1=-0.5, rho_tilde2=-0.2, beta1=0.3, beta2=0.6,
             a=0.5, b=0.2, eta=0.5, sigma1=[0.15, 0.25], sigma2=[0.1, 0.3], theta1=[0.05, 0.2], theta2=[0.1, 0.25],
             generator=[[-1.0, 1.0], [0.3, -0.3]], initial_regime=1)
d = np.linspace(0.05, 20.0, 15) - 1j
""",
            "call": "characteristic_function(d, 0.5, model, 32, 100)",
            "gold_call": "_oracle_characteristic_function(d, 0.5, model, 32, 100)",
        },
        # --- Edge: no liquidity loading and a single regime.
        {
            "setup": """import numpy as np
model = dict(s1=50.0, s2=50.0, nu1=0.04, nu2=0.04, alpha=0.3, kappa1=2.0, kappa2=2.0, xi1=0.4, xi2=0.4,
             rho1=-0.7, rho2=-0.7, rho=0.0, rho_tilde1=0.0, rho_tilde2=0.0, beta1=0.0, beta2=0.0,
             a=0.2, b=0.3, eta=0.9, sigma1=[0.2], sigma2=[0.2], theta1=[0.04], theta2=[0.04],
             generator=[[0.0]], initial_regime=1)
d = np.array([0.3, 2.0, 7.0])
""",
            "call": "characteristic_function(d, 2.0, model, 16, 80)",
            "gold_call": "_oracle_characteristic_function(d, 2.0, model, 16, 80)",
        },
        # --- Invalid: missing key ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        characteristic_function(np.array([1.0]), 1.0, dict(s1=100.0))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_characteristic_function(np.array([1.0]), 1.0, dict(s1=100.0))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: correlation outside [-1, 1] ---
        {
            "setup": """import numpy as np
model = dict(s1=100.0, s2=95.0, nu1=0.1, nu2=0.1, alpha=0.3, kappa1=2.0, kappa2=2.0, xi1=0.1, xi2=0.1,
             rho1=-0.5, rho2=-0.5, rho=-1.25, rho_tilde1=-0.7, rho_tilde2=-0.7, beta1=0.5, beta2=0.5,
             a=0.2, b=0.3, eta=0.9, sigma1=[0.1, 0.1], sigma2=[0.2, 0.2], theta1=[0.1, 0.3], theta2=[0.1, 0.3],
             generator=[[-0.5, 0.5], [0.45, -0.45]], initial_regime=2)
def run_model():
    try:
        characteristic_function(np.array([1.0]), 1.0, model)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_characteristic_function(np.array([1.0]), 1.0, model)
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
