"""
Recover from the characteristic function the two probabilities that price the unit-strike call on the price ratio, together with the normalisation phi(-i).

With Z = S1/S2 and the unit strike, the exchange value per unit of S2 is E[Z_T 1{Z_T > 1}] - P(Z_T > 1) under the numeraire measure. That is phi(-i) P1 - P2, where P2 is the exercise probability and P1 the exercise probability under the measure weighted by Z_T / E[Z_T]. Both follow from the Gil-Pelaez inversion at log-strike 0: P2 = 1/2 + (1/pi) int_0^inf Re[phi(delta) / (i delta)] d delta, and P1 the same with phi(delta - i) / phi(-i) in place of phi(delta). The integrals are truncated at delta_max and computed with n_nodes-point Gauss-Legendre quadrature on (0, delta_max).

Returns
-------
np.ndarray, float, shape (3,): [phi(-i) (real), P1, P2].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def exercise_probabilities(tau: float, model: dict, delta_max: float = 40.0, n_nodes: int = 240,
                           n_quad: int = 64, n_steps: int = 200) -> np.ndarray:
    '''phi(-i) and the two Gil-Pelaez probabilities.

    Parameters
    ----------
    tau : float
        Time to maturity, tau > 0.
    model : dict
        Model dictionary as in ``characteristic_function``.
    delta_max : float
        Truncation of the transform integrals, > 0.
    n_nodes : int
        Gauss-Legendre nodes on (0, delta_max), >= 2.
    n_quad : int
        Passed to the characteristic function.
    n_steps : int
        Passed to the characteristic function.

    Returns
    -------
    out : np.ndarray
        Shape (3,) float array [phi(-i), P1, P2].

    Raises
    ------
    ValueError
        If any argument is outside its domain.

    Notes
    -----
    Use ``characteristic_function``. Include every import your implementation
    needs inside the function body.
    '''
    return np.zeros(3, dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================


def _oracle_exercise_probabilities(tau: float, model: dict, delta_max: float = 40.0, n_nodes: int = 240,
                                   n_quad: int = 64, n_steps: int = 200) -> np.ndarray:
    import numpy as np

    for name, value in (("tau", tau), ("delta_max", delta_max)):
        if isinstance(value, bool) or not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be a finite positive number")
    if isinstance(n_nodes, bool) or not isinstance(n_nodes, (int, np.integer)) or int(n_nodes) < 2:
        raise ValueError("n_nodes must be an integer >= 2")
    nodes, weights = np.polynomial.legendre.leggauss(int(n_nodes))
    dl = 0.5 * float(delta_max) * (nodes + 1.0)
    wl = 0.5 * float(delta_max) * weights

    f_mi = _oracle_characteristic_function(np.array([-1j]), tau, model, n_quad, n_steps)
    phi_mi = f_mi[0][0] + 1j * f_mi[1][0]
    f2 = _oracle_characteristic_function(dl, tau, model, n_quad, n_steps)
    f1 = _oracle_characteristic_function(dl - 1j, tau, model, n_quad, n_steps)
    phi2 = f2[0] + 1j * f2[1]
    phi1 = f1[0] + 1j * f1[1]
    p2 = 0.5 + np.sum(wl * np.real(phi2 / (1j * dl))) / np.pi
    p1 = 0.5 + np.sum(wl * np.real(phi1 / (1j * dl * phi_mi))) / np.pi
    return np.array([phi_mi.real, p1, p2], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned limit: with no stochastic variance, no liquidity loading
        # and a single regime, Z is lognormal and P1 = N(d1), P2 = N(d2) of
        # the Margrabe formula.
        {
            "setup": """import numpy as np
from math import erf, log, sqrt
N = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
model = dict(s1=100.0, s2=95.0, nu1=0.0, nu2=0.0, alpha=0.0, kappa1=2.0, kappa2=2.0, xi1=0.1, xi2=0.1,
             rho1=0.0, rho2=0.0, rho=-0.25, rho_tilde1=0.0, rho_tilde2=0.0, beta1=0.0, beta2=0.0,
             a=0.2, b=0.0, eta=0.9, sigma1=[0.3], sigma2=[0.2], theta1=[0.0], theta2=[0.0],
             generator=[[0.0]], initial_regime=1)
s = sqrt(0.09 + 0.04 + 2 * 0.25 * 0.3 * 0.2)
d1 = (log(100.0 / 95.0) + 0.5 * s * s) / s
EXPECTED = np.array([100.0 / 95.0, N(d1), N(d1 - s)])
""",
            "call": "exercise_probabilities(1.0, model)",
            "gold_call": "EXPECTED",
        },
        # --- Normal: the task's model.
        {
            "setup": """import numpy as np
model = dict(s1=100.0, s2=95.0, nu1=0.1, nu2=0.1, alpha=0.3, kappa1=2.0, kappa2=2.0, xi1=0.1, xi2=0.1,
             rho1=-0.5, rho2=-0.5, rho=-0.25, rho_tilde1=-0.7, rho_tilde2=-0.7, beta1=0.5, beta2=0.5,
             a=0.2, b=0.3, eta=0.9, sigma1=[0.1, 0.1], sigma2=[0.2, 0.2], theta1=[0.1, 0.3], theta2=[0.1, 0.3],
             generator=[[-0.5, 0.5], [0.45, -0.45]], initial_regime=2)
""",
            "call": "exercise_probabilities(1.0, model)",
            "gold_call": "_oracle_exercise_probabilities(1.0, model)",
        },
        # --- Boundary: a short maturity with a coarse transform grid.
        {
            "setup": """import numpy as np
model = dict(s1=100.0, s2=100.0, nu1=0.1, nu2=0.1, alpha=0.3, kappa1=2.0, kappa2=2.0, xi1=0.1, xi2=0.1,
             rho1=-0.5, rho2=-0.5, rho=-0.25, rho_tilde1=-0.7, rho_tilde2=-0.7, beta1=0.5, beta2=0.5,
             a=0.2, b=0.3, eta=0.9, sigma1=[0.1, 0.1], sigma2=[0.2, 0.2], theta1=[0.1, 0.3], theta2=[0.1, 0.3],
             generator=[[-0.5, 0.5], [0.45, -0.45]], initial_regime=1)
""",
            "call": "exercise_probabilities(0.1, model, 60.0, 64, 16, 40)",
            "gold_call": "_oracle_exercise_probabilities(0.1, model, 60.0, 64, 16, 40)",
        },
        # --- Edge: deep in the money for asset 1, long maturity.
        {
            "setup": """import numpy as np
model = dict(s1=160.0, s2=80.0, nu1=0.05, nu2=0.15, alpha=0.5, kappa1=1.0, kappa2=3.0, xi1=0.4, xi2=0.2,
             rho1=-0.2, rho2=-0.8, rho=0.5, rho_tilde1=-0.4, rho_tilde2=-0.6, beta1=0.4, beta2=0.2,
             a=0.8, b=0.4, eta=0.3, sigma1=[0.2, 0.3], sigma2=[0.1, 0.2], theta1=[0.05, 0.2], theta2=[0.1, 0.3],
             generator=[[-0.2, 0.2], [0.6, -0.6]], initial_regime=1)
""",
            "call": "exercise_probabilities(3.0, model, 30.0, 160)",
            "gold_call": "_oracle_exercise_probabilities(3.0, model, 30.0, 160)",
        },
        # --- Invalid: non-positive maturity ---
        {
            "setup": """import numpy as np
model = dict(s1=100.0, s2=95.0, nu1=0.1, nu2=0.1, alpha=0.3, kappa1=2.0, kappa2=2.0, xi1=0.1, xi2=0.1,
             rho1=-0.5, rho2=-0.5, rho=-0.25, rho_tilde1=-0.7, rho_tilde2=-0.7, beta1=0.5, beta2=0.5,
             a=0.2, b=0.3, eta=0.9, sigma1=[0.1, 0.1], sigma2=[0.2, 0.2], theta1=[0.1, 0.3], theta2=[0.1, 0.3],
             generator=[[-0.5, 0.5], [0.45, -0.45]], initial_regime=2)
def run_model():
    try:
        exercise_probabilities(0.0, model)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_exercise_probabilities(0.0, model)
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
model = dict(s1=100.0, s2=95.0, nu1=0.1, nu2=0.1, alpha=0.3, kappa1=2.0, kappa2=2.0, xi1=0.1, xi2=0.1,
             rho1=-0.5, rho2=-0.5, rho=-0.25, rho_tilde1=-0.7, rho_tilde2=-0.7, beta1=0.5, beta2=0.5,
             a=0.2, b=0.3, eta=0.9, sigma1=[0.1, 0.1], sigma2=[0.2, 0.2], theta1=[0.1, 0.3], theta2=[0.1, 0.3],
             generator=[[-0.5, 0.5], [0.45, -0.45]], initial_regime=2)
def run_model():
    try:
        exercise_probabilities(1.0, model, 40.0, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_exercise_probabilities(1.0, model, 40.0, 1)
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
