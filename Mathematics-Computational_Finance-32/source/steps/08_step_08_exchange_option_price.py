"""
Chain the sub-problem functions 01-07 and return the time-0 value of the European option to exchange asset 2 for asset 1 under the regime-switching model with Heston variances and stochastic market liquidity. Orchestrator: yes - this is the final step; it integrates the earlier sub-problems (variance_riccati, liquidity_coefficients, regime_factor, characteristic_function, exercise_probabilities, margrabe_price, price_comparisons) rather than reimplementing them.

The value is computed by the transform method under the second asset as numeraire. Before the value is returned, the building blocks are checked against exact properties: the coefficients vanish where the forcing vanishes, the regime factor is one there, the characteristic function is one at the origin and returns S1/S2 at -i, and the transform price reproduces the Margrabe formula when variance, liquidity and switching are removed. A failed check raises ValueError. With no model supplied, the parameter set of the problem is used.

Returns
-------
float: the exchange option value at time 0, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def exchange_option_price(model: dict = None, tau: float = 1.0, delta_max: float = 40.0,
                          n_nodes: int = 240, n_quad: int = 64, n_steps: int = 200) -> float:
    '''Time-0 value of the exchange option under the full model.

    Parameters
    ----------
    model : dict or None
        Model dictionary as in ``characteristic_function`` (two regimes). None
        selects the parameter set of the problem.
    tau : float
        Time to maturity, tau > 0.
    delta_max, n_nodes, n_quad, n_steps
        Numerical settings passed to the transform pricing.

    Returns
    -------
    value : float
        The exchange option value, as a native Python float.

    Raises
    ------
    ValueError
        If any argument is outside its domain or a consistency check fails.

    Notes
    -----
    This is the final, orchestrating step: call the public functions of
    sub-problems 01-07 and feed each returned value into the next, rather
    than reimplementing them. Include every import your implementation needs
    inside the function body.
    '''
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================


def _oracle_exchange_option_price(model: dict = None, tau: float = 1.0, delta_max: float = 40.0,
                                  n_nodes: int = 240, n_quad: int = 64, n_steps: int = 200) -> float:
    import numpy as np

    if model is None:
        model = dict(s1=100.0, s2=95.0, nu1=0.1, nu2=0.1, alpha=0.3, kappa1=2.0, kappa2=2.0,
                     xi1=0.1, xi2=0.1, rho1=-0.5, rho2=-0.5, rho=-0.25, rho_tilde1=-0.7,
                     rho_tilde2=-0.7, beta1=0.5, beta2=0.5, a=0.2, b=0.3, eta=0.9,
                     sigma1=[0.1, 0.1], sigma2=[0.2, 0.2], theta1=[0.1, 0.3], theta2=[0.1, 0.3],
                     generator=[[-0.5, 0.5], [0.45, -0.45]], initial_regime=2)
    if isinstance(tau, bool) or not np.isfinite(float(tau)) or float(tau) <= 0.0:
        raise ValueError("tau must be a finite positive number")

    # ---- exact properties of the building blocks --------------------------
    at_minus_i = np.array([-1j])
    if np.max(np.abs(_oracle_variance_riccati(at_minus_i, tau, float(model["xi1"]), -float(model["kappa1"]), 0.0))) > 1e-12:
        raise ValueError("variance coefficient does not vanish at delta = -i")          # step 01
    liq = _oracle_liquidity_coefficients(at_minus_i, tau, float(model["a"]), float(model["b"]),
                                         float(model["eta"]), -float(model["a"]), 0.0, 0.5, n_quad)
    if np.max(np.abs(liq)) > 1e-12:
        raise ValueError("liquidity coefficients do not vanish at delta = -i")          # step 02
    G = np.asarray(model["generator"], dtype=float)
    k = G.shape[0]
    fac = _oracle_regime_factor(at_minus_i, tau, G, np.full(k, 0.05), 1.0, np.full(k, 0.1), 0.2,
                                -1.0, 0.0, 1.0, np.full(k, 0.1), 0.2, -1.0, 0.0, 1, 20)
    if abs(fac[0][0] - 1.0) > 1e-12 or abs(fac[1][0]) > 1e-12:
        raise ValueError("regime factor is not one at delta = -i")                      # step 03
    phi = _oracle_characteristic_function(np.array([0.0, -1j]), tau, model, n_quad, n_steps)
    ratio = float(model["s1"]) / float(model["s2"])
    if abs(phi[0][0] - 1.0) > 1e-10 or abs(phi[0][1] - ratio) > 1e-8 * ratio:
        raise ValueError("characteristic function fails phi(0) = 1 or phi(-i) = S1/S2")  # step 04

    # ---- Margrabe limit of the transform pricing -----------------------------
    # The regime with the largest constant variance rate gives the best-
    # conditioned inversion for the check.
    rho = float(model["rho"])
    sig_a = np.asarray(model["sigma1"], dtype=float)
    sig_b = np.asarray(model["sigma2"], dtype=float)
    rates = sig_a ** 2 + sig_b ** 2 - 2.0 * rho * sig_a * sig_b
    kmax = int(np.argmax(rates))
    s1v, s2v = float(sig_a[kmax]), float(sig_b[kmax])
    var_rate = float(rates[kmax])
    limit = dict(model, nu1=0.0, nu2=0.0, alpha=0.0, beta1=0.0, beta2=0.0, b=0.0,
                 sigma1=[s1v], sigma2=[s2v], theta1=[0.0], theta2=[0.0], generator=[[0.0]],
                 initial_regime=1, rho1=0.0, rho2=0.0, rho_tilde1=0.0, rho_tilde2=0.0)
    pr = _oracle_exercise_probabilities(tau, limit, delta_max, n_nodes, n_quad, n_steps)  # step 05
    transform_limit = float(model["s2"]) * (pr[0] * pr[1] - pr[2])
    reference = _oracle_margrabe_price(float(model["s1"]), float(model["s2"]), var_rate, tau)   # step 06
    if abs(transform_limit - reference) > 1e-4 * max(1.0, reference):
        raise ValueError("transform pricing fails the Margrabe limit")

    # ---- full model ------------------------------------------------------------
    values = _oracle_price_comparisons(tau, model, delta_max, n_nodes, n_quad, n_steps)     # step 07
    return float(values[0])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Benchmark configuration: the pinned answer of the task ---
        {
            "setup": """import numpy as np
""",
            "call": "exchange_option_price()",
            "gold_call": "_oracle_exchange_option_price()",
        },
        # --- Valid: the same model at a longer maturity ---
        {
            "setup": """import numpy as np
""",
            "call": "exchange_option_price(None, 2.0)",
            "gold_call": "_oracle_exchange_option_price(None, 2.0)",
        },
        # --- Boundary: a short maturity with equal prices, starting in regime 1 ---
        {
            "setup": """import numpy as np
model = dict(s1=100.0, s2=100.0, nu1=0.1, nu2=0.1, alpha=0.3, kappa1=2.0, kappa2=2.0, xi1=0.1, xi2=0.1,
             rho1=-0.5, rho2=-0.5, rho=-0.25, rho_tilde1=-0.7, rho_tilde2=-0.7, beta1=0.5, beta2=0.5,
             a=0.2, b=0.3, eta=0.9, sigma1=[0.1, 0.1], sigma2=[0.2, 0.2], theta1=[0.1, 0.3], theta2=[0.1, 0.3],
             generator=[[-0.5, 0.5], [0.45, -0.45]], initial_regime=1)
""",
            "call": "exchange_option_price(model, 0.1, 60.0, 64, 16, 40)",
            "gold_call": "_oracle_exchange_option_price(model, 0.1, 60.0, 64, 16, 40)",
        },
        # --- Edge: strongly different regimes and positive asset correlation ---
        {
            "setup": """import numpy as np
model = dict(s1=80.0, s2=100.0, nu1=0.05, nu2=0.05, alpha=0.2, kappa1=3.0, kappa2=1.0, xi1=0.3, xi2=0.3,
             rho1=-0.3, rho2=-0.3, rho=0.2, rho_tilde1=-0.5, rho_tilde2=-0.5, beta1=0.3, beta2=0.3,
             a=0.5, b=0.2, eta=0.4, sigma1=[0.05, 0.4], sigma2=[0.05, 0.3], theta1=[0.02, 0.5], theta2=[0.02, 0.4],
             generator=[[-3.0, 3.0], [2.0, -2.0]], initial_regime=1)
""",
            "call": "exchange_option_price(model, 2.0, 30.0, 120, 32, 150)",
            "gold_call": "_oracle_exchange_option_price(model, 2.0, 30.0, 120, 32, 150)",
        },
        # --- Invalid: non-positive maturity ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        exchange_option_price(None, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_exchange_option_price(None, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a model with a missing key ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        exchange_option_price(dict(s1=100.0, s2=95.0, generator=[[-0.5, 0.5], [0.45, -0.45]], xi1=0.1, kappa1=2.0, a=0.2, b=0.3, eta=0.9), 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_exchange_option_price(dict(s1=100.0, s2=95.0, generator=[[-0.5, 0.5], [0.45, -0.45]], xi1=0.1, kappa1=2.0, a=0.2, b=0.3, eta=0.9), 1.0)
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
