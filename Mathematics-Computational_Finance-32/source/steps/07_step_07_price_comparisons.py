"""
Price the exchange option under the full model and under three reduced versions that isolate its channels.

The exchange value at time 0 is S2 (phi(-i) P1 - P2), with the quantities of step 05. Besides the full model, the comparison prices the same contract (i) with the chain started in the other regime of a two-regime economy, (ii) with regime switching switched off, the chain frozen in its initial regime, and (iii) without the liquidity channel, beta1 = beta2 = 0. The differences show how much of the value comes from the current regime, from the possibility of switching, and from liquidity risk.

Returns
-------
np.ndarray, float, shape (4,): [full-model value, value started in the other regime, value with switching frozen, value without liquidity loading].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def price_comparisons(tau: float, model: dict, delta_max: float = 40.0, n_nodes: int = 240,
                      n_quad: int = 64, n_steps: int = 200) -> np.ndarray:
    '''Exchange value under the full model and three reductions.

    Parameters
    ----------
    tau : float
        Time to maturity, tau > 0.
    model : dict
        Model dictionary as in ``characteristic_function``, with two regimes.
    delta_max, n_nodes, n_quad, n_steps
        Numerical settings passed to ``exercise_probabilities``.

    Returns
    -------
    values : np.ndarray
        Shape (4,) float array; see the module description.

    Raises
    ------
    ValueError
        If the model does not have exactly two regimes or any argument is
        outside its domain.

    Notes
    -----
    Use ``exercise_probabilities``. Include every import your implementation
    needs inside the function body.
    '''
    return np.zeros(4, dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================


def _oracle_price_comparisons(tau: float, model: dict, delta_max: float = 40.0, n_nodes: int = 240,
                              n_quad: int = 64, n_steps: int = 200) -> np.ndarray:
    import numpy as np

    if not isinstance(model, dict) or "generator" not in model or "initial_regime" not in model:
        raise ValueError("model must be a dict as in characteristic_function")
    G = np.asarray(model["generator"], dtype=float)
    if G.shape != (2, 2):
        raise ValueError("the comparison needs a two-regime model")

    def value(m):
        out = _oracle_exercise_probabilities(tau, m, delta_max, n_nodes, n_quad, n_steps)
        return float(m["s2"]) * (out[0] * out[1] - out[2])

    full = value(model)
    other = value(dict(model, initial_regime=3 - int(model["initial_regime"])))
    frozen = value(dict(model, generator=[[0.0, 0.0], [0.0, 0.0]]))
    no_liquidity = value(dict(model, beta1=0.0, beta2=0.0))
    return np.array([full, other, frozen, no_liquidity], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the task's model at one year.
        {
            "setup": """import numpy as np
model = dict(s1=100.0, s2=95.0, nu1=0.1, nu2=0.1, alpha=0.3, kappa1=2.0, kappa2=2.0, xi1=0.1, xi2=0.1,
             rho1=-0.5, rho2=-0.5, rho=-0.25, rho_tilde1=-0.7, rho_tilde2=-0.7, beta1=0.5, beta2=0.5,
             a=0.2, b=0.3, eta=0.9, sigma1=[0.1, 0.1], sigma2=[0.2, 0.2], theta1=[0.1, 0.3], theta2=[0.1, 0.3],
             generator=[[-0.5, 0.5], [0.45, -0.45]], initial_regime=2)
""",
            "call": "price_comparisons(1.0, model)",
            "gold_call": "_oracle_price_comparisons(1.0, model)",
        },
        # --- Pinned structure: with identical regimes the regime choices do
        # not matter, so the first three values coincide.
        {
            "setup": """import numpy as np
model = dict(s1=100.0, s2=95.0, nu1=0.1, nu2=0.1, alpha=0.3, kappa1=2.0, kappa2=2.0, xi1=0.1, xi2=0.1,
             rho1=-0.5, rho2=-0.5, rho=-0.25, rho_tilde1=-0.7, rho_tilde2=-0.7, beta1=0.5, beta2=0.5,
             a=0.2, b=0.3, eta=0.9, sigma1=[0.1, 0.1], sigma2=[0.2, 0.2], theta1=[0.2, 0.2], theta2=[0.2, 0.2],
             generator=[[-0.5, 0.5], [0.45, -0.45]], initial_regime=2)
def spread(fn):
    v = fn(0.5, model, 40.0, 120, 32, 100)
    return np.array([v[0], float(abs(v[1] - v[0]) < 1e-6), float(abs(v[2] - v[0]) < 1e-6)])
REF = _oracle_price_comparisons(0.5, model, 40.0, 120, 32, 100)[0]
""",
            "call": "spread(price_comparisons)",
            "gold_call": "np.array([REF, 1.0, 1.0])",
        },
        # --- Boundary: a short maturity starting in regime 1.
        {
            "setup": """import numpy as np
model = dict(s1=100.0, s2=100.0, nu1=0.1, nu2=0.1, alpha=0.3, kappa1=2.0, kappa2=2.0, xi1=0.1, xi2=0.1,
             rho1=-0.5, rho2=-0.5, rho=-0.25, rho_tilde1=-0.7, rho_tilde2=-0.7, beta1=0.5, beta2=0.5,
             a=0.2, b=0.3, eta=0.9, sigma1=[0.1, 0.1], sigma2=[0.2, 0.2], theta1=[0.1, 0.3], theta2=[0.1, 0.3],
             generator=[[-0.5, 0.5], [0.45, -0.45]], initial_regime=1)
""",
            "call": "price_comparisons(0.1, model, 60.0, 64, 16, 40)",
            "gold_call": "_oracle_price_comparisons(0.1, model, 60.0, 64, 16, 40)",
        },
        # --- Edge: strongly different regimes and fast switching.
        {
            "setup": """import numpy as np
model = dict(s1=80.0, s2=100.0, nu1=0.05, nu2=0.05, alpha=0.2, kappa1=3.0, kappa2=1.0, xi1=0.3, xi2=0.3,
             rho1=-0.3, rho2=-0.3, rho=0.2, rho_tilde1=-0.5, rho_tilde2=-0.5, beta1=0.3, beta2=0.3,
             a=0.5, b=0.2, eta=0.4, sigma1=[0.05, 0.4], sigma2=[0.05, 0.3], theta1=[0.02, 0.5], theta2=[0.02, 0.4],
             generator=[[-3.0, 3.0], [2.0, -2.0]], initial_regime=1)
""",
            "call": "price_comparisons(2.0, model, 30.0, 120, 32, 150)",
            "gold_call": "_oracle_price_comparisons(2.0, model, 30.0, 120, 32, 150)",
        },
        # --- Invalid: a single-regime model ---
        {
            "setup": """import numpy as np
model = dict(s1=100.0, s2=95.0, nu1=0.1, nu2=0.1, alpha=0.3, kappa1=2.0, kappa2=2.0, xi1=0.1, xi2=0.1,
             rho1=-0.5, rho2=-0.5, rho=-0.25, rho_tilde1=-0.7, rho_tilde2=-0.7, beta1=0.5, beta2=0.5,
             a=0.2, b=0.3, eta=0.9, sigma1=[0.1], sigma2=[0.2], theta1=[0.1], theta2=[0.1],
             generator=[[0.0]], initial_regime=1)
def run_model():
    try:
        price_comparisons(1.0, model)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_price_comparisons(1.0, model)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: negative maturity ---
        {
            "setup": """import numpy as np
model = dict(s1=100.0, s2=95.0, nu1=0.1, nu2=0.1, alpha=0.3, kappa1=2.0, kappa2=2.0, xi1=0.1, xi2=0.1,
             rho1=-0.5, rho2=-0.5, rho=-0.25, rho_tilde1=-0.7, rho_tilde2=-0.7, beta1=0.5, beta2=0.5,
             a=0.2, b=0.3, eta=0.9, sigma1=[0.1, 0.1], sigma2=[0.2, 0.2], theta1=[0.1, 0.3], theta2=[0.1, 0.3],
             generator=[[-0.5, 0.5], [0.45, -0.45]], initial_regime=2)
def run_model():
    try:
        price_comparisons(-1.0, model)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_price_comparisons(-1.0, model)
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
