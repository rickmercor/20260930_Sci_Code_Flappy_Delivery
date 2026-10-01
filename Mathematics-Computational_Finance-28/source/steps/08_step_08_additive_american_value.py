"""
Chain the sub-problem functions 01-07 and return the value of the regularised additive forward representation of the American put at the spot. Orchestrator: yes - this is the final step; it integrates the earlier sub-problems (tanaka_drift_rate, bs_operator_bands, implicit_step, american_put_exercise, stopped_drift_integral, european_additive_value, additive_decomposition) rather than reimplementing them.

The value is the current gain plus the expected discounted integral of the gain's drift rate, local time included and regularised by the Gaussian kernel, up to the optimal exercise time of the put. It is computed on a fine grid. Before that, a coarse companion computation checks the building blocks against independent references: the European put by the same implicit steps against the Black-Scholes formula, the European form of the representation on the grid against its quadrature, and the American price against the intrinsic and European values. A failed check raises ValueError.

Returns
-------
float: the representation value at the spot and time 0, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def additive_american_value(spot: float = 100.0, strike: float = 100.0, rate: float = 0.05,
                            vol: float = 0.2, maturity: float = 1.0, eps: float = 0.5,
                            n_intervals: int = 4000, n_steps: int = 4000,
                            smax_factor: float = 4.0) -> float:
    '''Regularised additive representation of the American put at the spot.

    Parameters
    ----------
    spot : float
        Current price S_0, a node of the fine grid, 0 < S_0 < smax_factor * K.
    strike : float
        Strike K > 0.
    rate : float
        Risk-free rate r (finite).
    vol : float
        Volatility b > 0.
    maturity : float
        Maturity T > 0.
    eps : float
        Standard deviation of the Gaussian kernel in price units,
        0 < eps < K / 10.
    n_intervals : int
        Number of intervals of the fine grid on [0, smax_factor * K].
    n_steps : int
        Number of uniform time steps of the fine computation.
    smax_factor : float
        Right end of the grid in units of the strike, > 1.

    Returns
    -------
    value : float
        The representation value, as a native Python float.

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


def _oracle_additive_american_value(spot: float = 100.0, strike: float = 100.0, rate: float = 0.05,
                                    vol: float = 0.2, maturity: float = 1.0, eps: float = 0.5,
                                    n_intervals: int = 4000, n_steps: int = 4000,
                                    smax_factor: float = 4.0) -> float:
    import numpy as np
    from math import erf, exp, log, sqrt

    for name, value in (("spot", spot), ("strike", strike), ("vol", vol),
                        ("maturity", maturity), ("eps", eps)):
        if isinstance(value, bool) or not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be a finite positive number")
    if isinstance(rate, bool) or not np.isfinite(float(rate)):
        raise ValueError("rate must be finite")
    if isinstance(smax_factor, bool) or not np.isfinite(float(smax_factor)) or float(smax_factor) <= 1.0:
        raise ValueError("smax_factor must exceed 1")
    s0, k, r, b, t_mat, e = (float(x) for x in (spot, strike, rate, vol, maturity, eps))
    if e >= k / 10.0:
        raise ValueError("eps must be below strike / 10")
    if s0 >= float(smax_factor) * k:
        raise ValueError("spot must lie inside the grid")

    # ---- Coarse companion checks, on a grid of spacing eps / 2 -----------------
    s_max = float(smax_factor) * k
    n_coarse = int(np.ceil(s_max / (0.5 * e)))
    grid = np.linspace(0.0, s_max, n_coarse + 1)
    m_coarse = 400
    dt = t_mat / m_coarse
    bands = _oracle_bs_operator_bands(grid, r, b)                          # step 02
    euro = np.maximum(k - grid, 0.0)
    for level in range(m_coarse - 1, -1, -1):                              # step 03
        euro = _oracle_implicit_step(bands, dt, euro, k * exp(-r * (t_mat - level * dt)), 0.0)
    norm_cdf = lambda x: 0.5 * (1.0 + erf(x / sqrt(2.0)))
    d1 = (log(s0 / k) + (r + 0.5 * b * b) * t_mat) / (b * sqrt(t_mat))
    bs_put = k * exp(-r * t_mat) * norm_cdf(-(d1 - b * sqrt(t_mat))) - s0 * norm_cdf(-d1)
    euro_at_spot = float(np.interp(s0, grid, euro))
    if abs(euro_at_spot - bs_put) > 0.005 * max(1.0, bs_put):
        raise ValueError("implicit steps fail to reproduce the Black-Scholes put")

    parts = _oracle_tanaka_drift_rate(grid, k, r, b, e)                    # step 01
    no_exercise = np.zeros((m_coarse, grid.size))
    euro_integral = _oracle_stopped_drift_integral(grid, parts.sum(axis=0), no_exercise,
                                                   r, b, t_mat)            # step 05
    euro_grid_value = max(k - s0, 0.0) + float(np.interp(s0, grid, euro_integral[0]))
    euro_quad = _oracle_european_additive_value(s0, k, r, b, t_mat, e)     # step 06
    if abs(euro_grid_value - euro_quad[0]) > 0.005 * max(1.0, abs(euro_quad[0])):
        raise ValueError("grid and quadrature disagree on the European representation")

    american = _oracle_american_put_exercise(grid, k, r, b, t_mat, m_coarse)   # step 04
    am_at_spot = float(np.interp(s0, grid, american[0]))
    if am_at_spot < max(k - s0, 0.0) - 1e-9 or am_at_spot < euro_at_spot - 1e-6:
        raise ValueError("American price below the intrinsic or European value")

    # ---- Fine evaluation of the representation (step 07) ---------------------
    result = _oracle_additive_decomposition(s0, k, r, b, t_mat, e, n_intervals, n_steps, smax_factor)
    return float(result[0])

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
            "call": "additive_american_value()",
            "gold_call": "_oracle_additive_american_value()",
        },
        # --- Valid: a coarser fine grid, the same contract ---
        {
            "setup": """import numpy as np
""",
            "call": "additive_american_value(100.0, 100.0, 0.05, 0.2, 1.0, 0.5, 800, 200)",
            "gold_call": "_oracle_additive_american_value(100.0, 100.0, 0.05, 0.2, 1.0, 0.5, 800, 200)",
        },
        # --- Boundary: out of the money with a wider kernel ---
        {
            "setup": """import numpy as np
""",
            "call": "additive_american_value(110.0, 100.0, 0.05, 0.2, 1.0, 1.0, 800, 200)",
            "gold_call": "_oracle_additive_american_value(110.0, 100.0, 0.05, 0.2, 1.0, 1.0, 800, 200)",
        },
        # --- Edge: short maturity, high rate and volatility, small strike ---
        {
            "setup": """import numpy as np
""",
            "call": "additive_american_value(20.0, 20.0, 0.1, 0.45, 0.25, 0.4, 600, 150, 3.0)",
            "gold_call": "_oracle_additive_american_value(20.0, 20.0, 0.1, 0.45, 0.25, 0.4, 600, 150, 3.0)",
        },
        # --- Invalid: kernel wider than a tenth of the strike ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        additive_american_value(100.0, 100.0, 0.05, 0.2, 1.0, 15.0, 400, 50)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_additive_american_value(100.0, 100.0, 0.05, 0.2, 1.0, 15.0, 400, 50)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: spot outside the grid ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        additive_american_value(500.0, 100.0, 0.05, 0.2, 1.0, 0.5, 400, 50)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_additive_american_value(500.0, 100.0, 0.05, 0.2, 1.0, 0.5, 400, 50)
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
