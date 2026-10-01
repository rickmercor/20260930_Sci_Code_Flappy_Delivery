"""
Chain the sub-problem functions 01-07 end-to-end and return the European call value at the strike under the time-fractional model. Orchestrator: yes - this is the final step; it integrates the earlier sub-problems (integrated_kernels, integrated_kernel_weights, analytic_interior_weights, assemble_differentiation_matrices, l1_caputo_coefficients, fractional_bs_operator, l1_time_march) rather than reimplementing them.

The truncated domain is [0, smax_factor * K] with N uniform nodes, and the strike must fall on a node. The kernel shape parameter is c = c_over_h * h. The differentiation matrices and the semi-discrete operator are built from the integrated-kernel scheme, the payoff max(S - K, 0) is the value at tau = 0, the first node carries the value 0 and the last node the value S_max e^{-q tau} - K e^{-r tau} at every level, and the system is marched to tau = T with the L1 scheme on n uniform steps.

Returns
-------
float: the option value at S = K and tau = T, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def price_at_strike(strike: float = 100.0, rate: float = 0.05, dividend: float = 0.0,
                    sigma: float = 0.4, alpha: float = 0.8, maturity: float = 1.0,
                    n_nodes: int = 61, n_steps: int = 400, smax_factor: float = 3.0,
                    c_over_h: float = 4.0) -> float:
    '''European call value at the strike under the time-fractional model.

    Parameters
    ----------
    strike : float
        Strike K > 0.
    rate : float
        Risk-free rate r.
    dividend : float
        Continuous dividend yield q.
    sigma : float
        Volatility, sigma > 0.
    alpha : float
        Order of the Caputo derivative in time to maturity, 0 < alpha <= 1.
    maturity : float
        Time to maturity T > 0.
    n_nodes : int
        Number of asset nodes N >= 7 on [0, smax_factor * K], both ends included.
    n_steps : int
        Number of uniform time steps n >= 1.
    smax_factor : float
        Right end of the asset domain in units of the strike, > 1.
    c_over_h : float
        Kernel shape parameter in units of the grid spacing, > 0.

    Returns
    -------
    value : float
        Call value at S = K and tau = T, as a native Python float.

    Raises
    ------
    ValueError
        If any argument is outside its stated domain or if the strike does not
        fall on a grid node.

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


def _oracle_price_at_strike(strike: float = 100.0, rate: float = 0.05, dividend: float = 0.0,
                            sigma: float = 0.4, alpha: float = 0.8, maturity: float = 1.0,
                            n_nodes: int = 61, n_steps: int = 400, smax_factor: float = 3.0,
                            c_over_h: float = 4.0) -> float:
    import numpy as np

    for name, value in (("strike", strike), ("sigma", sigma), ("maturity", maturity)):
        if isinstance(value, bool) or not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be a finite positive number")
    for name, value in (("rate", rate), ("dividend", dividend)):
        if isinstance(value, bool) or not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if isinstance(smax_factor, bool) or not np.isfinite(float(smax_factor)) or float(smax_factor) <= 1.0:
        raise ValueError("smax_factor must exceed 1")
    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)) or int(n_steps) < 1:
        raise ValueError("n_steps must be an integer >= 1")

    k = float(strike)
    s_max = float(smax_factor) * k
    # -- Sub-problem 04: the seven-band differentiation matrices of the
    # integrated-kernel scheme.
    mats = _oracle_assemble_differentiation_matrices(n_nodes, s_max, c_over_h)
    n = int(n_nodes)
    grid = np.linspace(0.0, s_max, n)
    h = grid[1] - grid[0]
    c = float(c_over_h) * h
    i_strike = int(round(k / h))
    if abs(grid[i_strike] - k) > 1e-9 * k:
        raise ValueError("the strike must fall on a grid node")

    # -- Sub-problems 03, 02 and 01: consistency gate on the assembled rows.
    # The centred rows must carry the closed-form weights, the rows next to
    # the boundaries the integrated-kernel weights, and those weights must
    # satisfy the exactness conditions on the kernel translates.
    centred = _oracle_analytic_interior_weights(h, c)
    edge_nodes = grid[:7]
    edge = _oracle_integrated_kernel_weights(edge_nodes, grid[1], c)
    if n >= 7 and not (np.allclose(mats[:, 1, :7], edge, rtol=1e-12, atol=0.0)):
        raise ValueError("one-sided rows disagree with the integrated-kernel weights")
    if n >= 7 and 3 <= n - 4 and not np.allclose(mats[:, 3, 0:7], centred, rtol=1e-12, atol=0.0):
        raise ValueError("centred rows disagree with the closed-form weights")
    kern = _oracle_integrated_kernels((edge_nodes[None, :] - edge_nodes[:, None]).ravel(), c)[:, 2].reshape(7, 7)
    target = _oracle_integrated_kernels(grid[1] - edge_nodes, c)
    residual = np.max(np.abs(kern @ edge[1] - target[:, 0])) / np.max(np.abs(target[:, 0]))
    if not residual < 1e-6:
        raise ValueError("integrated-kernel exactness conditions are not met")

    # -- Sub-problem 06: the semi-discrete spatial operator.
    op = _oracle_fractional_bs_operator(grid, mats, sigma, rate, dividend)

    # -- Sub-problem 05: the L1 coefficients must sum to zero (a constant state
    # has zero Caputo derivative) before they are used in the march.
    last = _oracle_l1_caputo_coefficients(alpha, int(n_steps))
    if abs(float(np.sum(last))) > 1e-10 * float(np.max(np.abs(last))):
        raise ValueError("L1 coefficients do not sum to zero")

    # Payoff at tau = 0 and call boundary data at every level.
    tau = np.linspace(0.0, float(maturity), int(n_steps) + 1)
    u0 = np.maximum(grid - k, 0.0)
    left = np.zeros_like(tau)
    right = s_max * np.exp(-float(dividend) * tau) - k * np.exp(-float(rate) * tau)

    # -- Sub-problem 07: implicit L1 march to maturity.
    terminal = _oracle_l1_time_march(op, u0, left, right, alpha, maturity)
    return float(terminal[i_strike])

# =============================================================================
# TEST CASES
# =============================================================================

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
            "call": "price_at_strike()",
            "gold_call": "_oracle_price_at_strike()",
        },
        # --- Valid: a coarser grid and a stronger memory effect ---
        {
            "setup": """import numpy as np
""",
            "call": "price_at_strike(100.0, 0.05, 0.0, 0.4, 0.5, 1.0, 31, 120)",
            "gold_call": "_oracle_price_at_strike(100.0, 0.05, 0.0, 0.4, 0.5, 1.0, 31, 120)",
        },
        # --- Boundary: alpha = 1 with a dividend, the classical limit of the
        # model on the same spatial scheme ---
        {
            "setup": """import numpy as np
""",
            "call": "price_at_strike(50.0, 0.03, 0.02, 0.25, 1.0, 0.5, 31, 80)",
            "gold_call": "_oracle_price_at_strike(50.0, 0.03, 0.02, 0.25, 1.0, 0.5, 31, 80)",
        },
        # --- Edge: the smallest grid on which the strike is a node ---
        {
            "setup": """import numpy as np
""",
            "call": "price_at_strike(10.0, 0.01, 0.0, 0.6, 0.9, 2.0, 7, 25, 3.0, 6.0)",
            "gold_call": "_oracle_price_at_strike(10.0, 0.01, 0.0, 0.6, 0.9, 2.0, 7, 25, 3.0, 6.0)",
        },
        # --- Invalid: strike off the grid ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        price_at_strike(100.0, 0.05, 0.0, 0.4, 0.8, 1.0, 20, 10)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_price_at_strike(100.0, 0.05, 0.0, 0.4, 0.8, 1.0, 20, 10)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: memory order outside (0, 1] ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        price_at_strike(100.0, 0.05, 0.0, 0.4, 1.3, 1.0, 31, 10)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_price_at_strike(100.0, 0.05, 0.0, 0.4, 1.3, 1.0, 31, 10)
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
