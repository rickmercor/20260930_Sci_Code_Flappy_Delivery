"""
The put value is marched backward from its payoff at maturity with fully implicit steps of the discounted Black-Scholes generator, the first node carrying the value K and the last node the value 0 at every level. After each implicit step the early-exercise constraint is imposed by projection onto the payoff: the value at every node becomes the larger of the implicit continuation value and the payoff there. A node belongs to the exercise region of that level when the payoff there is strictly positive and strictly above the implicit continuation value. These regions define the stopping rule of the additive representation: a path is stopped the first time it stands on an exercise node.

The put value is marched backward from its payoff at maturity with fully implicit steps of the discounted Black-Scholes generator, the first node carrying the value K and the last node the value 0 at every level. After each implicit step the early-exercise constraint is imposed by projection onto the payoff. A node belongs to the exercise region of that level when the payoff there is strictly positive and strictly above the implicit continuation value. These regions define the stopping rule of the additive representation: a path is stopped the first time it stands on an exercise node.

Returns
-------
np.ndarray, float, shape (n_steps + 1, n): row 0 the put value at time 0 on the grid; row 1 + k the exercise indicator (1.0 or 0.0) at time t_k = k * maturity / n_steps, for k = 0, ..., n_steps - 1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def american_put_exercise(s_grid: np.ndarray, strike: float, rate: float, vol: float,
                          maturity: float, n_steps: int) -> np.ndarray:
    '''American put value at time 0 and the exercise indicators of every level.

    Parameters
    ----------
    s_grid : np.ndarray
        Uniform grid of n >= 3 prices starting at 0.
    strike : float
        Strike K > 0.
    rate : float
        Risk-free rate r (finite).
    vol : float
        Volatility b >= 0.
    maturity : float
        Maturity T > 0.
    n_steps : int
        Number of uniform time steps, n_steps >= 1.

    Returns
    -------
    out : np.ndarray
        Shape (n_steps + 1, n) float array; see the module description.

    Raises
    ------
    ValueError
        If the grid does not start at 0 or is not uniform, or if any
        argument is outside its domain.

    Notes
    -----
    Build the steps from ``bs_operator_bands`` and ``implicit_step``. Include
    every import your implementation needs inside the function body.
    '''
    return np.zeros((int(n_steps) + 1, np.asarray(s_grid).size), dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================


def _oracle_american_put_exercise(s_grid: np.ndarray, strike: float, rate: float, vol: float,
                                  maturity: float, n_steps: int) -> np.ndarray:
    import numpy as np

    s = np.asarray(s_grid, dtype=float)
    if s.ndim != 1 or s.size < 3 or not np.all(np.isfinite(s)) or s[0] != 0.0:
        raise ValueError("s_grid must be a finite grid of at least 3 nodes starting at 0")
    for name, value in (("strike", strike), ("maturity", maturity)):
        if isinstance(value, bool) or not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be a finite positive number")
    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)) or int(n_steps) < 1:
        raise ValueError("n_steps must be an integer >= 1")
    bands = _oracle_bs_operator_bands(s, rate, vol)   # validates uniformity, rate and vol
    k = float(strike)
    m = int(n_steps)
    dt = float(maturity) / m

    payoff = np.maximum(k - s, 0.0)
    value = payoff.copy()
    out = np.zeros((m + 1, s.size))
    for level in range(m - 1, -1, -1):
        continuation = _oracle_implicit_step(bands, dt, value, k, 0.0)
        exercise = (payoff > continuation) & (payoff > 0.0)
        value = np.maximum(continuation, payoff)
        out[1 + level] = exercise.astype(float)
    out[0] = value
    return out

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned structure: the value never falls below the payoff, the
        # first node keeps the value K, and at every level the exercise nodes
        # form one block starting at the first interior node.
        {
            "setup": """import numpy as np
S = np.linspace(0.0, 300.0, 151)
K = 100.0
def structure(fn):
    out = fn(S, K, 0.06, 0.25, 1.0, 40)
    v, masks = out[0], out[1:]
    pay = np.maximum(K - S, 0.0)
    above = float(np.all(v >= pay - 1e-9 * np.maximum(pay, 1.0)))
    block = 1.0
    for row in masks:
        idx = np.flatnonzero(row[1:-1] > 0.5) + 1
        if idx.size and not np.array_equal(idx, np.arange(1, idx[-1] + 1)):
            block = 0.0
    return np.array([above, block, v[0], float(masks[0][1:-1].sum() > 0)])
EXPECTED = np.array([1.0, 1.0, 100.0, 1.0])
""",
            "call": "structure(american_put_exercise)",
            "gold_call": "EXPECTED",
        },
        # --- Normal: benchmark-like parameters on a coarse grid.
        {
            "setup": """import numpy as np
S = np.linspace(0.0, 400.0, 201)
""",
            "call": "american_put_exercise(S, 100.0, 0.05, 0.2, 1.0, 50)",
            "gold_call": "_oracle_american_put_exercise(S, 100.0, 0.05, 0.2, 1.0, 50)",
        },
        # --- Boundary: a single time step.
        {
            "setup": """import numpy as np
S = np.linspace(0.0, 200.0, 41)
""",
            "call": "american_put_exercise(S, 100.0, 0.05, 0.2, 0.5, 1)",
            "gold_call": "_oracle_american_put_exercise(S, 100.0, 0.05, 0.2, 0.5, 1)",
        },
        # --- Edge: a high rate and a wide exercise region, with the volatility
        # kept large enough that the fully implicit operator stays monotone on
        # this grid (rate * h < vol^2 * S away from the first node), so the
        # continuation values never go negative and the projection is unambiguous.
        {
            "setup": """import numpy as np
S = np.linspace(0.0, 150.0, 76)
""",
            "call": "american_put_exercise(S, 50.0, 0.12, 0.4, 2.0, 30)",
            "gold_call": "_oracle_american_put_exercise(S, 50.0, 0.12, 0.4, 2.0, 30)",
        },
        # --- Invalid: grid not starting at zero ---
        {
            "setup": """import numpy as np
S = np.linspace(1.0, 200.0, 21)
def run_model():
    try:
        american_put_exercise(S, 100.0, 0.05, 0.2, 1.0, 10)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_american_put_exercise(S, 100.0, 0.05, 0.2, 1.0, 10)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: zero time steps ---
        {
            "setup": """import numpy as np
S = np.linspace(0.0, 200.0, 21)
def run_model():
    try:
        american_put_exercise(S, 100.0, 0.05, 0.2, 1.0, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_american_put_exercise(S, 100.0, 0.05, 0.2, 1.0, 0)
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
