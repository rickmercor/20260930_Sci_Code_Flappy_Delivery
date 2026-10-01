"""
Evaluate the additive forward representation of the American put at one spot price on a grid, together with its parts and the American price itself.

The asset grid is S_i = i h, i = 0, ..., n_intervals, with h = smax_factor * K / n_intervals, and the spot must be a grid node. The stopping rule is the exercise-region sequence of the American put on that grid, the drift-rate parts are those of step 01, and each part is integrated up to the stopping time as in step 05. The representation value is the current gain plus the two stopped integrals. The value obtained without the local-time term is the gain plus the first integral alone.

Returns
-------
np.ndarray, float, shape (4,): [representation value, value without the local-time term, local-time part, American put price], all at the spot and time 0.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def additive_decomposition(spot: float, strike: float, rate: float, vol: float, maturity: float,
                           eps: float, n_intervals: int, n_steps: int,
                           smax_factor: float) -> np.ndarray:
    '''Additive representation of the American put and its parts.

    Parameters
    ----------
    spot : float
        Current price S_0, a grid node with 0 < S_0 < smax_factor * K.
    strike : float
        Strike K > 0.
    rate : float
        Risk-free rate r (finite).
    vol : float
        Volatility b > 0.
    maturity : float
        Maturity T > 0.
    eps : float
        Standard deviation of the Gaussian kernel in price units, eps > 0.
    n_intervals : int
        Number of grid intervals on [0, smax_factor * K], >= 2.
    n_steps : int
        Number of uniform time steps, >= 1.
    smax_factor : float
        Right end of the grid in units of the strike, > 1.

    Returns
    -------
    parts : np.ndarray
        Shape (4,) float array; see the module description.

    Raises
    ------
    ValueError
        If any argument is outside its domain or the spot is not a grid node.

    Notes
    -----
    Use ``tanaka_drift_rate``, ``american_put_exercise`` and
    ``stopped_drift_integral``. Include every import your implementation
    needs inside the function body.
    '''
    return np.zeros(4, dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================


def _oracle_additive_decomposition(spot: float, strike: float, rate: float, vol: float, maturity: float,
                                   eps: float, n_intervals: int, n_steps: int,
                                   smax_factor: float) -> np.ndarray:
    import numpy as np

    for name, value in (("spot", spot), ("strike", strike), ("vol", vol), ("maturity", maturity)):
        if isinstance(value, bool) or not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be a finite positive number")
    if isinstance(smax_factor, bool) or not np.isfinite(float(smax_factor)) or float(smax_factor) <= 1.0:
        raise ValueError("smax_factor must exceed 1")
    if isinstance(n_intervals, bool) or not isinstance(n_intervals, (int, np.integer)) or int(n_intervals) < 2:
        raise ValueError("n_intervals must be an integer >= 2")
    k = float(strike)
    s_max = float(smax_factor) * k
    grid = np.linspace(0.0, s_max, int(n_intervals) + 1)
    h = grid[1] - grid[0]
    i0 = int(round(float(spot) / h))
    if not (0 < i0 < grid.size - 1) or abs(grid[i0] - float(spot)) > 1e-9 * max(1.0, float(spot)):
        raise ValueError("spot must be an interior grid node")

    # Stopping rule and American price (step 04).
    solved = _oracle_american_put_exercise(grid, k, rate, vol, maturity, n_steps)
    price = solved[0][i0]
    exercise = solved[1:]
    # Drift-rate parts (step 01), integrated up to the stopping time (step 05).
    parts = _oracle_tanaka_drift_rate(grid, k, rate, vol, eps)
    stopped = _oracle_stopped_drift_integral(grid, parts, exercise, rate, vol, maturity)
    gain = max(k - float(spot), 0.0)
    no_local_time = gain + stopped[0][i0]
    local_time = stopped[1][i0]
    return np.array([no_local_time + local_time, no_local_time, local_time, price], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the benchmark contract on a coarse grid.
        {
            "setup": """import numpy as np
""",
            "call": "additive_decomposition(100.0, 100.0, 0.05, 0.2, 1.0, 0.5, 800, 200, 4.0)",
            "gold_call": "_oracle_additive_decomposition(100.0, 100.0, 0.05, 0.2, 1.0, 0.5, 800, 200, 4.0)",
        },
        # --- Benchmark: the default grid of the task, the decomposition the
        # reasoning is asked to report.
        {
            "setup": """import numpy as np
""",
            "call": "additive_decomposition(100.0, 100.0, 0.05, 0.2, 1.0, 0.5, 4000, 4000, 4.0)",
            "gold_call": "_oracle_additive_decomposition(100.0, 100.0, 0.05, 0.2, 1.0, 0.5, 4000, 4000, 4.0)",
        },
        # --- Pinned: deep in the money the spot is exercised at once, so both
        # stopped integrals vanish and every value equals the payoff 40.
        {
            "setup": """import numpy as np
EXPECTED = np.array([40.0, 40.0, 0.0, 40.0])
""",
            "call": "additive_decomposition(60.0, 100.0, 0.05, 0.2, 1.0, 0.5, 400, 50, 4.0)",
            "gold_call": "EXPECTED",
        },
        # --- Pinned consistency: the value is the sum of the value without
        # local time and the local-time part.
        {
            "setup": """import numpy as np
def gap(fn):
    v = fn(105.0, 100.0, 0.05, 0.2, 1.0, 0.5, 800, 100, 4.0)
    return np.array([v[0] - v[1] - v[2], v[0]])
REF = _oracle_additive_decomposition(105.0, 100.0, 0.05, 0.2, 1.0, 0.5, 800, 100, 4.0)[0]
""",
            "call": "np.round(gap(additive_decomposition) * np.array([1e6, 1.0]), 3) + 0.0",
            "gold_call": "np.round(np.array([0.0, REF]) * np.array([1e6, 1.0]), 3) + 0.0",
        },
        # --- Boundary: a single time step.
        {
            "setup": """import numpy as np
""",
            "call": "additive_decomposition(100.0, 100.0, 0.05, 0.2, 0.25, 0.5, 400, 1, 4.0)",
            "gold_call": "_oracle_additive_decomposition(100.0, 100.0, 0.05, 0.2, 0.25, 0.5, 400, 1, 4.0)",
        },
        # --- Edge: a wide kernel, high volatility, short grid.
        {
            "setup": """import numpy as np
""",
            "call": "additive_decomposition(40.0, 40.0, 0.08, 0.6, 0.5, 2.0, 300, 60, 3.0)",
            "gold_call": "_oracle_additive_decomposition(40.0, 40.0, 0.08, 0.6, 0.5, 2.0, 300, 60, 3.0)",
        },
        # --- Invalid: spot not on the grid ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        additive_decomposition(100.3, 100.0, 0.05, 0.2, 1.0, 0.5, 400, 20, 4.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_additive_decomposition(100.3, 100.0, 0.05, 0.2, 1.0, 0.5, 400, 20, 4.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive kernel width ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        additive_decomposition(100.0, 100.0, 0.05, 0.2, 1.0, -0.5, 400, 20, 4.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_additive_decomposition(100.0, 100.0, 0.05, 0.2, 1.0, -0.5, 400, 20, 4.0)
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
