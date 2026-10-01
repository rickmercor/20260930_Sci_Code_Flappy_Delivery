"""
Build the two discrete grid functions that bracket any solution of the discretised Hamilton-Jacobi-Bellman equation.

The grid is uniform on the closed interval from the borrowing limit to the upper truncation point and has nint intervals, hence nint plus one nodes, with node zero at the borrowing limit.

Return a single 2-D array of shape (2 * (nint + 1), 2) obtained by stacking the two grid functions vertically. Rows 0 to nint hold the subsolution and rows nint + 1 to 2 * nint + 1 hold the supersolution. The two columns are the two income states, in the order low income then high income. Both grid functions take the same value in the two income states, so the two columns of each block are identical; the shape is kept two-dimensional so that the result lines up with the value function on the same grid.

The subsolution is built from the wealth-plus-lower-income flow. The supersolution is built from wealth measured in units that include the present value of the higher income stream, scaled by the barrier constant supplied as an argument. Both are ordinary constant-relative risk aversion expressions in those arguments.

Inputs are the risk aversion, the interest rate, the two income levels, the barrier constant, the borrowing limit, the upper truncation point and the number of intervals.

Raises ValueError if any real input is not finite, if nint is not a positive integer, if the upper truncation point does not exceed the borrowing limit, if risk aversion is not greater than one, if the interest rate or the barrier constant is not strictly positive, if the income levels do not satisfy higher income greater than lower income greater than zero, if the interest income plus lower income is not strictly positive at every node, if wealth plus the present value of the higher income stream is not strictly positive at every node, or if the resulting subsolution fails to lie at or below the resulting supersolution at every node.

Monotone schemes for state-constrained Hamilton-Jacobi-Bellman equations are analysed by exhibiting a discrete subsolution and a discrete supersolution and appealing to a discrete comparison principle. Any solution of the scheme is then trapped between them, which delivers a uniform bound independent of the mesh and is the first ingredient in proving that the iterative solver converges. In the recursive-utility setting the bound does more work than usual: the aggregator carries a fractional power of the value function, so the iteration is only well defined while the value function stays on the correct side of zero, and the barriers are what guarantee that.

The subsolution corresponds economically to an agent who consumes exactly the flow available without saving, evaluated at the lower of the two income levels, and it is a subsolution precisely because that policy is feasible everywhere including at the constraint. The supersolution corresponds to the complete-market benchmark, in which the agent may borrow against the entire future stream of the higher income and therefore does at least as well as in the incomplete-market problem; the constant scaling wealth in that expression is what encodes the discounting of the future stream and lies strictly between the interest rate and the discount rate.

Both barriers are the same in the two income states. That is not an approximation: the subsolution is deliberately built from the lower income so that it remains a subsolution for the high-income agent as well, and the supersolution from the higher income so that it remains a supersolution for the low-income agent. The ordering between the two, which holds under the standing assumptions when the interest rate is positive, is itself a consequence of the comparison principle and is worth checking numerically rather than assuming, since a mis-specified barrier constant breaks it silently.

Returns
-------
np.ndarray, shape (2 * (nint + 1), 2) of dtype float, the discrete subsolution stacked above the discrete supersolution, each replicated across the two income-state columns
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_discrete_barriers(gamma: float, r: float, y1: float, y2: float, b: float,
                              xlow: float, xbar: float, nint: int) -> np.ndarray:
    '''Return the stacked discrete sub- and supersolution grid functions.

    Parameters
    ----------
    gamma : float
        Risk aversion, strictly greater than one.
    r : float
        Interest rate, strictly positive.
    y1 : float
        Lower income level, strictly positive.
    y2 : float
        Higher income level, strictly greater than the lower income level.
    b : float
        Barrier constant scaling wealth in the supersolution.
    xlow : float
        Borrowing limit, the first grid node.
    xbar : float
        Upper truncation point, the last grid node.
    nint : int
        Number of grid intervals, strictly positive.

    Returns
    -------
    out : np.ndarray
        Shape (2 * (nint + 1), 2) float array holding the subsolution in its first
        nint + 1 rows and the supersolution in its last nint + 1 rows, with the two
        columns being the low and high income states.
    '''
    return out

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_discrete_barriers(gamma: float, r: float, y1: float, y2: float, b: float,
                                      xlow: float, xbar: float, nint: int) -> np.ndarray:
    for name, val in (("gamma", gamma), ("r", r), ("y1", y1), ("y2", y2),
                      ("b", b), ("xlow", xlow), ("xbar", xbar)):
        if isinstance(val, (bool, np.bool_)) or not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(name + " must be a real scalar")
        if not np.isfinite(float(val)):
            raise ValueError(name + " must be finite")
    if isinstance(nint, (bool, np.bool_)) or not isinstance(nint, (int, np.integer)):
        raise ValueError("nint must be an integer")
    nint = int(nint)
    if nint < 1:
        raise ValueError("nint must be a positive integer")
    gamma, r, y1, y2, b = float(gamma), float(r), float(y1), float(y2), float(b)
    xlow, xbar = float(xlow), float(xbar)
    if not gamma > 1.0:
        raise ValueError("gamma must be strictly greater than 1")
    if not r > 0.0:
        raise ValueError("r must be strictly positive")
    if not b > 0.0:
        raise ValueError("b must be strictly positive")
    if not (y2 > y1 > 0.0):
        raise ValueError("the ordering y2 > y1 > 0 must hold")
    if not xbar > xlow:
        raise ValueError("xbar must exceed xlow")

    dx = (xbar - xlow) / nint
    x = xlow + dx * np.arange(nint + 1)
    low_flow = r * x + y1
    if not np.all(low_flow > 0.0):
        raise ValueError("r*x + y1 must be strictly positive at every grid node")
    wide = x + y2 / r
    if not np.all(wide > 0.0):
        raise ValueError("x + y2/r must be strictly positive at every grid node")

    p = 1.0 - gamma
    lower = low_flow ** p / p
    upper = (b * wide) ** p / p
    if not np.all(lower <= upper + 1e-14 * np.abs(upper)):
        raise ValueError("the subsolution must lie at or below the supersolution")

    block_low = np.repeat(lower[:, None], 2, axis=1)
    block_up = np.repeat(upper[:, None], 2, axis=1)
    return np.vstack([block_low, block_up]).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
gamma, r, y1, y2 = 1.5, 0.021410049888, 0.4, 1.2
b = 0.024986667779
xlow, xbar, nint = -0.25, 24.75, 1250
""",
            "call": "compute_discrete_barriers(gamma, r, y1, y2, b, xlow, xbar, nint)",
            "gold_call": "_oracle_compute_discrete_barriers(gamma, r, y1, y2, b, xlow, xbar, nint)",
        },
        {
            "setup": """import numpy as np
gamma, r, y1, y2 = 3.0, 0.02, 0.5, 1.5
b = 0.024
xlow, xbar, nint = -0.15, 5.85, 8
""",
            "call": "compute_discrete_barriers(gamma, r, y1, y2, b, xlow, xbar, nint)",
            "gold_call": "_oracle_compute_discrete_barriers(gamma, r, y1, y2, b, xlow, xbar, nint)",
        },
        {
            "setup": """import numpy as np
gamma, r, y1, y2 = 1.5, 0.03, 0.4, 1.2
b = 0.035
xlow, xbar, nint = 0.0, 1.0, 1
""",
            "call": "compute_discrete_barriers(gamma, r, y1, y2, b, xlow, xbar, nint)",
            "gold_call": "_oracle_compute_discrete_barriers(gamma, r, y1, y2, b, xlow, xbar, nint)",
        },
        {
            "setup": """import numpy as np
gamma, r, y1, y2 = 1.05, 0.001, 0.9, 0.95
b = 0.0011
xlow, xbar, nint = -800.0, 200.0, 40
""",
            "call": "compute_discrete_barriers(gamma, r, y1, y2, b, xlow, xbar, nint)",
            "gold_call": "_oracle_compute_discrete_barriers(gamma, r, y1, y2, b, xlow, xbar, nint)",
        },
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_discrete_barriers(1.5, 0.02, 0.4, 1.2, 0.025, -30.0, 10.0, 20)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_discrete_barriers(1.5, 0.02, 0.4, 1.2, 0.025, -30.0, 10.0, 20)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_discrete_barriers(1.5, 0.02, 0.4, 1.2, 0.005, -0.25, 24.75, 50)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_discrete_barriers(1.5, 0.02, 0.4, 1.2, 0.005, -0.25, 24.75, 50)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_discrete_barriers(1.5, 0.02, 1.2, 0.4, 0.025, -0.25, 24.75, 50)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_discrete_barriers(1.5, 0.02, 1.2, 0.4, 0.025, -0.25, 24.75, 50)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_discrete_barriers(1.5, 0.02, 0.4, 1.2, 0.025, -0.25, 24.75, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_discrete_barriers(1.5, 0.02, 0.4, 1.2, 0.025, -0.25, 24.75, 0)
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
