"""
Reduce a stationary wealth distribution to the four scalars the equilibrium condition needs.

The distribution is supplied on a uniform grid running from the borrowing limit to the upper truncation point, with one row per node, node zero at the borrowing limit, and two columns holding the two income states in the order low income then high income. It is a density, so the sum of all its entries multiplied by the grid spacing is one.

Return a 1-D array of length 4, in this fixed order:

  index 0  aggregate capital, the mean wealth under the distribution taken over all nodes and both income states

  index 1  aggregate labour supply, the mean income under the distribution

  index 2  the weight the distribution places on the borrowing limit in the low income state

  index 3  the weight the distribution places on the borrowing limit in the high income state

A weight at a node is the entry of the density at that node multiplied by the grid spacing.

Inputs are the distribution, the two income levels, the borrowing limit and the upper truncation point.

Raises ValueError if the distribution is not a two-dimensional finite array with exactly two columns and at least two rows, if any of its entries is negative, if any real input is not finite, if the income levels do not satisfy higher income greater than lower income greater than zero, if the upper truncation point does not exceed the borrowing limit, or if the total mass of the distribution differs from one by more than one part in a hundred million.

The two aggregates that close a heterogeneous-agent model behave very differently. Aggregate capital is a genuine functional of the whole wealth distribution: it responds to every feature of the saving policy and is the channel through which individual optimisation feeds back into the price. Aggregate labour supply is not. Because income follows a two-state process that is independent of wealth, the population share in each income state is fixed by the switching rates alone, so aggregate labour is a weighted average of the two income levels with weights that no policy, grid or step length can change. Computing it by quadrature against the distribution and computing it from the switching rates in closed form give the same number, and a discrepancy between them signals an error in the forward operator rather than in the economics.

That weighting is easy to get wrong in a way that is not obviously wrong. Taking the plain average of the two income levels is correct only when the two switching rates coincide; with asymmetric rates the state that is harder to leave carries the larger share, and using the plain average shifts the equilibrium interest rate noticeably.

Aggregate capital raises a different question, which is what to do about the mass sitting exactly at the borrowing limit. The stationary measure of a constrained problem is not absolutely continuous: it has a density away from the constraint and a point mass at it. The mean of that measure includes the point mass weighted by the wealth level at which it sits, which is negative when agents are permitted to borrow. Treating the distribution as if it were a density over the interior alone drops that contribution and biases capital upward, which then biases the equilibrium interest rate downward. Reporting the two components of the point mass separately, rather than only their sum, also exposes how the constrained population splits between the two income states, which is the economically interesting margin: the constraint binds overwhelmingly for the low-income group.

Returns
-------
np.ndarray, shape (4,) of dtype float, holding aggregate capital, aggregate labour supply, the weight at the borrowing limit in the low income state, and the weight at the borrowing limit in the high income state, in that order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_aggregates(G: np.ndarray, y1: float, y2: float, xlow: float,
                       xbar: float) -> np.ndarray:
    '''Return aggregate capital, aggregate labour and the two weights at the borrowing limit.

    Parameters
    ----------
    G : np.ndarray
        Shape (n, 2) stationary distribution as a density on the wealth grid, columns being
        the low and high income states.
    y1 : float
        Lower income level.
    y2 : float
        Higher income level.
    xlow : float
        Borrowing limit, the first grid node.
    xbar : float
        Upper truncation point, the last grid node.

    Returns
    -------
    out : np.ndarray
        Shape (4,) float array holding aggregate capital, aggregate labour supply, the
        weight at the borrowing limit in the low income state, and the weight at the
        borrowing limit in the high income state, in that order.
    '''
    return out

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_aggregates(G: np.ndarray, y1: float, y2: float, xlow: float,
                               xbar: float) -> np.ndarray:
    G = np.asarray(G, dtype=float)
    if G.ndim != 2 or G.shape[1] != 2 or G.shape[0] < 2:
        raise ValueError("G must be a 2-D array with two columns and at least two rows")
    if not np.all(np.isfinite(G)):
        raise ValueError("G must be finite")
    if np.any(G < 0.0):
        raise ValueError("G must be non-negative")
    for name, val in (("y1", y1), ("y2", y2), ("xlow", xlow), ("xbar", xbar)):
        if isinstance(val, (bool, np.bool_)) or not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(name + " must be a real scalar")
        if not np.isfinite(float(val)):
            raise ValueError(name + " must be finite")
    y1, y2, xlow, xbar = float(y1), float(y2), float(xlow), float(xbar)
    if not (y2 > y1 > 0.0):
        raise ValueError("the ordering y2 > y1 > 0 must hold")
    if not xbar > xlow:
        raise ValueError("xbar must exceed xlow")

    n = G.shape[0]
    dx = (xbar - xlow) / (n - 1)
    total = float(G.sum() * dx)
    if abs(total - 1.0) > 1e-8:
        raise ValueError("the distribution must have total mass one")

    x = xlow + dx * np.arange(n)
    K = float(np.sum(x[:, None] * G) * dx)
    shares = G.sum(axis=0) * dx
    N = float(y1 * shares[0] + y2 * shares[1])
    return np.array([K, N, float(G[0, 0] * dx), float(G[0, 1] * dx)], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
xlow, xbar, nint = -0.25, 24.75, 1250
dx = (xbar - xlow) / nint
x = xlow + dx * np.arange(nint + 1)
dens = np.column_stack([np.exp(-1.1 * (x - xlow)), 0.6 * np.exp(-0.55 * (x - xlow))])
dens[0, 0] += 2.0
dens[0, 1] += 0.5
G = dens / (dens.sum() * dx)
""",
            "call": "compute_aggregates(G, 0.4, 1.2, xlow, xbar)",
            "gold_call": "_oracle_compute_aggregates(G, 0.4, 1.2, xlow, xbar)",
        },
        {
            "setup": """import numpy as np
xlow, xbar, nint = -0.15, 5.85, 60
dx = (xbar - xlow) / nint
G = np.zeros((nint + 1, 2))
G[0, 0] = 1.0
G[0, 1] = 1.0
G = G / (G.sum() * dx)
""",
            "call": "compute_aggregates(G, 0.5, 1.5, xlow, xbar)",
            "gold_call": "_oracle_compute_aggregates(G, 0.5, 1.5, xlow, xbar)",
        },
        {
            "setup": """import numpy as np
xlow, xbar = 0.0, 1.0
dx = 1.0
G = np.array([[0.375, 0.0], [0.0, 0.625]])
""",
            "call": "compute_aggregates(G, 0.4, 1.2, xlow, xbar)",
            "gold_call": "_oracle_compute_aggregates(G, 0.4, 1.2, xlow, xbar)",
        },
        {
            "setup": """import numpy as np
xlow, xbar, nint = -2.0, 40.0, 210
dx = (xbar - xlow) / nint
x = xlow + dx * np.arange(nint + 1)
dens = np.column_stack([np.ones_like(x), np.zeros_like(x)])
dens[-1, 1] = 5.0
G = dens / (dens.sum() * dx)
""",
            "call": "compute_aggregates(G, 0.4, 1.2, xlow, xbar)",
            "gold_call": "_oracle_compute_aggregates(G, 0.4, 1.2, xlow, xbar)",
        },
        {
            "setup": """import numpy as np
G = np.ones((5, 2))
def run_model():
    try:
        compute_aggregates(G, 0.4, 1.2, -0.25, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_aggregates(G, 0.4, 1.2, -0.25, 1.0)
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
G = np.zeros((5, 2))
G[0, 0] = 1.0
G = G / (G.sum() * 0.25)
G[3, 1] = -1e-6
def run_model():
    try:
        compute_aggregates(G, 0.4, 1.2, -0.25, 0.75)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_aggregates(G, 0.4, 1.2, -0.25, 0.75)
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
G = np.zeros((5, 2))
G[0, 0] = 4.0
def run_model():
    try:
        compute_aggregates(G, 1.2, 0.4, -0.25, 0.75)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_aggregates(G, 1.2, 0.4, -0.25, 0.75)
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
G = np.zeros((5, 2))
G[0, 0] = 4.0
def run_model():
    try:
        compute_aggregates(G, 0.4, 1.2, 0.75, 0.75)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_aggregates(G, 0.4, 1.2, 0.75, 0.75)
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
