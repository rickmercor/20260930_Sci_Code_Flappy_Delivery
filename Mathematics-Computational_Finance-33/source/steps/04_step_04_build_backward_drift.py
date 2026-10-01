"""
Assemble the banded implicit drift and discount system.

For reverse time $\tau=T-t$, the transport velocity is $-\mu$, where

$\mu_i=\kappa(\bar\lambda-\lambda_i)$ and $\mu_i^\pm$ are its positive and

negative parts. The implicit matrix $A=I-\Delta t L$ therefore has entries



$$

A_{ii}=1+\Delta t\left(r+\frac{|\mu_i|}{h}\right),\quad

A_{i,i-1}=\frac{\Delta t\mu_i^-}{h},\quad

A_{i,i+1}=-\frac{\Delta t\mu_i^+}{h}.

$$



The grid begins at zero and contains the baseline intensity, so boundary drift

points inward or vanishes. In band storage, row zero contains the upper diagonal

in columns 1 onward; row two contains the lower diagonal in columns 0 through

$I-1$. Unused corner entries are zero. Row sums are $1+r\Delta t$.

Returns
-------
Shape (3, I+1), upper/main/lower band storage of the implicit matrix.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_backward_drift(
    grid: np.ndarray,
    market: np.ndarray,
    dt: float,
) -> np.ndarray:
    r"""Assemble the banded implicit drift and discount system.

    Parameters
    ----------
    grid : np.ndarray
        Shape (I+1,), at least three equally spaced increasing nonnegative intensity
        nodes beginning at zero.
    market : np.ndarray
        Shape (5,), kappa, baseline, initial, beta, rate; baseline must lie within the
        grid.
    dt : float
        Positive time step in years.

    Returns
    -------
    bands : np.ndarray
        Shape (3, I+1), upper/main/lower band storage of the implicit matrix.

    Raises
    ------
    ValueError
        If grid is nonuniform or invalid, market is invalid, baseline is outside the
        grid, or dt is not positive and finite.

    Notes
    -----
    Inputs are not modified. Numerical comparisons use rtol=1e-9 and atol=1e-11.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _uniform_intensity_grid(grid):
    grid = _real_array(grid, "grid")
    if grid.ndim != 1 or grid.size < 3 or grid[0] != 0:
        raise ValueError("grid must start at zero and contain at least three nodes")
    differences = np.diff(grid)
    if np.any(differences <= 0) or not np.allclose(
        differences, differences[0], rtol=1e-12, atol=1e-14
    ):
        raise ValueError("grid must be increasing and uniform")
    return grid


def _oracle_build_backward_drift(
    grid: np.ndarray,
    market: np.ndarray,
    dt: float,
) -> np.ndarray:
    grid = _uniform_intensity_grid(grid)
    market = _checked_market(market)
    dt = _real_scalar(dt, "dt")
    if dt <= 0 or market[1] > grid[-1]:
        raise ValueError("positive dt and baseline inside grid required")
    drift = market[0] * (market[1] - grid)
    spacing = grid[1] - grid[0]
    bands = np.zeros((3, grid.size))
    bands[1] = 1 + dt * (np.abs(drift) / spacing + market[4])
    bands[0, 1:] = -dt * np.maximum(drift[:-1], 0) / spacing
    bands[2, :-1] = dt * np.minimum(drift[1:], 0) / spacing
    return bands

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid-input cases."""
    return [
        {
            "setup": """import numpy as np
grid = np.linspace(0, 4, 9)
market = np.array([8.0, 2.0, 2.7, 1.1, 0.02])
dt = 0.01
""",
            "call": "build_backward_drift(grid.copy(), market.copy(), dt)",
            "gold_call": "_oracle_build_backward_drift(grid.copy(), market.copy(), dt)",
        },
        {
            "setup": """import numpy as np
grid = np.array([0.0, 1.0, 2.0])
market = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
dt = 0.1
""",
            "call": "build_backward_drift(grid.copy(), market.copy(), dt)",
            "gold_call": "_oracle_build_backward_drift(grid.copy(), market.copy(), dt)",
        },
        {
            "setup": """import numpy as np
grid = np.array([0.0, 0.5, 1.0])
market = np.array([3.0, 1.0, 0.5, 0.2, 0.1])
dt = 0.02
""",
            "call": "build_backward_drift(grid.copy(), market.copy(), dt)",
            "gold_call": "_oracle_build_backward_drift(grid.copy(), market.copy(), dt)",
        },
        {
            "setup": """import numpy as np
grid = np.array([0.0, 0.5, 1.2])
market = np.array([3.0, 1.0, 0.5, 0.2, 0.1])
dt = 0.02

def _model_exception():
    try:
        build_backward_drift(grid.copy(), market.copy(), dt)
        return 0
    except ValueError:
        return 1

def _reference_exception():
    try:
        _oracle_build_backward_drift(grid.copy(), market.copy(), dt)
        return 0
    except ValueError:
        return 1
""",
            "call": "_model_exception()",
            "gold_call": "_reference_exception()",
        },
    ]
