"""
Propagate terminal exponential modes to time zero.

Terminal modal values are one at every intensity. The backward update is



$$

A F^n=F^{n+1}+\Delta t\,\lambda\odot(QF^{n+1}-F^{n+1}).

$$



The same implicit matrix is used for every frequency and time step. Multiplication

by intensity acts rowwise. Require the computable explicit-jump factor



$$

c_\Delta=\Delta t\max_{j,i}\lambda_i\left(1+\sum_\ell |Q_{j,i\ell}|\right)<1.

$$



For transfers constructed from positive weights with the zero frequency first,

the maximum is attained there and equals the quadrature exponential-moment bound.

At zero contour and frequency, the jump part annihilates constants; the discrete

discount factor is $(1+r\Delta t)^{-N}$.

Returns
-------
Complex array of shape (I+1, J), time-zero discounted exponential modes.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def evolve_modal_values(
    grid: np.ndarray,
    bands: np.ndarray,
    transfer: np.ndarray,
    dt: float,
    n_steps: int,
) -> np.ndarray:
    r"""Propagate terminal exponential modes to time zero.

    Parameters
    ----------
    grid : np.ndarray
        Uniform intensity grid of shape (I+1,), beginning at zero.
    bands : np.ndarray
        Shape (3, I+1), real implicit matrix with nonpositive off diagonals, positive
        row-dominance margin and zero unused corners.
    transfer : np.ndarray
        Finite complex gain array of shape (J, I+1, I+1), J >= 1.
    dt : float
        Positive finite time step in years.
    n_steps : int
        Positive number of backward steps.

    Returns
    -------
    modes : np.ndarray
        Complex array of shape (I+1, J), time-zero discounted exponential modes.

    Raises
    ------
    ValueError
        If dimensions, signs, row dominance, finite data or time parameters are invalid,
        the explicit-jump factor is at least one, or evolution becomes non-finite.

    Notes
    -----
    Inputs are not modified. Numerical comparisons use rtol=1e-9 and atol=1e-11.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import solve_banded


def _oracle_evolve_modal_values(
    grid: np.ndarray,
    bands: np.ndarray,
    transfer: np.ndarray,
    dt: float,
    n_steps: int,
) -> np.ndarray:
    grid = _uniform_intensity_grid(grid)
    bands = _real_array(bands, "bands")
    try:
        transfer = np.asarray(transfer, dtype=complex)
    except (TypeError, ValueError) as exc:
        raise ValueError("transfer must be numerical") from exc
    dt = _real_scalar(dt, "dt")
    n_steps = _positive_integer(n_steps, "n_steps")
    size = grid.size
    if (
        bands.shape != (3, size)
        or transfer.ndim != 3
        or transfer.shape[0] < 1
        or transfer.shape[1:] != (size, size)
    ):
        raise ValueError("incompatible modal dimensions")
    if dt <= 0 or not np.all(np.isfinite(transfer)) or np.any(bands[[0, 2]] > 0):
        raise ValueError("invalid matrix, transfer, or time step")
    if bands[0, 0] != 0 or bands[2, -1] != 0:
        raise ValueError("unused band corners must be zero")
    row_sum = bands[1].copy()
    row_sum[:-1] += bands[0, 1:]
    row_sum[1:] += bands[2, :-1]
    if np.any(row_sum <= 0):
        raise ValueError("implicit matrix must be strictly row dominant")
    cfl = dt * np.max(grid[None, :] * (1 + np.sum(np.abs(transfer), axis=2)))
    if cfl >= 1:
        raise ValueError("explicit jump stability factor must be below one")
    values = np.ones((size, transfer.shape[0]), dtype=complex)
    for _ in range(n_steps):
        gain = np.einsum("fij,jf->if", transfer, values, optimize=False)
        rhs = values + dt * grid[:, None] * (gain - values)
        values = solve_banded((1, 1), bands, rhs, check_finite=False)
    if not np.all(np.isfinite(values)):
        raise ValueError("modal evolution is non-finite")
    return values

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid-input cases."""
    return [
        {
            "setup": """import numpy as np
from scipy.linalg import solve_banded
grid = np.array([0.0, 1.0, 2.0])
dt = 0.03
bands = np.array([[0.0, -0.03, 0.0], [1.03, 1.0, 1.03], [0.0, -0.03, 0.0]])
transfer = np.array([np.eye(3) * 1.2, np.eye(3) * (0.9 + 0.2j)])
steps = 3
""",
            "call": "evolve_modal_values(grid.copy(), bands.copy(), transfer.copy(), dt, steps)",
            "gold_call": "_oracle_evolve_modal_values(grid.copy(), bands.copy(), transfer.copy(), dt, steps)",
        },
        {
            "setup": """import numpy as np
from scipy.linalg import solve_banded
grid = np.array([0.0, 1.0, 2.0])
dt = 0.1
bands = np.zeros((3, 3))
bands[1] = 1.02
transfer = np.eye(3)[None, :, :].astype(complex)
steps = 4
""",
            "call": "evolve_modal_values(grid.copy(), bands.copy(), transfer.copy(), dt, steps)",
            "gold_call": "_oracle_evolve_modal_values(grid.copy(), bands.copy(), transfer.copy(), dt, steps)",
        },
        {
            "setup": """import numpy as np
from scipy.linalg import solve_banded
grid = np.array([0.0, 0.5, 1.0])
dt = 0.01
bands = np.zeros((3, 3))
bands[1] = 1.0
transfer = np.array([[[0.0, 0.2, 1.1], [0.0, 0.0, 1.3], [0.0, 0.0, 1.3]]], complex)
steps = 1
""",
            "call": "evolve_modal_values(grid.copy(), bands.copy(), transfer.copy(), dt, steps)",
            "gold_call": "_oracle_evolve_modal_values(grid.copy(), bands.copy(), transfer.copy(), dt, steps)",
        },
        {
            "setup": """import numpy as np
from scipy.linalg import solve_banded
grid = np.array([0.0, 1.0, 2.0])
dt = 1.0
bands = np.zeros((3, 3))
bands[1] = 1.0
transfer = np.eye(3)[None, :, :]
steps = 1

def _model_exception():
    try:
        evolve_modal_values(grid.copy(), bands.copy(), transfer.copy(), dt, steps)
        return 0
    except ValueError:
        return 1

def _reference_exception():
    try:
        _oracle_evolve_modal_values(grid.copy(), bands.copy(), transfer.copy(), dt, steps)
        return 0
    except ValueError:
        return 1
""",
            "call": "_model_exception()",
            "gold_call": "_reference_exception()",
        },
    ]
