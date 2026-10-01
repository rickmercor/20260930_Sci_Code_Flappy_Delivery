"""
Recover one capped-claim value from discounted exponential modes.

For $f(u)=\min((u-K)^+,C)$, the damped payoff transform is



$$

\widehat f_\delta(y)=\frac{e^{-\eta K}(1-e^{-\eta C})}{\eta^2},\qquad \eta=\delta+iy.

$$



Interpolate modal values linearly to the initial intensity, then evaluate



$$

V=\frac{e^{\delta u_0}}{\pi}\int_0^{Y_{\max}}

\operatorname{Re}[\widehat f_\delta(y)F(\lambda_0,\delta+iy)e^{iyu_0}]\,dy.

$$



Use composite Simpson weights on an odd number of uniform frequency nodes, including

both endpoints. Modes already include discounting. Return the raw finite-grid value;

no clipping to payoff bounds, additional discount, or extrapolation is applied.

Returns
-------
Finite-grid capped-claim value in currency units.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def invert_capped_payoff(
    grid: np.ndarray,
    modes: np.ndarray,
    frequencies: np.ndarray,
    delta: float,
    initial_intensity: float,
    initial_loss: float,
    strike: float,
    cap: float,
) -> float:
    r"""Recover one capped-claim value from discounted exponential modes.

    Parameters
    ----------
    grid : np.ndarray
        Uniform intensity grid of shape (I+1,), starting at zero.
    modes : np.ndarray
        Finite complex discounted modes of shape (I+1, J).
    frequencies : np.ndarray
        Shape (J,), odd J >= 3, uniform increasing frequencies starting at zero.
    delta : float
        Positive contour shift in inverse loss units.
    initial_intensity : float
        Initial intensity within the grid.
    initial_loss : float
        Nonnegative accumulated loss.
    strike : float
        Nonnegative loss strike.
    cap : float
        Positive payoff cap in loss units.

    Returns
    -------
    price : float
        Finite-grid capped-claim value in currency units.

    Raises
    ------
    ValueError
        If grids, modes, dimensions, parameters or frequency parity are invalid, or the
        inversion is non-finite.

    Notes
    -----
    Inputs are not modified. Numerical comparisons use rtol=1e-9 and atol=1e-11.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_invert_capped_payoff(
    grid: np.ndarray,
    modes: np.ndarray,
    frequencies: np.ndarray,
    delta: float,
    initial_intensity: float,
    initial_loss: float,
    strike: float,
    cap: float,
) -> float:
    grid = _uniform_intensity_grid(grid)
    frequencies = _real_array(frequencies, "frequencies")
    try:
        modes = np.asarray(modes, dtype=complex)
    except (TypeError, ValueError) as exc:
        raise ValueError("modes must be numerical") from exc
    delta = _real_scalar(delta, "delta")
    initial_intensity = _real_scalar(initial_intensity, "initial_intensity")
    initial_loss = _real_scalar(initial_loss, "initial_loss")
    strike = _real_scalar(strike, "strike")
    cap = _real_scalar(cap, "cap")
    if frequencies.ndim != 1 or frequencies.size < 3 or frequencies.size % 2 != 1:
        raise ValueError("an odd frequency count of at least three is required")
    dy = np.diff(frequencies)
    if (
        frequencies[0] != 0
        or np.any(dy <= 0)
        or not np.allclose(dy, dy[0], rtol=1e-12, atol=1e-14)
    ):
        raise ValueError("frequencies must be uniform and start at zero")
    if modes.shape != (grid.size, frequencies.size) or not np.all(np.isfinite(modes)):
        raise ValueError("invalid modes")
    if (
        delta <= 0
        or not 0 <= initial_intensity <= grid[-1]
        or initial_loss < 0
        or strike < 0
        or cap <= 0
    ):
        raise ValueError("invalid payoff or initial state")
    left = min(
        int(np.searchsorted(grid, initial_intensity, side="right") - 1), grid.size - 2
    )
    fraction = (initial_intensity - grid[left]) / (grid[left + 1] - grid[left])
    field = (1 - fraction) * modes[left] + fraction * modes[left + 1]
    eta = delta + 1j * frequencies
    transform = np.exp(-eta * strike) * (-np.expm1(-eta * cap)) / eta**2
    integrand = np.real(transform * field * np.exp(1j * frequencies * initial_loss))
    coefficients = np.ones(frequencies.size)
    coefficients[1:-1:2] = 4
    coefficients[2:-1:2] = 2
    value = (
        np.exp(delta * initial_loss)
        * dy[0]
        / (3 * np.pi)
        * np.dot(coefficients, integrand)
    )
    if not np.isfinite(value):
        raise ValueError("inversion is non-finite")
    return float(value)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid-input cases."""
    return [
        {
            "setup": """import numpy as np
grid = np.array([0.0, 1.0, 2.0])
y = np.linspace(0, 4, 9)
modes = np.exp(-0.1 * y ** 2)[None, :] * np.array([[1.0], [1.2], [1.5]])
delta = 0.3
initial = 0.7
loss = 0.2
strike = 1.0
cap = 2.0
""",
            "call": "invert_capped_payoff(grid.copy(), modes.copy(), y.copy(), delta, initial, loss, strike, cap)",
            "gold_call": "_oracle_invert_capped_payoff(grid.copy(), modes.copy(), y.copy(), delta, initial, loss, strike, cap)",
        },
        {
            "setup": """import numpy as np
grid = np.array([0.0, 1.0, 2.0])
y = np.array([0.0, 0.5, 1.0])
modes = np.ones((3, 3), complex)
delta = 0.5
initial = 2.0
loss = 0.0
strike = 0.0
cap = 0.5
""",
            "call": "invert_capped_payoff(grid.copy(), modes.copy(), y.copy(), delta, initial, loss, strike, cap)",
            "gold_call": "_oracle_invert_capped_payoff(grid.copy(), modes.copy(), y.copy(), delta, initial, loss, strike, cap)",
        },
        {
            "setup": """import numpy as np
grid = np.array([0.0, 1.0, 2.0])
y = np.linspace(0, 6, 13)
modes = np.exp(-0.2 * y ** 2 + 0.1j * y)[None, :] * np.ones((3, 1))
delta = 0.2
initial = 0.0
loss = 1.0
strike = 0.5
cap = 3.0
""",
            "call": "invert_capped_payoff(grid.copy(), modes.copy(), y.copy(), delta, initial, loss, strike, cap)",
            "gold_call": "_oracle_invert_capped_payoff(grid.copy(), modes.copy(), y.copy(), delta, initial, loss, strike, cap)",
        },
        {
            "setup": """import numpy as np
grid = np.array([0.0, 1.0, 2.0])
y = np.linspace(0, 4, 8)
modes = np.ones((3, 8))
delta = 0.3
initial = 0.7
loss = 0.2
strike = 1.0
cap = 2.0

def _model_exception():
    try:
        invert_capped_payoff(grid.copy(), modes.copy(), y.copy(), delta, initial, loss, strike, cap)
        return 0
    except ValueError:
        return 1

def _reference_exception():
    try:
        _oracle_invert_capped_payoff(grid.copy(), modes.copy(), y.copy(), delta, initial, loss, strike, cap)
        return 0
    except ValueError:
        return 1
""",
            "call": "_model_exception()",
            "gold_call": "_reference_exception()",
        },
    ]
