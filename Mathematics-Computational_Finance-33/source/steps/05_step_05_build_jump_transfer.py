"""
Construct the modal jump-gain matrices with shifted-intensity interpolation.

For modal frequencies $\eta_j=\delta+iy_j$, the gain applied to a field $F$ is



$$

(Q_jF)_i=\sum_{m,q}\omega_{mq}e^{\eta_j x_{mq}}

\Pi[F](\lambda_i+\beta x_{mq}).

$$



Here $\Pi$ is piecewise-linear interpolation, extended constantly beyond the grid.

The resulting matrices include the mark-dependent exponential and state shift;

they exclude multiplication by intensity and exclude the loss term. Real input

coefficients imply conjugate symmetry between positive and negative frequencies.

Weights are positive, with total mass one. Frequencies start at zero and increase.

Returns
-------
Complex array of shape (J, I+1, I+1), the dimensionless modal jump-gain matrices.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_jump_transfer(
    grid: np.ndarray,
    rule: np.ndarray,
    beta: float,
    frequencies: np.ndarray,
    delta: float,
) -> np.ndarray:
    r"""Construct the modal jump-gain matrices with shifted-intensity interpolation.

    Parameters
    ----------
    grid : np.ndarray
        Uniform intensity grid of shape (I+1,), starting at zero.
    rule : np.ndarray
        Shape (M, Q, 2), positive mark nodes and probability weights summing to one.
    beta : float
        Nonnegative excitation coefficient in intensity per loss unit.
    frequencies : np.ndarray
        Shape (J,), at least one nonnegative increasing frequency, first entry zero, in
        inverse loss units.
    delta : float
        Nonnegative finite contour shift in inverse loss units; quadrature exponentials
        must remain finite.

    Returns
    -------
    transfer : np.ndarray
        Complex array of shape (J, I+1, I+1), the dimensionless modal jump-gain matrices.

    Raises
    ------
    ValueError
        If grid, rule, frequency ordering, beta or delta is invalid, weights are
        unnormalized, or the modal factors overflow.

    Notes
    -----
    Inputs are not modified. Numerical comparisons use rtol=1e-9 and atol=1e-11.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_jump_transfer(
    grid: np.ndarray,
    rule: np.ndarray,
    beta: float,
    frequencies: np.ndarray,
    delta: float,
) -> np.ndarray:
    grid = _uniform_intensity_grid(grid)
    rule = _real_array(rule, "rule")
    beta = _real_scalar(beta, "beta")
    delta = _real_scalar(delta, "delta")
    frequencies = _real_array(frequencies, "frequencies")
    if (
        rule.ndim != 3
        or rule.shape[2] != 2
        or min(rule.shape[:2]) < 1
        or np.any(rule <= 0)
    ):
        raise ValueError("rule must contain positive nodes and weights")
    if abs(np.sum(rule[:, :, 1]) - 1) > 1e-12:
        raise ValueError("quadrature weights must sum to one")
    if beta < 0 or delta < 0 or frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("invalid excitation, shift, or frequencies")
    if frequencies[0] != 0 or np.any(np.diff(frequencies) <= 0):
        raise ValueError("frequencies must increase from zero")
    nodes = rule[:, :, 0].ravel()
    weights = rule[:, :, 1].ravel()
    if np.max(delta * nodes) > 700:
        raise ValueError("modal exponent exceeds supported finite range")
    eta = delta + 1j * frequencies
    result = np.zeros((frequencies.size, grid.size, grid.size), dtype=complex)
    rows = np.arange(grid.size)
    spacing = grid[1] - grid[0]
    for node, weight in zip(nodes, weights):
        query = np.minimum(grid + beta * node, grid[-1])
        left = np.minimum(np.floor(query / spacing).astype(int), grid.size - 2)
        fraction = (query - grid[left]) / spacing
        factor = weight * np.exp(eta * node)
        result[:, rows, left] += factor[:, None] * (1 - fraction)
        result[:, rows, left + 1] += factor[:, None] * fraction
    if not np.all(np.isfinite(result)):
        raise ValueError("non-finite modal transfer")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid-input cases."""
    return [
        {
            "setup": """import numpy as np
grid = np.linspace(0, 3, 7)
rule = np.array([[[0.4, 0.3], [1.3, 0.7]]])
beta = 0.8
y = np.array([0.0, 0.7, 2.0])
delta = 0.2
""",
            "call": "build_jump_transfer(grid.copy(), rule.copy(), beta, y.copy(), delta)",
            "gold_call": "_oracle_build_jump_transfer(grid.copy(), rule.copy(), beta, y.copy(), delta)",
        },
        {
            "setup": """import numpy as np
grid = np.array([0.0, 1.0, 2.0])
rule = np.array([[[0.5, 1.0]]])
beta = 0.0
y = np.array([0.0])
delta = 0.0
""",
            "call": "build_jump_transfer(grid.copy(), rule.copy(), beta, y.copy(), delta)",
            "gold_call": "_oracle_build_jump_transfer(grid.copy(), rule.copy(), beta, y.copy(), delta)",
        },
        {
            "setup": """import numpy as np
grid = np.array([0.0, 1.0, 2.0])
rule = np.array([[[3.0, 1.0]]])
beta = 2.0
y = np.array([0.0, 1.0])
delta = 0.1
""",
            "call": "build_jump_transfer(grid.copy(), rule.copy(), beta, y.copy(), delta)",
            "gold_call": "_oracle_build_jump_transfer(grid.copy(), rule.copy(), beta, y.copy(), delta)",
        },
        {
            "setup": """import numpy as np
grid = np.array([0.0, 1.0, 2.0])
rule = np.array([[[0.5, 1.0]]])
beta = -1.0
y = np.array([0.0])
delta = 0.1

def _model_exception():
    try:
        build_jump_transfer(grid.copy(), rule.copy(), beta, y.copy(), delta)
        return 0
    except ValueError:
        return 1

def _reference_exception():
    try:
        _oracle_build_jump_transfer(grid.copy(), rule.copy(), beta, y.copy(), delta)
        return 0
    except ValueError:
        return 1
""",
            "call": "_model_exception()",
            "gold_call": "_reference_exception()",
        },
    ]
