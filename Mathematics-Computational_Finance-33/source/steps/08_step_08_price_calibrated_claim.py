"""
Calibrate the mark law and return the complete finite-grid capped-claim price.

Calibration uses the untruncated intensity mean, while claim valuation uses the

prescribed finite intensity grid. The same calibrated mark distribution enters

both the jump exponential and the post-jump intensity shift. For



$$

\mathrm{resolution}=(\lambda_{\max},I,N,Q,\delta,Y_{\max},N_y),

$$



construct intensity nodes $i\lambda_{\max}/I$, time step $T/N$, and frequencies

$jY_{\max}/N_y$ for $j=0,\ldots,N_y$. Here $N_y$ is even. The returned number is the raw finite-grid price.

Positive arrival activity is required for calibration.

Require $\delta<\min_m(b_m-\theta)$ and the explicit-jump bound below one.

All computations are deterministic; no simulated paths, seed, or external data are used.

Returns
-------
One finite capped-claim price in currency units.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def price_calibrated_claim(
    probabilities: np.ndarray,
    shapes: np.ndarray,
    rates: np.ndarray,
    market: np.ndarray,
    swap: np.ndarray,
    contract: np.ndarray,
    resolution: np.ndarray,
) -> float:
    r"""Calibrate the mark law and return the complete finite-grid capped-claim price.

    Parameters
    ----------
    probabilities : np.ndarray
        Shape (M,), normalized positive original component probabilities.
    shapes : np.ndarray
        Shape (M,), positive original Gamma shapes.
    rates : np.ndarray
        Shape (M,), positive original Gamma rates.
    market : np.ndarray
        Shape (5,), ordered kappa, baseline intensity, initial intensity, excitation,
        interest rate; kappa > 0, others nonnegative, and baseline plus initial
        intensity positive.
    swap : np.ndarray
        Shape (4,), positive settlement time, nonnegative quote, lower tilt, upper tilt;
        admissible subcritical increasing bracket.
    contract : np.ndarray
        Shape (4,), positive claim maturity, nonnegative initial loss, nonnegative
        strike, positive cap.
    resolution : np.ndarray
        Shape (7,), intensity maximum, integer interval count >= 2, integer time-step
        count >= 1, integer quadrature order 1..128, positive contour shift, positive
        frequency maximum, even integer frequency interval count >= 2.

    Returns
    -------
    price : float
        One finite capped-claim price in currency units.

    Raises
    ------
    ValueError
        If any component contract fails, resolution counts are nonintegral or out of
        range, the calibration quote is not bracketed, initial or baseline intensity is
        outside the grid, the contour is inadmissible, or the explicit-jump stability
        bound is violated.

    Notes
    -----
    Inputs are not modified. Numerical comparisons use rtol=1e-9 and atol=1e-11.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_price_calibrated_claim(
    probabilities: np.ndarray,
    shapes: np.ndarray,
    rates: np.ndarray,
    market: np.ndarray,
    swap: np.ndarray,
    contract: np.ndarray,
    resolution: np.ndarray,
) -> float:
    market = _checked_market(market)
    swap = _real_array(swap, "swap")
    contract = _real_array(contract, "contract")
    resolution = _real_array(resolution, "resolution")
    if swap.shape != (4,) or contract.shape != (4,) or resolution.shape != (7,):
        raise ValueError("swap, contract, and resolution have invalid shapes")
    if contract[0] <= 0 or contract[1] < 0 or contract[2] < 0 or contract[3] <= 0:
        raise ValueError("invalid claim contract")
    maximum, intervals, n_steps, order, delta, ymax, freq_intervals = resolution
    counts = (intervals, n_steps, order, freq_intervals)
    if any(value != int(value) or value < 1 for value in counts):
        raise ValueError("resolution counts must be positive integers")
    intervals, n_steps, order, freq_intervals = map(int, counts)
    if intervals < 2 or freq_intervals < 2 or freq_intervals % 2 or order > 128:
        raise ValueError("unsupported grid resolution")
    if maximum <= 0 or ymax <= 0 or delta <= 0 or max(market[1:3]) > maximum:
        raise ValueError("invalid contour or intensity domain")
    theta = _oracle_calibrate_mark_tilt(
        probabilities, shapes, rates, market, swap[0], swap[1], swap[2:]
    )
    mixture = _oracle_tilt_mark_distribution(probabilities, shapes, rates, theta)
    if delta >= np.min(mixture[:, 2]):
        raise ValueError("contour exceeds tilted mark moment domain")
    rule = _oracle_build_mark_quadrature(mixture, order)
    grid = np.linspace(0, maximum, intervals + 1)
    frequencies = np.linspace(0, ymax, freq_intervals + 1)
    dt = contract[0] / n_steps
    bands = _oracle_build_backward_drift(grid, market, dt)
    transfer = _oracle_build_jump_transfer(grid, rule, market[3], frequencies, delta)
    modes = _oracle_evolve_modal_values(grid, bands, transfer, dt, n_steps)
    return _oracle_invert_capped_payoff(
        grid,
        modes,
        frequencies,
        delta,
        market[2],
        contract[1],
        contract[2],
        contract[3],
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid-input cases."""
    return [
        {
            "setup": """from scipy.optimize import brentq
from scipy.special import gammaln, roots_genlaguerre
from scipy.linalg import solve_banded
import numpy as np
p = np.array([0.6, 0.4])
k = np.array([2.0, 6.0])
b = np.array([4.0, 2.5])
market = np.array([8.0, 2.0, 2.7, 1.1, 0.02])
swap = np.array([0.5, 2.2, 0.0, 0.35])
contract = np.array([150 / 365, 0.15, 1.2, 3.0])
resolution = np.array([24.0, 96, 256, 48, 0.3, 24.0, 192])
""",
            "call": "price_calibrated_claim(p.copy(), k.copy(), b.copy(), market.copy(), swap.copy(), contract.copy(), resolution.copy())",
            "gold_call": "_oracle_price_calibrated_claim(p.copy(), k.copy(), b.copy(), market.copy(), swap.copy(), contract.copy(), resolution.copy())",
        },
        {
            "setup": """from scipy.optimize import brentq
from scipy.special import gammaln, roots_genlaguerre
from scipy.linalg import solve_banded
import numpy as np
p = np.array([0.6, 0.4])
k = np.array([2.0, 6.0])
b = np.array([4.0, 2.5])
market = np.array([8.0, 2.0, 2.7, 1.1, 0.02])
swap = np.array([0.5, 2.2, 0.0, 0.35])
contract = np.array([150 / 365, 0.15, 1.2, 3.0])
resolution = np.array([24.0, 96, 256, 48, 0.3, 24.0, 192])
swap[1] = 1.8
contract = np.array([0.2, 0.0, 0.8, 2.0])
resolution = np.array([16.0, 32, 96, 16, 0.25, 12.0, 48])
""",
            "call": "price_calibrated_claim(p.copy(), k.copy(), b.copy(), market.copy(), swap.copy(), contract.copy(), resolution.copy())",
            "gold_call": "_oracle_price_calibrated_claim(p.copy(), k.copy(), b.copy(), market.copy(), swap.copy(), contract.copy(), resolution.copy())",
        },
        {
            "setup": """from scipy.optimize import brentq
from scipy.special import gammaln, roots_genlaguerre
from scipy.linalg import solve_banded
import numpy as np
p = np.array([0.6, 0.4])
k = np.array([2.0, 6.0])
b = np.array([4.0, 2.5])
market = np.array([8.0, 2.0, 2.7, 1.1, 0.02])
swap = np.array([0.5, 2.2, 0.0, 0.35])
contract = np.array([150 / 365, 0.15, 1.2, 3.0])
resolution = np.array([24.0, 96, 256, 48, 0.3, 24.0, 192])
market = np.array([6.0, 1.5, 2.0, 0.4, 0.03])
swap = np.array([0.3, 0.75, -0.2, 0.6])
contract = np.array([0.25, 0.3, 1.0, 1.5])
resolution = np.array([16.0, 32, 128, 20, 0.2, 16.0, 64])
""",
            "call": "price_calibrated_claim(p.copy(), k.copy(), b.copy(), market.copy(), swap.copy(), contract.copy(), resolution.copy())",
            "gold_call": "_oracle_price_calibrated_claim(p.copy(), k.copy(), b.copy(), market.copy(), swap.copy(), contract.copy(), resolution.copy())",
        },
        {
            "setup": """from scipy.optimize import brentq
from scipy.special import gammaln, roots_genlaguerre
from scipy.linalg import solve_banded
import numpy as np
p = np.array([0.6, 0.4])
k = np.array([2.0, 6.0])
b = np.array([4.0, 2.5])
market = np.array([8.0, 2.0, 2.7, 1.1, 0.02])
swap = np.array([0.5, 2.2, 0.0, 0.35])
contract = np.array([150 / 365, 0.15, 1.2, 3.0])
resolution = np.array([24.0, 96, 256, 48, 0.3, 24.0, 192])
resolution[6] = 47

def _model_exception():
    try:
        price_calibrated_claim(p.copy(), k.copy(), b.copy(), market.copy(), swap.copy(), contract.copy(), resolution.copy())
        return 0
    except ValueError:
        return 1

def _reference_exception():
    try:
        _oracle_price_calibrated_claim(p.copy(), k.copy(), b.copy(), market.copy(), swap.copy(), contract.copy(), resolution.copy())
        return 0
    except ValueError:
        return 1
""",
            "call": "_model_exception()",
            "gold_call": "_reference_exception()",
        },
    ]
