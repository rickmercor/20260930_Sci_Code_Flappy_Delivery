"""
Run the complete low-rank diffusion-gauge trajectory benchmark and return one scalar.

The final workflow must compose the preceding public scientific operations: hopping drift, onsite opposite-spin diffusion, randomized low-rank factors, structured gauge construction, covariance verification, one stochastic ensemble step, and interaction-energy averaging. The covariance gate prevents a numerically inconsistent factor from entering the trajectory update.

Returns
-------
float, the final real interaction-energy density rounded once to the requested decimal places
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_diffusion_gauge_benchmark(
    n_up: "np.ndarray",
    n_down: "np.ndarray",
    hopping: "np.ndarray",
    interaction: float,
    hbar: float,
    rank: int,
    sketch_seed: int,
    tolerance_factor: float,
    dt: float,
    n_trajectories: int,
    noise_seed: int,
    residual_tolerance: float = 1.0e-10,
    decimals: int = 12,
) -> float:
    """Run the complete deterministic diffusion-gauge benchmark.

    Parameters
    ----------
    n_up, n_down : np.ndarray
        Compatible finite complex spin-resolved phase-space matrices.
    hopping : np.ndarray
        Compatible finite hopping matrix.
    interaction : float
        Finite real onsite interaction strength.
    hbar : float
        Positive finite reduced Planck constant.
    rank : int
        Randomized sketch width.
    sketch_seed : int
        Seed for the range-finder sketch.
    tolerance_factor : float
        Positive singular-value cutoff multiplier.
    dt : float
        Positive Euler–Maruyama step size.
    n_trajectories : int
        Positive number of trajectory columns.
    noise_seed : int
        Seed for channel-major real Wiener draws.
    residual_tolerance : float, default=1e-10
        Positive maximum permitted relative diffusion-factor residual.
    decimals : int, default=12
        Number of decimal places for the single final rounding operation.

    Returns
    -------
    result : float
        Rounded real interaction-energy density as a native Python float.

    Raises
    ------
    ValueError
        If any upstream contract fails, the residual exceeds its tolerance, or
        final rounding controls are invalid.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
from numbers import Integral, Real

import numpy as np


def _oracle_run_diffusion_gauge_benchmark(
    n_up: "np.ndarray",
    n_down: "np.ndarray",
    hopping: "np.ndarray",
    interaction: float,
    hbar: float,
    rank: int,
    sketch_seed: int,
    tolerance_factor: float,
    dt: float,
    n_trajectories: int,
    noise_seed: int,
    residual_tolerance: float = 1.0e-10,
    decimals: int = 12,
) -> float:
    """Reference implementation."""
    if (
        isinstance(residual_tolerance, bool)
        or not isinstance(residual_tolerance, Real)
        or not math.isfinite(float(residual_tolerance))
        or float(residual_tolerance) <= 0.0
    ):
        raise ValueError("residual_tolerance must be positive and finite")
    if (
        isinstance(decimals, bool)
        or not isinstance(decimals, Integral)
        or int(decimals) < 0
        or int(decimals) > 15
    ):
        raise ValueError("decimals must be an integer from 0 to 15")

    drift = _oracle_compute_hubbard_drift(n_up, n_down, hopping, hbar)
    diffusion_block = _oracle_compute_opposite_spin_diffusion(
        n_up, n_down, interaction
    )
    factors = _oracle_compute_randomized_svd_factors(
        diffusion_block, rank, sketch_seed, tolerance_factor
    )
    gauge = _oracle_construct_diffusion_gauge(factors)
    residual = _oracle_compute_factorization_residual(diffusion_block, gauge)
    if not np.isfinite(residual) or residual > float(residual_tolerance):
        raise ValueError("diffusion gauge does not satisfy the required covariance tolerance")
    trajectories = _oracle_propagate_phase_space_ensemble(
        n_up,
        n_down,
        drift,
        gauge,
        dt,
        n_trajectories,
        noise_seed,
    )
    energy = _oracle_compute_interaction_energy_density(
        trajectories, interaction, np.asarray(n_up).shape[0]
    )
    return float(round(energy, int(decimals)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic test specifications."""
    return [
        {
            "setup": """import numpy as np
n_up = np.array([
    [0.82 + 0.00j, 0.12 + 0.05j, -0.04 + 0.02j],
    [0.12 - 0.05j, 0.47 + 0.00j, 0.09 - 0.03j],
    [-0.04 - 0.02j, 0.09 + 0.03j, 0.21 + 0.00j],
], dtype=complex)
n_down = np.array([
    [0.18 + 0.00j, -0.07 + 0.02j, 0.05 - 0.01j],
    [-0.07 - 0.02j, 0.53 + 0.00j, -0.11 + 0.04j],
    [0.05 + 0.01j, -0.11 - 0.04j, 0.79 + 0.00j],
], dtype=complex)
hopping = np.array([
    [0.0, 1.0, 0.35],
    [1.0, 0.0, 0.8],
    [0.35, 0.8, 0.0],
], dtype=float)
""",
            "call": "run_diffusion_gauge_benchmark(n_up.copy(), n_down.copy(), hopping.copy(), 1.3, 1.0, 6, 314159, 64.0, 0.015, 7, 271828, 1.0e-10, 12)",
            "gold_call": "_oracle_run_diffusion_gauge_benchmark(n_up.copy(), n_down.copy(), hopping.copy(), 1.3, 1.0, 6, 314159, 64.0, 0.015, 7, 271828, 1.0e-10, 12)",
        },
        {
            "setup": """import numpy as np
n_up = np.array([[0.70 + 0.0j, 0.08 + 0.03j], [0.05 - 0.02j, 0.30 + 0.0j]], dtype=complex)
n_down = np.array([[0.30 + 0.0j, -0.04 + 0.01j], [-0.06 - 0.02j, 0.70 + 0.0j]], dtype=complex)
hopping = np.array([[0.0, 0.9], [0.9, 0.0]], dtype=float)
""",
            "call": "run_diffusion_gauge_benchmark(n_up.copy(), n_down.copy(), hopping.copy(), 0.85, 1.0, 4, 1234, 64.0, 0.02, 5, 5678, 1.0e-10, 12)",
            "gold_call": "_oracle_run_diffusion_gauge_benchmark(n_up.copy(), n_down.copy(), hopping.copy(), 0.85, 1.0, 4, 1234, 64.0, 0.02, 5, 5678, 1.0e-10, 12)",
        },
        {
            "setup": """import numpy as np
n_up = np.diag([1.0, 0.0, 0.5]).astype(complex)
n_down = np.diag([0.0, 1.0, 0.5]).astype(complex)
hopping = np.zeros((3, 3), dtype=float)
""",
            "call": "run_diffusion_gauge_benchmark(n_up.copy(), n_down.copy(), hopping.copy(), 0.0, 1.0, 6, 4, 64.0, 0.03, 3, 8, 1.0e-10, 12)",
            "gold_call": "_oracle_run_diffusion_gauge_benchmark(n_up.copy(), n_down.copy(), hopping.copy(), 0.0, 1.0, 6, 4, 64.0, 0.03, 3, 8, 1.0e-10, 12)",
        },
        {
            "setup": """import numpy as np
n_up = np.array([
    [0.82 + 0.00j, 0.12 + 0.05j, -0.04 + 0.02j],
    [0.12 - 0.05j, 0.47 + 0.00j, 0.09 - 0.03j],
    [-0.04 - 0.02j, 0.09 + 0.03j, 0.21 + 0.00j],
], dtype=complex)
n_down = np.array([
    [0.18 + 0.00j, -0.07 + 0.02j, 0.05 - 0.01j],
    [-0.07 - 0.02j, 0.53 + 0.00j, -0.11 + 0.04j],
    [0.05 + 0.01j, -0.11 - 0.04j, 0.79 + 0.00j],
], dtype=complex)
hopping = np.array([
    [0.0, 1.0, 0.35],
    [1.0, 0.0, 0.8],
    [0.35, 0.8, 0.0],
], dtype=float)
""",
            "call": "run_diffusion_gauge_benchmark(n_up.copy(), n_down.copy(), hopping.copy(), 1.3, 1.0, 6, 271, 64.0, 0.0075, 4, 919, 1.0e-10, 12)",
            "gold_call": "_oracle_run_diffusion_gauge_benchmark(n_up.copy(), n_down.copy(), hopping.copy(), 1.3, 1.0, 6, 271, 64.0, 0.0075, 4, 919, 1.0e-10, 12)",
        },
        {
            "setup": """import numpy as np
n_up = np.array([[0.5]], dtype=complex)
n_down = np.array([[0.25]], dtype=complex)
hopping = np.zeros((1, 1), dtype=float)
def run_model():
    try:
        run_diffusion_gauge_benchmark(n_up.copy(), n_down.copy(), hopping.copy(), 1.2, 1.0, 1, 77, 64.0, 0.2, 4, 99, 0.0, 12)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_diffusion_gauge_benchmark(n_up.copy(), n_down.copy(), hopping.copy(), 1.2, 1.0, 1, 77, 64.0, 0.2, 4, 99, 0.0, 12)
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
