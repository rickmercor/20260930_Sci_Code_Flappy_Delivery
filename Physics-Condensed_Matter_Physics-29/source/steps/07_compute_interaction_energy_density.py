"""
Compute the sampled onsite interaction-energy density from trajectory matrices.

For opposite spins on the same site, the normally ordered Hubbard interaction estimator is the product of the corresponding diagonal phase-space occupations. Averaging this complex estimator over trajectories and taking the real part yields the finite-sample interaction-energy density requested by the benchmark.

Returns
-------
float, the real ensemble-averaged onsite interaction-energy density as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_interaction_energy_density(
    trajectories: "np.ndarray",
    interaction: float,
    n_sites: int,
) -> float:
    """Compute the real ensemble-averaged onsite interaction-energy density.

    Parameters
    ----------
    trajectories : np.ndarray
        Finite complex array with shape ``(2*n_sites**2, n_trajectories)``.
    interaction : float
        Finite real onsite interaction strength.
    n_sites : int
        Positive number of lattice sites.

    Returns
    -------
    energy_density : float
        Native Python float equal to the real part of the trajectory-averaged
        onsite interaction energy divided by ``n_sites``.

    Raises
    ------
    ValueError
        If dimensions or scalar inputs are outside the documented domain.
    """
    return energy_density

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
from numbers import Integral, Real

import numpy as np


def _oracle_compute_interaction_energy_density(
    trajectories: "np.ndarray",
    interaction: float,
    n_sites: int,
) -> float:
    """Reference implementation."""
    samples = np.asarray(trajectories, dtype=complex)
    if (
        isinstance(n_sites, bool)
        or not isinstance(n_sites, Integral)
        or int(n_sites) < 1
    ):
        raise ValueError("n_sites must be a positive integer")
    site_count = int(n_sites)
    if (
        samples.ndim != 2
        or samples.shape[0] != 2 * site_count * site_count
        or samples.shape[1] < 1
        or not np.all(np.isfinite(samples))
    ):
        raise ValueError("trajectories have an incompatible finite shape")
    if isinstance(interaction, bool) or not isinstance(interaction, Real):
        raise ValueError("interaction must be a finite real scalar")
    interaction_value = float(interaction)
    if not math.isfinite(interaction_value):
        raise ValueError("interaction must be a finite real scalar")

    estimates = np.empty(samples.shape[1], dtype=complex)
    split = site_count * site_count
    for trajectory in range(samples.shape[1]):
        up = samples[:split, trajectory].reshape(site_count, site_count, order="C")
        down = samples[split:, trajectory].reshape(site_count, site_count, order="C")
        estimates[trajectory] = (
            interaction_value
            * np.sum(np.diag(up) * np.diag(down))
            / site_count
        )
    return float(np.real(np.mean(estimates)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic test specifications."""
    return [
        {
            "setup": """import numpy as np
up = np.array([[0.8 + 0.1j, 0.0], [0.0, 0.2 - 0.05j]], dtype=complex)
down = np.array([[0.3 - 0.02j, 0.0], [0.0, 0.7 + 0.04j]], dtype=complex)
trajectories = np.concatenate((up.ravel(), down.ravel()))[:, None]
""",
            "call": "compute_interaction_energy_density(trajectories.copy(), 1.5, 2)",
            "gold_call": "_oracle_compute_interaction_energy_density(trajectories.copy(), 1.5, 2)",
        },
        {
            "setup": """import numpy as np
trajectory_a = np.array([0.4 + 0.2j, 0.0, 0.0, 0.6 - 0.1j, 0.5 - 0.3j, 0.0, 0.0, 0.5 + 0.2j], dtype=complex)
trajectory_b = np.conj(trajectory_a)
trajectories = np.column_stack((trajectory_a, trajectory_b))
""",
            "call": "compute_interaction_energy_density(trajectories.copy(), -0.7, 2)",
            "gold_call": "_oracle_compute_interaction_energy_density(trajectories.copy(), -0.7, 2)",
        },
        {
            "setup": """import numpy as np
trajectories = np.arange(54, dtype=float).reshape(18, 3).astype(complex) / 100.0
""",
            "call": "compute_interaction_energy_density(trajectories.copy(), 0.0, 3)",
            "gold_call": "_oracle_compute_interaction_energy_density(trajectories.copy(), 0.0, 3)",
        },
        {
            "setup": """import numpy as np
trajectories = np.zeros((7, 2), dtype=complex)
def run_model():
    try:
        compute_interaction_energy_density(trajectories.copy(), 1.0, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_interaction_energy_density(trajectories.copy(), 1.0, 2)
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
