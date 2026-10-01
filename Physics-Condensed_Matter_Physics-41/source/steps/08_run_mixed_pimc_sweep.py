"""
Run the ordered mixed harmonic/free bead updates on a periodic path and return the harmonic-reference energy of the final path.

Process the requested 0-based bead indices in order on a private copy of the periodic path. For each move, obtain the two current neighbors, construct the two bridge families, choose the forward family from the old bead and the reverse family from the trial bead, apply the strict decision $u<A$, and make every accepted replacement visible to later moves. Compose the seven previously defined public step functions rather than duplicating their scientific formulas. Return the harmonic-reference energy of the final path.

Returns
-------
float, the final finite-P harmonic-reference energy after the sequential sweep
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import math

from numbers import Integral, Real

import numpy as np


def _finite_float(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def run_mixed_pimc_sweep(
    path: np.ndarray,
    bead_indices: np.ndarray,
    normal_draws: np.ndarray,
    uniform_draws: np.ndarray,
    beta: float,
    mass: float,
    curvature: float,
    hbar: float,
    minimum: float,
    domain_radius: float,
    alpha_3: float,
    alpha_4: float,
) -> float:
    """Run a deterministic sequential mixed-kernel PIMC sweep.

    Parameters
    ----------
    path : numpy.ndarray
        One-dimensional periodic path with at least three finite beads.
    bead_indices : numpy.ndarray
        One-dimensional integer indices processed sequentially.
    normal_draws : numpy.ndarray
        Finite standard-normal deviates, one per proposed move.
    uniform_draws : numpy.ndarray
        Values in ``[0, 1)`` used with the strict rule ``u < A``.
    beta : float
        Positive inverse temperature.
    mass : float
        Positive particle mass.
    curvature : float
        Positive local curvature.
    hbar : float
        Positive reduced Planck constant.
    minimum : float
        Position of the local minimum.
    domain_radius : float
        Nonnegative radius of the closed harmonic domain.
    alpha_3, alpha_4 : float
        Finite coefficients of the anharmonic well.

    Returns
    -------
    energy : float
        Final harmonic-reference energy estimate as a native Python
        float. The input ``path`` must not be modified.

    Raises
    ------
    ValueError
        If an array has invalid shape, length, type, index, or non-finite
        content; if a uniform draw lies outside ``[0, 1)``; if ``beta``,
        ``mass``, ``curvature``, or ``hbar`` is not strictly positive; or
        if ``domain_radius`` is negative.
    """
    return float("nan")

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_run_mixed_pimc_sweep(
    path: np.ndarray,
    bead_indices: np.ndarray,
    normal_draws: np.ndarray,
    uniform_draws: np.ndarray,
    beta: float,
    mass: float,
    curvature: float,
    hbar: float,
    minimum: float,
    domain_radius: float,
    alpha_3: float,
    alpha_4: float,
) -> float:
    """Reference implementation composed exclusively from steps 1-7 oracles."""
    import math
    import numbers

    import numpy as np

    try:
        current_path = np.asarray(path, dtype=float).copy()
        raw_indices = np.asarray(bead_indices)
        normal_array = np.asarray(normal_draws, dtype=float)
        uniform_array = np.asarray(uniform_draws, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("path and draw inputs must be convertible to numeric arrays") from exc

    if current_path.ndim != 1 or current_path.size < 3 or not np.all(np.isfinite(current_path)):
        raise ValueError("path must be a finite one-dimensional array with at least three beads")
    if raw_indices.ndim != 1 or normal_array.ndim != 1 or uniform_array.ndim != 1:
        raise ValueError("bead_indices and draw arrays must be one-dimensional")
    if not (raw_indices.size == normal_array.size == uniform_array.size):
        raise ValueError("bead_indices, normal_draws, and uniform_draws must have equal lengths")
    if not np.all(np.isfinite(normal_array)) or not np.all(np.isfinite(uniform_array)):
        raise ValueError("draw arrays must be finite")
    if np.any((uniform_array < 0.0) | (uniform_array >= 1.0)):
        raise ValueError("uniform_draws must lie in [0, 1)")

    indices_list: list[int] = []
    for value in raw_indices.tolist():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
            if not isinstance(value, numbers.Real) or not float(value).is_integer():
                raise ValueError("bead_indices must contain integers")
        index = int(value)
        if index < 0 or index >= current_path.size:
            raise ValueError("bead index out of range")
        indices_list.append(index)

    named_values = (
        ("beta", beta),
        ("hbar", hbar),
        ("minimum", minimum),
        ("domain_radius", domain_radius),
        ("alpha_3", alpha_3),
        ("alpha_4", alpha_4),
    )
    converted: dict[str, float] = {}
    for name, value in named_values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise ValueError(f"{name} must be a finite real scalar")
        value_f = float(value)
        if not math.isfinite(value_f):
            raise ValueError(f"{name} must be finite")
        converted[name] = value_f

    beta_f = converted["beta"]
    hbar_f = converted["hbar"]
    minimum_f = converted["minimum"]
    radius_f = converted["domain_radius"]
    alpha_3_f = converted["alpha_3"]
    alpha_4_f = converted["alpha_4"]
    if beta_f <= 0.0 or hbar_f <= 0.0:
        raise ValueError("beta and hbar must be > 0")
    if radius_f < 0.0:
        raise ValueError("domain_radius must be >= 0")

    # Step 1 oracle: validates mass and curvature and returns omega.
    omega = _oracle_compute_reference_frequency(mass, curvature)
    mass_f = float(mass)
    curvature_f = float(curvature)
    tau = beta_f / current_path.size

    for index, normal_draw, uniform_draw in zip(
        indices_list,
        normal_array.tolist(),
        uniform_array.tolist(),
        strict=True,
    ):
        left = float(current_path[(index - 1) % current_path.size])
        old = float(current_path[index])
        right = float(current_path[(index + 1) % current_path.size])

        # Step 2 oracle: both fixed-endpoint Gaussian bridges.
        statistics = _oracle_compute_bridge_statistics(
            left,
            right,
            minimum_f,
            tau,
            mass_f,
            omega,
            hbar_f,
        )
        harmonic_mean, harmonic_variance, free_mean, free_variance = statistics.tolist()

        # Step 3 oracle: forward family is selected from the current state.
        selected = _oracle_select_proposal_parameters(
            old,
            minimum_f,
            radius_f,
            harmonic_mean,
            harmonic_variance,
            free_mean,
            free_variance,
        )

        # Step 4 oracle: deterministic proposal from the supplied normal deviate.
        trial = _oracle_generate_trial_position(
            float(selected[0]),
            float(selected[1]),
            float(normal_draw),
        )

        # Step 5 oracle: reverse family is selected from the proposed state.
        factor = _oracle_compute_hastings_factor(
            old,
            trial,
            minimum_f,
            radius_f,
            harmonic_mean,
            harmonic_variance,
            free_mean,
            free_variance,
        )

        # Step 6 oracle: residual-only Boltzmann factor under harmonic splitting.
        acceptance = _oracle_compute_mixed_acceptance(
            old,
            trial,
            tau,
            minimum_f,
            curvature_f,
            alpha_3_f,
            alpha_4_f,
            factor,
        )
        if float(uniform_draw) < acceptance:
            current_path[index] = trial

    # Step 7 oracle: final periodic-path harmonic-reference energy.
    return _oracle_compute_harmonic_energy(
        current_path,
        beta_f,
        mass_f,
        omega,
        hbar_f,
        minimum_f,
        alpha_3_f,
        alpha_4_f,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return branch-sensitive pipeline, empty-sweep, and invalid cases."""
    return [
        {
            "setup": """import numpy as np
path = np.array([0.08, 0.24, 0.22, 0.26, 0.03, -0.12, -0.60, -0.05], dtype=float)
bead_indices = np.array([3, 4, 6], dtype=int)
normal_draws = np.array([0.7, 0.5, 1.1], dtype=float)
uniform_draws = np.array([0.8964, 0.8590, 0.9477], dtype=float)
beta = 4.8
mass = 1.3
curvature = 3.2
hbar = 0.7
minimum = -0.15
domain_radius = 0.44
alpha_3 = 0.3125
alpha_4 = 12.5""",
            "call": "run_mixed_pimc_sweep(path, bead_indices, normal_draws, uniform_draws, beta, mass, curvature, hbar, minimum, domain_radius, alpha_3, alpha_4)",
            "gold_call": "_oracle_run_mixed_pimc_sweep(path, bead_indices, normal_draws, uniform_draws, beta, mass, curvature, hbar, minimum, domain_radius, alpha_3, alpha_4)",
        },
        {
            "setup": """import numpy as np
path = np.array([0.08, 0.24, 0.22, 0.26, 0.03, -0.12, -0.60, -0.05], dtype=float)
bead_indices = np.array([3, 4, 6], dtype=int)
normal_draws = np.array([0.7, 0.5, 1.1], dtype=float)
uniform_draws = np.array([0.8963, 0.8590, 0.9477], dtype=float)
beta = 4.8
mass = 1.3
curvature = 3.2
hbar = 0.7
minimum = -0.15
domain_radius = 0.44
alpha_3 = 0.3125
alpha_4 = 12.5""",
            "call": "run_mixed_pimc_sweep(path, bead_indices, normal_draws, uniform_draws, beta, mass, curvature, hbar, minimum, domain_radius, alpha_3, alpha_4)",
            "gold_call": "_oracle_run_mixed_pimc_sweep(path, bead_indices, normal_draws, uniform_draws, beta, mass, curvature, hbar, minimum, domain_radius, alpha_3, alpha_4)",
        },
        {
            "setup": """import numpy as np
path = np.array([-0.2, 0.0, 0.2], dtype=float)
bead_indices = np.array([], dtype=int)
normal_draws = np.array([], dtype=float)
uniform_draws = np.array([], dtype=float)
beta = 1.0
mass = 1.0
curvature = 2.0
hbar = 1.0
minimum = 0.0
domain_radius = 0.0
alpha_3 = 0.1
alpha_4 = 0.2""",
            "call": "run_mixed_pimc_sweep(path, bead_indices, normal_draws, uniform_draws, beta, mass, curvature, hbar, minimum, domain_radius, alpha_3, alpha_4)",
            "gold_call": "_oracle_run_mixed_pimc_sweep(path, bead_indices, normal_draws, uniform_draws, beta, mass, curvature, hbar, minimum, domain_radius, alpha_3, alpha_4)",
        },
        {
            "setup": """import numpy as np
path = np.array([0.0, 0.1, 0.2], dtype=float)
bead_indices = np.array([1, 2], dtype=int)
normal_draws = np.array([0.0], dtype=float)
uniform_draws = np.array([0.5, 0.5], dtype=float)
beta = 1.0
mass = 1.0
curvature = 1.0
hbar = 1.0
minimum = 0.0
domain_radius = 1.0
alpha_3 = 0.0
alpha_4 = 0.0
def run_model():
    try:
        run_mixed_pimc_sweep(path, bead_indices, normal_draws, uniform_draws, beta, mass, curvature, hbar, minimum, domain_radius, alpha_3, alpha_4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_mixed_pimc_sweep(path, bead_indices, normal_draws, uniform_draws, beta, mass, curvature, hbar, minimum, domain_radius, alpha_3, alpha_4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
path = np.array([0.0, 0.1, 0.2], dtype=float)
bead_indices = np.array([3], dtype=int)
normal_draws = np.array([0.0], dtype=float)
uniform_draws = np.array([0.5], dtype=float)
beta = 1.0
mass = 1.0
curvature = 1.0
hbar = 1.0
minimum = 0.0
domain_radius = 1.0
alpha_3 = 0.0
alpha_4 = 0.0
def run_model():
    try:
        run_mixed_pimc_sweep(path, bead_indices, normal_draws, uniform_draws, beta, mass, curvature, hbar, minimum, domain_radius, alpha_3, alpha_4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_mixed_pimc_sweep(path, bead_indices, normal_draws, uniform_draws, beta, mass, curvature, hbar, minimum, domain_radius, alpha_3, alpha_4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
