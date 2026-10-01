"""
Proper orthogonal decomposition represents a snapshot matrix X by its leading left singular vectors. The retained dimension n is the smallest index satisfying 1 - sum_{i=1}^n sigma_i^2 / sum_i sigma_i^2 < energy_tolerance. Singular vectors have arbitrary signs, so each retained vector is oriented with its largest-magnitude entry positive to make the numerical basis deterministic.

Returns
-------
np.ndarray of shape (n_state, n_reduced), the oriented float64 POD basis
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def select_pod_basis(snapshots: np.ndarray, energy_tolerance: float) -> np.ndarray:
    """Select and orient a POD basis using a truncated-energy criterion.

    Parameters
    ----------
    snapshots : np.ndarray
        Finite, nonempty snapshot matrix of shape (n_state, n_snapshots) with
        strictly positive total squared singular-value energy.
    energy_tolerance : float
        Required upper bound in the open interval (0, 1) on discarded energy.

    Returns
    -------
    basis : np.ndarray
        Orthonormal POD basis of shape (n_state, n_reduced).

    Raises
    ------
    ValueError
        If snapshots is not a finite, nonempty, positive-energy matrix or if
        energy_tolerance is not a scalar strictly between 0 and 1.
    """
    return basis

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_select_pod_basis(snapshots: np.ndarray, energy_tolerance: float) -> np.ndarray:
    """Reference implementation."""
    snapshots = np.asarray(snapshots, dtype=float)
    if snapshots.ndim != 2 or 0 in snapshots.shape:
        raise ValueError("snapshots must be a nonempty two-dimensional array")
    if not np.all(np.isfinite(snapshots)):
        raise ValueError("snapshots must contain finite values")
    if not np.isscalar(energy_tolerance):
        raise ValueError("energy_tolerance must be a scalar")
    energy_tolerance = float(energy_tolerance)
    if not 0.0 < energy_tolerance < 1.0:
        raise ValueError("energy_tolerance must lie in (0, 1)")

    left_vectors, singular_values, _ = np.linalg.svd(snapshots, full_matrices=False)
    total_energy = float(singular_values @ singular_values)
    if total_energy == 0.0:
        raise ValueError("snapshots must have positive energy")
    discarded = 1.0 - np.cumsum(singular_values**2) / total_energy
    retained = int(np.flatnonzero(discarded < energy_tolerance)[0] + 1)
    basis = left_vectors[:, :retained].copy()
    for column in range(retained):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0.0:
            basis[:, column] *= -1.0
    return basis

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def _pack(value):
    """Flatten a returned value into native numbers for comparison.

    Container sizes and array shapes are packed alongside the numbers, so a
    result carrying the right values in the wrong structure cannot compare equal.
    """
    if isinstance(value, dict):
        packed = [len(value)]
        for key in sorted(value):
            packed.extend(_pack(value[key]))
        return packed
    if isinstance(value, np.ndarray):
        packed = list(value.shape)
        packed.extend(round(float(entry), 12) for entry in value.ravel().tolist())
        return packed
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    return [round(float(value), 12)]


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
snapshots = np.array([
    [1.20, 1.00, 0.75, -0.35, -0.65, -0.90],
    [0.35, 0.60, 0.90, 1.10, 1.30, 1.45],
    [-0.55, -0.25, 0.05, 0.45, 0.85, 1.10],
    [0.85, 0.55, 0.20, -0.10, -0.45, -0.70],
    [-0.25, 0.05, 0.35, 0.70, 1.00, 1.25],
])
energy_tolerance = 0.035
""",
            "call": "_pack(select_pod_basis(snapshots, energy_tolerance))",
            "gold_call": "_pack(_oracle_select_pod_basis(snapshots, energy_tolerance))",
        },
        {
            "setup": """import numpy as np
snapshots = np.array([[2.0]])
energy_tolerance = 0.5
""",
            "call": "_pack(select_pod_basis(snapshots, energy_tolerance))",
            "gold_call": "_pack(_oracle_select_pod_basis(snapshots, energy_tolerance))",
        },
        {
            "setup": """import numpy as np
snapshots = np.zeros((2, 2))
def run_model():
    try:
        select_pod_basis(snapshots, 0.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_select_pod_basis(snapshots, 0.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
