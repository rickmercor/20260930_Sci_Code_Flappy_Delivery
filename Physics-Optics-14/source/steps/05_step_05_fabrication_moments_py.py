"""
Fabricated geometric-phase moments

Quantize phase to levels uniformly spaced values on [0,2pi), using nearest level and choosing the increasing phase at an exact half-level tie, modulo levels. The nanobrick rotation error is independent Normal(0,rotation_sigma[n]^2) at each cell; its geometric-phase error is twice that angle. Return the first two complex phasor moments at each cell. The rotation-error model is a constructed fabrication perturbation; the source supplies the geometric phase and phase-statistics framework. With q=(phase mod 2pi)/(2pi/levels), values within four binary64 ulps of floor(q)+1/2 use the half-level tie convention.

Returns
-------
complex ndarray (2,N). Rows E[Z] and E[Z^2], dimensionless, in source order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fabrication_moments(
    phase: "np.ndarray", levels: int, rotation_sigma: "np.ndarray"
) -> "np.ndarray":
    """Fabricated geometric-phase moments.

    Quantize phase to levels uniformly spaced values on [0,2pi), using
    nearest level and choosing the increasing phase at an exact half-level
    tie, modulo levels. The nanobrick rotation error is independent
    Normal(0,rotation_sigma[n]^2) at each cell; its geometric-phase error
    is twice that angle. Return the first two complex phasor moments at
    each cell. With q=(phase mod 2pi)/(2pi/levels), values within four
    binary64 ulps of floor(q)+1/2 use the half-level tie convention. The
    rotation-error model is a constructed fabrication perturbation; the
    source supplies the geometric phase and phase-statistics framework.

    Parameters
    phase : float ndarray (N,): cell phases in radians, in source order.
    levels : int >=2: number of uniformly spaced manufactured phase levels.
    rotation_sigma : float ndarray (N,): nonnegative standard deviation of
    nanobrick rotation error in radians.

    Returns
    complex ndarray (2,N). Rows E[Z] and E[Z^2], dimensionless, in source
    order.

    Raises
    ValueError for invalid dimensions, nonfinite inputs or parameters
    outside the stated domain.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_fabrication_moments(
    phase: "np.ndarray", levels: int, rotation_sigma: "np.ndarray"
) -> "np.ndarray":
    phase = np.asarray(phase, dtype=float)
    rotation_sigma = np.asarray(rotation_sigma, dtype=float)
    if (
        phase.ndim != 1
        or phase.size == 0
        or rotation_sigma.shape != phase.shape
        or not np.isfinite(phase).all()
        or not np.isfinite(rotation_sigma).all()
        or np.any(rotation_sigma < 0)
        or not np.isfinite(levels)
        or int(levels) != levels
        or levels < 2
    ):
        raise ValueError("Invalid phase, phase levels or angular errors.")
    step = 2 * np.pi / levels
    scaled = np.mod(phase, 2 * np.pi) / step
    half = np.floor(scaled) + 0.5
    scaled = np.where(
        np.abs(scaled - half) <= 4 * np.spacing(half), half, scaled
    )
    quantized = step * (np.floor(scaled + 0.5) % levels)
    sig = 2 * rotation_sigma
    return np.array(
        [np.exp(1j * k * quantized - 0.5 * k * k * sig**2) for k in [1, 2]]
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independently initialized differential cases."""
    return [
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

ph = np.array([0.0, np.pi / 2])
L = 4
sig = np.zeros(2)
candidate_args = deepcopy((ph, L, sig))
oracle_args = deepcopy((ph, L, sig))
"""
            ),
            "call": "fabrication_moments(*candidate_args)",
            "gold_call": "_oracle_fabrication_moments(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

ph = np.array([0.7])
L = 128
sig = np.array([0.17])
candidate_args = deepcopy((ph, L, sig))
oracle_args = deepcopy((ph, L, sig))
"""
            ),
            "call": "fabrication_moments(*candidate_args)",
            "gold_call": "_oracle_fabrication_moments(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

ph = np.array([-0.01, 2 * np.pi + 0.01, -2 * np.pi - 0.01])
L = 16
sig = np.array([0.02, 0.04, 0.08])
candidate_args = deepcopy((ph, L, sig))
oracle_args = deepcopy((ph, L, sig))
"""
            ),
            "call": "fabrication_moments(*candidate_args)",
            "gold_call": "_oracle_fabrication_moments(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

ph = np.array(
    [
        np.pi / 8,
        3 * np.pi / 8,
        15 * np.pi / 8,
        15 * np.pi / 8 - 1e-10,
        15 * np.pi / 8 + 1e-10,
    ]
)
L = 8
sig = np.zeros(5)
candidate_args = deepcopy((ph, L, sig))
oracle_args = deepcopy((ph, L, sig))
"""
            ),
            "call": "fabrication_moments(*candidate_args)",
            "gold_call": "_oracle_fabrication_moments(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

ph = np.array([0.2, 0.2, 0.2])
L = 32
sig = np.array([0.0, 0.1, 0.4])
candidate_args = deepcopy((ph, L, sig))
oracle_args = deepcopy((ph, L, sig))
"""
            ),
            "call": "fabrication_moments(*candidate_args)",
            "gold_call": "_oracle_fabrication_moments(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

ph = np.array([0.2, 1.7])
L = 16
sig = np.array([2.5, 5.0])
candidate_args = deepcopy((ph, L, sig))
oracle_args = deepcopy((ph, L, sig))
"""
            ),
            "call": "fabrication_moments(*candidate_args)",
            "gold_call": "_oracle_fabrication_moments(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

ph = np.array([0.4, 2.0, 4.0, 5.9])
L = 2
sig = np.array([0.1, 0.2, 0.1, 0.2])
candidate_args = deepcopy((ph, L, sig))
oracle_args = deepcopy((ph, L, sig))
"""
            ),
            "call": "fabrication_moments(*candidate_args)",
            "gold_call": "_oracle_fabrication_moments(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

ph = np.array([0.013, 0.061, 0.217])
L = 128
sig = np.array([0.01, 0.06, 0.2])
candidate_args = deepcopy((ph, L, sig))
oracle_args = deepcopy((ph, L, sig))
"""
            ),
            "call": "fabrication_moments(*candidate_args)",
            "gold_call": "_oracle_fabrication_moments(*oracle_args)",
            "tol": 1e-06,
        },
    ]
