"""
Generate Cartesian trajectory frames from a reference geometry and one displacement mode.

Normal-mode visualization samples a displacement field at signed amplitudes around the reference transition-state geometry.

Returns
-------
Cartesian trajectory with shape (n_amplitudes, n_atoms, 3).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def generate_mode_trajectory(
    reference_positions: np.ndarray,
    mode_vector: np.ndarray,
    amplitudes: np.ndarray,
) -> np.ndarray:
    """Generate Cartesian frames along a displacement mode.

    Parameters
    ----------
    reference_positions : np.ndarray
        Reference coordinates with shape (n_atoms, 3).
    mode_vector : np.ndarray
        Per-atom Cartesian displacement with the same shape.
    amplitudes : np.ndarray
        One-dimensional finite, nonconstant displacement amplitudes.

    Returns
    -------
    np.ndarray
        Trajectory with shape (n_amplitudes, n_atoms, 3).
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_generate_mode_trajectory(
    reference_positions: np.ndarray,
    mode_vector: np.ndarray,
    amplitudes: np.ndarray,
) -> np.ndarray:
    """Reference linear mode-displacement construction."""
    import numpy as np

    reference = np.asarray(reference_positions, dtype=float)
    mode = np.asarray(mode_vector, dtype=float)
    scale = np.asarray(amplitudes, dtype=float)

    if reference.ndim != 2 or reference.shape[0] < 2 or reference.shape[1] != 3:
        raise ValueError("reference_positions must have shape (n_atoms, 3)")
    if mode.shape != reference.shape:
        raise ValueError("mode_vector must match reference_positions")
    if scale.ndim != 1 or scale.size < 2:
        raise ValueError("amplitudes must contain at least two values")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(mode)):
        raise ValueError("coordinates and mode vectors must be finite")
    if not np.all(np.isfinite(scale)) or np.ptp(scale) <= 0.0:
        raise ValueError("amplitudes must be finite and nonconstant")

    return reference[None, :, :] + scale[:, None, None] * mode[None, :, :]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
reference_positions = np.array([[0.,0.,0.],[1.,0.,0.]])
mode_vector = np.array([[0.1,0.,0.],[-0.2,0.3,0.]])
amplitudes = np.array([-1.,0.,1.])""",
            "call": "generate_mode_trajectory(reference_positions, mode_vector, amplitudes)",
            "gold_call": "_oracle_generate_mode_trajectory(reference_positions, mode_vector, amplitudes)",
        },
        {
            "setup": """import numpy as np
reference_positions = np.array([[1.,2.,3.],[-1.,0.,2.],[0.5,0.5,0.5]])
mode_vector = np.array([[0.,1.,0.],[2.,0.,-1.],[0.2,-0.4,0.8]])
amplitudes = np.array([0.5,-0.5,1.5,0.0])""",
            "call": "generate_mode_trajectory(reference_positions, mode_vector, amplitudes)",
            "gold_call": "_oracle_generate_mode_trajectory(reference_positions, mode_vector, amplitudes)",
        },
        {
            "setup": """import numpy as np
reference_positions = np.array([[0.,0.,0.],[0.,1.,0.]])
mode_vector = np.zeros((2,3))
amplitudes = np.array([2.,-2.])""",
            "call": "generate_mode_trajectory(reference_positions, mode_vector, amplitudes)",
            "gold_call": "_oracle_generate_mode_trajectory(reference_positions, mode_vector, amplitudes)",
        },
    ]
