"""
Select the trajectory-frame pair with the largest raw Cartesian RMSD.

GraphRC compares maximally separated trajectory frames so that the internal coordinates carrying the transition motion are exposed.

Returns
-------
Integer array [i, j] for the lexicographically first maximum-RMSD pair.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def select_diverse_pair(trajectory: np.ndarray) -> np.ndarray:
    """Select the maximally separated frame pair without alignment.

    Parameters
    ----------
    trajectory : np.ndarray
        Finite Cartesian frames with shape (n_frames, n_atoms, 3).

    Returns
    -------
    np.ndarray
        Two increasing zero-based frame indices.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_select_diverse_pair(trajectory: np.ndarray) -> np.ndarray:
    """Reference unaligned Cartesian RMSD pair selection."""
    import numpy as np

    frames = np.asarray(trajectory, dtype=float)
    if (
        frames.ndim != 3
        or frames.shape[0] < 2
        or frames.shape[1] < 1
        or frames.shape[2] != 3
    ):
        raise ValueError("trajectory must have shape (n_frames, n_atoms, 3)")
    if not np.all(np.isfinite(frames)):
        raise ValueError("trajectory must be finite")

    best_pair = (0, 1)
    best_rmsd = -np.inf

    for first in range(frames.shape[0] - 1):
        for second in range(first + 1, frames.shape[0]):
            delta = frames[first] - frames[second]
            rmsd = float(
                np.sqrt(np.mean(np.sum(delta * delta, axis=1)))
            )
            if rmsd > best_rmsd + 1e-12:
                best_rmsd = rmsd
                best_pair = (first, second)

    return np.asarray(best_pair, dtype=int)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
trajectory = np.array([[[0.,0.,0.]], [[3.,0.,0.]], [[1.,0.,0.]]])""",
            "call": "select_diverse_pair(trajectory)",
            "gold_call": "_oracle_select_diverse_pair(trajectory)",
        },
        {
            "setup": """import numpy as np
trajectory = np.array([[[0.,0.,0.],[0.,1.,0.]], [[2.,0.,0.],[2.,1.,0.]], [[-2.,0.,0.],[-2.,1.,0.]]])""",
            "call": "select_diverse_pair(trajectory)",
            "gold_call": "_oracle_select_diverse_pair(trajectory)",
        },
        {
            "setup": """import numpy as np
trajectory = np.zeros((4,2,3), dtype=float)""",
            "call": "select_diverse_pair(trajectory)",
            "gold_call": "_oracle_select_diverse_pair(trajectory)",
        },
    ]
