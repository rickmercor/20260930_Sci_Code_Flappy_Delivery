"""
Enumerate the centered even reciprocal index cube used by the truncated spectral sum.

For even n_f, each Cartesian index runs from -n_f/2 through n_f/2-1. The zero row is excluded and lexicographic meshgrid order is retained so downstream comparisons are deterministic.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def centered_reciprocal_indices(mode_count: int) -> "np.ndarray":
    """Return all nonzero integer reciprocal indices in the centered even cube.

    Parameters
    ----------
    mode_count : int
        Even integer n_f at least 4.

    Returns
    -------
    indices : np.ndarray
        Integer array with shape (mode_count**3 - 1, 3), in deterministic
        meshgrid order, excluding only [0,0,0].

    Raises
    ------
    ValueError
        If mode_count is not an even integer at least 4.
    """
    return indices

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_centered_reciprocal_indices(mode_count: int) -> "np.ndarray":
    if isinstance(mode_count, bool) or int(mode_count) != mode_count:
        raise ValueError("mode_count must be an integer")
    mode_count = int(mode_count)
    if mode_count < 4 or mode_count % 2 != 0:
        raise ValueError("mode_count must be even and at least 4")
    axis = np.arange(-mode_count // 2, mode_count // 2, dtype=int)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    return grid[np.any(grid != 0, axis=1)]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return small, medium, and benchmark reciprocal cubes."""
    return [
        {"setup":"mode_count = 4","call":"centered_reciprocal_indices(mode_count)","gold_call":"_oracle_centered_reciprocal_indices(mode_count)"},
        {"setup":"mode_count = 8","call":"centered_reciprocal_indices(mode_count)","gold_call":"_oracle_centered_reciprocal_indices(mode_count)"},
        {"setup":"mode_count = 10","call":"centered_reciprocal_indices(mode_count)","gold_call":"_oracle_centered_reciprocal_indices(mode_count)"},
    ]
