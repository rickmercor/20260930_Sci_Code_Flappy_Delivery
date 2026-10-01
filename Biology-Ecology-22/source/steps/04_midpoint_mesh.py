"""
Return the midpoint-rule mesh of n_bins equal bins on [z_min, z_max]: an array of shape (2, n_bins) whose row 0 holds the bin midpoints in increasing order and whose row 1 holds the bin widths, all equal to (z_max - z_min) / n_bins, so that sum(row 1 * f(row 0)) approximates the integral of f over the interval.

An integral projection model replaces its kernel integrals by a quadrature on a fixed length mesh; the midpoint rule on equal bins turns the projection into a matrix-vector product and counts the individuals in each length bin directly.

Returns
-------
numpy.ndarray of float64 with shape (2, n_bins): rows of bin midpoints and bin widths.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def midpoint_mesh(z_min: float, z_max: float, n_bins: int) -> "numpy.ndarray":
    """Return the midpoint-rule mesh of n_bins equal bins on [z_min, z_max]: an array of shape (2, n_bins) whose row 0 holds the bin midpoints in increasing order and whose row 1 holds the bin widths, all equal to (z_max - z_min) / n_bins, so that sum(row 1 * f(row 0)) approximates the integral of f over the interval.

    Parameters
    ----------
    z_min : float
        Lower end of the length interval (metres).
    z_max : float
        Upper end of the length interval (metres); must exceed z_min.
    n_bins : int
        Number of equal bins, at least 1.

    Returns
    -------
    mesh : numpy.ndarray
        Array of shape (2, n_bins): row 0 the bin midpoints, row 1 the bin widths (float64).

    Raises
    ------
    ValueError
        If z_max does not exceed z_min, either end is not finite, or n_bins is not a positive integer.
    """
    return mesh

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_midpoint_mesh(z_min: float, z_max: float, n_bins: int) -> "numpy.ndarray":
    """Midpoint-rule mesh on [z_min, z_max]: row 0 the bin midpoints, row 1 the (constant) bin widths."""
    if not (np.isfinite(z_min) and np.isfinite(z_max)) or not z_max > z_min:
        raise ValueError("z_max must exceed z_min and both must be finite")
    if int(n_bins) != n_bins or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    n = int(n_bins)
    edges = np.linspace(float(z_min), float(z_max), n + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    widths = np.full(n, (float(z_max) - float(z_min)) / n)
    return np.vstack([mids, widths])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "z_min, z_max, n_bins = 0.01, 1.5, 300\n",
            "call": "midpoint_mesh(z_min, z_max, n_bins)",
            "gold_call": "_oracle_midpoint_mesh(z_min, z_max, n_bins)",
        },
        {
            "setup": "z_min, z_max, n_bins = 0.0, 1.0, 4\n",
            "call": "midpoint_mesh(z_min, z_max, n_bins)",
            "gold_call": "_oracle_midpoint_mesh(z_min, z_max, n_bins)",
        },
        {
            "setup": "z_min, z_max, n_bins = 0.01, 1.5, 40\n",
            "call": "midpoint_mesh(z_min, z_max, n_bins)",
            "gold_call": "_oracle_midpoint_mesh(z_min, z_max, n_bins)",
        },
        {
            "setup": "z_min, z_max, n_bins = 1.5, 0.01, 300\ndef run_model():\n    try:\n        midpoint_mesh(z_min, z_max, n_bins)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_midpoint_mesh(z_min, z_max, n_bins)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
