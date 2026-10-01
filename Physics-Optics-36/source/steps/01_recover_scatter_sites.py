"""
Recover the final scattering positions from exiting photon histories.

Coordinates use a planar sample surface at zero depth and positive inward depth.

Returns
-------
np.ndarray of shape (K, 3), the final scattering positions (x, y, z) in mm, preserving the input record order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def recover_scatter_sites(records: "np.ndarray") -> "np.ndarray":
    """Parameters
    ----------
    records : np.ndarray
        Shape (K, 7), K >= 1. Columns are exit x, exit y, outward unit
        direction vx, vy, vz < 0, final segment length s >= 0, and total
        in-medium physical path L >= s. Lengths are in mm.

    Returns
    -------
    sites : np.ndarray
        Shape (K, 3), final scattering positions x, y, z in row order.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_recover_scatter_sites(records: "np.ndarray") -> "np.ndarray":
    records = np.asarray(records, dtype=float)
    exits = np.column_stack((records[:, :2], np.zeros(len(records))))
    return exits - records[:, 5, None] * records[:, 2:5]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent normal, boundary and edge cases."""
    return [{'setup': 'import numpy as np\nr=np.array([[0.01, 0.0, 0.0, 0.0, -1.0, 0.2, 0.42], [0.15, 0.02, 0.6, 0.0, -0.8, 0.25, 0.49], [-0.13, 0.03, -0.6, 0.0, -0.8, 0.35, 0.84], [0.02, 0.17, 0.0, 0.6, -0.8, 0.4, 0.97], [0.22, 0.0, 0.6, 0.0, -0.8, 0.5, 1.18], [-0.24, 0.01, -0.6, 0.0, -0.8, 0.5, 1.27], [0.03, 0.34, 0.0, 0.6, -0.8, 0.65, 1.68], [0.43, 0.04, 0.6, 0.0, -0.8, 0.7, 1.84], [0.015, -0.02, 0.0, 0.0, -1.0, 0.8, 1.85], [0.55, 0.0, 0.8, 0.0, -0.6, 0.7, 1.7], [0.02, 0.01, 0.0, 0.0, -1.0, 0.1, 2.4], [0.02, -0.01, 0.0, 0.0, -1.0, 0.15, 0.3], [0.24, 0.26, 0.48, 0.64, -0.6, 0.4, 1.0], [0.01, 0.38, 0.0, 0.6, -0.8, 0.6, 1.5], [-0.35, 0.0, -0.6, 0.0, -0.8, 0.6, 1.61], [0.0, 0.0, 0.0, 0.0, -1.0, 0.6, 1.2]], dtype=float)\n', 'call': 'recover_scatter_sites(r.copy())', 'gold_call': '_oracle_recover_scatter_sites(r.copy())', 'tol': 1e-11}, {'setup': 'import numpy as np\nr=np.array([[.02,.01,0.,0.,-1.,0.,.3]])', 'call': 'recover_scatter_sites(r.copy())', 'gold_call': '_oracle_recover_scatter_sites(r.copy())', 'tol': 1e-11}, {'setup': 'import numpy as np\nr=np.array([[.24,.26,.48,.64,-.6,.4,1.]])', 'call': 'recover_scatter_sites(r.copy())', 'gold_call': '_oracle_recover_scatter_sites(r.copy())', 'tol': 1e-11}]
