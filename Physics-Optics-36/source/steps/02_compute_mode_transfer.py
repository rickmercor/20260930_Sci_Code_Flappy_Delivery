"""
Compute spherical-wave power transfer to the Gaussian collection mode.

The supplied beam parameters are defined inside the homogeneous sample; positions are referenced to the optical axis.

Returns
-------
np.ndarray of shape (K,), the dimensionless power-coupling coefficients of the selected paraxial model, preserving the input site order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_mode_transfer(sites: "np.ndarray", waist: float, rayleigh: float, focus: float) -> "np.ndarray":
    """Parameters
    ----------
    sites : np.ndarray
        Shape (K, 3), last-scatter x, y, z in mm, with z >= 0.
    waist, rayleigh : float
        Positive Gaussian waist radius and Rayleigh range in mm.
    focus : float
        Focus depth in mm.

    Returns
    -------
    transfer : np.ndarray
        Shape (K,), dimensionless power-coupling coefficients in row order,
        using the selected model's paraxial expression, including its
        normalization. The expression is continuously extended to z = 0.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_mode_transfer(sites: "np.ndarray", waist: float, rayleigh: float, focus: float) -> "np.ndarray":
    sites = np.asarray(sites, dtype=float)
    width2 = waist**2 * (1 + ((sites[:, 2] - focus) / rayleigh)**2)
    radius2 = np.sum(sites[:, :2]**2, axis=1)
    return (waist / rayleigh)**2 * waist**2 / width2 * np.exp(-2 * radius2 / width2)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent normal, boundary and edge cases."""
    return [{'setup': 'import numpy as np\nx=np.array([[.01,0,.2],[.02,-.01,.8]])', 'call': 'compute_mode_transfer(x.copy(), .03, .5, .6)', 'gold_call': '_oracle_compute_mode_transfer(x.copy(), .03, .5, .6)', 'tol': 1e-13}, {'setup': 'import numpy as np\nx=np.array([[0.,0.,.6]])', 'call': 'compute_mode_transfer(x.copy(), .03, .5, .6)', 'gold_call': '_oracle_compute_mode_transfer(x.copy(), .03, .5, .6)', 'tol': 1e-13}, {'setup': 'import numpy as np\nx=np.array([[.08,.03,.28],[.02,.01,0.]])', 'call': 'compute_mode_transfer(x.copy(), .03, .5, .6)', 'gold_call': '_oracle_compute_mode_transfer(x.copy(), .03, .5, .6)', 'tol': 1e-13}]
