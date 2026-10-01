"""
Construct reciprocal vectors for a volume-preserving simple shear and differentiate them analytically.

The direct-lattice matrix has columns H(gamma)=[[L,gamma L,0],[0,L,0],[0,0,L]]. The reciprocal basis is B=2*pi*H^{-T}; differentiating the matrix inverse supplies dB/dgamma, dxi/dgamma, and d|xi|/dgamma.

Returns
-------
np.ndarray, shape (M,8): columns 0:3 are reciprocal vectors, columns 3:6 are dxi/dgamma, column 6 is |xi|, and column 7 is d|xi|/dgamma.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sheared_reciprocal_geometry(indices: "np.ndarray", box_length: float, shear: float) -> "np.ndarray":
    """Return reciprocal vectors and their shear derivatives.

    Parameters
    ----------
    indices : np.ndarray
        Nonempty integer-valued array of shape (M,3), with no zero row.
    box_length : float
        Positive finite cubic scale L.
    shear : float
        Finite simple-shear parameter gamma satisfying |gamma| < 0.5.

    Returns
    -------
    geometry : np.ndarray
        Real array with shape (M,8). Columns 0:3 are xi_k, columns 3:6 are
        dxi_k/dgamma, column 6 is |xi_k|, and column 7 is d|xi_k|/dgamma.

    Raises
    ------
    ValueError
        If indices, box_length, or shear violate the stated domain.
    """
    return geometry

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_sheared_reciprocal_geometry(indices: "np.ndarray", box_length: float, shear: float) -> "np.ndarray":
    indices_array = np.asarray(indices)
    if indices_array.ndim != 2 or indices_array.shape[1] != 3 or indices_array.shape[0] == 0:
        raise ValueError("indices must have nonempty shape (M,3)")
    if not np.all(np.isfinite(indices_array)) or not np.all(indices_array == np.rint(indices_array)):
        raise ValueError("indices must be finite and integer-valued")
    integer_indices = indices_array.astype(int)
    if np.any(np.all(integer_indices == 0, axis=1)):
        raise ValueError("indices must exclude the zero row")

    box_length = float(box_length)
    shear = float(shear)
    if not np.isfinite(box_length) or box_length <= 0.0:
        raise ValueError("box_length must be positive and finite")
    if not np.isfinite(shear) or abs(shear) >= 0.5:
        raise ValueError("shear must satisfy |shear| < 0.5")

    cell = np.array([
        [box_length, shear * box_length, 0.0],
        [0.0, box_length, 0.0],
        [0.0, 0.0, box_length],
    ], dtype=float)
    cell_derivative = np.array([
        [0.0, box_length, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ], dtype=float)

    inverse_transpose = np.linalg.inv(cell).T
    reciprocal = 2.0 * np.pi * inverse_transpose
    reciprocal_derivative = (
        -2.0 * np.pi
        * inverse_transpose
        @ cell_derivative.T
        @ inverse_transpose
    )

    xi = integer_indices.astype(float) @ reciprocal.T
    xi_derivative = integer_indices.astype(float) @ reciprocal_derivative.T
    norms = np.linalg.norm(xi, axis=1)
    norm_derivative = np.sum(xi * xi_derivative, axis=1) / norms

    return np.column_stack([xi, xi_derivative, norms, norm_derivative])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return orthogonal and positively/negatively sheared geometries."""
    return [
        {"setup":"import numpy as np\nindices = np.array([[1,0,0],[0,1,0],[-1,2,1]], dtype=int)\nbox_length = 4.0\nshear = 0.0","call":"sheared_reciprocal_geometry(indices.copy(), box_length, shear)","gold_call":"_oracle_sheared_reciprocal_geometry(indices.copy(), box_length, shear)","tol":1e-10},
        {"setup":"import numpy as np\nindices = np.array([[2,-1,0],[-3,2,1],[1,1,-2]], dtype=int)\nbox_length = 4.8\nshear = 0.17","call":"sheared_reciprocal_geometry(indices.copy(), box_length, shear)","gold_call":"_oracle_sheared_reciprocal_geometry(indices.copy(), box_length, shear)","tol":1e-10},
        {"setup":"import numpy as np\nindices = np.array([[-2,-2,1],[3,0,-1],[1,-3,2]], dtype=int)\nbox_length = 5.2\nshear = -0.33","call":"sheared_reciprocal_geometry(indices.copy(), box_length, shear)","gold_call":"_oracle_sheared_reciprocal_geometry(indices.copy(), box_length, shear)","tol":1e-10},
    ]
