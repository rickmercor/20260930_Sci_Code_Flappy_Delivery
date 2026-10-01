"""
Evaluate reciprocal charge structure factors from fractional particle coordinates.

For a direct and reciprocal lattice transformed consistently by the same shear, xi_k dot r_j = 2*pi*k dot f_j. The structure factors therefore depend only on fractional coordinates, charges, and integer reciprocal indices.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fractional_structure_factors(fractional_positions: "np.ndarray", charges: "np.ndarray", indices: "np.ndarray") -> "np.ndarray":
    """Return S_k = sum_j q_j exp(i 2*pi*k dot f_j) for each reciprocal index.

    Parameters
    ----------
    fractional_positions : np.ndarray
        Finite array of particle fractional coordinates with shape (N,3).
    charges : np.ndarray
        Finite one-dimensional real charge vector of length N.
    indices : np.ndarray
        Finite integer-valued reciprocal indices with shape (M,3).

    Returns
    -------
    factors : np.ndarray
        Complex vector of length M containing the charge structure factors.

    Raises
    ------
    ValueError
        If the array shapes, finiteness, or integer-index requirement are violated.
    """
    return factors

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_fractional_structure_factors(fractional_positions: "np.ndarray", charges: "np.ndarray", indices: "np.ndarray") -> "np.ndarray":
    fractional_positions = np.asarray(fractional_positions, dtype=float)
    charges = np.asarray(charges, dtype=float)
    indices_array = np.asarray(indices)

    if fractional_positions.ndim != 2 or fractional_positions.shape[1] != 3 or fractional_positions.shape[0] == 0:
        raise ValueError("fractional_positions must have shape (N,3) with N > 0")
    if charges.ndim != 1 or charges.size != fractional_positions.shape[0]:
        raise ValueError("charges must be one-dimensional with length N")
    if indices_array.ndim != 2 or indices_array.shape[1] != 3:
        raise ValueError("indices must have shape (M,3)")
    if not np.all(np.isfinite(fractional_positions)) or not np.all(np.isfinite(charges)):
        raise ValueError("particle data must be finite")
    if not np.all(np.isfinite(indices_array)) or not np.all(indices_array == np.rint(indices_array)):
        raise ValueError("indices must be finite and integer-valued")

    phase = 2.0 * np.pi * (indices_array.astype(float) @ fractional_positions.T)
    return np.exp(1j * phase) @ charges

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return neutral and nonneutral differential structure-factor cases."""
    return [
        {"setup":"import numpy as np\nfractional_positions=np.array([[-0.25,0,0],[0.25,0,0]],float)\ncharges=np.array([1.0,-1.0])\nindices=np.array([[1,0,0],[-1,0,0],[0,1,0]],int)","call":"fractional_structure_factors(fractional_positions.copy(), charges.copy(), indices.copy())","gold_call":"_oracle_fractional_structure_factors(fractional_positions.copy(), charges.copy(), indices.copy())","tol":1e-10},
        {"setup":"import numpy as np\nfractional_positions=np.array([[-0.23,-0.08,0.05],[0.16,-0.18,-0.06],[0.25,0.14,0.11],[-0.07,0.23,-0.15]],float)\ncharges=np.array([1.0,-1.0,0.75,-0.75])\nindices=np.array([[1,0,0],[2,-1,1],[-3,2,0]],int)","call":"fractional_structure_factors(fractional_positions.copy(), charges.copy(), indices.copy())","gold_call":"_oracle_fractional_structure_factors(fractional_positions.copy(), charges.copy(), indices.copy())","tol":1e-10},
        {"setup":"import numpy as np\nfractional_positions=np.array([[0.1,-0.2,0.3],[-0.3,0.25,-0.1],[0.0,0.0,0.0]],float)\ncharges=np.array([0.5,1.0,-0.2])\nindices=np.array([[0,0,0],[1,1,1],[-2,1,3]],int)","call":"fractional_structure_factors(fractional_positions.copy(), charges.copy(), indices.copy())","gold_call":"_oracle_fractional_structure_factors(fractional_positions.copy(), charges.copy(), indices.copy())","tol":1e-10},
    ]
