"""
Form the real Taylor series of normalized volume-averaged electric intensity.

The squared field must be expanded consistently in deformation order before tensor-product spectral quadrature turns its Cartesian inner products into a robust scalar response.

Returns
-------
float64 ndarray of shape (M+1,) containing the volume-intensity Taylor coefficients
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def electric_intensity_coefficients(electric: "np.ndarray", operators: "np.ndarray") -> "np.ndarray":
    '''Compute the scalar intensity-series coefficients from electric fields.

    Parameters
    ----------
    electric : np.ndarray
        Complex array of shape (M + 1, N, 3) containing Cartesian electric
        field coefficients.
    operators : np.ndarray
        Array of shape (3, N, N) whose third matrix is the normalized diagonal
        quadrature matrix.

    Returns
    -------
    coefficients : np.ndarray
        Float64 array of shape (M + 1,) containing orders zero through M of
        the normalized quadrature volume average of the squared magnitude of
        the electric-field deformation series.

    Raises
    ------
    ValueError
        If input shapes are inconsistent or the quadrature diagonal is not
        finite and real to absolute tolerance 1e-13.
    '''
    return coefficients

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_electric_intensity_coefficients(electric: "np.ndarray", operators: "np.ndarray") -> "np.ndarray":
    """Reference implementation."""
    electric_array = np.asarray(electric)
    if electric_array.ndim != 3 or electric_array.shape[0] < 1 or electric_array.shape[1] < 1 or electric_array.shape[2] != 3:
        raise ValueError("electric must have shape (M+1, N, 3)")
    orders, n_total, _ = electric_array.shape
    operators_array = np.asarray(operators)
    if operators_array.shape != (3, n_total, n_total):
        raise ValueError("operators must have shape (3, N, N)")
    diagonal = np.diag(np.asarray(operators_array[2], dtype=complex))
    if not np.all(np.isfinite(diagonal)) or np.max(np.abs(diagonal.imag)) > 1e-13:
        raise ValueError("quadrature diagonal must be finite and real")
    weights = diagonal.real
    fields = np.asarray(electric_array, dtype=complex)
    coefficients = np.empty(orders, dtype=float)
    for total_order in range(orders):
        value = 0.0j
        for left_order in range(total_order + 1):
            right_order = total_order - left_order
            value += np.sum(
                weights[:, None]
                * fields[left_order]
                * np.conj(fields[right_order])
            )
        coefficients[total_order] = value.real
    return coefficients

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
n, orders = 7, 4
idx = np.arange(n, dtype=float)
electric = np.empty((orders,n,3), dtype=complex)
for m in range(orders):
    electric[m,:,0] = (m+1)*np.sin(0.2*idx) + 0.1j*np.cos(0.3*idx)
    electric[m,:,1] = np.cos((m+1)*0.1*idx) - 0.2j*np.sin(0.4*idx)
    electric[m,:,2] = 0.3*(m+1) + 0.05j*idx
operators = np.zeros((3,n,n), dtype=complex)
operators[2] = np.diag(np.linspace(0.08,0.20,n))
""",
            "call": "electric_intensity_coefficients(electric.copy(), operators.copy())",
            "gold_call": "_oracle_electric_intensity_coefficients(electric.copy(), operators.copy())",
            "tol": 2e-12,
        },
        {
            "setup": """import numpy as np
n = 5
electric = np.array([[[1+1j,0,0],[0,2-1j,0],[0,0,0.5j],[1,1,1],[2j,-1,0.25]]], dtype=complex)
operators = np.zeros((3,n,n), dtype=complex)
operators[2] = np.diag(np.full(n,0.2))
""",
            "call": "electric_intensity_coefficients(electric.copy(), operators.copy())",
            "gold_call": "_oracle_electric_intensity_coefficients(electric.copy(), operators.copy())",
            "tol": 2e-12,
        },
        {
            "setup": """import numpy as np
n = 6
base = np.arange(18,dtype=float).reshape(n,3)/10
electric = np.stack([1j*base, -base, 0.5j*base])
operators = np.zeros((3,n,n), dtype=complex)
operators[2] = np.diag(np.array([0.05,0.10,0.15,0.20,0.22,0.28]))
""",
            "call": "electric_intensity_coefficients(electric.copy(), operators.copy())",
            "gold_call": "_oracle_electric_intensity_coefficients(electric.copy(), operators.copy())",
            "tol": 2e-12,
        },
        {
            "setup": """import numpy as np
electric = np.zeros((2,4,3), dtype=complex)
operators = np.zeros((3,5,5), dtype=complex)
def catch_value_error(fn):
    try:
        fn(electric, operators)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "catch_value_error(electric_intensity_coefficients)",
            "gold_call": "catch_value_error(_oracle_electric_intensity_coefficients)",
        },
    ]
