"""
Reconstruct Cartesian electric-field coefficients from both scalar channels.

Each scalar circular channel must be lifted to its Cartesian Beltrami field before the inverse circular-field transformation can recover the physical electric field. The lift and the material factor are expanded consistently about the same nonzero deformation center.

Returns
-------
complex128 ndarray of shape (M+1, N, 3) containing Cartesian electric-field series coefficients
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reconstruct_electric_series(left_series: "np.ndarray", right_series: "np.ndarray", left_profiles: "np.ndarray", right_profiles: "np.ndarray", operators: "np.ndarray", k0: float) -> "np.ndarray":
    '''Reconstruct the physical electric-field coefficient series.

    Parameters
    ----------
    left_series, right_series : np.ndarray
        Complex scalar series of identical shape (M + 1, N).
    left_profiles, right_profiles : np.ndarray
        Corresponding channel profiles, each of shape (3, N).
    operators : np.ndarray
        Spectral operators of shape (3, N, N); rows zero and one are D_x and D_z.
    k0 : float
        Positive finite vacuum wavenumber.

    Returns
    -------
    electric : np.ndarray
        Complex128 array of shape (M + 1, N, 3), with the final axis ordered
        as x, y, z.  It uses the order-expanded transverse Beltrami
        reconstruction and the normalized inverse circular transformation
        associated with the stated channel and time-harmonic conventions.

    Raises
    ------
    ValueError
        If k0 is invalid or any input shape is inconsistent.
    '''
    return electric

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_reconstruct_electric_series(left_series: "np.ndarray", right_series: "np.ndarray", left_profiles: "np.ndarray", right_profiles: "np.ndarray", operators: "np.ndarray", k0: float) -> "np.ndarray":
    """Reference implementation."""
    if not np.isfinite(k0) or k0 <= 0.0:
        raise ValueError("k0 must be positive and finite")
    left = np.asarray(left_series)
    right = np.asarray(right_series)
    if left.ndim != 2 or right.shape != left.shape or left.shape[0] < 1 or left.shape[1] < 1:
        raise ValueError("left and right series must share shape (M+1, N)")
    orders, n_total = left.shape
    left_profiles_array = np.asarray(left_profiles)
    right_profiles_array = np.asarray(right_profiles)
    operators_array = np.asarray(operators)
    if left_profiles_array.shape != (3, n_total) or right_profiles_array.shape != (3, n_total):
        raise ValueError("each channel profile must have shape (3, N)")
    if operators_array.shape != (3, n_total, n_total):
        raise ValueError("operators must have shape (3, N, N)")

    dx = np.asarray(operators_array[0], dtype=complex)
    dz = np.asarray(operators_array[1], dtype=complex)
    left_center = np.asarray(left_profiles_array[1], dtype=float)
    left_one = np.asarray(left_profiles_array[2], dtype=float)
    right_center = np.asarray(right_profiles_array[1], dtype=float)
    right_one = np.asarray(right_profiles_array[2], dtype=float)
    electric = np.empty((orders, n_total, 3), dtype=np.complex128)
    zero_field = np.zeros(n_total, dtype=complex)
    for order in range(orders):
        left_previous = left[order - 1] if order >= 1 else zero_field
        right_previous = right[order - 1] if order >= 1 else zero_field
        left_vector = np.empty((n_total, 3), dtype=complex)
        right_vector = np.empty((n_total, 3), dtype=complex)
        left_vector[:, 0] = -(
            left_center * (dz @ left[order]) + left_one * (dz @ left_previous)
        ) / float(k0)
        left_vector[:, 1] = left[order]
        left_vector[:, 2] = (
            left_center * (dx @ left[order]) + left_one * (dx @ left_previous)
        ) / float(k0)
        right_vector[:, 0] = (
            right_center * (dz @ right[order]) + right_one * (dz @ right_previous)
        ) / float(k0)
        right_vector[:, 1] = right[order]
        right_vector[:, 2] = -(
            right_center * (dx @ right[order]) + right_one * (dx @ right_previous)
        ) / float(k0)
        electric[order] = left_vector - 1j * right_vector
    return electric

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    base = """import numpy as np
nx, nz, k0 = 4, 3, 8.4
n = nx*(nz+1)
operators=np.zeros((3,n,n),dtype=complex)
operators[0]=np.diag(1j*np.linspace(-2.1,2.1,n))
dz1=np.array([[2.0,-2.5,0.7,-0.2],[0.6,-0.1,-0.7,0.2],[-0.2,0.7,0.1,-0.6],[0.2,-0.7,2.5,-2.0]])
operators[1]=np.kron(np.eye(nx),dz1)
operators[2]=np.diag(np.full(n,1/n))
idx = np.arange(n, dtype=float)
left_profiles=np.zeros((3,n)); left_profiles[1]=0.95+0.02*np.cos(0.2*idx); left_profiles[2]=0.06*np.sin(0.3*idx)
right_profiles=np.zeros((3,n)); right_profiles[1]=1.05-0.03*np.sin(0.17*idx); right_profiles[2]=-0.05*np.cos(0.23*idx)
"""
    return [
        {
            "setup": base + """left_series = np.stack([np.sin(0.1*idx)+0.2j, 0.3*np.cos(0.2*idx)-0.1j, 0.1*np.sin(0.3*idx)])
right_series = np.stack([0.4*np.cos(0.13*idx)+0.1j, np.sin(0.09*idx)-0.2j, 0.07*np.cos(0.17*idx)])
""",
            "call": "reconstruct_electric_series(left_series.copy(), right_series.copy(), left_profiles.copy(), right_profiles.copy(), operators.copy(), k0)",
            "gold_call": "_oracle_reconstruct_electric_series(left_series.copy(), right_series.copy(), left_profiles.copy(), right_profiles.copy(), operators.copy(), k0)",
            "tol": 3e-10,
        },
        {
            "setup": base + """left_series = (np.sin(0.1*idx)+0.2j)[None,:]
right_series = (0.4*np.cos(0.13*idx)+0.1j)[None,:]
""",
            "call": "reconstruct_electric_series(left_series.copy(), right_series.copy(), left_profiles.copy(), right_profiles.copy(), operators.copy(), k0)",
            "gold_call": "_oracle_reconstruct_electric_series(left_series.copy(), right_series.copy(), left_profiles.copy(), right_profiles.copy(), operators.copy(), k0)",
            "tol": 3e-10,
        },
        {
            "setup": base + """left_profiles=np.zeros((3,n)); left_profiles[1]=1.0
right_profiles=np.zeros((3,n)); right_profiles[1]=1.0
left_series = np.stack([np.exp(0.05j*idx), np.zeros(n, dtype=complex)])
right_series = np.stack([0.5j*np.exp(0.05j*idx), np.zeros(n, dtype=complex)])
""",
            "call": "reconstruct_electric_series(left_series.copy(), right_series.copy(), left_profiles.copy(), right_profiles.copy(), operators.copy(), k0)",
            "gold_call": "_oracle_reconstruct_electric_series(left_series.copy(), right_series.copy(), left_profiles.copy(), right_profiles.copy(), operators.copy(), k0)",
            "tol": 3e-10,
        },
        {
            "setup": base + """left_series = np.zeros((2,n), dtype=complex)
right_series = np.zeros((3,n), dtype=complex)
def catch_value_error(fn):
    try:
        fn(left_series, right_series, left_profiles, right_profiles, operators, k0)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "catch_value_error(reconstruct_electric_series)",
            "gold_call": "catch_value_error(_oracle_reconstruct_electric_series)",
        },
    ]
