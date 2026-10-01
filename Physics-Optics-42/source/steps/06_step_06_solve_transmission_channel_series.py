"""
Generate a transmission-matched deformation series for one channel.

The scalar channel coefficients describe the slab response about the supplied deformation center. Compute the requested coefficient series from the supplied channel operator, material profiles, spectral operators, and exterior boundary data.

Returns
-------
complex128 ndarray of shape (max_order+1, N) containing one circular channel's deformation coefficients
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_transmission_channel_series(sign: int, matrix: "np.ndarray", profiles: "np.ndarray", operators: "np.ndarray", radiation: "np.ndarray", max_order: int) -> "np.ndarray":
    '''Solve all transmission-matched orders for one circular channel.

    Parameters
    ----------
    sign : int
        +1 for LCP or -1 for RCP.
    matrix : np.ndarray
        Complex transmission-matched boundary-value matrix of shape (N, N).
    profiles : np.ndarray
        Channel profiles of shape (3, N).
    operators : np.ndarray
        Spectral operators of shape (3, N, N).
    radiation : np.ndarray
        Radiation and incident data of shape (nx, 5).
    max_order : int
        Nonnegative highest deformation order.

    Returns
    -------
    series : np.ndarray
        Complex128 array of shape (max_order + 1, N). Row m contains the
        coefficient of the local deformation variable to power m in the
        selected scalar circular channel.

    Raises
    ------
    ValueError
        If sign, max_order, array shapes, or inferred grid dimensions are
        invalid.
    '''
    return series

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import lu_factor, lu_solve


def _oracle_solve_transmission_channel_series(sign: int, matrix: "np.ndarray", profiles: "np.ndarray", operators: "np.ndarray", radiation: "np.ndarray", max_order: int) -> "np.ndarray":
    """Reference implementation."""
    if sign not in (-1, 1) or isinstance(sign, (bool, np.bool_)):
        raise ValueError("sign must equal +1 or -1")
    if not isinstance(max_order, (int, np.integer)) or isinstance(max_order, (bool, np.bool_)) or max_order < 0:
        raise ValueError("max_order must be a nonnegative integer")
    radiation_array = np.asarray(radiation)
    if radiation_array.ndim != 2 or radiation_array.shape[1] != 5:
        raise ValueError("radiation must have shape (nx, 5)")
    nx = radiation_array.shape[0]
    profiles_array = np.asarray(profiles)
    if profiles_array.ndim != 2 or profiles_array.shape[0] != 3:
        raise ValueError("profiles must have shape (3, N)")
    n_total = profiles_array.shape[1]
    if nx < 4 or nx % 2 or n_total % nx:
        raise ValueError("array dimensions do not define a valid even Fourier grid")
    nz = n_total // nx - 1
    if nz < 2:
        raise ValueError("the inferred Chebyshev degree must be at least 2")
    matrix_array = np.asarray(matrix)
    operators_array = np.asarray(operators)
    if matrix_array.shape != (n_total, n_total):
        raise ValueError("matrix must have shape (N, N)")
    if operators_array.shape != (3, n_total, n_total):
        raise ValueError("operators must have shape (3, N, N)")

    alpha_p = np.asarray(radiation_array[:, 0], dtype=complex).real
    spacing = alpha_p[1] - alpha_p[0]
    d = 2.0 * np.pi / spacing
    x = np.arange(nx, dtype=float) * d / nx
    qinv = np.exp(1j * x[:, None] * alpha_p[None, :])
    forcing_column = 3 if sign == 1 else 4
    top_forcing = qinv @ np.asarray(radiation_array[:, forcing_column], dtype=complex)
    top_indices = np.arange(nx) * (nz + 1)
    bottom_indices = top_indices + nz

    factorization = lu_factor(np.asarray(matrix_array, dtype=complex))
    series = np.zeros((max_order + 1, n_total), dtype=np.complex128)
    zero_field = np.zeros(n_total, dtype=complex)
    for order in range(max_order + 1):
        if order == 0:
            right_hand_side = np.zeros(n_total, dtype=complex)
        else:
            previous_two = series[order - 2] if order >= 2 else zero_field
            right_hand_side = _oracle_assemble_transmission_source(
                series[order - 1], previous_two, profiles_array,
                operators_array, nx, nz
            )
        if order == 0:
            right_hand_side[top_indices] = top_forcing
            right_hand_side[bottom_indices] = 0.0
        series[order] = lu_solve(factorization, right_hand_side)
    return series

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    base = """import numpy as np
nx, nz = 4, 2
n = nx*(nz+1)
operators = np.zeros((3,n,n),dtype=complex)
operators[0] = np.diag(1j*np.linspace(-1.8,1.8,n))
dz1 = np.array([[1.5,-2.0,0.5],[0.5,0.0,-0.5],[-0.5,2.0,-1.5]])
operators[1] = np.kron(np.eye(nx),dz1)
operators[2] = np.diag(np.full(n,1/n))
radiation = np.zeros((nx,5),dtype=complex)
radiation[:,0] = np.array([-3.0,-1.0,1.0,3.0])
radiation[:,1] = np.array([2.0j,1.4,2.0,2.2j])
radiation[:,2] = -1j*radiation[:,1]
radiation[nx//2,3] = 1.2+0.4j
radiation[nx//2,4] = -0.3+1.1j
idx = np.arange(n,dtype=float)
def make_matrix(profiles):
    dx,dz = operators[0],operators[1]
    rc = profiles[1]
    R = np.diag(rc.astype(complex))
    A = R@(dx@R@dx+dz@R@dz)+5.0*np.eye(n)
    alpha_p = radiation[:,0].real
    spacing = alpha_p[1]-alpha_p[0]
    period = 2*np.pi/spacing
    x = np.arange(nx)*period/nx
    Q = np.exp(-1j*alpha_p[:,None]*x[None,:])/nx
    Qi = np.exp(1j*x[:,None]*alpha_p[None,:])
    T = Qi@np.diag(radiation[:,2])@Q
    top = np.arange(nx)*(nz+1); bottom=top+nz
    for j in range(nx):
        A[top[j],:] = -rc[top[j]]*dz[top[j],:]; A[top[j],top] -= T[j]
        A[bottom[j],:] = rc[bottom[j]]*dz[bottom[j],:]; A[bottom[j],bottom] -= T[j]
    return A
"""
    return [
        {
            "setup": base + """sign, max_order = 1, 4
profiles=np.zeros((3,n)); profiles[1]=0.94+0.02*np.cos(0.2*idx); profiles[2]=0.06*np.sin(0.3*idx)
matrix = make_matrix(profiles)
""",
            "call": "solve_transmission_channel_series(sign, matrix.copy(), profiles.copy(), operators.copy(), radiation.copy(), max_order)",
            "gold_call": "_oracle_solve_transmission_channel_series(sign, matrix.copy(), profiles.copy(), operators.copy(), radiation.copy(), max_order)",
            "tol": 3e-9,
        },
        {
            "setup": base + """sign, max_order = -1, 3
profiles=np.zeros((3,n)); profiles[1]=1.05-0.03*np.sin(0.17*idx); profiles[2]=-0.05*np.cos(0.23*idx)
matrix = make_matrix(profiles)
""",
            "call": "solve_transmission_channel_series(sign, matrix.copy(), profiles.copy(), operators.copy(), radiation.copy(), max_order)",
            "gold_call": "_oracle_solve_transmission_channel_series(sign, matrix.copy(), profiles.copy(), operators.copy(), radiation.copy(), max_order)",
            "tol": 3e-9,
        },
        {
            "setup": base + """sign, max_order = 1, 0
profiles=np.zeros((3,n)); profiles[1]=1.0
matrix = make_matrix(profiles)
""",
            "call": "solve_transmission_channel_series(sign, matrix.copy(), profiles.copy(), operators.copy(), radiation.copy(), max_order)",
            "gold_call": "_oracle_solve_transmission_channel_series(sign, matrix.copy(), profiles.copy(), operators.copy(), radiation.copy(), max_order)",
            "tol": 3e-9,
        },
        {
            "setup": base + """sign, max_order = 0, 2
profiles=np.zeros((3,n)); profiles[1]=0.96; profiles[2]=0.04*np.sin(0.2*idx)
matrix = make_matrix(profiles)
def catch_value_error(fn):
    try:
        fn(sign, matrix, profiles, operators, radiation, max_order)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "catch_value_error(solve_transmission_channel_series)",
            "gold_call": "catch_value_error(_oracle_solve_transmission_channel_series)",
        },
    ]
