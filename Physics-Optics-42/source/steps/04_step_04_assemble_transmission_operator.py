"""
Assemble the transmission-matched arbitrary-center scattering operator.

The centered chiral factor remains inside a weighted divergence, so its spatial variation couples all retained Fourier orders. Because the exterior is achiral while the interior chirality need not vanish at either interface, tangential-field continuity weights the normal derivative by the centered material factor before the exterior Fourier DtN map is applied.

Returns
-------
complex128 ndarray of shape (N, N) containing the coupled transmission-matched boundary-value operator
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def assemble_transmission_operator(profiles: "np.ndarray", operators: "np.ndarray", radiation: "np.ndarray", nx: int, nz: int) -> "np.ndarray":
    '''Assemble one circular channel's transmission-matched collocation matrix.

    Parameters
    ----------
    profiles : np.ndarray
        Array of shape (3, N) from centered_channel_profiles.
    operators : np.ndarray
        Array of shape (3, N, N) from spectral_operators_and_weights.
    radiation : np.ndarray
        Array of shape (nx, 5) from radiation_and_bohren_data.
    nx, nz : int
        Grid dimensions with N = nx * (nz + 1).

    Returns
    -------
    matrix : np.ndarray
        Complex128 array of shape (N, N).  Interior rows discretize the
        centered weighted-divergence Helmholtz operator.  For each lateral
        node, the r = 0 and r = nz rows enforce tangential-field continuity
        to achiral vacuum and the supplied outgoing Fourier multiplier.  The
        normal derivative is therefore multiplied by rho_c at that boundary
        node; multiplication precedes application of the nonlocal DtN map.

    Raises
    ------
    ValueError
        If nx or nz is invalid or the supplied array shapes are inconsistent.
    '''
    return matrix

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_assemble_transmission_operator(profiles: "np.ndarray", operators: "np.ndarray", radiation: "np.ndarray", nx: int, nz: int) -> "np.ndarray":
    """Reference implementation."""
    if not isinstance(nx, (int, np.integer)) or isinstance(nx, (bool, np.bool_)) or nx < 4 or nx % 2:
        raise ValueError("nx must be an even integer at least 4")
    if not isinstance(nz, (int, np.integer)) or isinstance(nz, (bool, np.bool_)) or nz < 2:
        raise ValueError("nz must be an integer at least 2")
    n_total = nx * (nz + 1)
    profiles_array = np.asarray(profiles)
    operators_array = np.asarray(operators)
    radiation_array = np.asarray(radiation)
    if profiles_array.shape != (3, n_total):
        raise ValueError("profiles must have shape (3, nx*(nz+1))")
    if operators_array.shape != (3, n_total, n_total):
        raise ValueError("operators must have shape (3, N, N)")
    if radiation_array.shape != (nx, 5):
        raise ValueError("radiation must have shape (nx, 5)")

    rho_center = np.asarray(profiles_array[1], dtype=float)
    dx = np.asarray(operators_array[0], dtype=complex)
    dz = np.asarray(operators_array[1], dtype=complex)
    alpha_p = np.asarray(radiation_array[:, 0], dtype=complex).real
    dtn_multiplier = np.asarray(radiation_array[:, 2], dtype=complex)
    spacing = alpha_p[1] - alpha_p[0]
    d = 2.0 * np.pi / spacing
    alpha = alpha_p[nx // 2]
    x = np.arange(nx, dtype=float) * d / nx
    qmat = np.exp(-1j * alpha_p[:, None] * x[None, :]) / nx
    qinv = np.exp(1j * x[:, None] * alpha_p[None, :])
    dtn = qinv @ np.diag(dtn_multiplier) @ qmat

    k0_squared = alpha * alpha + radiation_array[nx // 2, 1].real ** 2
    rho_diag = np.diag(rho_center.astype(complex))
    matrix = rho_diag @ (dx @ rho_diag @ dx + dz @ rho_diag @ dz)
    matrix += k0_squared * np.eye(n_total, dtype=complex)
    top_columns = np.arange(nx) * (nz + 1)
    bottom_columns = top_columns + nz
    for j in range(nx):
        top_row = j * (nz + 1)
        matrix[top_row, :] = -rho_center[top_row] * dz[top_row, :]
        matrix[top_row, top_columns] -= dtn[j, :]
        bottom_row = top_row + nz
        matrix[bottom_row, :] = rho_center[bottom_row] * dz[bottom_row, :]
        matrix[bottom_row, bottom_columns] -= dtn[j, :]
    return matrix.astype(np.complex128)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    base = """import numpy as np
nx, nz = 4, 2
n = nx*(nz+1)
operators = np.zeros((3,n,n), dtype=complex)
operators[0] = np.diag(1j*np.linspace(-2.3,2.1,n))
dz1 = np.array([[1.5,-2.0,0.5],[0.5,0.0,-0.5],[-0.5,2.0,-1.5]], dtype=float)
operators[1] = np.kron(np.eye(nx),dz1)
operators[2] = np.diag(np.full(n,1/n))
radiation = np.zeros((nx,5), dtype=complex)
radiation[:,0] = np.array([-3.0,-1.0,1.0,3.0])
radiation[:,1] = np.sqrt(5.0 - radiation[:,0]**2 + 0j)
radiation[:,2] = -1j*radiation[:,1]
grid = np.arange(n,dtype=float)
"""
    return [
        {
            "setup": base + """profiles = np.zeros((3,n),dtype=float)
profiles[0] = 0.01*(1+np.sin(0.3*grid))
profiles[1] = 0.91+0.03*np.cos(0.2*grid)
profiles[2] = 0.08*np.sin(0.4*grid)
""",
            "call": "assemble_transmission_operator(profiles.copy(), operators.copy(), radiation.copy(), nx, nz)",
            "gold_call": "_oracle_assemble_transmission_operator(profiles.copy(), operators.copy(), radiation.copy(), nx, nz)",
            "tol": 2e-10,
        },
        {
            "setup": base + """profiles = np.zeros((3,n),dtype=float)
profiles[0] = 0.02*(1+np.cos(0.25*grid))
profiles[1] = 1.08-0.025*np.sin(0.31*grid)
profiles[2] = -0.06*np.cos(0.17*grid)
""",
            "call": "assemble_transmission_operator(profiles.copy(), operators.copy(), radiation.copy(), nx, nz)",
            "gold_call": "_oracle_assemble_transmission_operator(profiles.copy(), operators.copy(), radiation.copy(), nx, nz)",
            "tol": 2e-10,
        },
        {
            "setup": base + """profiles = np.zeros((3,n),dtype=float)
profiles[1] = 1.0
""",
            "call": "assemble_transmission_operator(profiles.copy(), operators.copy(), radiation.copy(), nx, nz)",
            "gold_call": "_oracle_assemble_transmission_operator(profiles.copy(), operators.copy(), radiation.copy(), nx, nz)",
            "tol": 2e-10,
        },
        {
            "setup": base + """profiles = np.zeros((3, nx*(nz+1)-1))
def catch_value_error(fn):
    try:
        fn(profiles, operators, radiation, nx, nz)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "catch_value_error(assemble_transmission_operator)",
            "gold_call": "catch_value_error(_oracle_assemble_transmission_operator)",
        },
    ]
