"""
Assemble one transmission-matched perturbation source term.

The local coefficient expansion couples each new right-hand side to its two preceding fields through variable-coefficient weighted-divergence terms. At a chiral-to-achiral interface the coefficient of the normal derivative is also expanded, creating signed first-order boundary sources. Products stay in nodal form under the frozen no-dealiasing pseudospectral convention.

Returns
-------
complex128 ndarray of shape (N,) containing the interior and transmission-boundary recurrence source
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def assemble_transmission_source(previous: "np.ndarray", previous_two: "np.ndarray", profiles: "np.ndarray", operators: "np.ndarray", nx: int, nz: int) -> "np.ndarray":
    '''Form one arbitrary-center interior and interface recurrence source.

    Parameters
    ----------
    previous : np.ndarray
        Complex nodal field v_(m-1) of shape (N,).
    previous_two : np.ndarray
        Complex nodal field v_(m-2) of shape (N,); use exact zeros when the
        order does not yet exist.
    profiles : np.ndarray
        Shape (3, N) channel profiles containing rho_c and rho_1.
    operators : np.ndarray
        Shape (3, N, N) spectral operators; rows zero and one are D_x and D_z.
    nx, nz : int
        Grid dimensions with N = nx * (nz + 1).

    Returns
    -------
    source : np.ndarray
        Complex128 array of shape (N,) containing the centered recurrence
        source.  Nodal multiplication is used without dealiasing.  At r = 0
        the entry is rho_1 times the top normal derivative of v_(m-1); at
        r = nz it is the negative of that product.  These signs correspond
        to outward transmission conditions into achiral vacuum.

    Raises
    ------
    ValueError
        If grid dimensions or input shapes are inconsistent.
    '''
    return source

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_assemble_transmission_source(previous: "np.ndarray", previous_two: "np.ndarray", profiles: "np.ndarray", operators: "np.ndarray", nx: int, nz: int) -> "np.ndarray":
    """Reference implementation."""
    if not isinstance(nx, (int, np.integer)) or isinstance(nx, (bool, np.bool_)) or nx < 4 or nx % 2:
        raise ValueError("nx must be an even integer at least 4")
    if not isinstance(nz, (int, np.integer)) or isinstance(nz, (bool, np.bool_)) or nz < 2:
        raise ValueError("nz must be an integer at least 2")
    n_total = nx * (nz + 1)
    previous_array = np.asarray(previous)
    previous_two_array = np.asarray(previous_two)
    profiles_array = np.asarray(profiles)
    operators_array = np.asarray(operators)
    if previous_array.shape != (n_total,) or previous_two_array.shape != (n_total,):
        raise ValueError("previous fields must each have shape (N,)")
    if profiles_array.shape != (3, n_total):
        raise ValueError("profiles must have shape (3, N)")
    if operators_array.shape != (3, n_total, n_total):
        raise ValueError("operators must have shape (3, N, N)")

    v_one = np.asarray(previous_array, dtype=complex)
    v_two = np.asarray(previous_two_array, dtype=complex)
    rho_center = np.asarray(profiles_array[1], dtype=float)
    rho_one = np.asarray(profiles_array[2], dtype=float)
    dx = np.asarray(operators_array[0], dtype=complex)
    dz = np.asarray(operators_array[1], dtype=complex)

    div_one_previous = dx @ (rho_one * (dx @ v_one)) + dz @ (rho_one * (dz @ v_one))
    div_center_previous = dx @ (rho_center * (dx @ v_one)) + dz @ (rho_center * (dz @ v_one))
    div_one_previous_two = dx @ (rho_one * (dx @ v_two)) + dz @ (rho_one * (dz @ v_two))
    source = -rho_center * div_one_previous
    source -= rho_one * div_center_previous
    source -= rho_one * div_one_previous_two
    top = np.arange(nx) * (nz + 1)
    bottom = top + nz
    normal_derivative = dz @ v_one
    source[top] = rho_one[top] * normal_derivative[top]
    source[bottom] = -rho_one[bottom] * normal_derivative[bottom]
    return source.astype(np.complex128)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    base = """import numpy as np
nx, nz = 4, 3
n = nx*(nz+1)
operators = np.zeros((3,n,n),dtype=complex)
operators[0] = np.diag(1j*np.linspace(-2.0,2.0,n))
dz1 = np.array([[2.0,-2.5,0.7,-0.2],[0.6,-0.1,-0.7,0.2],[-0.2,0.7,0.1,-0.6],[0.2,-0.7,2.5,-2.0]])
operators[1] = np.kron(np.eye(nx),dz1)
operators[2] = np.diag(np.full(n,1/n))
idx = np.arange(nx*(nz+1), dtype=float)
previous = np.sin(0.17*idx) + 1j*np.cos(0.11*idx)
"""
    return [
        {
            "setup": base + """previous_two = np.cos(0.07*idx) - 0.3j*np.sin(0.13*idx)
profiles = np.zeros((3,n)); profiles[1]=0.95+0.02*np.cos(0.2*idx); profiles[2]=0.07*np.sin(0.3*idx)
""",
            "call": "assemble_transmission_source(previous.copy(), previous_two.copy(), profiles.copy(), operators.copy(), nx, nz)",
            "gold_call": "_oracle_assemble_transmission_source(previous.copy(), previous_two.copy(), profiles.copy(), operators.copy(), nx, nz)",
            "tol": 2e-10,
        },
        {
            "setup": base + """previous_two = np.zeros_like(previous)
profiles = np.zeros((3,n)); profiles[1]=1.04-0.03*np.sin(0.17*idx); profiles[2]=-0.05*np.cos(0.23*idx)
""",
            "call": "assemble_transmission_source(previous.copy(), previous_two.copy(), profiles.copy(), operators.copy(), nx, nz)",
            "gold_call": "_oracle_assemble_transmission_source(previous.copy(), previous_two.copy(), profiles.copy(), operators.copy(), nx, nz)",
            "tol": 2e-10,
        },
        {
            "setup": base + """previous_two = np.cos(0.07*idx)
profiles = np.zeros((3,n)); profiles[1]=1.0
""",
            "call": "assemble_transmission_source(previous.copy(), previous_two.copy(), profiles.copy(), operators.copy(), nx, nz)",
            "gold_call": "_oracle_assemble_transmission_source(previous.copy(), previous_two.copy(), profiles.copy(), operators.copy(), nx, nz)",
            "tol": 2e-10,
        },
        {
            "setup": base + """previous_two = np.zeros(previous.size-1, dtype=complex)
profiles = np.zeros((3,n)); profiles[1]=0.96; profiles[2]=0.04*np.sin(0.2*idx)
def catch_value_error(fn):
    try:
        fn(previous, previous_two, profiles, operators, nx, nz)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "catch_value_error(assemble_transmission_source)",
            "gold_call": "catch_value_error(_oracle_assemble_transmission_source)",
        },
    ]
