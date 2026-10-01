"""
Solve Poisson's equation for the electrostatic potential on the non-uniform grid from the current space charge.

The potential is not an input: it follows from the space charge, which moves as the carriers redistribute, so it is recomputed from the current densities at every time step. The net charge density at a node is q(p - n - NA): mobile holes count positive, mobile electrons negative, and the ionised acceptors form a fixed negative background (there are no donors).

Because the relative permittivity jumps at the junction and the grid spacing changes there too, the operator must be discretised in conservative form, as a balance of electric displacement over a control volume. The displacement at the midpoint of the interval between nodes l and l + 1 is the vacuum permittivity times the arithmetic mean of the two nodal relative permittivities, times the potential difference across that interval divided by that interval's own width, with the sign of -grad(phi). The displacement leaving the control volume of interior node l through its two faces must equal the charge that volume holds, where the control volume extends halfway to each neighbour and so has width (z[l+1] - z[l-1]) / 2. Pulling the permittivity outside the derivative, or using a single spacing for every interval, silently violates continuity of the displacement field at the interface.

The two end nodes carry Dirichlet values: phi_left at node 0 and phi_right at node N - 1. The interior equations form a tridiagonal linear system solved directly. Use q = 1.602176634e-19 C and a vacuum permittivity of 8.8541878128e-14 F/cm; lengths are in cm, potential in V and densities in cm^-3.

Returns
-------
np.ndarray of shape (N,), the electrostatic potential at every node in V as float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def poisson_potential(p: np.ndarray, n: np.ndarray, NA: np.ndarray, er: np.ndarray,
                      z: np.ndarray, phi_left: float, phi_right: float) -> np.ndarray:
    '''Electrostatic potential on a non-uniform grid with Dirichlet contacts.

    Parameters
    ----------
    p, n : np.ndarray
        Arrays of shape (N,) of hole and electron densities in cm^-3.
    NA : np.ndarray
        Array of shape (N,) of acceptor doping densities in cm^-3.
    er : np.ndarray
        Array of shape (N,) of relative permittivities, all positive.
    z : np.ndarray
        Array of shape (N,) of strictly increasing node positions in cm.
    phi_left, phi_right : float
        Potentials in V imposed at node 0 and node N - 1.

    Returns
    -------
    phi : np.ndarray
        Array of shape (N,) of nodal potentials in V; phi[0] = phi_left and phi[-1] = phi_right.

    Raises
    ------
    ValueError
        If the five arrays are not one-dimensional with a common length of at least 3, if z is not
        strictly increasing, or if any relative permittivity is not positive.
    '''
    return phi

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_poisson_potential(p: np.ndarray, n: np.ndarray, NA: np.ndarray, er: np.ndarray,
                              z: np.ndarray, phi_left: float, phi_right: float) -> np.ndarray:
    q = 1.602176634e-19
    eps0 = 8.8541878128e-14
    p = np.array(p, dtype=float); n = np.array(n, dtype=float)
    NA = np.array(NA, dtype=float); er = np.array(er, dtype=float); z = np.array(z, dtype=float)
    N = z.size
    if z.ndim != 1 or N < 3 or any(a.shape != z.shape for a in (p, n, NA, er)):
        raise ValueError("p, n, NA, er and z must be one-dimensional with a common length of at least 3")
    h = np.diff(z)
    if np.any(h <= 0.0):
        raise ValueError("z must be strictly increasing")
    if np.any(er <= 0.0):
        raise ValueError("relative permittivities must be positive")
    face = 0.5 * (er[:-1] + er[1:]) / h
    w = 2.0 / (z[2:] - z[:-2])
    i = np.arange(1, N - 1)
    A = np.zeros((N, N))
    A[i, i - 1] = w * face[:-1]
    A[i, i + 1] = w * face[1:]
    A[i, i] = -w * (face[:-1] + face[1:])
    A[0, 0] = 1.0
    A[-1, -1] = 1.0
    b = np.empty(N)
    b[0] = float(phi_left)
    b[-1] = float(phi_right)
    b[1:-1] = -(q / eps0) * (p[1:-1] - n[1:-1] - NA[1:-1])
    return np.linalg.solve(A, b)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
z = np.concatenate([np.arange(21) * 1.0e-7, 20.0e-7 + np.arange(1, 81) * 0.25e-7])
NA = np.concatenate([np.full(21, 2.81e19), np.full(80, 1.00e17)])
er = np.concatenate([np.full(21, 3.0), np.full(80, 9.4)])
p = NA.copy(); n = np.concatenate([np.full(21, 9.4552e-8), np.full(80, 2.5281e-29)])""",
            "call": "poisson_potential(p, n, NA, er, z, 0.0, 1.0)",
            "gold_call": "_oracle_poisson_potential(p, n, NA, er, z, 0.0, 1.0)",
        },
        {
            "setup": """import numpy as np
z = np.concatenate([np.arange(21) * 1.0e-7, 20.0e-7 + np.arange(1, 81) * 0.25e-7])
NA = np.concatenate([np.full(21, 2.81e19), np.full(80, 1.00e17)])
er = np.concatenate([np.full(21, 3.0), np.full(80, 9.4)])
p = NA * (1.0 + 1.0e-3 * np.sin(np.arange(101) * 0.7)); n = np.full(101, 5.0e15)""",
            "call": "poisson_potential(p, n, NA, er, z, 0.0, 1.0)",
            "gold_call": "_oracle_poisson_potential(p, n, NA, er, z, 0.0, 1.0)",
        },
        {
            "setup": """import numpy as np
z = np.array([0.0, 1.0e-7, 2.0e-7, 2.5e-7, 3.0e-7, 3.5e-7])
NA = np.zeros(6); p = np.zeros(6); n = np.zeros(6)
er = np.array([2.0, 2.0, 2.0, 8.0, 8.0, 8.0])""",
            "call": "poisson_potential(p, n, NA, er, z, -0.5, 0.5)",
            "gold_call": "_oracle_poisson_potential(p, n, NA, er, z, -0.5, 0.5)",
        },
        {
            "setup": """import numpy as np
z = np.array([0.0, 2.0e-7, 3.0e-7])
NA = np.array([0.0, 1.0e18, 0.0]); p = np.zeros(3); n = np.zeros(3); er = np.full(3, 4.0)""",
            "call": "poisson_potential(p, n, NA, er, z, 0.0, 0.0)",
            "gold_call": "_oracle_poisson_potential(p, n, NA, er, z, 0.0, 0.0)",
        },
        {
            "setup": """import numpy as np
def _probe(fn):
    hits = 0
    good = np.ones(4)
    for z, er in ((np.array([0.0, 1.0, 1.0, 2.0]), good), (np.array([0.0, 1.0, 2.0, 3.0]), np.array([1.0, 0.0, 1.0, 1.0])),
                  (np.array([0.0, 1.0]), np.ones(2))):
        try:
            fn(np.ones(z.size), np.ones(z.size), np.ones(z.size), er, z, 0.0, 1.0)
        except ValueError:
            hits += 1
    return float(hits)""",
            "call": "_probe(poisson_potential)",
            "gold_call": "_probe(_oracle_poisson_potential)",
        },
    ]
