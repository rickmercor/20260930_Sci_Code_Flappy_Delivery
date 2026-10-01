"""
Assemble the equilibrium moment vector of the multiple-relaxation-time model for the standard nine-velocity lattice.

Collision is performed in moment space. The equilibrium moments follow the standard orthogonal transformation of the nine-velocity lattice.

Returns
-------
np.ndarray of shape (9, N), the equilibrium moments as float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def equilibrium_moments(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """Assemble the equilibrium moment vector of the multiple-relaxation-time model for the standard nine-velocity lattice.

    Parameters
    ----------
    rho, ux, uy : np.ndarray
        Density and the two velocity components at each node.

    Returns
    -------
    np.ndarray of shape (9, N), the equilibrium moments as float64
    """
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# ---------------------------------------------------------------- helpers
def _lattice():
    """D2Q9 velocities, the orthogonal moment transform, its inverse, and the
    isotropy weights of the nearest-neighbour interaction stencil."""
    E = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                  [1, 1], [-1, 1], [-1, -1], [1, -1]])
    M = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [-4, -1, -1, -1, -1, 2, 2, 2, 2],
                  [4, -2, -2, -2, -2, 1, 1, 1, 1],
                  [0, 1, 0, -1, 0, 1, -1, -1, 1],
                  [0, -2, 0, 2, 0, 1, -1, -1, 1],
                  [0, 0, 1, 0, -1, 1, 1, -1, -1],
                  [0, 0, -2, 0, 2, 1, 1, -1, -1],
                  [0, 1, -1, 1, -1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, -1, 1, -1]], dtype=float)
    WI = np.array([0.0, 1/3, 1/3, 1/3, 1/3, 1/12, 1/12, 1/12, 1/12])
    return E, M, np.linalg.inv(M), WI


def _phys():
    """Lattice sound speed squared, reference speed, time step, interaction strength."""
    return 1.0/3.0, 1.0, 1.0, -1.0


def _fluid():
    """van der Waals constants, the equation-of-state scaling factor, and the
    critical temperature and density that follow from them."""
    A, B, R, KEOS = 9.0/49.0, 2.0/21.0, 1.0, 1.0/16.0
    return A, B, R, KEOS, 8.0*A/(27.0*R*B), 1.0/(3.0*B)


def _relaxation(tau):
    """The MRT relaxation vector. s_0 = s_j = 1, s_e = s_eps = s_p = 1/tau, and
    s_q from (1/s_q - 1/2)(1/s_p - 1/2) = 1/12, which gives s_q = 9/7 at tau = 0.8."""
    sp = 1.0/tau
    sq = 1.0/((1.0/12.0)/(1.0/sp - 0.5) + 0.5)
    return np.array([1.0, sp, sp, 1.0, sq, 1.0, sq, sp, sp])[:, None]




def _oracle_equilibrium_moments(rho: np.ndarray, ux: np.ndarray,
                                uy: np.ndarray) -> np.ndarray:
    _, C, _, _ = _phys()
    rho = np.asarray(rho, dtype=float)
    ux = np.asarray(ux, dtype=float); uy = np.asarray(uy, dtype=float)
    if not (rho.shape == ux.shape == uy.shape):
        raise ValueError("rho, ux and uy must have the same shape")
    if np.any(rho <= 0.0):
        raise ValueError("density must be strictly positive")
    x, y = ux/C, uy/C
    u2 = x*x + y*y
    return np.stack([rho, -2*rho + 3*rho*u2, rho - 3*rho*u2,
                     rho*x, -rho*x, rho*y, -rho*y, rho*(x*x - y*y), rho*x*y])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Candidate and reference receive independent equivalent array inputs."""
    return [{'setup': 'import numpy as np\n'
               'rho=np.array([1.0,4.0,7.0])\n'
               'ux=np.array([0.01,-0.02,0.0])\n'
               'uy=np.array([0.0,0.03,-0.01])',
      'call': 'equilibrium_moments(rho.copy(), ux.copy(), uy.copy())',
      'gold_call': '_oracle_equilibrium_moments(rho.copy(), ux.copy(), uy.copy())'},
     {'setup': 'import numpy as np\nrho=np.array([2.0,2.0])\nux=np.zeros(2)\nuy=np.zeros(2)',
      'call': 'equilibrium_moments(rho.copy(), ux.copy(), uy.copy())',
      'gold_call': '_oracle_equilibrium_moments(rho.copy(), ux.copy(), uy.copy())'},
     {'setup': 'import numpy as np\nrho=np.full(5,0.05)\nux=np.full(5,0.2)\nuy=np.full(5,-0.2)',
      'call': 'equilibrium_moments(rho.copy(), ux.copy(), uy.copy())',
      'gold_call': '_oracle_equilibrium_moments(rho.copy(), ux.copy(), uy.copy())'}]
