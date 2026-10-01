"""
Assemble the discrete force moment vector entering the collision step.

The forcing term is expressed directly in moment space so that it is consistent with the multiple-relaxation-time collision.

Returns
-------
np.ndarray of shape (9, N), the force moments as float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def discrete_force_moments(Fx: np.ndarray, Fy: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """Assemble the discrete force moment vector entering the collision step.

    Parameters
    ----------
    Fx, Fy : np.ndarray
        Total force components at each node.
    ux, uy : np.ndarray
        Velocity components at each node.

    Returns
    -------
    np.ndarray of shape (9, N), the force moments as float64
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




def _oracle_discrete_force_moments(Fx: np.ndarray, Fy: np.ndarray, ux: np.ndarray,
                                   uy: np.ndarray) -> np.ndarray:
    _, C, _, _ = _phys()
    Fx = np.asarray(Fx, dtype=float); Fy = np.asarray(Fy, dtype=float)
    ux = np.asarray(ux, dtype=float); uy = np.asarray(uy, dtype=float)
    if not (Fx.shape == Fy.shape == ux.shape == uy.shape):
        raise ValueError("all four inputs must have the same shape")
    fx, fy = Fx/C, Fy/C
    x, y = ux/C, uy/C
    Fu = fx*x + fy*y
    return np.stack([np.zeros_like(fx), 6*Fu, -6*Fu, fx, -fx, fy, -fy,
                     2*fx*x - 2*fy*y, fx*y + fy*x])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Candidate and reference receive independent equivalent array inputs."""
    return [{'setup': 'import numpy as np\n'
               'Fx=np.array([1e-7,0.0,-2e-7])\n'
               'Fy=np.array([0.01,-0.02,0.0])\n'
               'ux=np.array([0.01,0.0,-0.01])\n'
               'uy=np.zeros(3)',
      'call': 'discrete_force_moments(Fx.copy(), Fy.copy(), ux.copy(), uy.copy())',
      'gold_call': '_oracle_discrete_force_moments(Fx.copy(), Fy.copy(), ux.copy(), uy.copy())'},
     {'setup': 'import numpy as np\nFx=np.zeros(4)\nFy=np.zeros(4)\nux=np.zeros(4)\nuy=np.zeros(4)',
      'call': 'discrete_force_moments(Fx.copy(), Fy.copy(), ux.copy(), uy.copy())',
      'gold_call': '_oracle_discrete_force_moments(Fx.copy(), Fy.copy(), ux.copy(), uy.copy())'},
     {'setup': 'import numpy as np\n'
               'Fx=np.full(3,0.5)\n'
               'Fy=np.full(3,-0.5)\n'
               'ux=np.full(3,0.3)\n'
               'uy=np.full(3,0.3)',
      'call': 'discrete_force_moments(Fx.copy(), Fy.copy(), ux.copy(), uy.copy())',
      'gold_call': '_oracle_discrete_force_moments(Fx.copy(), Fy.copy(), ux.copy(), uy.copy())'}]
