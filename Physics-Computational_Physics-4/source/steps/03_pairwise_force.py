"""
Compute the wall-normal component of the mesoscopic interaction force on a one-dimensional periodic lattice using the isotropy-weighted nearest-neighbour stencil of the source.

The interaction force couples each node to its neighbours through distance-dependent weights chosen to maximise the isotropy of the discrete stencil.

Returns
-------
np.ndarray, the wall-normal force component at each node as float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pairwise_force(psi: np.ndarray) -> np.ndarray:
    """Compute the wall-normal component of the mesoscopic interaction force on a one-dimensional periodic lattice using the isotropy-weighted nearest-neighbour stencil of the source.

    Parameters
    ----------
    psi : np.ndarray
        Pseudopotential on a 1-D periodic lattice.

    Returns
    -------
    np.ndarray, the wall-normal force component at each node as float64
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




def _oracle_pairwise_force(psi: np.ndarray) -> np.ndarray:
    E, _, _, WI = _lattice()
    _, _, DT, G = _phys()
    psi = np.asarray(psi, dtype=float)
    if psi.ndim != 1 or psi.size < 3:
        raise ValueError("psi must be a 1-D array of length >= 3")
    if np.any(psi < 0.0):
        raise ValueError("the cohesive field must be non-negative")
    acc = np.zeros_like(psi)
    for i in range(1, 9):
        ey = E[i, 1]
        if ey != 0:
            acc += WI[i]*np.roll(psi, -ey)*ey
    return -G*psi*acc*DT

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Candidate and reference receive independent equivalent array inputs."""
    return [{'setup': 'import numpy as np\ny = np.arange(64.0)\npsi = 1.0 + 0.5*np.tanh((y-32)/4)',
      'call': 'pairwise_force(psi.copy())',
      'gold_call': '_oracle_pairwise_force(psi.copy())'},
     {'setup': 'import numpy as np\npsi = np.ones(8)',
      'call': 'pairwise_force(psi.copy())',
      'gold_call': '_oracle_pairwise_force(psi.copy())'},
     {'setup': 'import numpy as np\npsi = np.array([0.0, 3.0, 0.0])',
      'call': 'pairwise_force(psi.copy())',
      'gold_call': '_oracle_pairwise_force(psi.copy())'}]
