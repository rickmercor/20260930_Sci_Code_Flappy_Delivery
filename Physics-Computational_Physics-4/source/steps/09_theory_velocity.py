"""
Solve the one-dimensional steady momentum balance for the velocity profile driven by a constant body force, using the given density profile and no-slip walls.

The reference profile follows from the macroscopic momentum equation reduced to one dimension, integrated with the density profile taken from the simulation itself.

Returns
-------
np.ndarray, the reference velocity at each node as float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def theory_velocity(rho: np.ndarray, tau: float, Fdri: float) -> np.ndarray:
    """Solve the one-dimensional steady momentum balance for the velocity profile driven by a constant body force, using the given density profile and no-slip walls.

    Parameters
    ----------
    rho : np.ndarray
        Settled density profile across the channel.
    tau : float
        Dimensionless relaxation time.
    Fdri : float
        Constant driving force.

    Returns
    -------
    np.ndarray, the reference velocity at each node as float64
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




def _oracle_theory_velocity(rho: np.ndarray, tau: float, Fdri: float) -> np.ndarray:
    CS2, _, _, _ = _phys()
    rho = np.asarray(rho, dtype=float)
    if rho.ndim != 1 or rho.size < 3:
        raise ValueError("rho must be a 1-D array of length >= 3")
    if np.any(rho <= 0.0):
        raise ValueError("density must be strictly positive")
    if tau <= 0.5:
        raise ValueError("tau must exceed 1/2")
    N = rho.size
    mu = rho*CS2*(tau - 0.5)
    mh = 0.5*(mu[:-1] + mu[1:])
    A = np.zeros((N, N)); rhs = np.full(N, -float(Fdri))
    for j in range(1, N-1):
        A[j, j-1] = mh[j-1]; A[j, j+1] = mh[j]; A[j, j] = -(mh[j-1] + mh[j])
    A[0, 0] = A[-1, -1] = 1.0; rhs[0] = rhs[-1] = 0.0
    return np.linalg.solve(A, rhs)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Candidate and reference receive independent equivalent array inputs."""
    return [{'setup': 'import numpy as np\n'
               'y=np.arange(64.0)\n'
               'rho=1.0+5.0*0.5*(np.tanh((y-16)/4)-np.tanh((y-48)/4))',
      'call': 'theory_velocity(rho.copy(), 0.8, 2e-07)',
      'gold_call': '_oracle_theory_velocity(rho.copy(), 0.8, 2e-07)'},
     {'setup': 'import numpy as np\nrho=np.full(16, 2.0)',
      'call': 'theory_velocity(rho.copy(), 0.8, 2e-07)',
      'gold_call': '_oracle_theory_velocity(rho.copy(), 0.8, 2e-07)'},
     {'setup': 'import numpy as np\nrho=np.linspace(0.2, 8.0, 24)',
      'call': 'theory_velocity(rho.copy(), 1.5, 1e-06)',
      'gold_call': '_oracle_theory_velocity(rho.copy(), 1.5, 1e-06)'}]
