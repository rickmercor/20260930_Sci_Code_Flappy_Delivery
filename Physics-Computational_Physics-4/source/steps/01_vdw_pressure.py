"""
Evaluate the equation of state of the working fluid at the given densities and reduced temperature, using the scaled form adopted by the source.

The multiphase model draws its phase separation from a non-monotonic equation of state. A scaling factor is applied to the whole expression to control the interface thickness.

Returns
-------
np.ndarray, the pressure at each node as float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def vdw_pressure(rho: np.ndarray, Tr: float) -> np.ndarray:
    """Evaluate the equation of state of the working fluid at the given densities and reduced temperature, using the scaled form adopted by the source.

    Parameters
    ----------
    rho : np.ndarray
        Densities at each lattice node.
    Tr : float
        Reduced temperature T/T_cr.

    Returns
    -------
    np.ndarray, the pressure at each node as float64
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




def _oracle_vdw_pressure(rho: np.ndarray, Tr: float) -> np.ndarray:
    A, B, R, KEOS, TCR, _ = _fluid()
    rho = np.asarray(rho, dtype=float)
    if np.any(rho <= 0.0):
        raise ValueError("density must be strictly positive")
    if np.any(B*rho >= 1.0):
        raise ValueError("b*rho must be < 1 for the van der Waals EOS")
    if not np.isfinite(Tr) or Tr <= 0.0:
        raise ValueError("reduced temperature must be positive and finite")
    return KEOS*(rho*R*(Tr*TCR)/(1.0 - B*rho) - A*rho**2)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Candidate and reference receive independent equivalent array inputs."""
    return [{'setup': 'import numpy as np\nrho = np.array([0.5, 3.0, 7.0])',
      'call': 'vdw_pressure(rho.copy(), 0.54)',
      'gold_call': '_oracle_vdw_pressure(rho.copy(), 0.54)'},
     {'setup': 'import numpy as np\nrho = np.array([0.02])',
      'call': 'vdw_pressure(rho.copy(), 0.95)',
      'gold_call': '_oracle_vdw_pressure(rho.copy(), 0.95)'},
     {'setup': 'import numpy as np\nrho = np.array([10.4])',
      'call': 'vdw_pressure(rho.copy(), 0.54)',
      'gold_call': '_oracle_vdw_pressure(rho.copy(), 0.54)'}]
