"""
Invert the model's pressure relation to obtain the pseudopotential at each node. Raise ValueError where the relation admits no real value.

The model's bulk pressure relation in unit lattice spacing is

$$

p_{\mathrm{EOS}}=\rho c_s^2+\frac{G\psi^2}{2},\qquad c_s^2=\frac13,\quad G=-1.

$$

Obtain $p_{\mathrm{EOS}}$ from the preceding pressure step at the supplied density and reduced temperature, and use the nonnegative square root when inverting this relation. The interaction-force stencil uses weights $1/3$ axially and $1/12$ diagonally; with this normalization the interaction term is $G\psi^2/2$. If the square-root argument is negative, raise `ValueError`. The density inputs obey the preceding pressure step's domain. This operation returns one pseudopotential value per supplied density.

Returns
-------
np.ndarray, the pseudopotential at each node as float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def cohesion_field(rho: np.ndarray, Tr: float) -> np.ndarray:
    """Invert the model's pressure relation to obtain the pseudopotential at each node. Raise ValueError where the relation admits no real value.

    Parameters
    ----------
    rho : np.ndarray
        Densities at each lattice node.
    Tr : float
        Reduced temperature T/T_cr.

    Returns
    -------
    np.ndarray, the pseudopotential at each node as float64
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




def _oracle_cohesion_field(rho: np.ndarray, Tr: float) -> np.ndarray:
    CS2, _, DT, G = _phys()
    rho = np.asarray(rho, dtype=float)
    arg = 2.0*(_oracle_vdw_pressure(rho, Tr) - rho*CS2)/(G*DT**2)
    if np.any(arg < 0.0):
        raise ValueError("the squared cohesive field is negative: density out of range")
    return np.sqrt(arg)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Candidate and reference receive independent equivalent array inputs."""
    return [{'setup': 'import numpy as np\nrho = np.array([0.4, 2.0, 6.5])',
      'call': 'cohesion_field(rho.copy(), 0.54)',
      'gold_call': '_oracle_cohesion_field(rho.copy(), 0.54)'},
     {'setup': 'import numpy as np\nrho = np.array([0.05])',
      'call': 'cohesion_field(rho.copy(), 0.72)',
      'gold_call': '_oracle_cohesion_field(rho.copy(), 0.72)'},
     {'setup': 'import numpy as np\nrho = np.linspace(0.1, 9.0, 25)',
      'call': 'cohesion_field(rho.copy(), 0.65)',
      'gold_call': '_oracle_cohesion_field(rho.copy(), 0.65)'}]
