"""
Assemble the third-order discrete source moment vector in the form the source paper prescribes for suppressing interfacial artefacts. Apply the source's own choice for the entry the analysis leaves undetermined.

The discrete interaction source is expressed in the D2Q9 moment order $(\rho,e,\varepsilon,j_x,q_x,j_y,q_y,p_{xx},p_{xy})$, with $x$ wall-parallel and $y$ wall-normal. Here $\epsilon$ denotes the supplied mechanical-stability parameter and is distinct from the fourth-order moment $\varepsilon$. All lattice lengths, times and speeds are one, and $G=-1$. For the supplied interaction forces $F_x,F_y$, velocities $u_x,u_y$ and positive pseudopotential $\psi$, define

$$

D=G\psi^2,\qquad k_1=k_2=-\epsilon/16.

$$

The source components are

$$

\begin{aligned}

Q_0&=Q_3=Q_5=0,\\

Q_1&=3(k_1+2k_2)(F_x^2+F_y^2)/D, \&Q_2&=-Q_1/2,\\

Q_4&=\left[\frac{30\epsilon-15}{16}F_x^2-\frac{3\epsilon}{8}F_y^2\right]u_x/D,\\

Q_6&=\left[\frac{30\epsilon-15}{16}F_y^2-\frac{3\epsilon}{8}F_x^2\right]u_y/D,\\

Q_7&=k_1(F_x^2-F_y^2)/D, \&Q_8&=k_1F_xF_y/D.

\end{aligned}

$$

These are the improved discrete-source equations; the negative-half choice for $Q_2$ is the model's heuristic prescription. Only the third-order pair $Q_4,Q_6$ has a velocity factor. The force-quadratic entries can persist at a static interface. Use interaction force, excluding the imposed body force, in these equations. Apply them independently at each node and return the components in the displayed moment order. The five input arrays have the same shape; the returned array has shape $(9,N)$.

Returns
-------
np.ndarray of shape (9, N), the source moments as float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def improved_source_moments(Fix: np.ndarray, Fiy: np.ndarray, ux: np.ndarray, uy: np.ndarray, psi: np.ndarray, eps: float) -> np.ndarray:
    """Assemble the third-order discrete source moment vector in the form the source paper prescribes for suppressing interfacial artefacts. Apply the source's own choice for the entry the analysis leaves undetermined.

    Parameters
    ----------
    Fix, Fiy : np.ndarray
        Interaction force components at each node.
    ux, uy : np.ndarray
        Velocity components at each node.
    psi : np.ndarray
        Pseudopotential at each node.
    eps : float
        The parameter appearing in the mechanical stability condition.

    Returns
    -------
    np.ndarray of shape (9, N), the source moments as float64
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




def _oracle_improved_source_moments(Fix: np.ndarray, Fiy: np.ndarray, ux: np.ndarray,
                                    uy: np.ndarray, psi: np.ndarray,
                                    eps: float) -> np.ndarray:
    _, C, _, G = _phys()
    Fix = np.asarray(Fix, dtype=float); Fiy = np.asarray(Fiy, dtype=float)
    ux = np.asarray(ux, dtype=float); uy = np.asarray(uy, dtype=float)
    psi = np.asarray(psi, dtype=float)
    if not (Fix.shape == Fiy.shape == ux.shape == uy.shape == psi.shape):
        raise ValueError("all array inputs must have the same shape")
    if np.any(psi <= 0.0):
        raise ValueError("the pseudopotential must be strictly positive here")
    if not np.isfinite(eps):
        raise ValueError("eps must be finite")
    k1 = k2 = -eps/16.0
    den = G*psi**2*C**2
    Q1 = 3.0*(k1 + 2.0*k2)*(Fix**2 + Fiy**2)/den
    Q7 = k1*(Fix**2 - Fiy**2)/den
    Q8 = k1*(Fix*Fiy)/den
    a = (30.0*eps - 15.0)/16.0
    b = 3.0*eps/8.0
    Q4 = (a*Fix**2/den - b*Fiy**2/den)*(ux/C)
    Q6 = (a*Fiy**2/den - b*Fix**2/den)*(uy/C)
    Q2 = -Q1/2.0
    Z = np.zeros_like(Q1)
    return np.stack([Z, Q1, Q2, Z, Q4, Z, Q6, Q7, Q8])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Candidate and reference receive independent equivalent array inputs."""
    return [{'setup': 'import numpy as np\n'
               'Fix=np.zeros(5)\n'
               'Fiy=np.array([0.0,0.01,-0.02,0.005,0.0])\n'
               'ux=np.array([0.0,1e-3,2e-3,1e-3,0.0])\n'
               'uy=np.zeros(5)\n'
               'psi=np.array([0.5,0.9,1.4,1.9,2.2])',
      'call': 'improved_source_moments(Fix.copy(), Fiy.copy(), ux.copy(), uy.copy(), psi.copy(), 2.0)',
      'gold_call': '_oracle_improved_source_moments(Fix.copy(), Fiy.copy(), ux.copy(), uy.copy(), '
                   'psi.copy(), 2.0)'},
     {'setup': 'import numpy as np\n'
               'Fix=np.zeros(3)\n'
               'Fiy=np.array([0.01,0.02,0.03])\n'
               'ux=np.zeros(3)\n'
               'uy=np.zeros(3)\n'
               'psi=np.array([1.0,1.5,2.0])',
      'call': 'improved_source_moments(Fix.copy(), Fiy.copy(), ux.copy(), uy.copy(), psi.copy(), 2.0)',
      'gold_call': '_oracle_improved_source_moments(Fix.copy(), Fiy.copy(), ux.copy(), uy.copy(), '
                   'psi.copy(), 2.0)'},
     {'setup': 'import numpy as np\n'
               'Fix=np.array([0.02,-0.01])\n'
               'Fiy=np.array([-0.03,0.04])\n'
               'ux=np.array([1e-3,-1e-3])\n'
               'uy=np.array([2e-3,1e-3])\n'
               'psi=np.array([0.8,2.1])',
      'call': 'improved_source_moments(Fix.copy(), Fiy.copy(), ux.copy(), uy.copy(), psi.copy(), 0.0)',
      'gold_call': '_oracle_improved_source_moments(Fix.copy(), Fiy.copy(), ux.copy(), uy.copy(), '
                   'psi.copy(), 0.0)'}]
