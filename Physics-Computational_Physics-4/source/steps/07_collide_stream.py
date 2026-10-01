"""
Advance the distribution functions by one collision and one streaming step, with no-slip walls at the first and last nodes.

The populations have shape $(9,N)$, with discrete velocities in the order

$$

(0,0),(1,0),(0,1),(-1,0),(0,-1),(1,1),(-1,1),(-1,-1),(1,-1).

$$

Let $s_i=e_{ix}^2+e_{iy}^2$. The nine rows of the moment transform $M$ are the following basis functions evaluated at each velocity:

$$

1,\quad 3s_i-4,\quad (9s_i^2-21s_i+8)/2,\quad e_{ix},\quad (3s_i-5)e_{ix},\quad e_{iy},\quad (3s_i-5)e_{iy},\quad e_{ix}^2-e_{iy}^2,\quad e_{ix}e_{iy}.

$$

Use the previous pressure, pseudopotential, interaction-force and moment functions. The interaction force is wall-normal; the imposed body force is wall-parallel. At each node,

$$

\rho=\sum_i f_i,\qquad

\mathbf F_{\mathrm{tot}}=(\mathrm{drive},F_{\mathrm{int},y}),\qquad

\mathbf u=\frac{\sum_i\mathbf e_i f_i+\mathbf F_{\mathrm{tot}}/2}{\rho}.

$$

Set both velocity components to zero at the first and last nodes before forming any equilibrium or force/source moments. The force moments $F_m$ use total force; the interaction source $Q_m$ uses interaction force alone. For the improved branch use the preceding source-vector function. For the earlier branch, retain that vector's force-quadratic components $Q_1,Q_7,Q_8$ and its zero components $Q_0,Q_3,Q_5$, and replace $Q_2$ by $-Q_1$ and $Q_4,Q_6$ by zero.



The unit-time-step collision is

$$

m=Mf,\qquad

\bar m=m+F_m-S(m-m^{\mathrm{eq}}+F_m/2)+SQ_m,\qquad

\bar f=M^{-1}\bar m.

$$

Here $S=\operatorname{diag}(1,s_p,s_p,1,s_q,1,s_q,s_p,s_p)$, $s_p=1/\tau$ and $(1/s_q-1/2)(1/s_p-1/2)=1/12$. Streaming moves each population by its wall-normal velocity component; the wall-parallel direction is uniform. After streaming, at each wall node replace each population pointing into the fluid by the opposite-direction population of $\bar f$ at that same wall node. The interaction-force stencil remains periodic in the wall-normal coordinate. Return the post-streaming, wall-adjusted population array. The tested configurations have positive densities in the preceding equation-of-state domain and $\tau>1/2$.

Returns
-------
np.ndarray of shape (9, N), the post-streaming distributions as float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def collide_stream(f: np.ndarray, Tr: float, eps: float, tau: float, drive: float, improved: bool) -> np.ndarray:
    """Advance the distribution functions by one collision and one streaming step, with no-slip walls at the first and last nodes.

    Parameters
    ----------
    f : np.ndarray of shape (9, N)
        Distribution functions.
    Tr : float
        Reduced temperature.
    eps : float
        Mechanical-stability parameter.
    tau : float
        Dimensionless relaxation time.
    drive : float
        Constant driving force parallel to the walls.
    improved : bool
        Select the source paper's source term rather than the earlier one.

    Returns
    -------
    np.ndarray of shape (9, N), the post-streaming distributions as float64
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




def _original_source_moments(Fix, Fiy, psi, eps):
    """Huang and Wu's original third-order source term, Eq (11):
    Q_{m,4} = Q_{m,6} = 0 and Q_{m,2} = -Q_{m,1}."""
    _, C, _, G = _phys()
    Fix = np.asarray(Fix, dtype=float); Fiy = np.asarray(Fiy, dtype=float)
    psi = np.asarray(psi, dtype=float)
    if np.any(psi <= 0.0):
        raise ValueError("the pseudopotential must be strictly positive here")
    k1 = k2 = -eps/16.0
    den = G*psi**2*C**2
    Q1 = 3.0*(k1 + 2.0*k2)*(Fix**2 + Fiy**2)/den
    Q7 = k1*(Fix**2 - Fiy**2)/den
    Q8 = k1*(Fix*Fiy)/den
    Z = np.zeros_like(Q1)
    return np.stack([Z, Q1, -Q1, Z, Z, Z, Z, Q7, Q8])

def _oracle_collide_stream(f: np.ndarray, Tr: float, eps: float, tau: float,
                           drive: float, improved: bool) -> np.ndarray:
    E, M, Minv, _ = _lattice()
    _, _, DT, _ = _phys()
    f = np.asarray(f, dtype=float)
    if f.ndim != 2 or f.shape[0] != 9:
        raise ValueError("f must have shape (9, N)")
    if tau <= 0.5:
        raise ValueError("tau must exceed 1/2 for a positive viscosity")
    N = f.shape[1]
    rho = np.maximum(f.sum(0), 1e-12)
    psi = _oracle_cohesion_field(rho, Tr)
    Fiy = _oracle_pairwise_force(psi)
    Fix = np.zeros(N)
    Fx, Fy = Fix + drive, Fiy
    ux = ((f*E[:, 0, None]).sum(0) + DT/2*Fx)/rho
    uy = ((f*E[:, 1, None]).sum(0) + DT/2*Fy)/rho
    ux[0] = ux[-1] = uy[0] = uy[-1] = 0.0
    S = _relaxation(tau)
    m = M @ f
    me = _oracle_equilibrium_moments(rho, ux, uy)
    Fm = _oracle_discrete_force_moments(Fx, Fy, ux, uy)
    if improved:
        Qm = _oracle_improved_source_moments(Fix, Fiy, ux, uy, psi, eps)
    else:
        Qm = _original_source_moments(Fix, Fiy, psi, eps)
    fbar = Minv @ (m + DT*Fm - S*(m - me + DT/2*Fm) + S*Qm)
    out = np.empty_like(f)
    for i in range(9):
        out[i] = np.roll(fbar[i], E[i, 1])
    for i, o in ((2, 4), (5, 7), (6, 8)):
        out[i, 0] = fbar[o, 0]
    for i, o in ((4, 2), (7, 5), (8, 6)):
        out[i, -1] = fbar[o, -1]
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Candidate and reference receive independent equivalent array inputs."""
    return [{'setup': 'import numpy as np\n'
               'N=48\n'
               'y=np.arange(N,dtype=float)\n'
               '_rc=_fluid()[5]\n'
               'rho=_rc+0.15*_rc*(np.tanh((y-N/4)/8)-np.tanh((y-3*N/4)/8))\n'
               'f=_lattice()[2]@_oracle_equilibrium_moments(rho,np.zeros(N),np.zeros(N))',
      'call': 'collide_stream(f.copy(), 0.85, 2.0, 0.8, 2e-07, True)',
      'gold_call': '_oracle_collide_stream(f.copy(), 0.85, 2.0, 0.8, 2e-07, True)'},
     {'setup': 'import numpy as np\n'
               'N=48\n'
               'y=np.arange(N,dtype=float)\n'
               '_rc=_fluid()[5]\n'
               'rho=_rc+0.15*_rc*(np.tanh((y-N/4)/8)-np.tanh((y-3*N/4)/8))\n'
               'f=_lattice()[2]@_oracle_equilibrium_moments(rho,np.zeros(N),np.zeros(N))',
      'call': 'collide_stream(f.copy(), 0.85, 2.0, 0.8, 0.0, False)',
      'gold_call': '_oracle_collide_stream(f.copy(), 0.85, 2.0, 0.8, 0.0, False)'},
     {'setup': 'import numpy as np\n'
               'N=32\n'
               'rho=np.full(N,_fluid()[5])\n'
               'f=_lattice()[2]@_oracle_equilibrium_moments(rho,np.zeros(N),np.zeros(N))',
      'call': 'collide_stream(f.copy(), 0.9, 0.0, 1.5, 1e-06, True)',
      'gold_call': '_oracle_collide_stream(f.copy(), 0.9, 0.0, 1.5, 1e-06, True)'}]
