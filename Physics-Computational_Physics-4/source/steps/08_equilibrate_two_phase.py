"""
Relax an initially smooth two-layer profile to a settled interface at the target temperature, approaching it through a sequence of intermediate temperatures rather than in one jump.

A settled interface is reached by starting close to the critical point, where the cohesive force is weak, and reducing the temperature in stages so the interface can re-form at each one.

Returns
-------
np.ndarray of shape (9, N), the settled distributions as float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def equilibrate_two_phase(N: int, Tr_target: float, eps: float, tau: float, n_stages: int, per_stage: int) -> np.ndarray:
    """Relax an initially smooth two-layer profile to a settled interface at the target temperature, approaching it through a sequence of intermediate temperatures rather than in one jump.

    Parameters
    ----------
    N : int
        Number of lattice nodes across the channel.
    Tr_target : float
        Final reduced temperature.
    eps : float
        Mechanical-stability parameter.
    tau : float
        Dimensionless relaxation time.
    n_stages : int
        Number of temperature stages.
    per_stage : int
        Steps held at each stage.

    Returns
    -------
    np.ndarray of shape (9, N), the settled distributions as float64
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




def _oracle_equilibrate_two_phase(N: int, Tr_target: float, eps: float, tau: float,
                                  n_stages: int, per_stage: int) -> np.ndarray:
    _, _, Minv, _ = _lattice()
    _, _, _, _, _, RHOCR = _fluid()
    if N < 16:
        raise ValueError("N must be at least 16")
    if n_stages < 1 or per_stage < 1:
        raise ValueError("n_stages and per_stage must be positive")
    y = np.arange(N, dtype=float)
    rho = RHOCR + 0.15*RHOCR*(np.tanh((y - N/4)/8.0) - np.tanh((y - 3*N/4)/8.0))
    f = Minv @ _oracle_equilibrium_moments(rho, np.zeros(N), np.zeros(N))
    for Tr in ([Tr_target] if n_stages == 1
           else np.linspace(0.95, Tr_target, n_stages)):
        for _ in range(per_stage):
            f = _oracle_collide_stream(f, Tr, eps, tau, 0.0, False)
    return f

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np",
         "call": "equilibrate_two_phase(48, 0.85, 2.0, 0.8, 3, 200).sum(0)",
         "gold_call": "_oracle_equilibrate_two_phase(48, 0.85, 2.0, 0.8, 3, 200).sum(0)"},
        {"setup": "import numpy as np",
         "call": "equilibrate_two_phase(32, 0.94, 0.0, 0.8, 1, 100).sum(0)",
         "gold_call": "_oracle_equilibrate_two_phase(32, 0.94, 0.0, 0.8, 1, 100).sum(0)"},
        {"setup": "import numpy as np",
         "call": "equilibrate_two_phase(64, 0.80, 1.0, 1.5, 4, 300).sum(0)",
         "gold_call": "_oracle_equilibrate_two_phase(64, 0.80, 1.0, 1.5, 4, 300).sum(0)"},
    ]
