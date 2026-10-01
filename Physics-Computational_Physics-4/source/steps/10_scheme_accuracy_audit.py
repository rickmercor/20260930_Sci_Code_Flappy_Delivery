"""
Settle the interface, apply the constant force for the given number of steps, and report how far the resulting velocity profile sits from the reference profile in a relative two-norm.

This step calls the earlier steps in sequence. It is the end-to-end audit of the scheme against the reference solution.

Returns
-------
float, the relative two-norm deviation from the reference profile
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def scheme_accuracy_audit(N: int, Tr: float, eps: float, tau: float, Fdri: float, n_stages: int, per_stage: int, drive_steps: int) -> float:
    """Settle the interface, apply the constant force for the given number of steps, and report how far the resulting velocity profile sits from the reference profile in a relative two-norm.

    Parameters
    ----------
    N, Tr, eps, tau, Fdri, n_stages, per_stage, drive_steps
        The full configuration of the run.

    Returns
    -------
    float, the relative two-norm deviation from the reference profile
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




def _oracle_scheme_accuracy_audit(N: int, Tr: float, eps: float, tau: float,
                                  Fdri: float, n_stages: int, per_stage: int,
                                  drive_steps: int) -> float:
    E, _, _, _ = _lattice()
    CS2, _, DT, _ = _phys()
    if int(N) < 16:
        raise ValueError("N must be at least 16")
    if not (0.0 < Tr < 1.0):
        raise ValueError("the reduced temperature must lie strictly between 0 and 1")
    if tau <= 0.5:
        raise ValueError("tau must exceed 1/2 for a positive viscosity")
    if Fdri <= 0.0:
        raise ValueError("the driving force must be positive")
    if int(drive_steps) < 1 or int(n_stages) < 1 or int(per_stage) < 1:
        raise ValueError("step counts must be positive")

    f0 = _oracle_equilibrate_two_phase(N, Tr, eps, tau, n_stages, per_stage)

    # Consistency gate over the settled interface: the pairwise force must balance
    # the gradient of the non-ideal part of the pressure. This exercises the early
    # steps on the state the rest of the audit is built from, and raises if the
    # chain is inconsistent rather than silently reporting a number.
    rho0 = f0.sum(0)
    p0 = _oracle_vdw_pressure(rho0, Tr)
    psi0 = _oracle_cohesion_field(rho0, Tr)
    Fi0 = _oracle_pairwise_force(psi0)
    interior = slice(2, -2)
    resid = Fi0[interior] + np.gradient(p0 - rho0*CS2)[interior]
    scale = max(np.max(np.abs(Fi0[interior])), 1e-30)
    if np.max(np.abs(resid))/scale > 5e-2:
        raise ValueError("the settled interface does not satisfy the force balance")

    zero = np.zeros_like(rho0)
    me0 = _oracle_equilibrium_moments(rho0, zero, zero)
    Fm0 = _oracle_discrete_force_moments(zero, Fi0, zero, zero)
    Q0 = _oracle_improved_source_moments(zero, Fi0, zero, zero, psi0, eps)
    if np.max(np.abs(Q0[4])) + np.max(np.abs(Q0[6])) > 1e-30:
        raise ValueError("the source term must vanish when the fluid is at rest")
    if abs(me0[0].sum() - rho0.sum()) > 1e-8*max(rho0.sum(), 1.0):
        raise ValueError("the equilibrium moments do not conserve mass")
    if np.max(np.abs(Fm0[3])) > 1e-30:
        raise ValueError("a wall-parallel force moment appeared with no driving force")

    out = {}
    for improved in (False, True):
        f = f0.copy()
        for _ in range(int(drive_steps)):
            f = _oracle_collide_stream(f, Tr, eps, tau, Fdri, improved)
        rho = f.sum(0)
        ux = ((f*E[:, 0, None]).sum(0) + DT/2*Fdri)/rho
        ux[0] = ux[-1] = 0.0
        ut = _oracle_theory_velocity(rho, tau, Fdri)
        out[improved] = float(np.linalg.norm(ux - ut)/np.linalg.norm(ut))
    return out[True]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np",
         "call": "scheme_accuracy_audit(48, 0.85, 2.0, 0.8, 2e-7, 3, 200, 500)",
         "gold_call": "_oracle_scheme_accuracy_audit(48, 0.85, 2.0, 0.8, 2e-7, 3, 200, 500)"},
        {"setup": "import numpy as np",
         "call": "scheme_accuracy_audit(32, 0.90, 0.0, 0.8, 1e-6, 2, 150, 300)",
         "gold_call": "_oracle_scheme_accuracy_audit(32, 0.90, 0.0, 0.8, 1e-6, 2, 150, 300)"},
        {"setup": "import numpy as np",
         "call": "scheme_accuracy_audit(48, 0.85, 1.0, 1.5, 2e-7, 3, 200, 400)",
         "gold_call": "_oracle_scheme_accuracy_audit(48, 0.85, 1.0, 1.5, 2e-7, 3, 200, 400)"},
    ]
