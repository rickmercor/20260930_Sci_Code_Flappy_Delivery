"""
Final orchestrator. Chains the earlier steps end to end - families and bond kernel (step 1), damage-model normalization constant (step 2), initial bond phase-field of the pre-notched state (step 6), and the dynamics of step 8 (which invoke step 7 and through it steps 3-6 once per force evaluation) - then evaluates the method's total crack dissipation energy of the body on the final and the initial bond phase-field and returns their difference, the crack dissipation growth dE. Every earlier step is on the answer path; nothing is recomputed locally.

The total crack dissipation energy of the body is the source's nonlocal crack-surface functional: a body-and-family double sum of the bond crack-surface density, normalized with the constant of step 2 so that a fully formed crack dissipates Gc per unit area. Its value at t = 0 comes entirely from the pre-notch and is excluded from the answer by the subtraction. Its weighting, argument and prefactor must be taken from the source; the final scalar must be produced by the run and reported to at least 6 significant figures.

Returns
-------
float: crack dissipation growth E_Gamma(T) - E_Gamma(0) (J)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def pfpd_crack_dissipation(nx: int, ny: int, nz: int, dx: float, delta: float,
                           E: float, nu: float, rho0: float, Gc: float, s_c: float,
                           v0: float, dt: float, n_steps: int) -> float:
    """Crack dissipation growth of the full benchmark run (orchestrator).

    Chains the earlier steps end to end: build_families (step 1) for the
    point cloud, families, truncation-corrected bond kernel and bond masks;
    pfpd_normalization_constant('cubic') (step 2) for the damage-model
    normalization constant c0; bond_phase_field (step 6) for the bond
    phase-field of the pre-notched initial state (history 1.0e30 J/m^3 on
    the pre-notch pairs, zero elsewhere); velocity_verlet_run (step 8,
    which calls internal_force_evaluation of step 7 and through it steps
    3-6 once per force evaluation) for the final bond state. Then
    evaluates the method's total crack dissipation energy of the body -
    the source's nonlocal crack-surface functional, whose weighting and
    normalization must be taken from the source - on the final and on
    the initial bond phase-field and returns their difference, so that
    the pre-notch contribution present at t = 0 is excluded.

    Parameters
    ----------
    nx : int
        Number of grid points along x (>= 2).
    ny : int
        Number of grid points along y (>= 4).
    nz : int
        Number of grid points along z (>= 2).
    dx : float
        Grid spacing in m (> 0).
    delta : float
        Horizon radius in m (> dx).
    E : float
        Young's modulus in Pa (> 0).
    nu : float
        Poisson's ratio in (-1, 0.5).
    rho0 : float
        Mass density in kg/m^3 (> 0).
    Gc : float
        Critical energy release rate in J/m^2 (> 0).
    s_c : float
        Critical phase-field value in (0, 1).
    v0 : float
        Boundary layer speed in m/s (>= 0).
    dt : float
        Time step in s (> 0).
    n_steps : int
        Number of velocity Verlet steps (>= 0).

    Returns
    -------
    float
        Crack dissipation growth E_Gamma(T) - E_Gamma(0), in J.

    Raises
    ------
    ValueError
        If any parameter is invalid, or the crack dissipation decreased
        (irreversibility violated).
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_pfpd_crack_dissipation(nx, ny, nz, dx, delta, E, nu, rho0, Gc, s_c,
                                   v0, dt, n_steps):
    if not all(isinstance(v, (int, np.integer)) and v >= 2 for v in (nx, ny, nz)):
        raise ValueError("each grid dimension must be an integer >= 2 (the moment matrix "
                         "is singular for a single-layer grid)")
    if not (isinstance(n_steps, (int, np.integer)) and n_steps >= 0):
        raise ValueError("n_steps must be a non-negative integer")
    if not (dx > 0.0 and delta > dx and dt > 0.0):
        raise ValueError("require dx > 0, delta > dx and dt > 0")
    if not (E > 0.0 and -1.0 < nu < 0.5 and rho0 > 0.0 and Gc > 0.0 and 0.0 < s_c < 1.0
            and v0 >= 0.0):
        raise ValueError("invalid material or loading parameters")
    if ny < 4:
        raise ValueError("need at least 4 rows for boundary layers and interior")
    # step 1: point cloud, families, kernel, truncation-corrected bond weights, bond masks
    fam = _oracle_build_families(nx, ny, nz, dx, delta)
    # step 2: Griffith-consistency normalization constant of the cubic B-spline kernel
    c0 = _oracle_pfpd_normalization_constant('cubic')
    # step 6: bond phase-field of the pre-notched initial state (history 1e30 on notch bonds)
    calY0 = np.where(fam['notch'], 1.0e30, 0.0)
    s0 = _oracle_bond_phase_field(calY0, Gc, delta, c0)
    # step 8 (calling step 7, which calls steps 3-6 once per force evaluation): the dynamics
    state = _oracle_velocity_verlet_run(fam, E, nu, rho0, Gc, s_c, v0, dt, n_steps, c0)
    s_T = state['s']
    # crack dissipation functional of the method: double sum over body and family of
    # omega_b * s, i.e. both directions of every unordered pair, scaled by Gc/(c0 delta)
    omb = fam['omega_b']; V = fam['V']
    eG_T = (Gc/(c0*delta))*2.0*float(np.sum(omb*s_T))*V*V
    eG_0 = (Gc/(c0*delta))*2.0*float(np.sum(omb*s0))*V*V
    if eG_T < eG_0 - 1.0e-12*max(abs(eG_0), 1.0):
        raise ValueError("crack dissipation decreased: irreversibility violated")
    return float(eG_T - eG_0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"name": "normal_frozen_benchmark_final_answer",
         "setup": "",
         "call": "pfpd_crack_dissipation(16, 10, 2, 1.0e-3, 2.015e-3, 3.2e10, 0.25, 2450.0, 120.0, 0.95, 1.2, 2.0e-8, 360)",
         "gold_call": "_oracle_pfpd_crack_dissipation(16, 10, 2, 1.0e-3, 2.015e-3, 3.2e10, 0.25, 2450.0, 120.0, 0.95, 1.2, 2.0e-8, 360)"},
        {"name": "boundary_early_time_120_steps",
         "setup": "",
         "call": "pfpd_crack_dissipation(16, 10, 2, 1.0e-3, 2.015e-3, 3.2e10, 0.25, 2450.0, 120.0, 0.95, 1.2, 2.0e-8, 120)",
         "gold_call": "_oracle_pfpd_crack_dissipation(16, 10, 2, 1.0e-3, 2.015e-3, 3.2e10, 0.25, 2450.0, 120.0, 0.95, 1.2, 2.0e-8, 120)"},
        {"name": "edge_smaller_plate_lower_rate",
         "setup": "",
         "call": "pfpd_crack_dissipation(10, 6, 2, 1.0e-3, 2.015e-3, 3.2e10, 0.25, 2450.0, 120.0, 0.95, 0.8, 2.0e-8, 150)",
         "gold_call": "_oracle_pfpd_crack_dissipation(10, 6, 2, 1.0e-3, 2.015e-3, 3.2e10, 0.25, 2450.0, 120.0, 0.95, 0.8, 2.0e-8, 150)"},
    ]
