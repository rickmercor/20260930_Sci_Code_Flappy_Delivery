"""
Runs the frozen explicit dynamics of the pre-notched benchmark on the families of step 1: initial bond phase-field from the pre-notch history through step 6, velocity Dirichlet layers, one initializing internal-force evaluation of step 7, and n_steps velocity Verlet steps with exactly one internal-force evaluation of step 7 each. Returns the final kinematic state and the final bond history and phase-field, which the orchestrator of step 9 turns into the crack dissipation energy. Validates the boundary protocol, the integrator layout, and the irreversible accumulation of damage.

Velocity Verlet with the frozen protocol: positions update first (u <- u + dt v + dt^2 a/2, prescribed layers overwritten with their exact positions -/+ v0 t), then exactly one internal-force evaluation of step 7 updates the bond state and yields the new acceleration (treated as zero on the prescribed layers), then velocities update (v <- v + dt (a_old + a_new)/2, prescribed layers overwritten with -/+ v0); one force evaluation of the initial state initializes the acceleration before the first step. The pre-notch enters only through the initial history 1.0e30 J/m^3 on the pre-notch pairs, turned into the initial bond phase-field by step 6; no bond is removed, no damping, viscosity, contact or time-step adaptation is added.

Returns
-------
dict: final 'u', 'v', 'a' (N,3) and bond 'calY', 's' (P,) after n_steps velocity Verlet steps
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def velocity_verlet_run(fam: dict, E: float, nu: float, rho0: float, Gc: float,
                        s_c: float, v0: float, dt: float, n_steps: int,
                        c0: float) -> dict:
    """Explicit dynamics of the pre-notched benchmark; returns the final state.

    Runs the frozen benchmark protocol on the bond families of step 1:
    the pre-notch bond pairs start fully damaged through a damage history
    of 1.0e30 J/m^3 turned into the initial bond phase-field by step 6;
    rows j = 0 and j = ny-1 are velocity Dirichlet layers with
    v = (0, -v0, 0) and (0, +v0, 0) for all t >= 0, exact displacement
    -/+ v0 t (zero x, z components) and zero acceleration; all other
    points start at rest with zero displacement; no external body force.
    One internal-force evaluation of step 7 on the initial state
    initializes the acceleration; then n_steps velocity Verlet steps of
    size dt follow, each with the position update first (prescribed
    layers overwritten with their exact positions), then exactly one
    internal-force evaluation of step 7 that updates the bond state and
    yields the new acceleration, then the velocity update (prescribed
    layers overwritten with their prescribed velocity). The final state
    is consumed by the orchestrator of step 9.

    Parameters
    ----------
    fam : dict
        Bond-family dictionary returned by build_families (step 1); needs
        ny >= 4.
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
    c0 : float
        Griffith-consistency normalization constant of the kernel (output
        of step 2), in (0, 1).

    Returns
    -------
    dict
        'u' : (N, 3) displacement in m, 'v' : (N, 3) velocity in m/s,
        'a' : (N, 3) acceleration in m/s^2, all at t = n_steps * dt;
        'calY' : (P,) bond damage-history variable in J/m^3 and
        's' : (P,) bond phase-field in [0, 1], both after the last force
        evaluation.

    Raises
    ------
    ValueError
        If n_steps is not a non-negative integer, ny < 4, or a material,
        loading, integration or normalization parameter is invalid.
    """
    return {}

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_velocity_verlet_run(fam, E, nu, rho0, Gc, s_c, v0, dt, n_steps, c0):
    if not (isinstance(n_steps, (int, np.integer)) and n_steps >= 0):
        raise ValueError("n_steps must be a non-negative integer")
    if not (E > 0.0 and -1.0 < nu < 0.5 and rho0 > 0.0 and Gc > 0.0 and 0.0 < s_c < 1.0
            and v0 >= 0.0 and dt > 0.0 and 0.0 < c0 < 1.0):
        raise ValueError("invalid material, loading, integration or normalization parameters")
    ijk = fam['ijk']; ny = fam['ny']; notch = fam['notch']
    N = fam['X'].shape[0]
    if ny < 4:
        raise ValueError("need at least 4 rows for boundary layers and interior")
    jt = ijk[:, 1] == ny - 1
    jb = ijk[:, 1] == 0
    bc = jt | jb
    vbc = np.zeros((N, 3)); vbc[jt, 1] = v0; vbc[jb, 1] = -v0
    u = np.zeros((N, 3)); v = vbc.copy()
    # pre-notch: fully damaged initial bond state through the history variable (step 6)
    calY = np.where(notch, 1.0e30, 0.0)
    s = _oracle_bond_phase_field(calY, Gc, fam['delta'], c0)
    # one force evaluation of the initial state initializes the acceleration (step 7)
    out = _oracle_internal_force_evaluation(fam, u, calY, s, E, nu, Gc, s_c, c0)
    calY, s = out['calY'], out['s']
    a = out['B']/rho0; a[bc] = 0.0
    for step in range(n_steps):
        u = u + dt*v + 0.5*dt*dt*a
        u[bc] = vbc[bc]*(dt*(step + 1))
        out = _oracle_internal_force_evaluation(fam, u, calY, s, E, nu, Gc, s_c, c0)
        calY, s = out['calY'], out['s']
        an = out['B']/rho0; an[bc] = 0.0
        v = v + 0.5*dt*(a + an); v[bc] = vbc[bc]
        a = an
    return {'u': u, 'v': v, 'a': a, 'calY': calY, 's': s}

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = "import numpy as np\ndef _ke(build, run, c0, nx, ny, nz, v0, n_steps):\n    fam = build(nx, ny, nz, 1.0e-3, 2.015e-3)\n    st = run(fam, 3.2e10, 0.25, 2450.0, 120.0, 0.95, v0, 2.0e-8, n_steps, c0)\n    return float(0.5*2450.0*np.sum(st['v']**2)*fam['V'])\ndef _grown_pairs(build, run, c0, nx, ny, nz, v0, n_steps):\n    fam = build(nx, ny, nz, 1.0e-3, 2.015e-3)\n    st = run(fam, 3.2e10, 0.25, 2450.0, 120.0, 0.95, v0, 2.0e-8, n_steps, c0)\n    return float(np.sum((st['s'] > 0.5) & ~fam['notch']))\ndef _s_sum(build, run, c0, nx, ny, nz, v0, n_steps):\n    fam = build(nx, ny, nz, 1.0e-3, 2.015e-3)\n    st = run(fam, 3.2e10, 0.25, 2450.0, 120.0, 0.95, v0, 2.0e-8, n_steps, c0)\n    return float(np.sum(st['s']))"
    return [
        {"name": "normal_frozen_run_360_steps_kinetic_energy",
         "setup": setup,
         "call": "_ke(build_families, velocity_verlet_run, pfpd_normalization_constant('cubic'), 16, 10, 2, 1.2, 360)",
         "gold_call": "_ke(_oracle_build_families, _oracle_velocity_verlet_run, _oracle_pfpd_normalization_constant('cubic'), 16, 10, 2, 1.2, 360)"},
        {"name": "normal_frozen_run_360_steps_grown_bond_pairs",
         "setup": setup,
         "call": "_grown_pairs(build_families, velocity_verlet_run, pfpd_normalization_constant('cubic'), 16, 10, 2, 1.2, 360)",
         "gold_call": "_grown_pairs(_oracle_build_families, _oracle_velocity_verlet_run, _oracle_pfpd_normalization_constant('cubic'), 16, 10, 2, 1.2, 360)"},
        {"name": "boundary_early_time_120_steps_phase_field_sum",
         "setup": setup,
         "call": "_s_sum(build_families, velocity_verlet_run, pfpd_normalization_constant('cubic'), 16, 10, 2, 1.2, 120)",
         "gold_call": "_s_sum(_oracle_build_families, _oracle_velocity_verlet_run, _oracle_pfpd_normalization_constant('cubic'), 16, 10, 2, 1.2, 120)"},
        {"name": "edge_zero_steps_initial_prenotch_state",
         "setup": setup,
         "call": "_s_sum(build_families, velocity_verlet_run, pfpd_normalization_constant('cubic'), 16, 10, 2, 1.2, 0)",
         "gold_call": "_s_sum(_oracle_build_families, _oracle_velocity_verlet_run, _oracle_pfpd_normalization_constant('cubic'), 16, 10, 2, 1.2, 0)"},
    ]
