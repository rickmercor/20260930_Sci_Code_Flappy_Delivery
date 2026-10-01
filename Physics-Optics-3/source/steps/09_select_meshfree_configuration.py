"""
Select the convergence-qualified TGWP configuration and return its adjusted speedup.

Compose the convergence table and select only after all five inclusive limits are applied. Among feasible rows choose the first maximum of the unrounded quality-adjusted speedup and round that score once. After selection, recompute its two grid-reference errors as a certificate by directly composing all seven primitive public functions. The private oracle does the same exclusively through each `_oracle_` twin.

Returns
-------
One finite float rounded to rounding_digits decimal places.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def select_meshfree_configuration(log2_counts=(4,5,6,6,6),
                                  step_counts=(32,32,32,48,64),
                                  amplitudes=(.35,.65),
                                  grid_x=None, grid_y=None,
                                  q0=(-3.5,.2), p0=(2.2,.1),
                                  initial_gamma_diag=(.7,1.2),
                                  basis_gamma_diag=(1.4,2.4),
                                  final_time=2.4,
                                  potential_parameters=(0.,0.,1.3,1.,1.6,.65,3.45,.3),
                                  grid_steps=192, sobol_seed=20260902,
                                  feasibility_limits=(.55,.48,.19,.001,.40),
                                  rounding_digits=6, mass=1.0, hbar=1.0):
    """Return the selected quality-adjusted speedup as one float scalar.

    Passing no arguments evaluates the benchmark configuration in the Problem
    statement; ``None`` for either grid generates its stated periodic grid.

    Raises
    ------
    ValueError
        If no candidate is feasible or any delegated contract is invalid.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_select_meshfree_configuration(log2_counts=(4,5,6,6,6),
                                          step_counts=(32,32,32,48,64),
                                          amplitudes=(.35,.65),
                                          grid_x=None, grid_y=None,
                                          q0=(-3.5,.2), p0=(2.2,.1),
                                          initial_gamma_diag=(.7,1.2),
                                          basis_gamma_diag=(1.4,2.4),
                                          final_time=2.4,
                                          potential_parameters=(0.,0.,1.3,1.,1.6,.65,3.45,.3),
                                          grid_steps=192, sobol_seed=20260902,
                                          feasibility_limits=(.55,.48,.19,.001,.40),
                                          rounding_digits=6, mass=1.0, hbar=1.0):
    import numpy as np
    if grid_x is None:
        grid_x = np.linspace(-8.0, 8.0, 48, endpoint=False)
    if grid_y is None:
        grid_y = np.linspace(-6.0, 6.0, 36, endpoint=False)
    table = _oracle_tgwp_convergence_table(
        log2_counts, step_counts, amplitudes, grid_x, grid_y, q0, p0,
        initial_gamma_diag, basis_gamma_diag, final_time, potential_parameters,
        grid_steps, sobol_seed, feasibility_limits, mass, hbar
    )
    feasible = np.flatnonzero(table[:, 9] > 0.5)
    if len(feasible) == 0:
        raise ValueError("no TGWP candidate satisfies every convergence and accuracy limit")
    selected = feasible[int(np.argmax(table[feasible, 8]))]

    # Recompute the selected certificate through every private oracle twin.  This
    # makes each scientific step causally necessary to the final answer rather
    # than relying only on the candidate-table call above.
    log_count = int(table[selected, 0])
    step_count = int(table[selected, 1])
    x = np.asarray(grid_x, dtype=float)
    y = np.asarray(grid_y, dtype=float)
    params = np.asarray(potential_parameters, dtype=float)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    direct_errors = []
    for amplitude in np.asarray(amplitudes, dtype=float):
        local = params.copy()
        local[0] = amplitude
        packets = _oracle_phase_space_sobol_packets(
            log_count, int(sobol_seed), q0, p0,
            initial_gamma_diag, basis_gamma_diag, hbar
        )
        derivative_probe = _oracle_interference_potential_derivatives(
            packets[:, :2], 0.0, local
        )
        trajectory = _oracle_stormer_verlet_centers_actions(
            packets, final_time, step_count, local, mass, hbar
        )
        widths = _oracle_hagedorn_width_evolution(
            trajectory, final_time, local, basis_gamma_diag, mass, hbar
        )
        tgwp = _oracle_coherent_tgwp_reconstruction(
            x, y, packets, trajectory[-1], widths, hbar
        )
        reference = _oracle_strang_split_step_reference(
            x, y, q0, p0, initial_gamma_diag, final_time,
            int(grid_steps), local, mass, hbar
        )
        diagnostics = _oracle_wavefunction_error_diagnostics(
            tgwp, reference, dx, dy
        )
        if not np.all(np.isfinite(derivative_probe)):
            raise ValueError("selected-candidate derivative certificate is non-finite")
        direct_errors.append(diagnostics[2:4])
    direct_errors = np.asarray(direct_errors, dtype=float)
    if not np.allclose(
        np.max(direct_errors, axis=0), table[selected, 2:4],
        rtol=2e-12, atol=2e-12
    ):
        raise ValueError("selected-candidate certificate disagrees with the convergence table")
    return float(round(float(table[selected, 8]), int(rounding_digits)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"import numpy as np\nx=np.linspace(-4,4,16,endpoint=False);y=np.linspace(-3,3,12,endpoint=False);q=np.array([-1.,0.]);p=np.array([1.,0.]);g0=np.ones(2);gb=2*np.ones(2);v=np.array([0.,0.,1.,1.,1.2,.4,2.1,.2])", "call":"select_meshfree_configuration(np.array([2,3]),np.array([6,8]),np.array([.2]),x,y,q,p,g0,gb,.7,v,16,3,np.array([5.,5.,5.,5.,5.]))", "gold_call":"_oracle_select_meshfree_configuration(np.array([2,3]),np.array([6,8]),np.array([.2]),x,y,q,p,g0,gb,.7,v,16,3,np.array([5.,5.,5.,5.,5.]))"},
        {"setup":"import numpy as np\nx=np.linspace(-3,3,12,endpoint=False);y=np.linspace(-2,2,8,endpoint=False);q=np.zeros(2);p=np.array([.5,-.2]);g0=np.ones(2);gb=np.ones(2);v=np.array([0.,0.,1.,1.,1.,1.,1.,0.])", "call":"select_meshfree_configuration(np.array([1,2]),np.array([4,6]),np.array([0.]),x,y,q,p,g0,gb,.5,v,8,1,np.array([10.,10.,10.,10.,10.]),5)", "gold_call":"_oracle_select_meshfree_configuration(np.array([1,2]),np.array([4,6]),np.array([0.]),x,y,q,p,g0,gb,.5,v,8,1,np.array([10.,10.,10.,10.,10.]),5)"},
        {"setup":"import numpy as np\nx=np.linspace(-5,5,20,endpoint=False);y=np.linspace(-3,3,12,endpoint=False);q=np.array([-2.,.4]);p=np.array([2.,.1]);g0=np.array([.7,1.2]);gb=np.array([1.4,2.4]);v=np.array([0.,.2,.6,1.7,-1.1,.9,3.,-.4])", "call":"select_meshfree_configuration(np.array([2,3]),np.array([7,11]),np.array([.3,.8]),x,y,q,p,g0,gb,1.1,v,20,11,np.array([2.,2.,2.,2.,2.]),6,.8)", "gold_call":"_oracle_select_meshfree_configuration(np.array([2,3]),np.array([7,11]),np.array([.3,.8]),x,y,q,p,g0,gb,1.1,v,20,11,np.array([2.,2.,2.,2.,2.]),6,.8)"},
        {"setup":"", "call":"select_meshfree_configuration()", "gold_call":"_oracle_select_meshfree_configuration()"}
    ]
