"""
Apply grid accuracy, time refinement, and packet-basis saturation tests to all candidates.

Evaluate each `(log2 packet count, step count)` candidate over every declared amplitude. Build one independent grid reference per amplitude, then for each candidate compute the TGWP field, its twice-step field, and its twice-packet field. Aggregate the five maxima in prompt order, apply inclusive feasibility ratios, and return ordered columns `[log2N,steps,worst_L2,worst_abs_L2,worst_norm_error,worst_time_refinement,worst_basis_refinement,speedup,quality_adjusted_speedup,feasible,governing_component,worst_overlap_loss]`. Solver implementations must compose the seven earlier public functions; the private oracle chains their `_oracle_` twins.

Returns
-------
One ordered candidate table with 12 numerical columns.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def tgwp_convergence_table(log2_counts, step_counts, amplitudes,
                           grid_x, grid_y, q0, p0,
                           initial_gamma_diag, basis_gamma_diag,
                           final_time, potential_parameters, grid_steps,
                           sobol_seed, feasibility_limits,
                           mass=1.0, hbar=1.0):
    """Return a float array of shape (number_of_candidates,12).

    Columns follow the exact order stated in the Step scientific background.

    Raises
    ------
    ValueError
        If candidate, grid, physical, or feasibility inputs violate their domains.
    """
    return np.empty((len(log2_counts), 12), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_tgwp_convergence_table(log2_counts, step_counts, amplitudes,
                                   grid_x, grid_y, q0, p0,
                                   initial_gamma_diag, basis_gamma_diag,
                                   final_time, potential_parameters, grid_steps,
                                   sobol_seed, feasibility_limits,
                                   mass=1.0, hbar=1.0):
    import numpy as np
    logs = np.asarray(log2_counts, dtype=int)
    steps = np.asarray(step_counts, dtype=int)
    amplitudes = np.asarray(amplitudes, dtype=float)
    x = np.asarray(grid_x, dtype=float)
    y = np.asarray(grid_y, dtype=float)
    limits = np.asarray(feasibility_limits, dtype=float)
    params = np.asarray(potential_parameters, dtype=float)
    if logs.ndim != 1 or steps.shape != logs.shape or len(logs) == 0:
        raise ValueError("candidate arrays must be nonempty one-dimensional arrays of equal length")
    if amplitudes.ndim != 1 or len(amplitudes) == 0 or not np.all(np.isfinite(amplitudes)):
        raise ValueError("amplitudes must be a nonempty finite vector")
    if x.ndim != 1 or y.ndim != 1 or len(x) < 3 or len(y) < 3:
        raise ValueError("both grids need at least three points")
    if params.shape != (8,) or limits.shape != (5,) or np.any(limits <= 0.0):
        raise ValueError("potential_parameters and feasibility_limits have wrong shape or domain")
    references = []
    for amplitude in amplitudes:
        local = params.copy()
        local[0] = amplitude
        references.append(_oracle_strang_split_step_reference(
            x, y, q0, p0, initial_gamma_diag, final_time, int(grid_steps), local, mass, hbar
        ))

    def simulate(log_count, count, local):
        packet_rows = _oracle_phase_space_sobol_packets(
            int(log_count), int(sobol_seed), q0, p0, initial_gamma_diag, basis_gamma_diag, hbar
        )
        trajectory = _oracle_stormer_verlet_centers_actions(
            packet_rows, final_time, int(count), local, mass, hbar
        )
        width_rows = _oracle_hagedorn_width_evolution(
            trajectory, final_time, local, basis_gamma_diag, mass, hbar
        )
        return _oracle_coherent_tgwp_reconstruction(
            x, y, packet_rows, trajectory[-1], width_rows, hbar
        )

    table = np.empty((len(logs), 12), dtype=float)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    grid_cost = float(grid_steps) * len(x) * len(y) * np.log2(len(x) * len(y))
    for ci, (log_count, count) in enumerate(zip(logs, steps)):
        condition_rows = []
        for amplitude, reference in zip(amplitudes, references):
            local = params.copy()
            local[0] = amplitude
            base = simulate(log_count, count, local)
            time_fine = simulate(log_count, 2 * count, local)
            basis_fine = simulate(log_count + 1, count, local)
            physical = _oracle_wavefunction_error_diagnostics(base, reference, dx, dy)
            time_error = _oracle_wavefunction_error_diagnostics(base, time_fine, dx, dy)[2]
            basis_error = _oracle_wavefunction_error_diagnostics(base, basis_fine, dx, dy)[2]
            condition_rows.append(np.r_[physical, time_error, basis_error])
        condition_rows = np.asarray(condition_rows, dtype=float)
        worst_l2 = float(np.max(condition_rows[:, 2]))
        worst_abs = float(np.max(condition_rows[:, 3]))
        worst_norm_error = float(np.max(np.abs(condition_rows[:, 0] - 1.0)))
        worst_time = float(np.max(condition_rows[:, 6]))
        worst_basis = float(np.max(condition_rows[:, 7]))
        worst_overlap_loss = float(np.max(1.0 - condition_rows[:, 4]))
        speedup = grid_cost / (2.0 ** int(log_count) * int(count))
        score = speedup / (1.0 + worst_l2 + worst_abs + worst_time + worst_basis)
        ratios = np.array([worst_l2, worst_abs, worst_norm_error, worst_time, worst_basis]) / limits
        feasible = float(np.all(np.isfinite(ratios)) and np.all(ratios <= 1.0))
        table[ci] = [log_count, count, worst_l2, worst_abs, worst_norm_error,
                     worst_time, worst_basis, speedup, score, feasible,
                     int(np.argmax(ratios)), worst_overlap_loss]
    return table

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"import numpy as np\nx=np.linspace(-4,4,16,endpoint=False);y=np.linspace(-3,3,12,endpoint=False);q=np.array([-1.,0.]);p=np.array([1.,0.]);g0=np.ones(2);gb=2*np.ones(2);v=np.array([0.,0.,1.,1.,1.2,.4,2.1,.2])", "call":"tgwp_convergence_table(np.array([2,3]),np.array([6,8]),np.array([.2]),x,y,q,p,g0,gb,.7,v,16,3,np.array([5.,5.,5.,5.,5.]))", "gold_call":"_oracle_tgwp_convergence_table(np.array([2,3]),np.array([6,8]),np.array([.2]),x,y,q,p,g0,gb,.7,v,16,3,np.array([5.,5.,5.,5.,5.]))"},
        {"setup":"import numpy as np\nx=np.linspace(-3,3,12,endpoint=False);y=np.linspace(-2,2,8,endpoint=False);q=np.zeros(2);p=np.array([.5,-.2]);g0=np.ones(2);gb=np.ones(2);v=np.array([0.,0.,1.,1.,1.,1.,1.,0.])", "call":"tgwp_convergence_table(np.array([1]),np.array([4]),np.array([0.]),x,y,q,p,g0,gb,.5,v,8,1,np.array([10.,10.,10.,10.,10.]))", "gold_call":"_oracle_tgwp_convergence_table(np.array([1]),np.array([4]),np.array([0.]),x,y,q,p,g0,gb,.5,v,8,1,np.array([10.,10.,10.,10.,10.]))"},
        {"setup":"import numpy as np\nx=np.linspace(-5,5,20,endpoint=False);y=np.linspace(-3,3,12,endpoint=False);q=np.array([-2.,.4]);p=np.array([2.,.1]);g0=np.array([.7,1.2]);gb=np.array([1.4,2.4]);v=np.array([0.,.2,.6,1.7,-1.1,.9,3.,-.4])", "call":"tgwp_convergence_table(np.array([2,3]),np.array([7,11]),np.array([.3,.8]),x,y,q,p,g0,gb,1.1,v,20,11,np.array([2.,2.,2.,2.,2.]),.8)", "gold_call":"_oracle_tgwp_convergence_table(np.array([2,3]),np.array([7,11]),np.array([.3,.8]),x,y,q,p,g0,gb,1.1,v,20,11,np.array([2.,2.,2.,2.,2.]),.8)"},
        {"setup":"import numpy as np\nx=np.linspace(-4,4,16,endpoint=False);y=np.linspace(-3,3,12,endpoint=False);q=np.array([-1.,0.]);p=np.array([1.,0.]);g0=np.ones(2);gb=2*np.ones(2);v=np.array([0.,0.,1.,1.,1.2,.4,2.1,.2])", "call":"tgwp_convergence_table(np.array([2,3]),np.array([6,8]),np.array([.2]),x,y,q,p,g0,gb,.7,v,16,3,np.array([1.2,1.,.8,.0004,.7]))", "gold_call":"_oracle_tgwp_convergence_table(np.array([2,3]),np.array([6,8]),np.array([.2]),x,y,q,p,g0,gb,.7,v,16,3,np.array([1.2,1.,.8,.0004,.7]))"}
    ]
