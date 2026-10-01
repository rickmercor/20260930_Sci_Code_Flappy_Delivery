"""
Orchestrator of sub-problems 01-10, calling build_loading_path, evaluate_double_well, solve_incremental_step, return_map_step, integrate_energetic_evolution, track_local_branch, energy_balance_defect, measure_energy_consistency_order, summarize_cycle and identify_material_parameters to recover a bistable mark's landscape from one assay record, drive it through the same stimulus cycle under both selection principles, and return the ratio of the interaction potentials at which they carry it past the barrier.

Orchestrator: yes - this is the final step; it integrates the earlier sub-problems (build_loading_path, evaluate_double_well, solve_incremental_step, return_map_step, integrate_energetic_evolution, track_local_branch, energy_balance_defect, measure_energy_consistency_order, summarize_cycle, identify_material_parameters) rather than reimplementing them.

The comparison is only meaningful once the landscape is fixed by the experiment rather than assumed, so the run begins by identifying the constants from the record and only then integrates. The two arms are thereafter the same model, the same landscape, the same stimulus protocol, the same initial mark and the same partition; they differ only in which configurations the evolution is allowed to compare itself against. One arm compares against the whole state space and can therefore be transported to a basin no continuous path of equilibria reaches; the other compares against nothing but the branch it occupies and moves off it only when that branch runs out. Because everything else is held fixed, whatever separates the two crossings is attributable to the selection principle alone. A crossing located on a partition is only as trustworthy as the integrator that located it, so the run is not reported until the identified landscape reproduces the record it came from and the discrete energy balance is confirmed to close at the rate the theory underwrites.

Returns
-------
float, the interaction potential at the first node where the globally selected evolution has left the repressed well, divided by the interaction potential at the first node where the branch-tracking evolution has left it, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def run_epigenetic_switch_comparison(q_peak: float = 0.448487,
                                     q_residual: float = 0.165654,
                                     dissipation: float = 0.0835480,
                                     delta: float = 0.00603213,
                                     ell_inf: float = 0.5,
                                     lam: float = 1.0,
                                     S_max: float = 5.0,
                                     T: float = 1.0,
                                     n_steps: int = 4000,
                                     refinement=None) -> float:
    """Compare the two selection principles on one stimulus cycle.

    The landscape and the dissipation threshold are recovered from the assay
    record before anything is integrated. Both arms are then integrated on the
    same uniform partition of [0, T] into n_steps intervals, from a mark
    resting at the bottom of the repressed well, under the same stimulus
    protocol. The crossing of each arm is the first node at which its state is
    strictly positive, and the reported ratio divides the interaction
    potential at the globally selected crossing by the interaction potential
    at the branch-tracking crossing.

    Parameters
    ----------
    q_peak, q_residual, dissipation : float
        The assay record of one closed stimulus cycle: the peak state, the
        residual state at baseline and the accumulated dissipation.
    delta : float
        Free energy of the active well above the repressed one, measured
        independently of the cycle.
    ell_inf, lam, S_max, T : float
        Settings of the stimulus protocol and its interaction potential.
    n_steps : int
        Number of intervals of the partition, n_steps >= 2.
    refinement : sequence of int or None
        Partition sizes of the consistency check; None uses the dyadic sweep
        100, 200, 400, 800, 1600, 3200 and 6400.

    Returns
    -------
    ratio : float
        The interaction potential at the globally selected crossing divided
        by the interaction potential at the branch-tracking crossing, as a
        native Python float.

    Raises
    ------
    ValueError
        If n_steps is not an integer of at least two, if ell_inf, lam, S_max
        or T is not a finite number strictly greater than zero, if the
        landscape recovered from the record carries no barrier or cannot
        retain the mark at baseline, if the stimulus is too weak for either
        principle to carry the mark across, if the discrete energy balance is
        violated or fails to close under refinement, if the identified
        landscape does not reproduce the recorded peak and residual, or if
        either arm leaves the mark short of the barrier.
    """
    return ratio  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================

import numpy as np
def _oracle_run_epigenetic_switch_comparison(q_peak: float = 0.448487,
                                             q_residual: float = 0.165654,
                                             dissipation: float = 0.0835480,
                                             delta: float = 0.00603213,
                                             ell_inf: float = 0.5,
                                             lam: float = 1.0,
                                             S_max: float = 5.0,
                                             T: float = 1.0,
                                             n_steps: int = 4000,
                                             refinement=None) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    # -- The pipeline is an end-to-end chain of the ten earlier golden
    #    implementations, called by their _oracle_ names, so that the
    #    result never depends on any other implementation of a step.

    # -- Validate the orchestrator inputs.
    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)) or int(n_steps) < 2:
        raise ValueError("n_steps must be an integer of at least two")
    for name, value in (("ell_inf", ell_inf), ("lam", lam), ("S_max", S_max), ("T", T)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and np.isfinite(float(value)) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number strictly greater than zero")
    sweep = (100, 200, 400, 800, 1600, 3200, 6400) if refinement is None else tuple(
        int(value) for value in refinement)

    # -- Sub-problem 01: tabulate the stimulus protocol and its potential.
    loading = _oracle_build_loading_path(int(n_steps), float(T), float(S_max),
                                 float(ell_inf), float(lam))
    ceiling = float(np.max(loading[:, 2]))

    # -- Sub-problem 10: the landscape is read out of the record, not assumed.
    recovered = np.asarray(_oracle_identify_material_parameters(
        float(q_peak), float(q_residual), float(dissipation), float(delta), ceiling),
        dtype=float)
    k, a, b, rho = (float(recovered[0]), float(recovered[1]),
                    float(recovered[2]), float(recovered[3]))
    start = -a

    # -- Sub-problem 02: the landscape must really be bistable, since a
    #    vanishing barrier makes the two selection principles coincide and the
    #    comparison the run exists to make would be empty.
    barrier = np.asarray(_oracle_evaluate_double_well([start, 0.0], k, a, b), dtype=float)
    if not float(barrier[0, 1]) > 0.0:
        raise ValueError("the landscape carries no barrier, so the two principles coincide")

    # -- The mark must survive its own unloading arm: were the active well
    #    raised by more than the threshold can hold, the globally minimising
    #    evolution would carry it back across the barrier at baseline and the
    #    record could not have been produced by either principle.
    if not 0.5 * k * (a - b) < rho:
        raise ValueError("the recovered landscape cannot retain the mark at baseline")

    # -- Sub-problems 03 and 04: probe both increments at the peak of the
    #    stimulus. If neither principle can move the mark at the strongest
    #    drive the protocol reaches, no crossing can occur anywhere and the
    #    ratio is undefined; this is settled before the partition is swept.
    probe_global = float(_oracle_solve_incremental_step(start, ceiling, k, a, rho, b))
    probe_local = float(_oracle_return_map_step(start, ceiling, rho, k, a, 0.0))
    if probe_global <= 0.0 and probe_local <= 0.0:
        raise ValueError("the stimulus is too weak for either principle to cross the barrier")

    # -- Sub-problems 05 and 06: the two arms, on one and the same partition.
    global_arm = _oracle_integrate_energetic_evolution(loading, k, a, rho, start, b)
    local_arm = _oracle_track_local_branch(loading, k, a, rho, start, b)

    # -- Sub-problem 07: the scheme over-dissipates rather than manufacturing
    #    energy, so a negative defect would mean the trajectory is not the one
    #    the incremental scheme produces.
    defect = float(_oracle_energy_balance_defect(loading, global_arm, k, a, rho, b))
    if defect < -1.0e-12:
        raise ValueError("the discrete energy balance is violated, so the trajectory is invalid")

    # -- Sub-problem 08: the crossings are only as trustworthy as the
    #    integrator, so the defect must decay under refinement before the
    #    located nodes are reported.
    order = float(_oracle_measure_energy_consistency_order(sweep, k, a, rho, start, float(T),
                                                   float(S_max), float(ell_inf),
                                                   float(lam), b))
    if not order > 0.5:
        raise ValueError("the energy balance does not close under refinement")

    # -- Sub-problem 09: reduce each arm to its reportable quantities; entry
    #    four of the summary is the interaction potential at the crossing.
    global_summary = np.asarray(_oracle_summarize_cycle(loading, global_arm, rho), dtype=float)
    local_summary = np.asarray(_oracle_summarize_cycle(loading, local_arm, rho), dtype=float)

    # -- The identification is only credible if the identified landscape
    #    reproduces the record it was read from, which the globally selected
    #    arm must do to within the resolution of the partition.
    tolerance = 1.0e-3 * (1.0 + abs(float(q_peak)))
    if abs(float(global_summary[0]) - float(q_peak)) > tolerance:
        raise ValueError("the identified landscape does not reproduce the recorded peak")
    if abs(float(global_summary[1]) - float(q_residual)) > tolerance:
        raise ValueError("the identified landscape does not reproduce the recorded residual")

    if float(global_summary[4]) < 0.0 or float(local_summary[4]) < 0.0:
        raise ValueError("one of the two arms never carries the mark past the barrier")
    if not float(local_summary[4]) > 0.0:
        raise ValueError("the branch-tracking crossing sits at zero potential, so the ratio is undefined")

    return float(global_summary[4] / local_summary[4])

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================


import numpy as np


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the benchmark record on a coarse partition ---
        {
            "setup": """import numpy as np
sweep = [50, 100, 200]
""",
            "call": "float(1.0e6 * run_epigenetic_switch_comparison(n_steps=200, refinement=sweep))",
            "gold_call": "float(1.0e6 * _oracle_run_epigenetic_switch_comparison(n_steps=200, refinement=sweep))",
        },
        # --- Valid: a record left by a stiffer, more resistant mark ---
        {
            "setup": """import numpy as np
k, a, b, rho, Lm = 1.9, 0.16, 0.10, 0.062, 0.5 * (1.0 - np.exp(-5.0))
peak = b + (Lm - rho) / k
res = b + rho / k
cost = rho * (a + 2.0 * peak - res)
record = dict(q_peak=peak, q_residual=res, dissipation=cost,
              delta=0.5 * k * (a * a - b * b))
sweep = [40, 80, 160]
""",
            "call": "float(1.0e6 * run_epigenetic_switch_comparison(n_steps=160, refinement=sweep, **record))",
            "gold_call": "float(1.0e6 * _oracle_run_epigenetic_switch_comparison(n_steps=160, refinement=sweep, **record))",
        },
        # --- Valid: a weaker, slower stimulus acting on the same kind of mark ---
        {
            "setup": """import numpy as np
k, a, b, rho = 1.3, 0.14, 0.08, 0.055
Lm = 0.4 * (1.0 - np.exp(-0.7 * 4.0))
peak = b + (Lm - rho) / k
res = b + rho / k
cost = rho * (a + 2.0 * peak - res)
record = dict(q_peak=peak, q_residual=res, dissipation=cost,
              delta=0.5 * k * (a * a - b * b))
sweep = [60, 120, 240]
""",
            "call": "float(1.0e6 * run_epigenetic_switch_comparison(ell_inf=0.4, lam=0.7, S_max=4.0, n_steps=240, refinement=sweep, **record))",
            "gold_call": "float(1.0e6 * _oracle_run_epigenetic_switch_comparison(ell_inf=0.4, lam=0.7, S_max=4.0, n_steps=240, refinement=sweep, **record))",
        },
        # --- Boundary: a symmetric landscape, where the record carries no offset ---
        {
            "setup": """import numpy as np
k, a, b, rho, Lm = 1.0, 0.15, 0.15, 0.10, 0.5 * (1.0 - np.exp(-5.0))
peak = b + (Lm - rho) / k
res = b + rho / k
cost = rho * (a + 2.0 * peak - res)
record = dict(q_peak=peak, q_residual=res, dissipation=cost, delta=0.0)
sweep = [50, 100]
""",
            "call": "float(1.0e6 * run_epigenetic_switch_comparison(n_steps=200, refinement=sweep, **record))",
            "gold_call": "float(1.0e6 * _oracle_run_epigenetic_switch_comparison(n_steps=200, refinement=sweep, **record))",
        },
        # --- Edge: a record whose landscape is too resistant for either
        # principle to cross the barrier within the cycle ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_epigenetic_switch_comparison(q_peak=0.448487, q_residual=0.165654,
                                         dissipation=0.0835480, delta=0.00603213,
                                         ell_inf=0.02, n_steps=100, refinement=[50, 100])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_epigenetic_switch_comparison(q_peak=0.448487, q_residual=0.165654,
                                                 dissipation=0.0835480, delta=0.00603213,
                                                 ell_inf=0.02, n_steps=100, refinement=[50, 100])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: an offset so large that the mark could not have survived
        # its own unloading arm, so the record is inconsistent ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_epigenetic_switch_comparison(delta=0.05, n_steps=100, refinement=[50, 100])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_epigenetic_switch_comparison(delta=0.05, n_steps=100, refinement=[50, 100])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a residual level that exceeds the peak level ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_epigenetic_switch_comparison(q_peak=0.165654, q_residual=0.448487,
                                         n_steps=100, refinement=[50, 100])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_epigenetic_switch_comparison(q_peak=0.165654, q_residual=0.448487,
                                                 n_steps=100, refinement=[50, 100])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a partition with fewer than two intervals ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_epigenetic_switch_comparison(n_steps=1, refinement=[50, 100])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_epigenetic_switch_comparison(n_steps=1, refinement=[50, 100])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
