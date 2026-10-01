"""
Aggregate paper-native wave, coercivity, and energy certificates over coupled states.

The wave cube columns are [amplitude_error, phase_error, log_error, steps, time, absolute_mode]. The coercivity cube columns are [A6, B6, C6, A7, B7, C7]. The energy cube columns are [minimum_internal, heat_fraction].



For candidate i, take over all stress states



wave_worst = componentwise maximum of the first three wave columns,

B6_min = minimum of coercivity columns 0:3,

B7_min = minimum of coercivity columns 3:6,

heat_worst = maximum heat_fraction,

internal_min = minimum minimum_internal.



Form



ratios = [wave_worst_amplitude,

          wave_worst_phase,

          wave_worst_log,

          heat_worst]

         / [amplitude_limit,

            phase_limit,

            log_damping_limit,

            heat_fraction_limit],



score = max(ratios),

cells = ceil(1/h)^2,

work = cells * (sum_states(wave_steps) + 3*S),



where S is the number of stress states.



A candidate is feasible exactly when score <= 1, B6_min >= b6_margin_min, and internal_min > 0. B7_min is diagnostic and is not a feasibility gate.



Return one row per candidate with columns



[h, tau, amplitude_worst, phase_worst, log_worst,

 B6_min, B7_min, heat_worst, internal_min,

 score, work, feasible].

Returns
-------
Return one C-by-12 real NumPy array in candidate order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def paper_candidate_table(candidates, wave_cube, coercivity_cube, energy_cube,
                          amplitude_limit=1e-5, phase_limit=8e-5,
                          log_damping_limit=1e-5,
                          heat_fraction_limit=1e-4,
                          b6_margin_min=0.10):
    """Aggregate stress-state certificates into candidate rows.

    Parameters
    ----------
    candidates : array_like, shape (C,2)
        Ordered [h,tau] rows.
    wave_cube : array_like, shape (C,S,6)
        Modal certificates over stress states.
    coercivity_cube : array_like, shape (C,S,6)
        Normalized B.6/B.7 coefficient triples.
    energy_cube : array_like, shape (C,S,2)
        [minimum_internal,heat_fraction] certificates.
    amplitude_limit : float
        Positive amplitude-error scale.
    phase_limit : float
        Positive phase-error scale.
    log_damping_limit : float
        Positive log-damping-error scale.
    heat_fraction_limit : float
        Positive Joule-heat-fraction scale.
    b6_margin_min : float
        Positive admissibility threshold.
    Returns
    -------
    ndarray, shape (C,12)
        [h,tau,wave3,b6min,b7min,heat,internal,score,work,feasible].
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_paper_candidate_table(candidates, wave_cube, coercivity_cube,
                                  energy_cube, amplitude_limit=1e-5,
                                  phase_limit=8e-5, log_damping_limit=1e-5,
                                  heat_fraction_limit=1e-4,
                                  b6_margin_min=0.10):
    """Aggregate the paper-native wave, Newton, and energy diagnostics."""
    import math
    import numpy as np
    candidates = np.asarray(candidates, dtype=float)
    wave = np.asarray(wave_cube, dtype=float)
    coercivity = np.asarray(coercivity_cube, dtype=float)
    energy = np.asarray(energy_cube, dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != 2 or candidates.shape[0] == 0:
        raise ValueError("candidates must have shape (C,2)")
    c = candidates.shape[0]
    if wave.ndim != 3 or wave.shape[0] != c or wave.shape[2] != 6 or wave.shape[1] == 0:
        raise ValueError("wave_cube must have shape (C,S,6)")
    if coercivity.shape != (c, wave.shape[1], 6) or energy.shape != (c, wave.shape[1], 2):
        raise ValueError("certificate cubes have inconsistent shapes")
    limits = np.asarray([amplitude_limit, phase_limit, log_damping_limit,
                         heat_fraction_limit], dtype=float)
    margin = float(b6_margin_min)
    if (not np.all(np.isfinite(candidates)) or np.any(candidates <= 0)
            or not np.all(np.isfinite(wave)) or not np.all(np.isfinite(coercivity))
            or not np.all(np.isfinite(energy)) or not np.all(np.isfinite(limits))
            or np.any(limits <= 0) or not math.isfinite(margin) or margin <= 0):
        raise ValueError("all inputs and positive limits must be finite")
    rows = []
    for i, (mesh_h, tau) in enumerate(candidates):
        wave_worst = np.max(wave[i, :, :3], axis=0)
        b6_min = float(np.min(coercivity[i, :, :3]))
        b7_min = float(np.min(coercivity[i, :, 3:]))
        heat_worst = float(np.max(energy[i, :, 1]))
        internal_min = float(np.min(energy[i, :, 0]))
        ratios = np.array([*wave_worst, heat_worst]) / limits
        score = float(np.max(ratios))
        cells = int(math.ceil(1.0 / mesh_h)) ** 2
        work = float(cells * (np.sum(wave[i, :, 3]) + 3.0 * wave.shape[1]))
        feasible = float(score <= 1.0 and b6_min >= margin and internal_min > 0.0)
        rows.append([mesh_h, tau, *wave_worst, b6_min, b7_min, heat_worst,
                     internal_min, score, work, feasible])
    return np.asarray(rows, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\ncandidates=np.array([[.2,.01],[.1,.004],[.05,.001]],float)\nwave=np.array([[[2e-5,4e-5,8e-6,20,1.,.9],[8e-6,2e-4,9e-6,21,1.,.9]],[[8e-6,8e-5,7e-6,45,1.,.9],[9e-6,7e-5,6e-6,47,1.,.9]],[[3e-6,2e-5,2e-6,180,1.,.9],[4e-6,3e-5,3e-6,182,1.,.9]]],float)\ncoercivity=np.ones((3,2,6),float)\ncoercivity[0,1,2]=.08\ncoercivity[1,:,2]=[.15,.12]\ncoercivity[1,:,5]=[-3.,-2.]\ncoercivity[2,:,2]=[.3,.25]\ncoercivity[2,:,5]=[-1.,-.5]\nenergy=np.array([[[.4,8e-5],[.35,7e-5]],[[.3,9e-5],[.32,8e-5]],[[.25,2e-5],[.28,3e-5]]],float)\n', 'call': 'paper_candidate_table(candidates,wave,coercivity,energy)', 'gold_call': '_oracle_paper_candidate_table(candidates,wave,coercivity,energy)'}, {'setup': 'import numpy as np\ncandidates=np.array([[.2,.01],[.1,.004],[.05,.001]],float)\nwave=np.array([[[2e-5,4e-5,8e-6,20,1.,.9],[8e-6,2e-4,9e-6,21,1.,.9]],[[8e-6,8e-5,7e-6,45,1.,.9],[9e-6,7e-5,6e-6,47,1.,.9]],[[3e-6,2e-5,2e-6,180,1.,.9],[4e-6,3e-5,3e-6,182,1.,.9]]],float)\ncoercivity=np.ones((3,2,6),float)\ncoercivity[0,1,2]=.08\ncoercivity[1,:,2]=[.15,.12]\ncoercivity[1,:,5]=[-3.,-2.]\ncoercivity[2,:,2]=[.3,.25]\ncoercivity[2,:,5]=[-1.,-.5]\nenergy=np.array([[[.4,8e-5],[.35,7e-5]],[[.3,9e-5],[.32,8e-5]],[[.25,2e-5],[.28,3e-5]]],float)\n', 'call': 'paper_candidate_table(candidates,wave,coercivity,energy,phase_limit=9e-5)', 'gold_call': '_oracle_paper_candidate_table(candidates,wave,coercivity,energy,phase_limit=9e-5)'}, {'setup': 'import numpy as np\ncandidates=np.array([[.2,.01],[.1,.004],[.05,.001]],float)\nwave=np.array([[[2e-5,4e-5,8e-6,20,1.,.9],[8e-6,2e-4,9e-6,21,1.,.9]],[[8e-6,8e-5,7e-6,45,1.,.9],[9e-6,7e-5,6e-6,47,1.,.9]],[[3e-6,2e-5,2e-6,180,1.,.9],[4e-6,3e-5,3e-6,182,1.,.9]]],float)\ncoercivity=np.ones((3,2,6),float)\ncoercivity[0,1,2]=.08\ncoercivity[1,:,2]=[.15,.12]\ncoercivity[1,:,5]=[-3.,-2.]\ncoercivity[2,:,2]=[.3,.25]\ncoercivity[2,:,5]=[-1.,-.5]\nenergy=np.array([[[.4,8e-5],[.35,7e-5]],[[.3,9e-5],[.32,8e-5]],[[.25,2e-5],[.28,3e-5]]],float)\n', 'call': 'paper_candidate_table(candidates,wave,coercivity,energy,b6_margin_min=.11)', 'gold_call': '_oracle_paper_candidate_table(candidates,wave,coercivity,energy,b6_margin_min=.11)'}, {'setup': 'import numpy as np\ncandidates=np.array([[.2,.01],[.1,.004],[.05,.001]],float)\nwave=np.array([[[2e-5,4e-5,8e-6,20,1.,.9],[8e-6,2e-4,9e-6,21,1.,.9]],[[8e-6,8e-5,7e-6,45,1.,.9],[9e-6,7e-5,6e-6,47,1.,.9]],[[3e-6,2e-5,2e-6,180,1.,.9],[4e-6,3e-5,3e-6,182,1.,.9]]],float)\ncoercivity=np.ones((3,2,6),float)\ncoercivity[0,1,2]=.08\ncoercivity[1,:,2]=[.15,.12]\ncoercivity[1,:,5]=[-3.,-2.]\ncoercivity[2,:,2]=[.3,.25]\ncoercivity[2,:,5]=[-1.,-.5]\nenergy=np.array([[[.4,8e-5],[.35,7e-5]],[[.3,9e-5],[.32,8e-5]],[[.25,2e-5],[.28,3e-5]]],float)\n', 'call': 'paper_candidate_table(candidates,wave,coercivity,energy,phase_limit=2.5e-4)', 'gold_call': '_oracle_paper_candidate_table(candidates,wave,coercivity,energy,phase_limit=2.5e-4)'}, {'setup': 'import numpy as np\ncandidates=np.array([[.2,.01],[.1,.004],[.05,.001]],float)\nwave=np.array([[[2e-5,4e-5,8e-6,20,1.,.9],[8e-6,2e-4,9e-6,21,1.,.9]],[[8e-6,8e-5,7e-6,45,1.,.9],[9e-6,7e-5,6e-6,47,1.,.9]],[[3e-6,2e-5,2e-6,180,1.,.9],[4e-6,3e-5,3e-6,182,1.,.9]]],float)\ncoercivity=np.ones((3,2,6),float)\ncoercivity[0,1,2]=.08\ncoercivity[1,:,2]=[.15,.12]\ncoercivity[1,:,5]=[-3.,-2.]\ncoercivity[2,:,2]=[.3,.25]\ncoercivity[2,:,5]=[-1.,-.5]\nenergy=np.array([[[.4,8e-5],[.35,7e-5]],[[.3,9e-5],[.32,8e-5]],[[.25,2e-5],[.28,3e-5]]],float)\n', 'call': 'paper_candidate_table(candidates,wave,coercivity,energy,b6_margin_min=.2)', 'gold_call': '_oracle_paper_candidate_table(candidates,wave,coercivity,energy,b6_margin_min=.2)'}, {'setup': 'import numpy as np\ncandidates=np.array([[.2,.01],[.1,.004],[.05,.001]],float)\nwave=np.array([[[2e-5,4e-5,8e-6,20,1.,.9],[8e-6,2e-4,9e-6,21,1.,.9]],[[8e-6,8e-5,7e-6,45,1.,.9],[9e-6,7e-5,6e-6,47,1.,.9]],[[3e-6,2e-5,2e-6,180,1.,.9],[4e-6,3e-5,3e-6,182,1.,.9]]],float)\ncoercivity=np.ones((3,2,6),float)\ncoercivity[0,1,2]=.08\ncoercivity[1,:,2]=[.15,.12]\ncoercivity[1,:,5]=[-3.,-2.]\ncoercivity[2,:,2]=[.3,.25]\ncoercivity[2,:,5]=[-1.,-.5]\nenergy=np.array([[[.4,8e-5],[.35,7e-5]],[[.3,9e-5],[.32,8e-5]],[[.25,2e-5],[.28,3e-5]]],float)\n', 'call': 'paper_candidate_table(candidates,wave,coercivity,energy,heat_fraction_limit=5e-5)', 'gold_call': '_oracle_paper_candidate_table(candidates,wave,coercivity,energy,heat_fraction_limit=5e-5)'}, {'setup': 'import numpy as np\ncandidates=np.array([[.2,.01],[.1,.004],[.05,.001]],float)\nwave=np.array([[[2e-5,4e-5,8e-6,20,1.,.9],[8e-6,2e-4,9e-6,21,1.,.9]],[[8e-6,8e-5,7e-6,45,1.,.9],[9e-6,7e-5,6e-6,47,1.,.9]],[[3e-6,2e-5,2e-6,180,1.,.9],[4e-6,3e-5,3e-6,182,1.,.9]]],float)\ncoercivity=np.ones((3,2,6),float)\ncoercivity[0,1,2]=.08\ncoercivity[1,:,2]=[.15,.12]\ncoercivity[1,:,5]=[-3.,-2.]\ncoercivity[2,:,2]=[.3,.25]\ncoercivity[2,:,5]=[-1.,-.5]\nenergy=np.array([[[.4,8e-5],[.35,7e-5]],[[.3,9e-5],[.32,8e-5]],[[.25,2e-5],[.28,3e-5]]],float)\n', 'call': 'paper_candidate_table(candidates,wave,coercivity,energy,amplitude_limit=3e-5)', 'gold_call': '_oracle_paper_candidate_table(candidates,wave,coercivity,energy,amplitude_limit=3e-5)'}]
