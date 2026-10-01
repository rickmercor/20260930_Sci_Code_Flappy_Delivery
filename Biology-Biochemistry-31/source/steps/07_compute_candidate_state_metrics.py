"""
Compute objective, enzyme demand, profile deviation, feasibility, and reaction-resolved enzyme masses for each candidate/state point.

candidate_fluxes and candidate_pattern_mask are aligned candidate-by-reaction arrays. Positive continuous flux is permitted only where the aligned binary row state is 1.0.

turnover_forward_h, molecular_weights_g_per_mmol, and objective_coefficients are reaction-aligned. saturation_factors and thermodynamic_factors are state-by-reaction arrays. pattern_state_eligibility is candidate-by-state. concentration_states_mM is state-by-metabolite and reference_profile_mM is metabolite-aligned.

For each candidate, compute the state-independent nonlinear objective
\[
q=v\cdot c.
\]
For each state, compute the profile deviation
\[
L_1=\sum_m|x_m-x_m^{ref}|.
\]
For an ineligible candidate/state pair, set total demand to positive infinity and feasibility code to 0.0. For an eligible pair and each positive-flux reaction r, compute
\[
m_r=\frac{W_rv_r}{k_{cat,r}^{+}\kappa_r\gamma_r}.
\]
Zero-flux reactions contribute zero enzyme mass. Sum reaction contributions to obtain total demand D. The feasibility code is 1.0 when \(D\leq\) enzyme_pool_limit and 0.0 otherwise; the endpoint is inclusive.

Return channels in this exact order:
0 = q; 1 = total demand; 2 = profile L1 deviation; 3 = numerical pool-feasibility code; channels 4 onward = reaction-resolved enzyme-mass contributions in reaction order.

Validate every dimensional, numerical, positivity, and binary-code contract. Raise ValueError if an eligible positive-flux reaction has a non-positive effective efficiency or any public input is invalid

Returns
-------
3D NumPy float array of shape (n_candidates, n_states, 4 + n_reactions).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
 
 
def compute_candidate_state_metrics(
    candidate_fluxes: np.ndarray,
    candidate_pattern_mask: np.ndarray,
    turnover_forward_h: np.ndarray,
    molecular_weights_g_per_mmol: np.ndarray,
    saturation_factors: np.ndarray,
    thermodynamic_factors: np.ndarray,
    pattern_state_eligibility: np.ndarray,
    concentration_states_mM: np.ndarray,
    reference_profile_mM: np.ndarray,
    enzyme_pool_limit: float,
    objective_coefficients: np.ndarray,
) -> np.ndarray:
    """Compute nonlinear objective, demand, profile error, feasibility, and reaction masses."""
    return np.empty((0, 0, 0), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_candidate_state_metrics(
    candidate_fluxes,
    candidate_pattern_mask,
    turnover_forward_h,
    molecular_weights_g_per_mmol,
    saturation_factors,
    thermodynamic_factors,
    pattern_state_eligibility,
    concentration_states_mM,
    reference_profile_mM,
    enzyme_pool_limit,
    objective_coefficients,
):
    import numpy as np
 
    fluxes = np.asarray(candidate_fluxes, dtype=float)
    pattern = np.asarray(candidate_pattern_mask, dtype=float)
    turnover = np.asarray(turnover_forward_h, dtype=float)
    weights = np.asarray(molecular_weights_g_per_mmol, dtype=float)
    kappa = np.asarray(saturation_factors, dtype=float)
    gamma = np.asarray(thermodynamic_factors, dtype=float)
    eligibility = np.asarray(pattern_state_eligibility, dtype=float)
    concentrations = np.asarray(concentration_states_mM, dtype=float)
    reference = np.asarray(reference_profile_mM, dtype=float)
    objective = np.asarray(objective_coefficients, dtype=float)
 
    if fluxes.ndim != 2 or min(fluxes.shape) < 1:
        raise ValueError("candidate_fluxes must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(fluxes)) or np.any(fluxes < 0.0):
        raise ValueError("candidate_fluxes must contain finite non-negative values")
    n_candidates, n_reactions = fluxes.shape
 
    if pattern.shape != fluxes.shape or not np.all(np.isfinite(pattern)):
        raise ValueError("candidate_pattern_mask must align with candidate_fluxes")
    if not np.all((pattern == 0.0) | (pattern == 1.0)):
        raise ValueError("candidate_pattern_mask must contain only 0.0 and 1.0")
    if np.any((fluxes > 0.0) & (pattern == 0.0)):
        raise ValueError("positive continuous flux requires a row-level binary state of 1.0")
 
    if turnover.shape != (n_reactions,) or not np.all(np.isfinite(turnover)) or np.any(turnover <= 0.0):
        raise ValueError("turnover_forward_h must be finite, positive, and reaction-aligned")
    if weights.shape != (n_reactions,) or not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("molecular_weights_g_per_mmol must be finite, positive, and reaction-aligned")
    if kappa.ndim != 2 or kappa.shape[0] < 1 or kappa.shape[1] != n_reactions:
        raise ValueError("saturation_factors must align with state and reaction dimensions")
    if not np.all(np.isfinite(kappa)) or np.any(kappa <= 0.0) or np.any(kappa > 1.0):
        raise ValueError("saturation_factors must be finite and lie in (0, 1]")
    if gamma.shape != kappa.shape or not np.all(np.isfinite(gamma)):
        raise ValueError("thermodynamic_factors must be finite and aligned with saturation_factors")
 
    n_states = kappa.shape[0]
    if eligibility.shape != (n_candidates, n_states):
        raise ValueError("pattern_state_eligibility must align with candidates and states")
    if not np.all(np.isfinite(eligibility)) or not np.all((eligibility == 0.0) | (eligibility == 1.0)):
        raise ValueError("pattern_state_eligibility must contain only 0.0 and 1.0")
    if concentrations.ndim != 2 or concentrations.shape[0] != n_states:
        raise ValueError("concentration_states_mM must align with state rows")
    if not np.all(np.isfinite(concentrations)) or np.any(concentrations <= 0.0):
        raise ValueError("concentration_states_mM must contain finite positive values")
    if reference.shape != (concentrations.shape[1],) or not np.all(np.isfinite(reference)) or np.any(reference <= 0.0):
        raise ValueError("reference_profile_mM must be finite, positive, and metabolite-aligned")
    if objective.shape != (n_reactions,) or not np.all(np.isfinite(objective)):
        raise ValueError("objective_coefficients must be finite and reaction-aligned")
 
    try:
        pool_limit = float(enzyme_pool_limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("enzyme_pool_limit must be a real scalar") from exc
    if not np.isfinite(pool_limit) or pool_limit <= 0.0:
        raise ValueError("enzyme_pool_limit must be finite and positive")
 
    profile_error = np.sum(np.abs(concentrations - reference[None, :]), axis=1)
    result = np.zeros((n_candidates, n_states, 4 + n_reactions), dtype=float)
 
    for candidate in range(n_candidates):
        active = fluxes[candidate] > 0.0
        q_value = float(fluxes[candidate] @ objective)
        if not np.isfinite(q_value):
            raise ValueError("candidate objective values must be finite")
        for state in range(n_states):
            result[candidate, state, 0] = q_value
            result[candidate, state, 2] = profile_error[state]
            if eligibility[candidate, state] == 0.0:
                result[candidate, state, 1] = np.inf
                continue
            denominator = turnover * kappa[state] * gamma[state]
            if np.any(denominator[active] <= 0.0):
                raise ValueError("eligible positive-flux reactions require positive effective efficiencies")
            contributions = np.zeros(n_reactions, dtype=float)
            contributions[active] = weights[active] * fluxes[candidate, active] / denominator[active]
            demand = float(np.sum(contributions, dtype=float))
            result[candidate, state, 1] = demand
            result[candidate, state, 3] = float(demand <= pool_limit)
            result[candidate, state, 4:] = contributions
 
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
candidate_fluxes=np.array([[2.,2.,.5,1.5],[1.5,1.5,.4,1.1]])
candidate_pattern_mask=(candidate_fluxes>=0).astype(float)
kcat=np.array([100.,80.,120.,90.]);W=np.array([60.,55.,50.,65.])
kappa=np.array([[1.,.5,.8,.4],[1.,.7,.6,.3],[1.,.4,.9,.5]])
gamma=np.array([[.8,.6,.9,.7],[.9,.2,.8,.5],[.7,.4,.6,.8]])
elig=np.array([[1.,0.,1.],[1.,1.,1.]])
x=np.array([[1.,2.],[3.,1.],[2.,4.]]);ref=np.array([2.2,2.1]);ep=10.;obj=np.array([0.,0.,1.,1.01])""",
            "call": "compute_candidate_state_metrics(candidate_fluxes.copy(),candidate_pattern_mask.copy(),kcat.copy(),W.copy(),kappa.copy(),gamma.copy(),elig.copy(),x.copy(),ref.copy(),ep,obj.copy())",
            "gold_call": "_oracle_compute_candidate_state_metrics(candidate_fluxes.copy(),candidate_pattern_mask.copy(),kcat.copy(),W.copy(),kappa.copy(),gamma.copy(),elig.copy(),x.copy(),ref.copy(),ep,obj.copy())",
        },
        {
            "setup": """import numpy as np
candidate_fluxes=np.array([[0.,2.,1.],[1.5,1.,0.]])
candidate_pattern_mask=np.array([[0.,1.,1.],[1.,1.,0.]])
kcat=np.array([100.,80.,120.]);W=np.array([50.,60.,55.])
kappa=np.array([[.9,.5,.8],[.7,.6,.4]])
gamma=np.array([[-.5,.7,.8],[.6,.8,.9]])
elig=np.array([[0.,1.],[0.,1.]])
x=np.array([[2.,3.],[4.,5.]]);ref=np.array([3.5,4.5]);ep=3.;obj=np.array([0.,1.,2.])""",
            "call": "compute_candidate_state_metrics(candidate_fluxes.copy(),candidate_pattern_mask.copy(),kcat.copy(),W.copy(),kappa.copy(),gamma.copy(),elig.copy(),x.copy(),ref.copy(),ep,obj.copy())",
            "gold_call": "_oracle_compute_candidate_state_metrics(candidate_fluxes.copy(),candidate_pattern_mask.copy(),kcat.copy(),W.copy(),kappa.copy(),gamma.copy(),elig.copy(),x.copy(),ref.copy(),ep,obj.copy())",
        },
        {
            "setup": """import numpy as np
candidate_fluxes=np.array([[2.,1.,3.,0.],[1.,4.,0.,2.],[3.,0.,2.,1.]])
candidate_pattern_mask=(candidate_fluxes>0).astype(float)
kcat=np.array([90.,110.,80.,130.]);W=np.array([55.,60.,50.,65.])
kappa=np.array([[.8,.5,.7,1.],[.6,.9,.4,1.],[.7,.3,.8,1.]])
gamma=np.array([[.9,.6,.8,.7],[.7,.8,.5,.9],[.5,.4,.9,.6]])
elig=np.array([[1,1,0],[1,0,1],[0,1,1]],float)
x=np.array([[1.,2.],[3.,1.],[2.,4.]]);ref=np.array([2.1,2.2]);ep=20.;obj=np.array([0.,3.,2.,0.])
co=np.array([2,0,1]);so=np.array([1,2,0]);rp=np.array([2,0,3,1])
candidate_fluxes=candidate_fluxes[co][:,rp];candidate_pattern_mask=candidate_pattern_mask[co][:,rp]
kcat=kcat[rp];W=W[rp];kappa=kappa[so][:,rp];gamma=gamma[so][:,rp];elig=elig[co][:,so];x=x[so];obj=obj[rp]""",
            "call": "compute_candidate_state_metrics(candidate_fluxes.copy(),candidate_pattern_mask.copy(),kcat.copy(),W.copy(),kappa.copy(),gamma.copy(),elig.copy(),x.copy(),ref.copy(),ep,obj.copy())",
            "gold_call": "_oracle_compute_candidate_state_metrics(candidate_fluxes.copy(),candidate_pattern_mask.copy(),kcat.copy(),W.copy(),kappa.copy(),gamma.copy(),elig.copy(),x.copy(),ref.copy(),ep,obj.copy())",
        },
        {
            "setup": """import numpy as np
candidate_fluxes=np.array([[1.,2.]])
candidate_pattern_mask=np.array([[1.,0.]])
kcat=np.ones(2);W=np.ones(2);kappa=np.ones((1,2));gamma=np.ones((1,2));elig=np.ones((1,1));x=np.ones((1,1));ref=np.ones(1);ep=2.;obj=np.ones(2)
def candidate_wrapper():
    try: compute_candidate_state_metrics(candidate_fluxes,candidate_pattern_mask,kcat,W,kappa,gamma,elig,x,ref,ep,obj)
    except ValueError: return 1.0
    return 0.0
def gold_wrapper():
    try: _oracle_compute_candidate_state_metrics(candidate_fluxes,candidate_pattern_mask,kcat,W,kappa,gamma,elig,x,ref,ep,obj)
    except ValueError: return 1.0
    return 0.0""",
            "call": "candidate_wrapper()",
            "gold_call": "gold_wrapper()",
        },
    ]
