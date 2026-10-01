"""
Compose the complete source-consistent calculation and return the selected auxiliary enzyme-mass fraction.

This is the only Final orchestrator. It must genuinely call and use the public functions from Items 1 through 9 in order:

1. compute_reaction_driving_forces
2. compute_direct_binding_saturation_factors
3. compute_thermodynamic_efficiency_factors
4. encode_pattern_group_states
5. select_source_screen_records
6. classify_pattern_state_eligibility
7. compute_candidate_state_metrics
8. filter_source_postprocessing_points
9. select_source_terminal_point

Do not privately reproduce an earlier step, bypass a public function, hardcode a record/state ID, or hardcode the final scalar.

After the terminal candidate/state IDs are resolved, locate their unique aligned positions. Read the reaction-resolved enzyme-mass contributions from channels 4 onward in candidate_state_metrics. branch_reaction_indices contains unique zero-based reaction indices. Return
\[
\frac{\sum_{r\in branch}m_r}{\sum_rm_r}
\]
as one finite float.

The orchestrator must preserve all alignments and propagate ValueError for invalid public inputs. The selected total enzyme mass must be finite and strictly positive.

Returns
-------
One finite float containing the selected auxiliary enzyme-mass fraction.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
 
 
def resolve_auxiliary_enzyme_fraction(
    concentration_states_mM: np.ndarray,
    candidate_fluxes: np.ndarray,
    candidate_ids: np.ndarray,
    candidate_parent_ids: np.ndarray,
    parent_ids: np.ndarray,
    parent_pattern_mask: np.ndarray,
    candidate_pattern_mask: np.ndarray,
    coupling_group_ids: np.ndarray,
    relaxed_objective_values: np.ndarray,
    state_ids: np.ndarray,
    stoichiometric_columns: np.ndarray,
    standard_gibbs_kj: np.ndarray,
    substrate_indices: np.ndarray,
    substrate_km_mM: np.ndarray,
    substrate_fixed_terms: np.ndarray,
    product_indices: np.ndarray,
    product_km_mM: np.ndarray,
    product_fixed_terms: np.ndarray,
    turnover_forward_h: np.ndarray,
    molecular_weights_g_per_mmol: np.ndarray,
    temperature: float,
    gas_constant: float,
    minimum_driving_force: float,
    enzyme_pool_limit: float,
    objective_coefficients: np.ndarray,
    reference_profile_mM: np.ndarray,
    objective_retention_fraction: float,
    branch_reaction_indices: np.ndarray,
    relaxed_objective_tolerance: float,
    profile_tolerance: float,
    demand_tolerance: float,
) -> float:
    """Resolve the source-consistent profile-matched state and return its auxiliary enzyme fraction."""
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_resolve_auxiliary_enzyme_fraction(
    concentration_states_mM,
    candidate_fluxes,
    candidate_ids,
    candidate_parent_ids,
    parent_ids,
    parent_pattern_mask,
    candidate_pattern_mask,
    coupling_group_ids,
    relaxed_objective_values,
    state_ids,
    stoichiometric_columns,
    standard_gibbs_kj,
    substrate_indices,
    substrate_km_mM,
    substrate_fixed_terms,
    product_indices,
    product_km_mM,
    product_fixed_terms,
    turnover_forward_h,
    molecular_weights_g_per_mmol,
    temperature,
    gas_constant,
    minimum_driving_force,
    enzyme_pool_limit,
    objective_coefficients,
    reference_profile_mM,
    objective_retention_fraction,
    branch_reaction_indices,
    relaxed_objective_tolerance,
    profile_tolerance,
    demand_tolerance,
):
    import numpy as np
 
    driving_forces = _oracle_compute_reaction_driving_forces(
        concentration_states_mM,
        stoichiometric_columns,
        standard_gibbs_kj,
        temperature,
        gas_constant,
    )
    saturation_factors = _oracle_compute_direct_binding_saturation_factors(
        concentration_states_mM,
        substrate_indices,
        substrate_km_mM,
        substrate_fixed_terms,
        product_indices,
        product_km_mM,
        product_fixed_terms,
    )
    thermodynamic_factors = _oracle_compute_thermodynamic_efficiency_factors(
        driving_forces,
        temperature,
        gas_constant,
    )
    candidate_group_summary = _oracle_encode_pattern_group_states(
        candidate_pattern_mask,
        coupling_group_ids,
    )
    parent_group_summary = _oracle_encode_pattern_group_states(
        parent_pattern_mask,
        coupling_group_ids,
    )
    source_selection_codes = _oracle_select_source_screen_records(
        candidate_ids,
        candidate_parent_ids,
        parent_ids,
        relaxed_objective_values,
        candidate_group_summary,
        parent_group_summary,
        relaxed_objective_tolerance,
    )
    pattern_state_eligibility = _oracle_classify_pattern_state_eligibility(
        candidate_pattern_mask,
        driving_forces,
        minimum_driving_force,
    )
    metrics = _oracle_compute_candidate_state_metrics(
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
    )
    near_optimal_codes = _oracle_filter_source_postprocessing_points(
        metrics,
        source_selection_codes,
        objective_retention_fraction,
    )
    selected = _oracle_select_source_terminal_point(
        candidate_ids,
        state_ids,
        metrics,
        near_optimal_codes,
        profile_tolerance,
        demand_tolerance,
    )
 
    raw_candidate_ids = np.asarray(candidate_ids)
    raw_state_ids = np.asarray(state_ids)
    raw_branch_indices = np.asarray(branch_reaction_indices)
    fluxes = np.asarray(candidate_fluxes, dtype=float)
 
    if raw_candidate_ids.ndim != 1 or not np.issubdtype(raw_candidate_ids.dtype, np.integer):
        raise ValueError("candidate_ids must be a one-dimensional integer array")
    if raw_state_ids.ndim != 1 or not np.issubdtype(raw_state_ids.dtype, np.integer):
        raise ValueError("state_ids must be a one-dimensional integer array")
    if raw_branch_indices.ndim != 1 or raw_branch_indices.size < 1 or not np.issubdtype(raw_branch_indices.dtype, np.integer):
        raise ValueError("branch_reaction_indices must be a non-empty integer array")
 
    candidate_values = raw_candidate_ids.astype(int, copy=False)
    state_values = raw_state_ids.astype(int, copy=False)
    branch_indices = raw_branch_indices.astype(int, copy=False)
    if np.any(branch_indices < 0) or np.any(branch_indices >= fluxes.shape[1]) or np.unique(branch_indices).size != branch_indices.size:
        raise ValueError("branch_reaction_indices must contain unique valid reaction indices")
 
    selected_candidate_id = int(selected[0])
    selected_state_id = int(selected[1])
    candidate_rows = np.flatnonzero(candidate_values == selected_candidate_id)
    state_rows = np.flatnonzero(state_values == selected_state_id)
    if candidate_rows.size != 1 or state_rows.size != 1:
        raise ValueError("selected candidate and state IDs must each map to one row")
 
    contributions = metrics[int(candidate_rows[0]), int(state_rows[0]), 4:]
    if not np.all(np.isfinite(contributions)) or np.any(contributions < 0.0):
        raise ValueError("selected reaction enzyme-mass contributions must be finite and non-negative")
    denominator = float(np.sum(contributions, dtype=float))
    numerator = float(np.sum(contributions[branch_indices], dtype=float))
    if not np.isfinite(denominator) or denominator <= 0.0 or not np.isfinite(numerator):
        raise ValueError("selected enzyme-mass fraction is not finite")
    fraction = numerator / denominator
    if not np.isfinite(fraction):
        raise ValueError("final fraction must be finite")
    return float(fraction)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    base_setup = """import numpy as np
concentration_states_mM=np.array([[4.,2.,1.],[2.,4.,1.5],[5.,3.,2.],[3.,1.,4.]])
candidate_fluxes=np.array([
[6.,5.,2.,3.,3.,1.],[5.,5.,2.,3.,3.,0.],[7.,7.,4.,3.,3.,0.],
[6.,6.,1.,5.,5.,0.],[6.,6.,4.,2.,2.,0.],[7.,7.,5.,2.,2.,0.]])
candidate_ids=np.array([11,12,13,21,22,23]);candidate_parent_ids=np.array([1,1,1,2,2,2]);parent_ids=np.array([1,2])
parent_pattern_mask=np.ones((2,6));candidate_pattern_mask=(candidate_fluxes>0).astype(float)
coupling_group_ids=np.arange(1,7);relaxed_objective_values=np.array([10.,10.,9.4,8.,8.,7.5]);state_ids=np.array([6,2,8,3])
stoichiometric_columns=np.array([[1,-1,0,0,0,-1],[0,1,-1,-1,0,0],[0,0,0,1,-1,0]],float)
standard_gibbs_kj=np.array([-5.,-4.,-20.,-5.,-20.,-20.])
substrate_indices=np.array([-1,0,1,1,2,0]);substrate_km_mM=np.array([np.nan,1.,.8,1.,.9,1.2]);substrate_fixed_terms=np.array([3.,np.nan,np.nan,np.nan,np.nan,np.nan])
product_indices=np.array([0,1,-1,2,-1,-1]);product_km_mM=np.array([.5,.5,np.nan,.3,np.nan,np.nan]);product_fixed_terms=np.array([np.nan,np.nan,.2,np.nan,.2,.2])
turnover_forward_h=np.array([50000.,20000.,15000.,18000.,16000.,60000.]);molecular_weights_g_per_mmol=np.array([60.,55.,50.,58.,70.,62.])
temperature=300.;gas_constant=8.314462618e-3;minimum_driving_force=.5;enzyme_pool_limit=1.;objective_coefficients=np.array([0.,0.,1.,0.,1.01,0.])
reference_profile_mM=np.array([2.1,4.1,1.4]);objective_retention_fraction=.995;branch_reaction_indices=np.array([3,4]);relaxed_objective_tolerance=1e-9;profile_tolerance=1e-12;demand_tolerance=1e-12
def current_args():
 return (concentration_states_mM,candidate_fluxes,candidate_ids,candidate_parent_ids,parent_ids,parent_pattern_mask,candidate_pattern_mask,coupling_group_ids,relaxed_objective_values,state_ids,stoichiometric_columns,standard_gibbs_kj,substrate_indices,substrate_km_mM,substrate_fixed_terms,product_indices,product_km_mM,product_fixed_terms,turnover_forward_h,molecular_weights_g_per_mmol,temperature,gas_constant,minimum_driving_force,enzyme_pool_limit,objective_coefficients,reference_profile_mM,objective_retention_fraction,branch_reaction_indices,relaxed_objective_tolerance,profile_tolerance,demand_tolerance)
"""
    return [
        {
            "setup": base_setup,
            "call": "resolve_auxiliary_enzyme_fraction(*current_args())",
            "gold_call": "_oracle_resolve_auxiliary_enzyme_fraction(*current_args())",
        },
        {
            "setup": base_setup + """
order=np.array([4,0,5,2,3,1]);candidate_fluxes=candidate_fluxes[order];candidate_ids=candidate_ids[order];candidate_parent_ids=candidate_parent_ids[order];candidate_pattern_mask=candidate_pattern_mask[order];relaxed_objective_values=relaxed_objective_values[order]
""",
            "call": "resolve_auxiliary_enzyme_fraction(*current_args())",
            "gold_call": "_oracle_resolve_auxiliary_enzyme_fraction(*current_args())",
        },
        {
            "setup": base_setup + """
po=np.array([1,0]);parent_ids=parent_ids[po];parent_pattern_mask=parent_pattern_mask[po];candidate_parent_ids=np.where(candidate_parent_ids==1,20,10);parent_ids=np.array([10,20])
""",
            "call": "resolve_auxiliary_enzyme_fraction(*current_args())",
            "gold_call": "_oracle_resolve_auxiliary_enzyme_fraction(*current_args())",
        },
        {
            "setup": base_setup + """
branch_reaction_indices=np.array([99])
def candidate_wrapper():
    try: resolve_auxiliary_enzyme_fraction(*current_args())
    except ValueError: return 1.0
    return 0.0
def gold_wrapper():
    try: _oracle_resolve_auxiliary_enzyme_fraction(*current_args())
    except ValueError: return 1.0
    return 0.0
""",
            "call": "candidate_wrapper()",
            "gold_call": "gold_wrapper()",
        },
    ]
