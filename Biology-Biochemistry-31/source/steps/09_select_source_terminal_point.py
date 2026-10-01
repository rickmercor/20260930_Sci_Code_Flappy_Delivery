"""
Select one deterministic candidate/state point from the source-retained terminal set.

candidate_ids and state_ids are unique positive integer identifiers aligned with candidate_state_metrics. near_optimal_codes is the candidate-by-state output of filter_source_postprocessing_points.

Apply the matching source's metabolite-profile terminal criterion to the retained points, using the profile-deviation and demand channels in candidate_state_metrics. Resolve profile-deviation ties within profile_tolerance. Resolve any remaining tie by lower total enzyme demand within demand_tolerance, then by lower numerical candidate ID, then by lower numerical state ID.

Every retained point must be nonlinear-feasible with finite demand. Return the selected numerical candidate ID and selected numerical state ID as a two-element float array.

Raise ValueError for an empty retained set, invalid identifiers, invalid metric channels, inconsistent feasibility codes, non-finite or negative tolerances, or any dimensional mismatch.

Returns
-------
1D NumPy float array of shape (2,), containing [selected_candidate_id, selected_state_id].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
 
 
def select_source_terminal_point(
    candidate_ids: np.ndarray,
    state_ids: np.ndarray,
    candidate_state_metrics: np.ndarray,
    near_optimal_codes: np.ndarray,
    profile_tolerance: float,
    demand_tolerance: float,
) -> np.ndarray:
    """Select one deterministic terminal candidate/state point from the source-retained set."""
    return np.empty(2, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_select_source_terminal_point(
    candidate_ids,
    state_ids,
    candidate_state_metrics,
    near_optimal_codes,
    profile_tolerance,
    demand_tolerance,
):
    import numpy as np
 
    raw_candidate_ids = np.asarray(candidate_ids)
    raw_state_ids = np.asarray(state_ids)
    metrics = np.asarray(candidate_state_metrics, dtype=float)
    codes = np.asarray(near_optimal_codes, dtype=float)
 
    if raw_candidate_ids.ndim != 1 or raw_candidate_ids.size < 1 or not np.issubdtype(raw_candidate_ids.dtype, np.integer):
        raise ValueError("candidate_ids must be a non-empty integer vector")
    candidate_values = raw_candidate_ids.astype(int, copy=False)
    if np.any(candidate_values <= 0) or np.unique(candidate_values).size != candidate_values.size:
        raise ValueError("candidate_ids must contain unique positive integers")
    if raw_state_ids.ndim != 1 or raw_state_ids.size < 1 or not np.issubdtype(raw_state_ids.dtype, np.integer):
        raise ValueError("state_ids must be a non-empty integer vector")
    state_values = raw_state_ids.astype(int, copy=False)
    if np.any(state_values <= 0) or np.unique(state_values).size != state_values.size:
        raise ValueError("state_ids must contain unique positive integers")
 
    if metrics.ndim != 3 or metrics.shape[0] != candidate_values.size or metrics.shape[1] != state_values.size or metrics.shape[2] < 4:
        raise ValueError("candidate_state_metrics must align with candidates and states")
    if codes.shape != metrics.shape[:2] or not np.all(np.isfinite(codes)):
        raise ValueError("near_optimal_codes must be a finite aligned array")
    if not np.all((codes == 0.0) | (codes == 1.0)):
        raise ValueError("near_optimal_codes must contain only 0.0 and 1.0")
 
    profile = metrics[:, :, 2]
    demand = metrics[:, :, 1]
    feasible = metrics[:, :, 3]
    if np.any(np.isnan(profile)) or np.any(profile < 0.0):
        raise ValueError("profile-deviation channel must be non-negative and not NaN")
    if np.any(np.isnan(demand)) or np.any(demand < 0.0):
        raise ValueError("demand channel must be non-negative or positive infinity")
    if not np.all(np.isfinite(feasible)) or not np.all((feasible == 0.0) | (feasible == 1.0)):
        raise ValueError("feasible-code channel is invalid")
    if np.any((codes == 1.0) & ((feasible != 1.0) | (~np.isfinite(demand)))):
        raise ValueError("every near-optimal point must be feasible with finite demand")
 
    try:
        profile_tol = float(profile_tolerance)
        demand_tol = float(demand_tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError("tolerances must be real scalars") from exc
    if not np.isfinite(profile_tol) or profile_tol < 0.0 or not np.isfinite(demand_tol) or demand_tol < 0.0:
        raise ValueError("tolerances must be finite and non-negative")
 
    allowed = codes == 1.0
    if not np.any(allowed):
        raise ValueError("no near-optimal point exists")
    best_profile = float(np.min(profile[allowed]))
    profile_ties = allowed & (np.abs(profile - best_profile) <= profile_tol)
    best_demand = float(np.min(demand[profile_ties]))
    demand_ties = profile_ties & (np.abs(demand - best_demand) <= demand_tol)
    positions = np.argwhere(demand_ties)
    order = np.lexsort((state_values[positions[:, 1]], candidate_values[positions[:, 0]]))
    candidate_position, state_position = positions[order[0]]
    return np.array(
        [float(candidate_values[candidate_position]), float(state_values[state_position])],
        dtype=float,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
cid=np.array([8,3]);sid=np.array([9,2]);m=np.zeros((2,2,5));m[:,:,1]=[[.3,.2],[.25,.1]];m[:,:,2]=[[2.,1.],[.8,.7]];m[:,:,3]=1.;codes=np.ones((2,2));pt=1e-12;dt=1e-12""",
            "call": "select_source_terminal_point(cid.copy(),sid.copy(),m.copy(),codes.copy(),pt,dt)",
            "gold_call": "_oracle_select_source_terminal_point(cid.copy(),sid.copy(),m.copy(),codes.copy(),pt,dt)",
        },
        {
            "setup": """import numpy as np
cid=np.array([9,2,7]);sid=np.array([4,1]);m=np.zeros((3,2,4));m[:,:,1]=[[.12,.13],[.1200000000005,.2],[.08,.09]];m[:,:,2]=1.;m[:,:,3]=1.;codes=np.ones((3,2));pt=1e-12;dt=1e-12""",
            "call": "select_source_terminal_point(cid.copy(),sid.copy(),m.copy(),codes.copy(),pt,dt)",
            "gold_call": "_oracle_select_source_terminal_point(cid.copy(),sid.copy(),m.copy(),codes.copy(),pt,dt)",
        },
        {
            "setup": """import numpy as np
cid=np.array([14,5,9]);sid=np.array([12,3]);m=np.zeros((3,2,4));m[:,:,1]=np.arange(6).reshape(3,2)/100+.1;m[:,:,2]=np.array([[2.,1.],[.8,.7],[1.2,1.1]]);m[:,:,3]=1.;codes=np.ones((3,2));o=np.array([2,0,1]);s=np.array([1,0]);cid=cid[o];m=m[o][:,s];codes=codes[o][:,s];sid=sid[s];pt=1e-12;dt=1e-12""",
            "call": "select_source_terminal_point(cid.copy(),sid.copy(),m.copy(),codes.copy(),pt,dt)",
            "gold_call": "_oracle_select_source_terminal_point(cid.copy(),sid.copy(),m.copy(),codes.copy(),pt,dt)",
        },
        {
            "setup": """import numpy as np
cid=np.array([1]);sid=np.array([1]);m=np.zeros((1,1,4));codes=np.zeros((1,1));pt=1e-12;dt=1e-12
def candidate_wrapper():
    try: select_source_terminal_point(cid,sid,m,codes,pt,dt)
    except ValueError: return 1.0
    return 0.0
def gold_wrapper():
    try: _oracle_select_source_terminal_point(cid,sid,m,codes,pt,dt)
    except ValueError: return 1.0
    return 0.0""",
            "call": "candidate_wrapper()",
            "gold_call": "gold_wrapper()",
        },
    ]
