"""
Encode the source-retained candidate/state points for terminal postprocessing.

candidate_state_metrics is the output of compute_candidate_state_metrics. source_selection_codes is the candidate-aligned 0.0/1.0 output of select_source_screen_records. objective_retention_fraction is a finite scalar in (0, 1].

Apply the matching source's objective-retention postprocessing to the source-carried nonlinear-feasible points. Use the objective channel and feasibility-code channel in candidate_state_metrics, and apply the supplied retention fraction to the raw maximal eligible objective. The retention endpoint is inclusive.

Return one numerical 0.0/1.0 code for every candidate/state point, preserving candidate and state order. Raise ValueError when no source-selected nonlinear-feasible point exists, the maximal eligible objective is non-positive, or the public contract is violated.

Returns
-------
2D NumPy float array of shape (n_candidates, n_states), containing only 0.0 and 1.0.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
 
 
def filter_source_postprocessing_points(
    candidate_state_metrics: np.ndarray,
    source_selection_codes: np.ndarray,
    objective_retention_fraction: float,
) -> np.ndarray:
    """Encode source-selected nonlinear-feasible points inside the objective-retention envelope."""
    return np.empty((0, 0), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_filter_source_postprocessing_points(
    candidate_state_metrics,
    source_selection_codes,
    objective_retention_fraction,
):
    import numpy as np
 
    metrics = np.asarray(candidate_state_metrics, dtype=float)
    codes = np.asarray(source_selection_codes, dtype=float)
 
    if metrics.ndim != 3 or min(metrics.shape[:2]) < 1 or metrics.shape[2] < 4:
        raise ValueError("candidate_state_metrics must have shape (candidates, states, channels)")
    if codes.shape != (metrics.shape[0],) or not np.all(np.isfinite(codes)):
        raise ValueError("source_selection_codes must be a finite candidate-aligned vector")
    if not np.all((codes == 0.0) | (codes == 1.0)):
        raise ValueError("source_selection_codes must contain only 0.0 and 1.0")
 
    objective = metrics[:, :, 0]
    feasible = metrics[:, :, 3]
    if not np.all(np.isfinite(objective)):
        raise ValueError("objective channel must contain finite values")
    if not np.all(np.isfinite(feasible)) or not np.all((feasible == 0.0) | (feasible == 1.0)):
        raise ValueError("feasible-code channel must contain only 0.0 and 1.0")
 
    try:
        fraction = float(objective_retention_fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError("objective_retention_fraction must be a real scalar") from exc
    if not np.isfinite(fraction) or fraction <= 0.0 or fraction > 1.0:
        raise ValueError("objective_retention_fraction must lie in (0, 1]")
 
    eligible = (codes[:, None] == 1.0) & (feasible == 1.0)
    if not np.any(eligible):
        raise ValueError("no source-selected nonlinear-feasible point exists")
    maximum = float(np.max(objective[eligible]))
    if maximum <= 0.0:
        raise ValueError("the maximal eligible objective must be positive")
    threshold = fraction * maximum
    return (eligible & (objective >= threshold)).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
m=np.zeros((3,3,5));m[:,:,0]=np.array([[100.,100.,100.],[99.6,99.6,99.6],[101.,101.,101.]]);m[:,:,3]=1.;codes=np.array([1.,1.,0.]);fraction=.995""",
            "call": "filter_source_postprocessing_points(m.copy(),codes.copy(),fraction)",
            "gold_call": "_oracle_filter_source_postprocessing_points(m.copy(),codes.copy(),fraction)",
        },
        {
            "setup": """import numpy as np
m=np.zeros((2,3,4));m[:,:,0]=[[10.,10.,10.],[9.95,9.95,9.95]];m[:,:,3]=[[1.,0.,1.],[1.,1.,1.]];codes=np.array([1.,1.]);fraction=.995""",
            "call": "filter_source_postprocessing_points(m.copy(),codes.copy(),fraction)",
            "gold_call": "_oracle_filter_source_postprocessing_points(m.copy(),codes.copy(),fraction)",
        },
        {
            "setup": """import numpy as np
m=np.zeros((3,2,4));m[:,:,0]=np.array([[8.,8.],[9.,9.],[8.96,8.96]]);m[:,:,3]=1.;codes=np.array([1.,0.,1.]);o=np.array([2,0,1]);s=np.array([1,0]);m=m[o][:,s];codes=codes[o];fraction=.995""",
            "call": "filter_source_postprocessing_points(m.copy(),codes.copy(),fraction)",
            "gold_call": "_oracle_filter_source_postprocessing_points(m.copy(),codes.copy(),fraction)",
        },
        {
            "setup": """import numpy as np
m=np.zeros((1,1,4));m[0,0,0]=1.;m[0,0,3]=0.;codes=np.array([1.]);fraction=.995
def candidate_wrapper():
    try: filter_source_postprocessing_points(m,codes,fraction)
    except ValueError: return 1.0
    return 0.0
def gold_wrapper():
    try: _oracle_filter_source_postprocessing_points(m,codes,fraction)
    except ValueError: return 1.0
    return 0.0""",
            "call": "candidate_wrapper()",
            "gold_call": "gold_wrapper()",
        },
    ]
