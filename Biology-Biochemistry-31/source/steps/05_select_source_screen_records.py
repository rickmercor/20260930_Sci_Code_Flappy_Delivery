"""
Resolve the source-carried records from aligned family and constrained-screen metadata.

candidate_ids, candidate_parent_ids, relaxed_objective_values, and candidate_group_summary are aligned by candidate row. parent_ids and parent_group_summary are aligned by family row. Group-summary arrays are outputs of encode_pattern_group_states and therefore contain one effective group-state column per supplied group plus a final validity column.

Apply the matching source's constrained-screen advancement rule. The result must respect source-valid group encodings, the admissibility relation between a candidate record and its aligned family record, the best constrained-screen objective within objective_tolerance, and every auxiliary pattern solution that the source carries into nonlinear evaluation.

When the source rule leaves multiple records equivalent for one required carried pattern, select the lower numerical candidate ID. Return one numerical 0.0/1.0 code per candidate row and preserve candidate order.

Candidate and family IDs must be unique positive integers; every candidate family must occur in parent_ids. relaxed_objective_values must be finite. Group summaries and objective_tolerance must satisfy their stated numerical contracts. Raise ValueError when any family has no source-admissible candidate or the public contract is violated.

Returns
-------
1D NumPy float array of length n_candidates containing 1.0 for source-carried rows and 0.0 otherwise.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
 
def select_source_screen_records(candidate_ids: np.ndarray, candidate_parent_ids: np.ndarray, parent_ids: np.ndarray, relaxed_objective_values: np.ndarray, candidate_group_summary: np.ndarray, parent_group_summary: np.ndarray, objective_tolerance: float) -> np.ndarray:
    """Resolve the source-carried records from aligned upstream screen metadata."""
    return np.empty(0, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_select_source_screen_records(candidate_ids, candidate_parent_ids, parent_ids, relaxed_objective_values, candidate_group_summary, parent_group_summary, objective_tolerance):
    import numpy as np
    raw_candidate_ids = np.asarray(candidate_ids)
    raw_candidate_parent_ids = np.asarray(candidate_parent_ids)
    raw_parent_ids = np.asarray(parent_ids)
    relaxed = np.asarray(relaxed_objective_values, dtype=float)
    candidate_summary = np.asarray(candidate_group_summary, dtype=float)
    parent_summary = np.asarray(parent_group_summary, dtype=float)
    if raw_candidate_ids.ndim != 1 or raw_candidate_ids.size < 1 or (not np.issubdtype(raw_candidate_ids.dtype, np.integer)):
        raise ValueError('candidate_ids must be a non-empty one-dimensional integer array')
    candidate_id_values = raw_candidate_ids.astype(int, copy=False)
    if np.any(candidate_id_values <= 0) or np.unique(candidate_id_values).size != candidate_id_values.size:
        raise ValueError('candidate_ids must contain unique positive integers')
    if raw_candidate_parent_ids.ndim != 1 or raw_candidate_parent_ids.size != candidate_id_values.size or (not np.issubdtype(raw_candidate_parent_ids.dtype, np.integer)):
        raise ValueError('candidate_parent_ids must be an integer array aligned with candidates')
    candidate_parent_values = raw_candidate_parent_ids.astype(int, copy=False)
    if raw_parent_ids.ndim != 1 or raw_parent_ids.size < 1 or (not np.issubdtype(raw_parent_ids.dtype, np.integer)):
        raise ValueError('parent_ids must be a non-empty integer array')
    parent_id_values = raw_parent_ids.astype(int, copy=False)
    if np.any(parent_id_values <= 0) or np.unique(parent_id_values).size != parent_id_values.size:
        raise ValueError('parent_ids must contain unique positive integers')
    if not np.all(np.isin(candidate_parent_values, parent_id_values)):
        raise ValueError('every candidate parent ID must occur in parent_ids')
    if relaxed.ndim != 1 or relaxed.size != candidate_id_values.size or (not np.all(np.isfinite(relaxed))):
        raise ValueError('relaxed_objective_values must be finite and aligned with candidates')
    if candidate_summary.ndim != 2 or candidate_summary.shape[0] != candidate_id_values.size or candidate_summary.shape[1] < 2:
        raise ValueError('candidate_group_summary must align with candidates and contain a validity column')
    if parent_summary.shape != (parent_id_values.size, candidate_summary.shape[1]):
        raise ValueError('parent_group_summary must align with parent IDs and candidate group columns')
    if not np.all(np.isfinite(candidate_summary)) or not np.all(np.isfinite(parent_summary)):
        raise ValueError('group summaries must contain only finite numerical values')
    candidate_bits = candidate_summary[:, :-1]
    candidate_valid = candidate_summary[:, -1]
    parent_bits = parent_summary[:, :-1]
    parent_valid = parent_summary[:, -1]
    if not np.all((candidate_bits == -1.0) | (candidate_bits == 0.0) | (candidate_bits == 1.0)):
        raise ValueError('candidate group codes must be -1.0, 0.0, or 1.0')
    if not np.all((parent_bits == -1.0) | (parent_bits == 0.0) | (parent_bits == 1.0)):
        raise ValueError('parent group codes must be -1.0, 0.0, or 1.0')
    if not np.all((candidate_valid == 0.0) | (candidate_valid == 1.0)):
        raise ValueError('candidate validity codes must be 0.0 or 1.0')
    if not np.all(parent_valid == 1.0) or np.any(parent_bits < 0.0):
        raise ValueError('every parent pattern must have a valid coupled-group representation')
    try:
        tolerance = float(objective_tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError('objective_tolerance must be a real scalar') from exc
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError('objective_tolerance must be finite and non-negative')
    result = np.zeros(candidate_id_values.size, dtype=float)
    for parent_position, parent in enumerate(parent_id_values):
        positions = np.flatnonzero(candidate_parent_values == parent)
        if positions.size < 1:
            raise ValueError('every parent must have at least one candidate row')
        valid_positions = positions[(candidate_valid[positions] == 1.0) & np.all(candidate_bits[positions] <= parent_bits[parent_position][None, :], axis=1)]
        if valid_positions.size < 1:
            raise ValueError('a parent has no source-admissible candidate pattern')
        best_objective = np.max(relaxed[valid_positions])
        optimal_positions = valid_positions[np.abs(relaxed[valid_positions] - best_objective) <= tolerance]
        counts = np.sum(candidate_bits[optimal_positions], axis=1)
        for target_count in (np.max(counts), np.min(counts)):
            tied = optimal_positions[counts == target_count]
            chosen = tied[np.argmin(candidate_id_values[tied])]
            result[chosen] = 1.0
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\ncandidate_ids = np.array([11, 12, 13, 21, 22, 23], dtype=int)\ncandidate_parent_ids = np.array([1, 1, 1, 2, 2, 2], dtype=int)\nparent_ids = np.array([1, 2], dtype=int)\nrelaxed_objective_values = np.array([9.0, 9.0, 8.8, 7.5, 7.5, 7.5], dtype=float)\ncandidate_group_summary = np.array([\n    [1, 1, 1, 1],\n    [1, 0, 0, 1],\n    [1, 1, 0, 1],\n    [1, 1, 0, 1],\n    [1, 0, 0, 1],\n    [1, -1, 0, 0],\n], dtype=float)\nparent_group_summary = np.array([[1, 1, 1, 1], [1, 1, 0, 1]], dtype=float)\nobjective_tolerance = 1e-9\n', 'call': 'select_source_screen_records(candidate_ids.copy(), candidate_parent_ids.copy(), parent_ids.copy(), relaxed_objective_values.copy(), candidate_group_summary.copy(), parent_group_summary.copy(), objective_tolerance)', 'gold_call': '_oracle_select_source_screen_records(candidate_ids.copy(), candidate_parent_ids.copy(), parent_ids.copy(), relaxed_objective_values.copy(), candidate_group_summary.copy(), parent_group_summary.copy(), objective_tolerance)'}, {'setup': 'import numpy as np\ncandidate_ids = np.array([5, 2, 9, 4], dtype=int)\ncandidate_parent_ids = np.array([3, 3, 3, 3], dtype=int)\nparent_ids = np.array([3], dtype=int)\nrelaxed_objective_values = np.array([10.0, 10.0 + 5e-10, 10.0, 10.0], dtype=float)\ncandidate_group_summary = np.array([\n    [1, 1, 1, 1],\n    [1, 0, 0, 1],\n    [1, 1, 0, 1],\n    [1, 1, 1, 1],\n], dtype=float)\nparent_group_summary = np.array([[1, 1, 0, 1]], dtype=float)\nobjective_tolerance = 1e-9\n', 'call': 'select_source_screen_records(candidate_ids.copy(), candidate_parent_ids.copy(), parent_ids.copy(), relaxed_objective_values.copy(), candidate_group_summary.copy(), parent_group_summary.copy(), objective_tolerance)', 'gold_call': '_oracle_select_source_screen_records(candidate_ids.copy(), candidate_parent_ids.copy(), parent_ids.copy(), relaxed_objective_values.copy(), candidate_group_summary.copy(), parent_group_summary.copy(), objective_tolerance)'}, {'setup': 'import numpy as np\ncandidate_ids = np.array([31, 32, 33, 41, 42, 43], dtype=int)\ncandidate_parent_ids = np.array([1, 1, 1, 2, 2, 2], dtype=int)\nparent_ids = np.array([1, 2], dtype=int)\nrelaxed_objective_values = np.array([5.0, 5.0, 4.0, 6.0, 6.0, 6.0], dtype=float)\ncandidate_group_summary = np.array([[1,1,1,1],[1,0,0,1],[1,1,0,1],[1,1,1,1],[1,0,0,1],[1,1,0,1]], dtype=float)\nparent_group_summary = np.array([[1,1,1,1],[1,1,1,1]], dtype=float)\norder = np.array([4, 0, 5, 2, 3, 1], dtype=int)\ncandidate_ids = candidate_ids[order]\ncandidate_parent_ids = candidate_parent_ids[order]\nrelaxed_objective_values = relaxed_objective_values[order]\ncandidate_group_summary = candidate_group_summary[order]\nparent_order = np.array([1, 0], dtype=int)\nparent_ids = parent_ids[parent_order]\nparent_group_summary = parent_group_summary[parent_order]\nobjective_tolerance = 1e-9\n', 'call': 'select_source_screen_records(candidate_ids.copy(), candidate_parent_ids.copy(), parent_ids.copy(), relaxed_objective_values.copy(), candidate_group_summary.copy(), parent_group_summary.copy(), objective_tolerance)', 'gold_call': '_oracle_select_source_screen_records(candidate_ids.copy(), candidate_parent_ids.copy(), parent_ids.copy(), relaxed_objective_values.copy(), candidate_group_summary.copy(), parent_group_summary.copy(), objective_tolerance)'}, {'setup': 'import numpy as np\ncandidate_ids = np.array([1], dtype=int)\ncandidate_parent_ids = np.array([1], dtype=int)\nparent_ids = np.array([1], dtype=int)\nrelaxed_objective_values = np.array([1.0], dtype=float)\ncandidate_group_summary = np.array([[1.0, -1.0, 0.0]], dtype=float)\nparent_group_summary = np.array([[1.0, 1.0, 1.0]], dtype=float)\nobjective_tolerance = 1e-9\ndef candidate_wrapper():\n    try:\n        select_source_screen_records(candidate_ids, candidate_parent_ids, parent_ids, relaxed_objective_values, candidate_group_summary, parent_group_summary, objective_tolerance)\n    except ValueError:\n        return 1.0\n    return 0.0\ndef gold_wrapper():\n    try:\n        _oracle_select_source_screen_records(candidate_ids, candidate_parent_ids, parent_ids, relaxed_objective_values, candidate_group_summary, parent_group_summary, objective_tolerance)\n    except ValueError:\n        return 1.0\n    return 0.0\n', 'call': 'candidate_wrapper()', 'gold_call': 'gold_wrapper()'}]
