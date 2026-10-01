"""
Encode reaction-level binary records as aligned source-consistent group-state summaries.

pattern_mask contains one row per record and one column per reaction, with numerical binary values 0.0 or 1.0. coupling_group_ids is a reaction-aligned positive integer vector.

Apply the reaction-group preprocessing implied by the matching source. Return one group-state column for each distinct group ID in ascending numerical group-ID order, followed by one row-validity column.

For a source-consistent group in a row, emit its shared binary state. If a row does not represent a source-consistent state for a supplied group, emit -1.0 for that group and mark the row invalid. The final column is 1.0 only when every group in that row has a valid source-consistent state; otherwise it is 0.0.

Preserve row order. Raise ValueError for non-binary pattern values, non-positive or misaligned group IDs, empty dimensions, or non-finite inputs.

Returns
-------
2D NumPy float array of shape (n_records, n_unique_groups + 1); the last column is a 0.0/1.0 row-validity code.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
 
def encode_pattern_group_states(pattern_mask: np.ndarray, coupling_group_ids: np.ndarray) -> np.ndarray:
    """Encode reaction-level binary records as aligned group-state summaries."""
    return np.empty((0, 0), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_encode_pattern_group_states(pattern_mask, coupling_group_ids):
    import numpy as np
    pattern = np.asarray(pattern_mask, dtype=float)
    raw_groups = np.asarray(coupling_group_ids)
    if pattern.ndim != 2 or min(pattern.shape) < 1:
        raise ValueError('pattern_mask must be a non-empty two-dimensional array')
    if not np.all(np.isfinite(pattern)) or not np.all((pattern == 0.0) | (pattern == 1.0)):
        raise ValueError('pattern_mask must contain only numerical 0.0 and 1.0 values')
    if raw_groups.ndim != 1 or raw_groups.size != pattern.shape[1] or (not np.issubdtype(raw_groups.dtype, np.integer)):
        raise ValueError('coupling_group_ids must be an integer array aligned with reaction columns')
    groups = raw_groups.astype(int, copy=False)
    if np.any(groups <= 0):
        raise ValueError('coupling_group_ids must contain positive integers')
    unique_groups = np.unique(groups)
    result = np.empty((pattern.shape[0], unique_groups.size + 1), dtype=float)
    valid = np.ones(pattern.shape[0], dtype=float)
    for output_col, group in enumerate(unique_groups):
        positions = np.flatnonzero(groups == group)
        group_values = pattern[:, positions]
        uniform = np.all(group_values == group_values[:, [0]], axis=1)
        result[:, output_col] = np.where(uniform, group_values[:, 0], -1.0)
        valid[~uniform] = 0.0
    result[:, -1] = valid
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\npattern_mask = np.array([[1, 1, 0, 0, 1], [1, 0, 0, 1, 1], [0, 0, 1, 1, 0]], dtype=float)\ncoupling_group_ids = np.array([1, 1, 2, 2, 3], dtype=int)\n', 'call': 'encode_pattern_group_states(pattern_mask.copy(), coupling_group_ids.copy())', 'gold_call': '_oracle_encode_pattern_group_states(pattern_mask.copy(), coupling_group_ids.copy())'}, {'setup': 'import numpy as np\npattern_mask = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=float)\ncoupling_group_ids = np.array([9, 2, 9, 7], dtype=int)\n', 'call': 'encode_pattern_group_states(pattern_mask.copy(), coupling_group_ids.copy())', 'gold_call': '_oracle_encode_pattern_group_states(pattern_mask.copy(), coupling_group_ids.copy())'}, {'setup': 'import numpy as np\npattern_mask = np.array([[1, 1, 0, 0, 1, 1], [0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]], dtype=float)\ncoupling_group_ids = np.array([1, 1, 2, 2, 3, 3], dtype=int)\nrp = np.array([4, 1, 5, 0, 3, 2], dtype=int)\npattern_mask = pattern_mask[:, rp]\ncoupling_group_ids = coupling_group_ids[rp]\nrow_order = np.array([2, 0, 1], dtype=int)\npattern_mask = pattern_mask[row_order]\n', 'call': 'encode_pattern_group_states(pattern_mask.copy(), coupling_group_ids.copy())', 'gold_call': '_oracle_encode_pattern_group_states(pattern_mask.copy(), coupling_group_ids.copy())'}, {'setup': 'import numpy as np\npattern_mask = np.array([[1.0, 0.5]], dtype=float)\ncoupling_group_ids = np.array([1, 1], dtype=int)\ndef candidate_wrapper():\n    try:\n        encode_pattern_group_states(pattern_mask, coupling_group_ids)\n    except ValueError:\n        return 1.0\n    return 0.0\ndef gold_wrapper():\n    try:\n        _oracle_encode_pattern_group_states(pattern_mask, coupling_group_ids)\n    except ValueError:\n        return 1.0\n    return 0.0\n', 'call': 'candidate_wrapper()', 'gold_call': 'gold_wrapper()'}]
