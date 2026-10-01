"""
Enumerate every bounded rule multiset whose summed moiety change matches the overall transformation and return the source-ranked rows.

rule_change_matrix has one moiety row and one elementary-rule column. overall_change is the aligned target vector. rule_ids contains unique positive identifiers aligned with rule columns, and rule_use_limits gives the maximum non-negative integer use count for each rule. Enumerate every non-negative integer rule-count vector that does not exceed its aligned use limit, uses no more than max_steps total rules, and whose matrix product equals overall_change exactly. Rank rows first by total rule count, then lexicographically by the expanded numerical rule-ID list. Preserve the supplied rule-column order in the returned count matrix. Return an empty matrix when no balanced multiset exists. Raise ValueError if the public contract is violated.

Returns
-------
2D NumPy float array with one source-ranked balanced multiset per row and one column per supplied rule.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def enumerate_balanced_rule_multisets(
    rule_change_matrix: np.ndarray,
    overall_change: np.ndarray,
    rule_ids: np.ndarray,
    rule_use_limits: np.ndarray,
    max_steps: int,
) -> np.ndarray:
    """Return all bounded balanced rule multisets in source rank order."""
    return np.empty((0, 0), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_enumerate_balanced_rule_multisets(
    rule_change_matrix,
    overall_change,
    rule_ids,
    rule_use_limits,
    max_steps,
):
    import numpy as np

    changes = np.asarray(rule_change_matrix, dtype=float)
    target = np.asarray(overall_change, dtype=float)
    raw_rule_ids = np.asarray(rule_ids)
    raw_limits = np.asarray(rule_use_limits)

    if changes.ndim != 2 or changes.shape[0] < 1 or changes.shape[1] < 1:
        raise ValueError('rule_change_matrix must be a non-empty two-dimensional array')
    n_moieties, n_rules = changes.shape
    if target.shape != (n_moieties,):
        raise ValueError('overall_change must align with the moiety rows')
    if not np.all(np.isfinite(changes)) or not np.all(np.isfinite(target)):
        raise ValueError('rule changes and overall change must be finite')
    if not np.all(changes == np.floor(changes)) or not np.all(target == np.floor(target)):
        raise ValueError('rule changes and overall change must be integer-valued')

    if (
        raw_rule_ids.ndim != 1
        or raw_rule_ids.size != n_rules
        or not np.issubdtype(raw_rule_ids.dtype, np.integer)
    ):
        raise ValueError('rule_ids must be a one-dimensional integer array aligned with rule columns')
    rule_id_values = raw_rule_ids.astype(int, copy=False)
    if np.any(rule_id_values <= 0) or np.unique(rule_id_values).size != n_rules:
        raise ValueError('rule_ids must contain unique positive integers')

    if (
        raw_limits.ndim != 1
        or raw_limits.size != n_rules
        or not np.issubdtype(raw_limits.dtype, np.integer)
    ):
        raise ValueError('rule_use_limits must be a one-dimensional integer array aligned with rule columns')
    limit_values = raw_limits.astype(int, copy=False)
    if np.any(limit_values < 0):
        raise ValueError('rule_use_limits must contain non-negative integers')

    if isinstance(max_steps, (bool, np.bool_)):
        raise ValueError('max_steps must be an integer')
    try:
        step_limit = int(max_steps)
    except (TypeError, ValueError) as exc:
        raise ValueError('max_steps must be an integer') from exc
    if step_limit != max_steps or step_limit < 0:
        raise ValueError('max_steps must be a non-negative integer')

    counts = np.zeros(n_rules, dtype=int)
    solutions = []

    def visit(position, used_steps):
        if position == n_rules:
            if np.array_equal(changes @ counts, target):
                solutions.append(counts.copy())
            return

        maximum = min(int(limit_values[position]), step_limit - used_steps)
        for value in range(maximum + 1):
            counts[position] = value
            visit(position + 1, used_steps + value)
        counts[position] = 0

    visit(0, 0)

    id_order = np.argsort(rule_id_values)

    def ranking_key(row):
        expanded_ids = []
        for position in id_order:
            expanded_ids.extend(
                [int(rule_id_values[position])] * int(row[position])
            )
        return int(np.sum(row)), tuple(expanded_ids)

    solutions.sort(key=ranking_key)

    if not solutions:
        return np.empty((0, n_rules), dtype=float)
    return np.asarray(solutions, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for this public function."""
    return [
        {
            "setup": """import numpy as np
rule_change_matrix = np.array([
    [-1,  0, -1, -1],
    [ 1, -1,  0,  2],
    [ 0,  1,  1, -1],
], dtype=float)
overall_change = np.array([-1, 0, 1], dtype=float)
rule_ids = np.array([10, 20, 30, 40], dtype=int)
rule_use_limits = np.ones(4, dtype=int)
max_steps = 3
""",
            "call": 'enumerate_balanced_rule_multisets(rule_change_matrix.copy(), overall_change.copy(), rule_ids.copy(), rule_use_limits.copy(), max_steps)',
            "gold_call": '_oracle_enumerate_balanced_rule_multisets(rule_change_matrix.copy(), overall_change.copy(), rule_ids.copy(), rule_use_limits.copy(), max_steps)',
        },
        {
            "setup": """import numpy as np
rule_change_matrix = np.array([
    [-1, -2],
    [ 1,  2],
], dtype=float)
overall_change = np.array([-2, 2], dtype=float)
rule_ids = np.array([7, 3], dtype=int)
rule_use_limits = np.array([2, 1], dtype=int)
max_steps = 2
""",
            "call": 'enumerate_balanced_rule_multisets(rule_change_matrix.copy(), overall_change.copy(), rule_ids.copy(), rule_use_limits.copy(), max_steps)',
            "gold_call": '_oracle_enumerate_balanced_rule_multisets(rule_change_matrix.copy(), overall_change.copy(), rule_ids.copy(), rule_use_limits.copy(), max_steps)',
        },
        {
            "setup": """import numpy as np
rule_change_matrix = np.array([
    [-1,  0, -1, -1],
    [ 1, -1,  0,  2],
    [ 0,  1,  1, -1],
], dtype=float)
overall_change = np.array([-1, 0, 1], dtype=float)
rule_ids = np.array([41, 12, 33, 25], dtype=int)
rule_use_limits = np.ones(4, dtype=int)
order = np.array([2, 0, 3, 1], dtype=int)
rule_change_matrix = rule_change_matrix[:, order]
rule_ids = rule_ids[order]
rule_use_limits = rule_use_limits[order]
max_steps = 3
""",
            "call": 'enumerate_balanced_rule_multisets(rule_change_matrix.copy(), overall_change.copy(), rule_ids.copy(), rule_use_limits.copy(), max_steps)',
            "gold_call": '_oracle_enumerate_balanced_rule_multisets(rule_change_matrix.copy(), overall_change.copy(), rule_ids.copy(), rule_use_limits.copy(), max_steps)',
        },
        {
            "setup": """import numpy as np
rule_change_matrix = np.array([[-1, -1], [1, 1]], dtype=float)
overall_change = np.array([-1, 1], dtype=float)
rule_ids = np.array([5, 5], dtype=int)
rule_use_limits = np.ones(2, dtype=int)
max_steps = 1

def candidate_wrapper():
    try:
        enumerate_balanced_rule_multisets(rule_change_matrix, overall_change, rule_ids, rule_use_limits, max_steps)
    except ValueError:
        return 1.0
    return 0.0

def gold_wrapper():
    try:
        _oracle_enumerate_balanced_rule_multisets(rule_change_matrix, overall_change, rule_ids, rule_use_limits, max_steps)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": 'candidate_wrapper()',
            "gold_call": 'gold_wrapper()',
        },
    ]
