"""
Scan the ranked balanced rows, skip unorderable rows, and collect the requested number of source-feasible ordered mechanisms.

balanced_rule_counts and balanced_order_summary are aligned outputs of Items 2 and 3. Scan their rows in order. Skip every row whose feasibility code is 0.0 and continue until mechanism_count feasible rows have been collected. Validate that every feasible order uses exactly its aligned rule-count multiset. Return one float row per collected mechanism with: mechanism ID H as a positive integer code; one-based balanced-row ID B; step count; the complete aligned rule-count vector; and the complete padded ordered-rule-ID vector. Assign H IDs consecutively in collection order. Raise ValueError if fewer than mechanism_count feasible rows exist or the public contract is violated.

Returns
-------
2D NumPy float array with mechanism ID, balanced-row ID, step count, aligned rule counts, and padded ordered rule IDs.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def collect_source_feasible_mechanisms(
    balanced_rule_counts: np.ndarray,
    balanced_order_summary: np.ndarray,
    rule_ids: np.ndarray,
    mechanism_count: int,
) -> np.ndarray:
    """Collect the requested number of source-feasible ordered mechanisms."""
    return np.empty((0, 0), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_collect_source_feasible_mechanisms(
    balanced_rule_counts,
    balanced_order_summary,
    rule_ids,
    mechanism_count,
):
    import numpy as np

    balanced = np.asarray(balanced_rule_counts, dtype=float)
    orders = np.asarray(balanced_order_summary, dtype=float)
    raw_rule_ids = np.asarray(rule_ids)

    if balanced.ndim != 2 or balanced.shape[0] < 1 or balanced.shape[1] < 1:
        raise ValueError('balanced_rule_counts must be a non-empty two-dimensional array')
    n_balanced, n_rules = balanced.shape
    if (
        not np.all(np.isfinite(balanced))
        or not np.all(balanced == np.floor(balanced))
        or np.any(balanced < 0.0)
    ):
        raise ValueError('balanced_rule_counts must contain non-negative integers')

    if (
        raw_rule_ids.ndim != 1
        or raw_rule_ids.size != n_rules
        or not np.issubdtype(raw_rule_ids.dtype, np.integer)
    ):
        raise ValueError('rule_ids must be an integer array aligned with rule columns')
    rule_id_values = raw_rule_ids.astype(int, copy=False)
    if np.any(rule_id_values <= 0) or np.unique(rule_id_values).size != n_rules:
        raise ValueError('rule_ids must contain unique positive integers')

    if orders.ndim != 2 or orders.shape[0] != n_balanced or orders.shape[1] < 1:
        raise ValueError('balanced_order_summary must align with balanced rows')
    if not np.all(np.isfinite(orders)):
        raise ValueError('balanced_order_summary must contain finite values')
    feasible_codes = orders[:, 0]
    order_ids = orders[:, 1:]
    if not np.all((feasible_codes == 0.0) | (feasible_codes == 1.0)):
        raise ValueError('the feasibility column must contain only 0.0 and 1.0')
    if not np.all(order_ids == np.floor(order_ids)):
        raise ValueError('ordered rule identifiers must be integer-valued')

    if isinstance(mechanism_count, (bool, np.bool_)):
        raise ValueError('mechanism_count must be an integer')
    try:
        requested = int(mechanism_count)
    except (TypeError, ValueError) as exc:
        raise ValueError('mechanism_count must be an integer') from exc
    if requested != mechanism_count or requested < 1:
        raise ValueError('mechanism_count must be a positive integer')

    id_to_position = {
        int(rule_id): position
        for position, rule_id in enumerate(rule_id_values)
    }

    output_rows = []
    max_steps = orders.shape[1] - 1

    for balanced_index in range(n_balanced):
        counts = balanced[balanced_index].astype(int)
        step_count = int(np.sum(counts))
        row_order = order_ids[balanced_index]

        if feasible_codes[balanced_index] == 0.0:
            if not np.all(row_order == -1.0):
                raise ValueError('an infeasible balanced row must contain only -1 order padding')
            continue

        if step_count > max_steps:
            raise ValueError('a feasible order cannot fit in balanced_order_summary')
        used_ids = row_order[:step_count]
        padding = row_order[step_count:]
        if np.any(used_ids <= 0.0) or not np.all(padding == -1.0):
            raise ValueError('a feasible order must contain positive rule IDs followed by -1 padding')
        if any(int(rule_id) not in id_to_position for rule_id in used_ids):
            raise ValueError('balanced_order_summary contains an unknown rule ID')

        reconstructed = np.zeros(n_rules, dtype=int)
        for rule_id in used_ids:
            reconstructed[id_to_position[int(rule_id)]] += 1
        if not np.array_equal(reconstructed, counts):
            raise ValueError('a feasible order must use exactly the balanced rule multiset')

        mechanism_id = len(output_rows) + 1
        output_rows.append(
            np.concatenate(
                (
                    np.array(
                        [
                            float(mechanism_id),
                            float(balanced_index + 1),
                            float(step_count),
                        ],
                        dtype=float,
                    ),
                    balanced[balanced_index].astype(float),
                    row_order.astype(float),
                )
            )
        )

        if len(output_rows) == requested:
            break

    if len(output_rows) < requested:
        raise ValueError('fewer than mechanism_count balanced rule multisets have a feasible order')

    return np.vstack(output_rows)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for this public function."""
    return [
        {
            "setup": """import numpy as np
balanced_rule_counts = np.array([
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 0, 1, 0],
    [1, 0, 0, 1],
    [0, 1, 0, 1],
], dtype=float)
balanced_order_summary = np.array([
    [1, 10, 20],
    [0, -1, -1],
    [1, 30, -1],
    [1, 10, 40],
    [1, 20, 40],
], dtype=float)
rule_ids = np.array([10, 20, 30, 40], dtype=int)
mechanism_count = 3
""",
            "call": 'collect_source_feasible_mechanisms(balanced_rule_counts.copy(), balanced_order_summary.copy(), rule_ids.copy(), mechanism_count)',
            "gold_call": '_oracle_collect_source_feasible_mechanisms(balanced_rule_counts.copy(), balanced_order_summary.copy(), rule_ids.copy(), mechanism_count)',
        },
        {
            "setup": """import numpy as np
balanced_rule_counts = np.array([[2, 0], [0, 1]], dtype=float)
balanced_order_summary = np.array([[1, 5, 5, -1], [1, 9, -1, -1]], dtype=float)
rule_ids = np.array([5, 9], dtype=int)
mechanism_count = 1
""",
            "call": 'collect_source_feasible_mechanisms(balanced_rule_counts.copy(), balanced_order_summary.copy(), rule_ids.copy(), mechanism_count)',
            "gold_call": '_oracle_collect_source_feasible_mechanisms(balanced_rule_counts.copy(), balanced_order_summary.copy(), rule_ids.copy(), mechanism_count)',
        },
        {
            "setup": """import numpy as np
balanced_rule_counts = np.array([
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
], dtype=float)
balanced_order_summary = np.array([
    [1, 41, 12],
    [1, 33, -1],
    [0, -1, -1],
], dtype=float)
rule_ids = np.array([41, 12, 33], dtype=int)
rule_order = np.array([2, 0, 1], dtype=int)
balanced_rule_counts = balanced_rule_counts[:, rule_order]
rule_ids = rule_ids[rule_order]
mechanism_count = 2
""",
            "call": 'collect_source_feasible_mechanisms(balanced_rule_counts.copy(), balanced_order_summary.copy(), rule_ids.copy(), mechanism_count)',
            "gold_call": '_oracle_collect_source_feasible_mechanisms(balanced_rule_counts.copy(), balanced_order_summary.copy(), rule_ids.copy(), mechanism_count)',
        },
        {
            "setup": """import numpy as np
balanced_rule_counts = np.array([[1, 0]], dtype=float)
balanced_order_summary = np.array([[1, 8, -1]], dtype=float)
rule_ids = np.array([7, 8], dtype=int)
mechanism_count = 1

def candidate_wrapper():
    try:
        collect_source_feasible_mechanisms(balanced_rule_counts, balanced_order_summary, rule_ids, mechanism_count)
    except ValueError:
        return 1.0
    return 0.0

def gold_wrapper():
    try:
        _oracle_collect_source_feasible_mechanisms(balanced_rule_counts, balanced_order_summary, rule_ids, mechanism_count)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": 'candidate_wrapper()',
            "gold_call": 'gold_wrapper()',
        },
    ]
