"""
Determine the canonical source-admissible order, if one exists, for every balanced rule multiset.

balanced_rule_counts contains ranked non-negative integer rule-count rows aligned with rule_change_matrix columns. For each row, use every selected rule exactly as many times as specified. Starting from initial_moiety_counts, a step is admissible only when the updated cumulative inventory remains non-negative for every moiety whose catalytic_moiety_mask entry is False. Moieties marked True are excluded from that cumulative gate. Among multiple admissible orders, choose the lexicographically smallest sequence of numerical rule IDs. Return one row per balanced multiset: column 0 is 1.0 when an admissible order exists and 0.0 otherwise; the remaining max_steps columns contain ordered rule IDs followed by -1.0 padding. An inadmissible row contains only -1.0 after its leading 0.0. Preserve balanced-row order. Raise ValueError if the public contract is violated.

Returns
-------
2D NumPy float array of shape (n_balanced, max_steps + 1) containing a 0.0/1.0 feasibility code followed by ordered rule IDs and -1.0 padding.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def order_balanced_rule_multisets(
    rule_change_matrix: np.ndarray,
    rule_ids: np.ndarray,
    balanced_rule_counts: np.ndarray,
    initial_moiety_counts: np.ndarray,
    catalytic_moiety_mask: np.ndarray,
    max_steps: int,
) -> np.ndarray:
    """Return source-admissible canonical orders for balanced rule rows."""
    return np.empty((0, 0), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_order_balanced_rule_multisets(
    rule_change_matrix,
    rule_ids,
    balanced_rule_counts,
    initial_moiety_counts,
    catalytic_moiety_mask,
    max_steps,
):
    import numpy as np

    changes = np.asarray(rule_change_matrix, dtype=float)
    raw_rule_ids = np.asarray(rule_ids)
    balanced = np.asarray(balanced_rule_counts, dtype=float)
    initial = np.asarray(initial_moiety_counts, dtype=float)
    catalytic = np.asarray(catalytic_moiety_mask)

    if changes.ndim != 2 or changes.shape[0] < 1 or changes.shape[1] < 1:
        raise ValueError('rule_change_matrix must be a non-empty two-dimensional array')
    n_moieties, n_rules = changes.shape
    if not np.all(np.isfinite(changes)) or not np.all(changes == np.floor(changes)):
        raise ValueError('rule_change_matrix must contain finite integer-valued changes')

    if (
        raw_rule_ids.ndim != 1
        or raw_rule_ids.size != n_rules
        or not np.issubdtype(raw_rule_ids.dtype, np.integer)
    ):
        raise ValueError('rule_ids must be a one-dimensional integer array aligned with rule columns')
    rule_id_values = raw_rule_ids.astype(int, copy=False)
    if np.any(rule_id_values <= 0) or np.unique(rule_id_values).size != n_rules:
        raise ValueError('rule_ids must contain unique positive integers')

    if balanced.ndim != 2 or balanced.shape[0] < 1 or balanced.shape[1] != n_rules:
        raise ValueError('balanced_rule_counts must be a non-empty array aligned with rule columns')
    if (
        not np.all(np.isfinite(balanced))
        or not np.all(balanced == np.floor(balanced))
        or np.any(balanced < 0.0)
    ):
        raise ValueError('balanced_rule_counts must contain non-negative integers')

    if initial.shape != (n_moieties,):
        raise ValueError('initial_moiety_counts must align with moiety rows')
    if (
        not np.all(np.isfinite(initial))
        or np.any(initial < 0.0)
        or not np.all(initial == np.floor(initial))
    ):
        raise ValueError('initial_moiety_counts must contain non-negative integers')

    if catalytic.shape != (n_moieties,) or catalytic.dtype != np.bool_:
        raise ValueError('catalytic_moiety_mask must be a boolean array aligned with moiety rows')

    if isinstance(max_steps, (bool, np.bool_)):
        raise ValueError('max_steps must be an integer')
    try:
        step_limit = int(max_steps)
    except (TypeError, ValueError) as exc:
        raise ValueError('max_steps must be an integer') from exc
    if step_limit != max_steps or step_limit < 0:
        raise ValueError('max_steps must be a non-negative integer')
    if np.any(np.sum(balanced, axis=1) > step_limit):
        raise ValueError('a balanced rule multiset exceeds max_steps')

    result = np.full(
        (balanced.shape[0], step_limit + 1),
        -1.0,
        dtype=float,
    )
    result[:, 0] = 0.0
    id_order = list(np.argsort(rule_id_values))

    for row_index in range(balanced.shape[0]):
        counts = balanced[row_index].astype(int)
        total_steps = int(np.sum(counts))
        remaining = counts.copy()
        chosen_positions = []

        def search(inventory):
            if len(chosen_positions) == total_steps:
                return True

            for position in id_order:
                if remaining[position] <= 0:
                    continue

                updated = inventory + changes[:, position]
                if np.any(updated[~catalytic] < 0.0):
                    continue

                remaining[position] -= 1
                chosen_positions.append(position)
                if search(updated):
                    return True
                chosen_positions.pop()
                remaining[position] += 1

            return False

        if search(initial.copy()):
            result[row_index, 0] = 1.0
            for step_index, position in enumerate(chosen_positions, start=1):
                result[row_index, step_index] = float(rule_id_values[position])

    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for this public function."""
    return [
        {
            "setup": """import numpy as np
rule_change_matrix = np.array([
    [-1,  0, -1,  0,  0],
    [ 1, -1,  0, -1,  1],
    [ 0,  1,  1,  1, -1],
    [-1,  1,  0,  0,  0],
], dtype=float)
rule_ids = np.array([10, 20, 30, 40, 50], dtype=int)
balanced_rule_counts = np.array([
    [1, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 1],
], dtype=float)
initial_moiety_counts = np.array([1, 0, 0, 0], dtype=float)
catalytic_moiety_mask = np.array([False, False, False, True], dtype=bool)
max_steps = 2
""",
            "call": 'order_balanced_rule_multisets(rule_change_matrix.copy(), rule_ids.copy(), balanced_rule_counts.copy(), initial_moiety_counts.copy(), catalytic_moiety_mask.copy(), max_steps)',
            "gold_call": '_oracle_order_balanced_rule_multisets(rule_change_matrix.copy(), rule_ids.copy(), balanced_rule_counts.copy(), initial_moiety_counts.copy(), catalytic_moiety_mask.copy(), max_steps)',
        },
        {
            "setup": """import numpy as np
rule_change_matrix = np.array([
    [-1, -1,  0],
    [ 1,  0, -1],
    [ 0,  1,  1],
], dtype=float)
rule_ids = np.array([5, 2, 9], dtype=int)
balanced_rule_counts = np.array([[1, 1, 0]], dtype=float)
initial_moiety_counts = np.array([2, 0, 0], dtype=float)
catalytic_moiety_mask = np.array([False, False, False], dtype=bool)
max_steps = 3
""",
            "call": 'order_balanced_rule_multisets(rule_change_matrix.copy(), rule_ids.copy(), balanced_rule_counts.copy(), initial_moiety_counts.copy(), catalytic_moiety_mask.copy(), max_steps)',
            "gold_call": '_oracle_order_balanced_rule_multisets(rule_change_matrix.copy(), rule_ids.copy(), balanced_rule_counts.copy(), initial_moiety_counts.copy(), catalytic_moiety_mask.copy(), max_steps)',
        },
        {
            "setup": """import numpy as np
rule_change_matrix = np.array([
    [-1,  0, -1,  0,  0],
    [ 1, -1,  0, -1,  1],
    [ 0,  1,  1,  1, -1],
    [-1,  1,  0,  0,  0],
], dtype=float)
rule_ids = np.array([10, 20, 30, 40, 50], dtype=int)
balanced_rule_counts = np.array([
    [1, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 1],
], dtype=float)
initial_moiety_counts = np.array([1, 0, 0, 0], dtype=float)
catalytic_moiety_mask = np.array([False, False, False, True], dtype=bool)
rule_order = np.array([4, 1, 3, 0, 2], dtype=int)
row_order = np.array([2, 0, 1], dtype=int)
rule_change_matrix = rule_change_matrix[:, rule_order]
rule_ids = rule_ids[rule_order]
balanced_rule_counts = balanced_rule_counts[row_order][:, rule_order]
max_steps = 2
""",
            "call": 'order_balanced_rule_multisets(rule_change_matrix.copy(), rule_ids.copy(), balanced_rule_counts.copy(), initial_moiety_counts.copy(), catalytic_moiety_mask.copy(), max_steps)',
            "gold_call": '_oracle_order_balanced_rule_multisets(rule_change_matrix.copy(), rule_ids.copy(), balanced_rule_counts.copy(), initial_moiety_counts.copy(), catalytic_moiety_mask.copy(), max_steps)',
        },
        {
            "setup": """import numpy as np
rule_change_matrix = np.array([[-1.0], [1.0]], dtype=float)
rule_ids = np.array([1], dtype=int)
balanced_rule_counts = np.array([[2.0]], dtype=float)
initial_moiety_counts = np.array([2.0, 0.0], dtype=float)
catalytic_moiety_mask = np.array([0, 0], dtype=int)
max_steps = 2

def candidate_wrapper():
    try:
        order_balanced_rule_multisets(rule_change_matrix, rule_ids, balanced_rule_counts, initial_moiety_counts, catalytic_moiety_mask, max_steps)
    except ValueError:
        return 1.0
    return 0.0

def gold_wrapper():
    try:
        _oracle_order_balanced_rule_multisets(rule_change_matrix, rule_ids, balanced_rule_counts, initial_moiety_counts, catalytic_moiety_mask, max_steps)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": 'candidate_wrapper()',
            "gold_call": 'gold_wrapper()',
        },
    ]
