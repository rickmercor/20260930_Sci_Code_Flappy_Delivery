#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def compute_overall_moiety_change(reactant_moiety_counts, product_moiety_counts):
    import numpy as np

    reactants = np.asarray(reactant_moiety_counts, dtype=float)
    products = np.asarray(product_moiety_counts, dtype=float)

    if reactants.ndim != 1 or reactants.size < 1:
        raise ValueError('reactant_moiety_counts must be a non-empty one-dimensional array')
    if products.shape != reactants.shape:
        raise ValueError('product_moiety_counts must align with reactant_moiety_counts')
    if not np.all(np.isfinite(reactants)) or not np.all(np.isfinite(products)):
        raise ValueError('moiety counts must contain only finite values')
    if np.any(reactants < 0.0) or np.any(products < 0.0):
        raise ValueError('moiety counts must be non-negative')
    if not np.all(reactants == np.floor(reactants)) or not np.all(products == np.floor(products)):
        raise ValueError('moiety counts must be integer-valued')

    return products - reactants

def enumerate_balanced_rule_multisets(
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

def order_balanced_rule_multisets(
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

def collect_source_feasible_mechanisms(
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

def encode_mechanism_arrow_sets(mechanism_rule_counts, rule_arrow_incidence):
    import numpy as np

    counts = np.asarray(mechanism_rule_counts, dtype=float)
    incidence = np.asarray(rule_arrow_incidence, dtype=float)

    if counts.ndim != 2 or counts.shape[0] < 1 or counts.shape[1] < 1:
        raise ValueError('mechanism_rule_counts must be a non-empty two-dimensional array')
    if incidence.ndim != 2 or incidence.shape[0] != counts.shape[1] or incidence.shape[1] < 1:
        raise ValueError('rule_arrow_incidence must align with rule columns and contain arrow columns')
    if (
        not np.all(np.isfinite(counts))
        or not np.all(counts == np.floor(counts))
        or np.any(counts < 0.0)
    ):
        raise ValueError('mechanism_rule_counts must contain non-negative integers')
    if (
        not np.all(np.isfinite(incidence))
        or not np.all((incidence == 0.0) | (incidence == 1.0))
    ):
        raise ValueError('rule_arrow_incidence must contain only 0.0 and 1.0')
    if np.any(np.sum(counts, axis=1) < 1.0):
        raise ValueError('every mechanism must contain at least one rule')

    return (
        ((counts > 0.0).astype(float) @ incidence) > 0.0
    ).astype(float)

def compute_mechanism_reference_jaccard(mechanism_arrow_sets, reference_arrow_sets):
    import numpy as np

    mechanism = np.asarray(mechanism_arrow_sets, dtype=float)
    reference = np.asarray(reference_arrow_sets, dtype=float)

    if mechanism.ndim != 2 or mechanism.shape[0] < 1 or mechanism.shape[1] < 1:
        raise ValueError('mechanism_arrow_sets must be a non-empty two-dimensional array')
    if reference.ndim != 2 or reference.shape[0] < 1 or reference.shape[1] != mechanism.shape[1]:
        raise ValueError('reference_arrow_sets must align with arrow columns')
    if (
        not np.all(np.isfinite(mechanism))
        or not np.all((mechanism == 0.0) | (mechanism == 1.0))
        or not np.all(np.isfinite(reference))
        or not np.all((reference == 0.0) | (reference == 1.0))
    ):
        raise ValueError('arrow-set arrays must contain only 0.0 and 1.0')
    if np.any(np.sum(mechanism, axis=1) < 1.0) or np.any(np.sum(reference, axis=1) < 1.0):
        raise ValueError('every mechanism and reference must contain at least one arrow feature')

    result = np.empty(
        (mechanism.shape[0], reference.shape[0]),
        dtype=float,
    )

    for mechanism_index in range(mechanism.shape[0]):
        mechanism_mask = mechanism[mechanism_index] == 1.0
        for reference_index in range(reference.shape[0]):
            reference_mask = reference[reference_index] == 1.0
            intersection = int(np.count_nonzero(mechanism_mask & reference_mask))
            union = int(np.count_nonzero(mechanism_mask | reference_mask))
            result[mechanism_index, reference_index] = intersection / union

    return result

def summarize_mechanism_similarity(
    jaccard_matrix,
    mechanism_ids,
    reference_ids,
    score_tolerance,
):
    import numpy as np

    scores = np.asarray(jaccard_matrix, dtype=float)
    raw_mechanism_ids = np.asarray(mechanism_ids)
    raw_reference_ids = np.asarray(reference_ids)

    if scores.ndim != 2 or scores.shape[0] < 1 or scores.shape[1] < 1:
        raise ValueError('jaccard_matrix must be a non-empty two-dimensional array')
    if (
        not np.all(np.isfinite(scores))
        or np.any(scores < 0.0)
        or np.any(scores > 1.0)
    ):
        raise ValueError('jaccard_matrix must contain finite values in [0, 1]')

    if (
        raw_mechanism_ids.ndim != 1
        or raw_mechanism_ids.size != scores.shape[0]
        or not np.issubdtype(raw_mechanism_ids.dtype, np.integer)
    ):
        raise ValueError('mechanism_ids must be an integer array aligned with mechanism rows')
    mechanism_id_values = raw_mechanism_ids.astype(int, copy=False)
    if np.any(mechanism_id_values <= 0) or np.unique(mechanism_id_values).size != mechanism_id_values.size:
        raise ValueError('mechanism_ids must contain unique positive integers')

    if (
        raw_reference_ids.ndim != 1
        or raw_reference_ids.size != scores.shape[1]
        or not np.issubdtype(raw_reference_ids.dtype, np.integer)
    ):
        raise ValueError('reference_ids must be an integer array aligned with reference columns')
    reference_id_values = raw_reference_ids.astype(int, copy=False)
    if np.any(reference_id_values <= 0) or np.unique(reference_id_values).size != reference_id_values.size:
        raise ValueError('reference_ids must contain unique positive integers')

    try:
        tolerance = float(score_tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError('score_tolerance must be a real scalar') from exc
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError('score_tolerance must be finite and non-negative')

    result = np.empty((scores.shape[0], 3), dtype=float)

    for mechanism_index in range(scores.shape[0]):
        best_score = float(np.max(scores[mechanism_index]))
        tied_positions = np.flatnonzero(
            np.abs(scores[mechanism_index] - best_score) <= tolerance
        )
        chosen_position = tied_positions[
            np.argmin(reference_id_values[tied_positions])
        ]
        result[mechanism_index] = (
            float(mechanism_id_values[mechanism_index]),
            float(reference_id_values[chosen_position]),
            best_score,
        )

    return result

def rerank_source_mechanisms(similarity_summary, score_tolerance):
    import numpy as np

    summary = np.asarray(similarity_summary, dtype=float)

    if summary.ndim != 2 or summary.shape[0] < 2 or summary.shape[1] != 3:
        raise ValueError('similarity_summary must have shape (n_mechanisms, 3) with at least two rows')
    if not np.all(np.isfinite(summary)):
        raise ValueError('similarity_summary must contain only finite values')

    mechanism_ids = summary[:, 0]
    reference_ids = summary[:, 1]
    scores = summary[:, 2]

    if (
        not np.all(mechanism_ids == np.floor(mechanism_ids))
        or np.any(mechanism_ids <= 0.0)
        or np.unique(mechanism_ids).size != mechanism_ids.size
    ):
        raise ValueError('mechanism IDs must be unique positive integers')
    if not np.all(reference_ids == np.floor(reference_ids)) or np.any(reference_ids <= 0.0):
        raise ValueError('reference IDs must be positive integers')
    if np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError('similarity scores must lie in [0, 1]')

    try:
        tolerance = float(score_tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError('score_tolerance must be a real scalar') from exc
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError('score_tolerance must be finite and non-negative')

    remaining = list(range(summary.shape[0]))
    ordered_positions = []

    while remaining:
        best_score = max(float(scores[position]) for position in remaining)
        tied_positions = [
            position
            for position in remaining
            if abs(float(scores[position]) - best_score) <= tolerance
        ]
        tied_positions.sort(key=lambda position: int(mechanism_ids[position]))
        ordered_positions.extend(tied_positions)
        tied_set = set(tied_positions)
        remaining = [
            position
            for position in remaining
            if position not in tied_set
        ]

    result = np.empty((summary.shape[0], 4), dtype=float)
    for rank, position in enumerate(ordered_positions, start=1):
        result[rank - 1] = (
            float(rank),
            mechanism_ids[position],
            reference_ids[position],
            scores[position],
        )

    return result

def compute_similarity_confidence_margin(reranked_summary):
    import numpy as np

    ranking = np.asarray(reranked_summary, dtype=float)

    if ranking.ndim != 2 or ranking.shape[0] < 2 or ranking.shape[1] != 4:
        raise ValueError('reranked_summary must have shape (n_mechanisms, 4) with at least two rows')
    if not np.all(np.isfinite(ranking)):
        raise ValueError('reranked_summary must contain only finite values')

    expected_ranks = np.arange(1, ranking.shape[0] + 1, dtype=float)
    if not np.array_equal(ranking[:, 0], expected_ranks):
        raise ValueError('the rank column must be consecutive and one-based')
    if (
        not np.all(ranking[:, 1] == np.floor(ranking[:, 1]))
        or np.any(ranking[:, 1] <= 0.0)
        or np.unique(ranking[:, 1]).size != ranking.shape[0]
    ):
        raise ValueError('mechanism IDs must be unique positive integers')
    if not np.all(ranking[:, 2] == np.floor(ranking[:, 2])) or np.any(ranking[:, 2] <= 0.0):
        raise ValueError('reference IDs must be positive integers')
    if np.any(ranking[:, 3] < 0.0) or np.any(ranking[:, 3] > 1.0):
        raise ValueError('similarity scores must lie in [0, 1]')
    if np.any(ranking[:-1, 3] < ranking[1:, 3]):
        raise ValueError('reranked_summary must be ordered by non-increasing score')

    return float(ranking[0, 3] - ranking[1, 3])

def resolve_mechanism_similarity_margin(
    reactant_moiety_counts,
    product_moiety_counts,
    rule_change_matrix,
    rule_ids,
    rule_use_limits,
    catalytic_moiety_mask,
    rule_arrow_incidence,
    reference_arrow_sets,
    reference_ids,
    max_steps,
    mechanism_count,
    score_tolerance,
):
    import numpy as np

    overall_change = compute_overall_moiety_change(
        reactant_moiety_counts,
        product_moiety_counts,
    )

    balanced_rule_counts = enumerate_balanced_rule_multisets(
        rule_change_matrix,
        overall_change,
        rule_ids,
        rule_use_limits,
        max_steps,
    )

    if balanced_rule_counts.shape[0] < 1:
        raise ValueError('no balanced rule multiset exists')

    balanced_order_summary = order_balanced_rule_multisets(
        rule_change_matrix,
        rule_ids,
        balanced_rule_counts,
        reactant_moiety_counts,
        catalytic_moiety_mask,
        max_steps,
    )

    mechanism_summary = collect_source_feasible_mechanisms(
        balanced_rule_counts,
        balanced_order_summary,
        rule_ids,
        mechanism_count,
    )

    n_rules = np.asarray(rule_ids).size
    mechanism_rule_counts = mechanism_summary[:, 3:3 + n_rules]

    mechanism_arrow_sets = encode_mechanism_arrow_sets(
        mechanism_rule_counts,
        rule_arrow_incidence,
    )

    jaccard_matrix = compute_mechanism_reference_jaccard(
        mechanism_arrow_sets,
        reference_arrow_sets,
    )

    similarity_summary = summarize_mechanism_similarity(
        jaccard_matrix,
        mechanism_summary[:, 0].astype(int),
        reference_ids,
        score_tolerance,
    )

    reranked_summary = rerank_source_mechanisms(
        similarity_summary,
        score_tolerance,
    )

    return compute_similarity_confidence_margin(reranked_summary)
SCICODE_GOLD_EOF
