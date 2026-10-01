"""
Compose the complete bounded source-matched mechanism search and return its similarity-confidence margin.

This is the only Final orchestrator. It must genuinely call and use the public functions from Items 1 through 9 in numerical order: compute_overall_moiety_change; enumerate_balanced_rule_multisets; order_balanced_rule_multisets; collect_source_feasible_mechanisms; encode_mechanism_arrow_sets; compute_mechanism_reference_jaccard; summarize_mechanism_similarity; rerank_source_mechanisms; and compute_similarity_confidence_margin. Use reactant_moiety_counts as the initial inventory for ordering. Do not reproduce any earlier calculation privately and do not hardcode a mechanism, reference, balanced-row ID, or final scalar. Return the float produced by compute_similarity_confidence_margin.

Returns
-------
One finite float containing the final similarity-confidence margin
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def resolve_mechanism_similarity_margin(
    reactant_moiety_counts: np.ndarray,
    product_moiety_counts: np.ndarray,
    rule_change_matrix: np.ndarray,
    rule_ids: np.ndarray,
    rule_use_limits: np.ndarray,
    catalytic_moiety_mask: np.ndarray,
    rule_arrow_incidence: np.ndarray,
    reference_arrow_sets: np.ndarray,
    reference_ids: np.ndarray,
    max_steps: int,
    mechanism_count: int,
    score_tolerance: float,
) -> float:
    """Resolve the complete bounded mechanism search and return its margin."""
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_resolve_mechanism_similarity_margin(
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

    overall_change = _oracle_compute_overall_moiety_change(
        reactant_moiety_counts,
        product_moiety_counts,
    )

    balanced_rule_counts = _oracle_enumerate_balanced_rule_multisets(
        rule_change_matrix,
        overall_change,
        rule_ids,
        rule_use_limits,
        max_steps,
    )

    if balanced_rule_counts.shape[0] < 1:
        raise ValueError('no balanced rule multiset exists')

    balanced_order_summary = _oracle_order_balanced_rule_multisets(
        rule_change_matrix,
        rule_ids,
        balanced_rule_counts,
        reactant_moiety_counts,
        catalytic_moiety_mask,
        max_steps,
    )

    mechanism_summary = _oracle_collect_source_feasible_mechanisms(
        balanced_rule_counts,
        balanced_order_summary,
        rule_ids,
        mechanism_count,
    )

    n_rules = np.asarray(rule_ids).size
    mechanism_rule_counts = mechanism_summary[:, 3:3 + n_rules]

    mechanism_arrow_sets = _oracle_encode_mechanism_arrow_sets(
        mechanism_rule_counts,
        rule_arrow_incidence,
    )

    jaccard_matrix = _oracle_compute_mechanism_reference_jaccard(
        mechanism_arrow_sets,
        reference_arrow_sets,
    )

    similarity_summary = _oracle_summarize_mechanism_similarity(
        jaccard_matrix,
        mechanism_summary[:, 0].astype(int),
        reference_ids,
        score_tolerance,
    )

    reranked_summary = _oracle_rerank_source_mechanisms(
        similarity_summary,
        score_tolerance,
    )

    return _oracle_compute_similarity_confidence_margin(reranked_summary)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for this public function."""
    return [
        {
            "setup": """import numpy as np
reactant_moiety_counts = np.array([1,0,0,0,0], dtype=int)
product_moiety_counts = np.array([0,0,0,1,0], dtype=int)
rule_change_matrix = np.array([
    [-1, 0,-1,-1, 0, 0],
    [ 1,-1, 0, 0, 0,-1],
    [ 0, 0, 0, 1,-1, 1],
    [ 0, 1, 1, 0, 1, 0],
    [-1, 1, 0, 0, 0, 0],
], dtype=int)
rule_ids = np.arange(1, 7, dtype=int)
rule_use_limits = np.ones(6, dtype=int)
catalytic_moiety_mask = np.array([False,False,False,False,True], dtype=bool)
rule_arrow_incidence = np.array([
    [1,1,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [1,0,0,1,0,0,0,0],
    [0,0,0,0,1,0,0,0],
    [0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,1,0],
], dtype=int)
reference_arrow_sets = np.array([
    [1,1,1,0,0,0,0,1],
    [1,0,0,1,0,0,0,1],
    [0,0,0,0,1,1,0,1],
], dtype=int)
reference_ids = np.array([1,2,3], dtype=int)
max_steps = 3
mechanism_count = 3
score_tolerance = 1e-12

def current_args():
    return (
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
    )
""",
            "call": 'resolve_mechanism_similarity_margin(*current_args())',
            "gold_call": '_oracle_resolve_mechanism_similarity_margin(*current_args())',
        },
        {
            "setup": """import numpy as np
reactant_moiety_counts = np.array([1,0,0,0,0], dtype=int)
product_moiety_counts = np.array([0,0,0,1,0], dtype=int)
rule_change_matrix = np.array([
    [-1, 0,-1,-1, 0, 0],
    [ 1,-1, 0, 0, 0,-1],
    [ 0, 0, 0, 1,-1, 1],
    [ 0, 1, 1, 0, 1, 0],
    [-1, 1, 0, 0, 0, 0],
], dtype=int)
rule_ids = np.arange(1, 7, dtype=int)
rule_use_limits = np.ones(6, dtype=int)
catalytic_moiety_mask = np.array([False,False,False,False,True], dtype=bool)
rule_arrow_incidence = np.array([
    [1,1,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [1,0,0,1,0,0,0,0],
    [0,0,0,0,1,0,0,0],
    [0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,1,0],
], dtype=int)
reference_arrow_sets = np.array([
    [1,1,1,0,0,0,0,1],
    [1,0,0,1,0,0,0,1],
    [0,0,0,0,1,1,0,1],
], dtype=int)
reference_ids = np.array([1,2,3], dtype=int)
max_steps = 3
mechanism_count = 3
score_tolerance = 1e-12

def current_args():
    return (
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
    )
reference_arrow_sets = reference_arrow_sets.copy()
reference_arrow_sets[0, 1] = 0.0
reference_arrow_sets[0, 6] = 1.0
""",
            "call": 'resolve_mechanism_similarity_margin(*current_args())',
            "gold_call": '_oracle_resolve_mechanism_similarity_margin(*current_args())',
        },
        {
            "setup": """import numpy as np
reactant_moiety_counts = np.array([1,0,0,0,0], dtype=int)
product_moiety_counts = np.array([0,0,0,1,0], dtype=int)
rule_change_matrix = np.array([
    [-1, 0,-1,-1, 0, 0],
    [ 1,-1, 0, 0, 0,-1],
    [ 0, 0, 0, 1,-1, 1],
    [ 0, 1, 1, 0, 1, 0],
    [-1, 1, 0, 0, 0, 0],
], dtype=int)
rule_ids = np.arange(1, 7, dtype=int)
rule_use_limits = np.ones(6, dtype=int)
catalytic_moiety_mask = np.array([False,False,False,False,True], dtype=bool)
rule_arrow_incidence = np.array([
    [1,1,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [1,0,0,1,0,0,0,0],
    [0,0,0,0,1,0,0,0],
    [0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,1,0],
], dtype=int)
reference_arrow_sets = np.array([
    [1,1,1,0,0,0,0,1],
    [1,0,0,1,0,0,0,1],
    [0,0,0,0,1,1,0,1],
], dtype=int)
reference_ids = np.array([1,2,3], dtype=int)
max_steps = 3
mechanism_count = 3
score_tolerance = 1e-12

def current_args():
    return (
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
    )
moiety_order = np.array([4,2,0,3,1], dtype=int)
rule_order = np.array([5,1,3,0,4,2], dtype=int)
arrow_order = np.array([7,3,0,6,2,5,1,4], dtype=int)
reference_order = np.array([2,0,1], dtype=int)
reactant_moiety_counts = reactant_moiety_counts[moiety_order]
product_moiety_counts = product_moiety_counts[moiety_order]
rule_change_matrix = rule_change_matrix[moiety_order][:, rule_order]
rule_ids = rule_ids[rule_order]
rule_use_limits = rule_use_limits[rule_order]
catalytic_moiety_mask = catalytic_moiety_mask[moiety_order]
rule_arrow_incidence = rule_arrow_incidence[rule_order][:, arrow_order]
reference_arrow_sets = reference_arrow_sets[reference_order][:, arrow_order]
reference_ids = reference_ids[reference_order]
""",
            "call": 'resolve_mechanism_similarity_margin(*current_args())',
            "gold_call": '_oracle_resolve_mechanism_similarity_margin(*current_args())',
        },
        {
            "setup": """import numpy as np
reactant_moiety_counts = np.array([1,0,0,0,0], dtype=int)
product_moiety_counts = np.array([0,0,0,1,0], dtype=int)
rule_change_matrix = np.array([
    [-1, 0,-1,-1, 0, 0],
    [ 1,-1, 0, 0, 0,-1],
    [ 0, 0, 0, 1,-1, 1],
    [ 0, 1, 1, 0, 1, 0],
    [-1, 1, 0, 0, 0, 0],
], dtype=int)
rule_ids = np.arange(1, 7, dtype=int)
rule_use_limits = np.ones(6, dtype=int)
catalytic_moiety_mask = np.array([False,False,False,False,True], dtype=bool)
rule_arrow_incidence = np.array([
    [1,1,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [1,0,0,1,0,0,0,0],
    [0,0,0,0,1,0,0,0],
    [0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,1,0],
], dtype=int)
reference_arrow_sets = np.array([
    [1,1,1,0,0,0,0,1],
    [1,0,0,1,0,0,0,1],
    [0,0,0,0,1,1,0,1],
], dtype=int)
reference_ids = np.array([1,2,3], dtype=int)
max_steps = 3
mechanism_count = 3
score_tolerance = 1e-12

def current_args():
    return (
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
    )
mechanism_count = 8

def candidate_wrapper():
    try:
        resolve_mechanism_similarity_margin(*current_args())
    except ValueError:
        return 1.0
    return 0.0

def gold_wrapper():
    try:
        _oracle_resolve_mechanism_similarity_margin(*current_args())
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": 'candidate_wrapper()',
            "gold_call": 'gold_wrapper()',
        },
    ]
