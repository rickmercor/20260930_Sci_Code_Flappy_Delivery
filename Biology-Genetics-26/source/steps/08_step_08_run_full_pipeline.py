"""
Chain attachment screening, rooted-NNI direction inference, and the three-descendant parent-history mixture into the requested scalar.

Full-gene-tree evidence first identifies a branch pair whose discordant attachments exceed its avuncular control. Reconciliation then determines which member is the recipient. The selected internal recipient's coalescent lineage configurations determine the nonlinear mixture used to estimate gamma.

Returns
-------
float, the estimated per-lineage donor probability
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_full_pipeline(seed: int = 17) -> float:
    '''Run the deterministic full-gene-tree introgression analysis.

    Parameters
    ----------
    seed : int
        Seed passed to the Step 04 np.random.default_rng tie process.

    Returns
    -------
    gamma_hat : float
        Estimated per-lineage donor probability.

    Raises
    ------
    ValueError
        If seed is not an integer or the fixed configuration does not yield one supported candidate, a modeled recipient, and a mixture consistent with the observed attachment frequency.
    '''
    return gamma_hat

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_run_full_pipeline(seed: int = 17) -> float:
    """Reference implementation chaining every earlier step."""
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")

    attachment_codes_by_candidate = [
        np.array([1] * 50 + [2] * 22 + [0] * 8 + [2] * 10 + [0] * 10),
        np.array([2] * 30 + [1] * 50 + [0] * 20),
    ]
    branch_presence_a = np.column_stack(
        (
            [1] * 80 + [0] * 20,
            [1] * 100,
            [1] * 72 + [0] * 8 + [1] * 18 + [0] * 2,
        )
    )
    branch_presence_b = np.ones((100, 3), dtype=int)
    branch_presence_b[60:, 1] = 0
    branch_presence_by_candidate = [branch_presence_a, branch_presence_b]

    species_tree = (((((0, 1), 2), 3), 4), ((5, 6), 7))
    gene_tree_patterns_by_candidate = [
        [
            ((((0, 1), 2), (((5, 6), 7), 3)), 4),
            (((0, 1), 2), ((((5, 6), 7), 3), 4)),
            ((((0, 1), 2), ((5, 6), 7)), (3, 4)),
            ((((0, 1), 2), (3, 4)), ((5, 6), 7)),
            (((((0, 1), 2), 4), ((5, 6), 7)), 3),
        ],
        [
            ((((((5, 6), 7), (0, 1)), 2), 3), 4),
            ((((5, 6), 7), (0, 1)), ((2, 3), 4)),
        ],
    ]
    gene_tree_multiplicities_by_candidate = [
        np.array([11, 7, 13, 12, 7], dtype=int),
        np.array([31, 19], dtype=int),
    ]
    candidate_clade_masks = np.array([[224, 7], [224, 3]], dtype=int)

    candidate_summary = _oracle_summarize_candidate_evidence(
        attachment_codes_by_candidate,
        branch_presence_by_candidate,
    )
    screen = _oracle_screen_attachment_candidates(candidate_summary, z_cutoff=-1.96)
    supported_candidates = np.flatnonzero(screen[:, 4] == 1.0)
    if supported_candidates.size != 1:
        raise ValueError("the configuration must yield exactly one supported candidate")
    candidate_index = int(supported_candidates[0])

    reconciliation_summary = _oracle_reconcile_rooted_nni(
        species_tree,
        gene_tree_patterns_by_candidate[candidate_index],
        candidate_clade_masks[candidate_index],
    )
    vote_summary = _oracle_aggregate_recipient_votes(
        reconciliation_summary,
        gene_tree_multiplicities_by_candidate[candidate_index],
        int(seed),
    )
    recipient_index = int(vote_summary[2])
    recipient_mask = int(candidate_clade_masks[candidate_index, recipient_index])

    if recipient_mask == 7:
        first_interval, second_interval = 0.37, 0.42
        parent_attachment_probabilities = np.array(
            [
                0.08, 0.36, 0.43, 0.70, 0.31, 0.62, 0.68, 0.91,
                0.12, 0.58, 0.47, 0.88,
                0.10, 0.51, 0.61, 0.89,
                0.11, 0.56, 0.49, 0.87,
                0.15, 0.84,
                0.13, 0.82,
                0.14, 0.80,
            ],
            dtype=float,
        )
    elif recipient_mask == 224:
        first_interval, second_interval = 0.51, 0.33
        parent_attachment_probabilities = np.array(
            [
                0.09, 0.41, 0.35, 0.68, 0.33, 0.64, 0.59, 0.90,
                0.13, 0.55, 0.50, 0.86,
                0.11, 0.49, 0.63, 0.88,
                0.12, 0.57, 0.46, 0.85,
                0.16, 0.83,
                0.14, 0.81,
                0.15, 0.79,
            ],
            dtype=float,
        )
    else:
        raise ValueError("the selected recipient lacks a three-descendant model")

    configuration_probabilities = _oracle_compute_three_descendant_configurations(
        first_interval,
        second_interval,
    )
    gamma_hat = _oracle_estimate_three_descendant_gamma(
        int(candidate_summary[candidate_index, 0]),
        int(candidate_summary[candidate_index, 5]),
        configuration_probabilities,
        parent_attachment_probabilities,
    )
    parent_history_weights = _oracle_compute_parent_history_weights(
        configuration_probabilities,
        gamma_hat,
    )
    observed_frequency = (
        float(candidate_summary[candidate_index, 0])
        / float(candidate_summary[candidate_index, 5])
    )
    modeled_frequency = float(
        np.dot(parent_history_weights, parent_attachment_probabilities)
    )
    if not np.isclose(modeled_frequency, observed_frequency, rtol=0.0, atol=1e-8):
        raise ValueError("the inferred gamma does not recover the observed frequency")
    return float(gamma_hat)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": '''expected = 0.7482114295606223
def check(value):
    result = float(value)
    if abs(result - expected) > 1e-8:
        raise AssertionError("unexpected prompt-instance pipeline result")
    return 1
''',
            "call": "check(run_full_pipeline(seed=17))",
            "gold_call": "check(_oracle_run_full_pipeline(seed=17))",
        },
        {
            "setup": '''expected = 0.7603340758721338
def check(value):
    result = float(value)
    if abs(result - expected) > 1e-8:
        raise AssertionError("unexpected alternate-recipient pipeline result")
    return 2
''',
            "call": "check(run_full_pipeline(seed=49842))",
            "gold_call": "check(_oracle_run_full_pipeline(seed=49842))",
        },
        {
            "setup": '''def error_code(function):
    try:
        function(seed=1.5)
    except ValueError:
        return 3
    except Exception:
        return 4
    return 0
''',
            "call": "error_code(run_full_pipeline)",
            "gold_call": "error_code(_oracle_run_full_pipeline)",
        },
    ]
