"""
Drug hypotheses aggregate evidence over ranked discovery genes. Candidate genes are ordered by decreasing core_score with smaller gene index breaking ties, and rank r among n candidates receives GeneWeight = 1 - r/n + 1e-6. Each evidence row contributes GeneWeight times its supplied clinical-stage and source weight. Drugs are ordered by summed DrugScore, then by more distinct targets, better best-gene rank, and smaller drug index.

Inputs

------

core_scores: Non-negative float array of shape (n_genes,), with positive values marking candidates.

evidence_rows: Float array of shape (n_rows, 3), holding drug index, gene index, and evidence weight.

Returns

-------

drug_ranking: Float array of shape (n_drugs, 5), holding drug index, DrugScore, target count, best gene rank, and drug rank.

Returns
-------
np.ndarray of shape (n_drugs, 5), ranked drug index, score, target count, best gene rank, and rank
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def aggregate_drug_evidence(
    core_scores: "np.ndarray",
    evidence_rows: "np.ndarray",
) -> "np.ndarray":
    """Rank drugs by rank-weighted target evidence.

    Parameters
    ----------
    core_scores : np.ndarray
        Non-negative integrated target scores; positive entries are candidates.
    evidence_rows : np.ndarray
        Rows containing integer drug index, integer candidate-gene index, and
        a finite non-negative evidence weight. Drug indices must be contiguous
        from zero.

    Raises
    ------
    ValueError
        If `core_scores` is not a non-empty finite non-negative vector, no
        candidate exists, `evidence_rows` is not a non-empty matrix with three
        columns, identifiers are not integers in range, drug identifiers are
        not contiguous from zero, an evidence gene is not a candidate, or an
        evidence weight is non-finite or negative.

    Returns
    -------
    drug_ranking : np.ndarray
        Rows containing drug index, DrugScore, distinct-target count, best
        gene rank, and one-based drug rank.
    """
    return drug_ranking  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811

def _oracle_aggregate_drug_evidence(
    core_scores: "np.ndarray",
    evidence_rows: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation."""
    core_scores = np.asarray(core_scores, dtype=float)
    evidence_rows = np.asarray(evidence_rows, dtype=float)
    if core_scores.ndim != 1 or core_scores.size == 0:
        raise ValueError("core_scores must be a non-empty vector")
    if not np.all(np.isfinite(core_scores)) or np.any(core_scores < 0.0):
        raise ValueError("core_scores must be finite and non-negative")
    candidate_indices = np.flatnonzero(core_scores > 0.0)
    if candidate_indices.size == 0:
        raise ValueError("at least one candidate target is required")
    if evidence_rows.ndim != 2 or evidence_rows.shape[1] != 3 or evidence_rows.shape[0] == 0:
        raise ValueError("evidence_rows must be a non-empty matrix with three columns")
    if not np.all(np.isfinite(evidence_rows)) or np.any(evidence_rows[:, 2] < 0.0):
        raise ValueError("evidence rows must be finite with non-negative weights")
    identifiers = evidence_rows[:, :2]
    if not np.all(identifiers == np.floor(identifiers)) or np.any(identifiers < 0.0):
        raise ValueError("drug and gene identifiers must be non-negative integers")
    drug_indices = identifiers[:, 0].astype(int)
    gene_indices = identifiers[:, 1].astype(int)
    if np.any(gene_indices >= core_scores.size):
        raise ValueError("gene identifiers must index core_scores")
    if np.any(core_scores[gene_indices] <= 0.0):
        raise ValueError("every evidence gene must be a candidate target")
    unique_drugs = np.unique(drug_indices)
    if not np.array_equal(unique_drugs, np.arange(unique_drugs[-1] + 1)):
        raise ValueError("drug identifiers must be contiguous from zero")

    candidate_order = np.lexsort(
        (candidate_indices, -core_scores[candidate_indices])
    )
    ordered_genes = candidate_indices[candidate_order]
    gene_ranks = np.zeros(core_scores.size, dtype=int)
    gene_ranks[ordered_genes] = np.arange(1, ordered_genes.size + 1)
    gene_weights = np.zeros(core_scores.size, dtype=float)
    gene_weights[ordered_genes] = (
        1.0 - gene_ranks[ordered_genes] / ordered_genes.size + 1e-6
    )

    n_drugs = unique_drugs.size
    scores = np.zeros(n_drugs, dtype=float)
    target_counts = np.zeros(n_drugs, dtype=int)
    best_ranks = np.zeros(n_drugs, dtype=int)
    for drug_index in range(n_drugs):
        rows = drug_indices == drug_index
        genes = gene_indices[rows]
        scores[drug_index] = np.sum(
            gene_weights[genes] * evidence_rows[rows, 2]
        )
        target_counts[drug_index] = np.unique(genes).size
        best_ranks[drug_index] = int(gene_ranks[genes].min())
    ranking_order = np.lexsort(
        (unique_drugs, best_ranks, -target_counts, -scores)
    )
    drug_ranks = np.empty(n_drugs, dtype=int)
    drug_ranks[ranking_order] = np.arange(1, n_drugs + 1)
    table = np.column_stack(
        [unique_drugs, scores, target_counts, best_ranks, drug_ranks]
    ).astype(float)
    return table[ranking_order]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
core = np.array([0.4, 0.7, 0.4, 0.0])
rows = np.array([[0, 0, 6.0], [0, 2, 0.8], [1, 1, 4.0], [1, 2, 3.0], [2, 0, 1.0], [2, 2, 2.0], [3, 1, 1.2], [3, 0, 0.5]])
""",
            "call": "aggregate_drug_evidence(core, rows).tolist()",
            "gold_call": "_oracle_aggregate_drug_evidence(core, rows).tolist()",
        },
        {
            "setup": """import numpy as np
core = np.array([1.0])
rows = np.array([[0, 0, 5.0]])
""",
            "call": "aggregate_drug_evidence(core, rows).tolist()",
            "gold_call": "_oracle_aggregate_drug_evidence(core, rows).tolist()",
        },
        {
            "setup": """import numpy as np
core = np.array([1.0, 0.0])
rows = np.array([[0, 1, 2.0]])
def run_model():
    try:
        aggregate_drug_evidence(core, rows)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_aggregate_drug_evidence(core, rows)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
