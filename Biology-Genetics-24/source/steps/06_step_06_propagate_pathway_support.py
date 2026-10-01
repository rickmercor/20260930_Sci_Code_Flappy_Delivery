"""
Pathway evidence is propagated from recurrent enrichment rather than from one foreground cutoff. For term t, significant scans have FDR below the threshold, robustness(t) is their count, strength(t) = -log10 of their minimum FDR, and weight(t) = strength(t) * robustness(t). A gene's PathwayScore is the sum of the weights for significant terms containing that gene; unsupported terms and genes contribute zero.

Inputs

------

term_fdr: Float array of shape (n_terms, n_scans).

term_gene_membership: Binary array of shape (n_terms, n_genes).

fdr_threshold: Threshold in (0, 1).

Returns

-------

term_summary: Float array of shape (n_terms, 3), containing robustness, strength, and weight.

pathway_scores: Float array of shape (n_genes,).

Returns
-------
tuple containing a float64 (n_terms, 3) term summary and a float64 (n_genes,) pathway score
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def propagate_pathway_support(
    term_fdr: "np.ndarray",
    term_gene_membership: "np.ndarray",
    fdr_threshold: float,
) -> "tuple[np.ndarray, np.ndarray]":
    """Propagate recurrent pathway-enrichment evidence to genes.

    Parameters
    ----------
    term_fdr : np.ndarray
        Adjusted pathway values with shape (n_terms, n_scans).
    term_gene_membership : np.ndarray
        Binary term-gene membership matrix with shape (n_terms, n_genes).
    fdr_threshold : float
        Finite strict significance threshold in (0, 1).

    Raises
    ------
    ValueError
        If arrays are not non-empty two-dimensional arrays with the same term
        count, FDR values are non-finite or outside [0, 1], membership is not
        binary, or `fdr_threshold` is not finite and in (0, 1).

    Returns
    -------
    term_summary : np.ndarray
        Per-term robustness, strength, and weight columns.
    pathway_scores : np.ndarray
        Sum of recurrent significant term weights for each gene.
    """
    return pathway_results  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811

def _oracle_propagate_pathway_support(
    term_fdr: "np.ndarray",
    term_gene_membership: "np.ndarray",
    fdr_threshold: float,
) -> "tuple[np.ndarray, np.ndarray]":
    """Reference implementation."""
    term_fdr = np.asarray(term_fdr, dtype=float)
    membership = np.asarray(term_gene_membership)
    if term_fdr.ndim != 2 or membership.ndim != 2:
        raise ValueError("term_fdr and term_gene_membership must be two dimensional")
    if any(size == 0 for size in term_fdr.shape) or any(
        size == 0 for size in membership.shape
    ):
        raise ValueError("pathway arrays must be non-empty")
    if term_fdr.shape[0] != membership.shape[0]:
        raise ValueError("pathway arrays must have the same term count")
    if not np.all(np.isfinite(term_fdr)) or np.any(
        (term_fdr < 0.0) | (term_fdr > 1.0)
    ):
        raise ValueError("term_fdr must be finite and lie in [0, 1]")
    if not np.all((membership == 0) | (membership == 1)):
        raise ValueError("term_gene_membership must be binary")
    if not (
        isinstance(fdr_threshold, (int, float, np.integer, np.floating))
        and np.isfinite(fdr_threshold)
        and 0.0 < float(fdr_threshold) < 1.0
    ):
        raise ValueError("fdr_threshold must lie in (0, 1)")

    significant = term_fdr < float(fdr_threshold)
    robustness = significant.sum(axis=1).astype(float)
    strength = np.zeros(term_fdr.shape[0], dtype=float)
    for term_index in range(term_fdr.shape[0]):
        if robustness[term_index] > 0.0:
            minimum = max(
                float(term_fdr[term_index][significant[term_index]].min()),
                np.finfo(float).tiny,
            )
            strength[term_index] = -np.log10(minimum)
    weights = robustness * strength
    term_summary = np.column_stack([robustness, strength, weights])
    pathway_scores = membership.astype(float).T @ weights
    return term_summary, pathway_scores

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
term_fdr = np.array([[0.01, 0.04, 0.2], [0.2, 0.3, 0.4]])
membership = np.array([[1, 0, 1], [0, 1, 1]])

def _pack_pathway_result(result):
    packed = []
    for value in result:
        array = np.asarray(value, dtype=float)
        packed.extend([array.ndim, *array.shape])
        packed.extend(array.ravel().tolist())
    return np.asarray(packed, dtype=float)
""",
            "call": "_pack_pathway_result(propagate_pathway_support(term_fdr, membership, 0.1))",
            "gold_call": "_pack_pathway_result(_oracle_propagate_pathway_support(term_fdr, membership, 0.1))",
        },
        {
            "setup": """import numpy as np
term_fdr = np.array([[1.0]])
membership = np.array([[1]])

def _pack_pathway_result(result):
    packed = []
    for value in result:
        array = np.asarray(value, dtype=float)
        packed.extend([array.ndim, *array.shape])
        packed.extend(array.ravel().tolist())
    return np.asarray(packed, dtype=float)
""",
            "call": "_pack_pathway_result(propagate_pathway_support(term_fdr, membership, 0.1))",
            "gold_call": "_pack_pathway_result(_oracle_propagate_pathway_support(term_fdr, membership, 0.1))",
        },
        {
            "setup": """import numpy as np
term_fdr = np.array([[0.01, 0.2]])
membership = np.array([[1, 2]])
def run_model():
    try:
        propagate_pathway_support(term_fdr, membership, 0.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_propagate_pathway_support(term_fdr, membership, 0.1)
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
