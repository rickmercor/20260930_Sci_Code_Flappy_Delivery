"""
Integrated target prioritization balances the genotype-derived discovery signal with biological and translational evidence. Each component is percentile-normalized among discovery genes by mapping the smallest average rank to zero and the largest to one, with tied values assigned their average rank. The published operating point is core_score_g = 0.45 DE_g + 0.25 Path_g + 0.25 Drug_g + 0.05 Hub_g; non-discovery genes remain zero.

Inputs

------

discovery_scores: Non-negative float array of shape (n_genes,), with zero marking non-discovery genes.

pathway_scores: Non-negative float array of shape (n_genes,).

druggability_scores: Non-negative float array of shape (n_genes,).

hub_scores: Non-negative float array of shape (n_genes,).

Returns

-------

integrated_scores: Float array of shape (n_genes, 5), with four percentile components and core_score.

Returns
-------
np.ndarray of shape (n_genes, 5), four candidate-only percentiles followed by core_score
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def integrate_target_evidence(
    discovery_scores: "np.ndarray",
    pathway_scores: "np.ndarray",
    druggability_scores: "np.ndarray",
    hub_scores: "np.ndarray",
) -> "np.ndarray":
    """Percentile-normalize evidence and compute integrated target scores.

    Parameters
    ----------
    discovery_scores : np.ndarray
        Non-negative discovery scores; positive entries define candidates.
    pathway_scores : np.ndarray
        Non-negative pathway support for the same genes.
    druggability_scores : np.ndarray
        Non-negative druggability evidence for the same genes.
    hub_scores : np.ndarray
        Non-negative network hub evidence for the same genes.

    Raises
    ------
    ValueError
        If inputs are not matching non-empty one-dimensional arrays, contain
        non-finite or negative values, or no positive discovery score exists.

    Returns
    -------
    integrated_scores : np.ndarray
        Columns DE_g, Path_g, Drug_g, Hub_g, and core_score_g, with zero rows
        for non-discovery genes.
    """
    return integrated_scores  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _average_rank_percentiles(values):
    """Map average ascending ranks to [0, 1], preserving ties."""
    count = values.size
    if count == 1:
        return np.ones(1, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(count, dtype=float)
    start = 0
    while start < count:
        stop = start + 1
        while stop < count and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks / (count - 1.0)


def _oracle_integrate_target_evidence(
    discovery_scores: "np.ndarray",
    pathway_scores: "np.ndarray",
    druggability_scores: "np.ndarray",
    hub_scores: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation."""
    arrays = [
        np.asarray(discovery_scores, dtype=float),
        np.asarray(pathway_scores, dtype=float),
        np.asarray(druggability_scores, dtype=float),
        np.asarray(hub_scores, dtype=float),
    ]
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("all evidence inputs must be one dimensional")
    if arrays[0].size == 0 or any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("all evidence inputs must be non-empty and have matching shapes")
    if any(
        not np.all(np.isfinite(array)) or np.any(array < 0.0)
        for array in arrays
    ):
        raise ValueError("all evidence values must be finite and non-negative")
    candidate = arrays[0] > 0.0
    if not np.any(candidate):
        raise ValueError("at least one positive discovery score is required")

    normalized = np.zeros((arrays[0].size, 4), dtype=float)
    for component_index, array in enumerate(arrays):
        normalized[candidate, component_index] = _average_rank_percentiles(
            array[candidate]
        )
    core_score = (
        0.45 * normalized[:, 0]
        + 0.25 * normalized[:, 1]
        + 0.25 * normalized[:, 2]
        + 0.05 * normalized[:, 3]
    )
    core_score[~candidate] = 0.0
    return np.column_stack([normalized, core_score])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
de = np.array([0.2, 0.8, 0.5, 0.0])
path = np.array([3.0, 1.0, 3.0, 9.0])
drug = np.array([0.1, 0.9, 0.5, 1.0])
hub = np.array([0.7, 0.2, 0.4, 0.9])
""",
            "call": "integrate_target_evidence(de, path, drug, hub).tolist()",
            "gold_call": "_oracle_integrate_target_evidence(de, path, drug, hub).tolist()",
        },
        {
            "setup": """import numpy as np
de = np.array([1.0])
path = np.array([0.0])
drug = np.array([0.0])
hub = np.array([0.0])
""",
            "call": "integrate_target_evidence(de, path, drug, hub).tolist()",
            "gold_call": "_oracle_integrate_target_evidence(de, path, drug, hub).tolist()",
        },
        {
            "setup": """import numpy as np
de = np.array([0.2, -0.1])
path = np.ones(2)
drug = np.ones(2)
hub = np.ones(2)
def run_model():
    try:
        integrate_target_evidence(de, path, drug, hub)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_integrate_target_evidence(de, path, drug, hub)
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
