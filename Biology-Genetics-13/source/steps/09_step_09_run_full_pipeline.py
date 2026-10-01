"""
The full calculation polarizes the first carrier batch, packs its leaf memberships, constructs adaptive node reaches, discovers snapshot candidates, plans and applies reuse-aware edits, traces an audit batch independently on the edited graph, and evaluates the audit overlap factor. The final scalar is rho(k) for those audit traversals.

Inputs

------

children: Ordered graph child lists.

n_samples: Number of sample leaves.

alternate_carriers: Binary alternate-carrier matrix for the edit batch.

observed_haplotypes: Binary observed-haplotype matrix for the edit batch.

ancestral_is_alternate: Binary polarization flags for the edit batch.

audit_carriers: Binary carrier matrix for independent post-edit traversals.

word_size: Number of update bits per packed word.

dense_threshold: Density threshold for dense reach storage.

Returns

-------

overlap_factor: float, the post-edit traversal overlap factor.

Returns
-------
float, the post-edit audit traversal overlap factor as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def run_full_pipeline(
    children: list,
    n_samples: int,
    alternate_carriers: np.ndarray,
    observed_haplotypes: np.ndarray,
    ancestral_is_alternate: np.ndarray,
    audit_carriers: np.ndarray,
    word_size: int = 32,
    dense_threshold: float = 0.03125,
) -> float:
    """Run batched editing and return the post-edit overlap factor.

    Parameters
    ----------
    children : list
        Ordered child lists for the initial graph.
    n_samples : int
        Number of sample leaves.
    alternate_carriers : np.ndarray
        Alternate-carrier matrix for the polarization batch.
    observed_haplotypes : np.ndarray
        Observed-haplotype matrix for the polarization batch.
    ancestral_is_alternate : np.ndarray
        Binary flags selecting carrier complementation.
    audit_carriers : np.ndarray
        Target carrier sets for post-edit independent traversals.
    word_size : int, optional
        Number of update bits per packed word.
    dense_threshold : float, optional
        Inclusive density threshold for dense reach storage.

    Raises
    ------
    ValueError
        If `dense_threshold` is outside [0, 1].

    Returns
    -------
    overlap_factor : float
        Audit traversal overlap factor rho(k).
    """
    return overlap_factor  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_run_full_pipeline(
    children: list,
    n_samples: int,
    alternate_carriers: np.ndarray,
    observed_haplotypes: np.ndarray,
    ancestral_is_alternate: np.ndarray,
    audit_carriers: np.ndarray,
    word_size: int = 32,
    dense_threshold: float = 0.03125,
) -> float:
    """Reference implementation chaining every earlier step."""
    targets = _oracle_polarize_carrier_sets(  # noqa: F821
        alternate_carriers, observed_haplotypes, ancestral_is_alternate
    )
    packed = _oracle_pack_leaf_memberships(targets, word_size=word_size)  # noqa: F821
    encoding = _oracle_encode_adaptive_descendants(  # noqa: F821
        children, n_samples, dense_threshold
    )
    discovery = _oracle_discover_shared_candidates(  # noqa: F821
        children,
        n_samples,
        packed,
        encoding["descendant_lists"],
        int(targets.shape[0]),
        word_size,
    )
    plan = _oracle_plan_greedy_attachments(  # noqa: F821
        discovery["candidate_lists"],
        encoding["descendant_lists"],
        targets,
        n_samples,
    )
    updated = _oracle_apply_snapshot_batch(children, plan, n_samples)  # noqa: F821
    traces = _oracle_trace_independent_visits(  # noqa: F821
        updated["children"], n_samples, audit_carriers
    )
    overlap = _oracle_compute_overlap_factor(  # noqa: F821
        traces["visit_matrix"], word_size=word_size
    )
    return float(overlap["overlap_factor"])

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np  # noqa: E402, F811


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
children = [[], [], [], [], [0, 1], [2, 3], [4, 5]]
n_samples = 4
alternate = np.array([[0, 0, 0, 1], [1, 1, 0, 0]], dtype=np.uint8)
observed = np.ones((2, 4), dtype=np.uint8)
ancestral = np.ones(2, dtype=np.uint8)
audit = np.array([[1, 1, 1, 0], [0, 1, 1, 1]], dtype=np.uint8)
""",
            "call": "float(run_full_pipeline(children, n_samples, alternate, observed, ancestral, audit, word_size=2, dense_threshold=0.5))",
            "gold_call": "float(_oracle_run_full_pipeline(children, n_samples, alternate, observed, ancestral, audit, word_size=2, dense_threshold=0.5))",
        },
        {
            "setup": """import numpy as np
children = [[], [], [0, 1]]
n_samples = 2
alternate = np.array([[0, 0]], dtype=np.uint8)
observed = np.ones((1, 2), dtype=np.uint8)
ancestral = np.ones(1, dtype=np.uint8)
audit = np.array([[1, 1]], dtype=np.uint8)
""",
            "call": "float(run_full_pipeline(children, n_samples, alternate, observed, ancestral, audit, word_size=1, dense_threshold=1.0))",
            "gold_call": "float(_oracle_run_full_pipeline(children, n_samples, alternate, observed, ancestral, audit, word_size=1, dense_threshold=1.0))",
        },
        {
            "setup": """import numpy as np
children = [[], [], [0, 1]]
n_samples = 2
alternate = np.array([[0, 0]], dtype=np.uint8)
observed = np.ones((1, 2), dtype=np.uint8)
ancestral = np.ones(1, dtype=np.uint8)
audit = np.array([[1, 1]], dtype=np.uint8)
def run_model():
    try:
        run_full_pipeline(children, n_samples, alternate, observed, ancestral, audit, dense_threshold=1.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_run_full_pipeline(children, n_samples, alternate, observed, ancestral, audit, dense_threshold=1.5)
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
