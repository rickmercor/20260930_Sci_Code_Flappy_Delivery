"""
Orchestrate the complete multiallelic GRG batch-cost benchmark.



The full calculation converts multiallelic polarization records into



replacement carrier sets, encodes the snapshot's reaches and packed mutation



state, discovers and applies fixed-snapshot attachments, verifies that every replacement



reaches exactly its carriers, audits independent



discovery on the unedited discovery snapshot, and evaluates the paper's



batched-to-independent work quotient. This orchestrator is the final Studio



step.

Returns
-------
float, the full-precision batched-to-independent candidate-discovery work ratio as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_full_pipeline(
    children: list[list[int]],
    n_samples: int,
    alternate_carriers: "np.ndarray",
    observed_haplotypes: "np.ndarray",
    ancestral_alternate_index: "np.ndarray",
    word_size: int,
    dense_threshold: float,
) -> float:
    '''Run multiallelic remapping and return the snapshot-discovery work ratio.

    Parameters
    ----------
    children : list[list[int]]
        Topologically ordered discovery-snapshot graph.
    n_samples : int
        Number of leading sample leaves.
    alternate_carriers : np.ndarray
        Binary original alternate carrier rows by site.
    observed_haplotypes : np.ndarray
        Binary observed-haplotype masks by site.
    ancestral_alternate_index : np.ndarray
        Ancestral alternate index for every site.
    word_size : int
        Number of usable bits per packed word, from 1 through 62.
    dense_threshold : float
        Adaptive dense-storage threshold in the interval (0, 1].

    Raises
    ------
    ValueError
        If the graph or n_samples is malformed, the polarization arrays have
        incompatible shapes or values, word_size is outside 1 through 62,
        dense_threshold is outside (0, 1], or the applied batch leaves a
        replacement mutation on a node whose descendant samples are not exactly
        its carriers.

    Returns
    -------
    work_ratio : float
        Batched candidate-discovery work divided by independent work, with
        both quantities evaluated on the read-only discovery snapshot.
    '''
    return work_ratio  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math  # noqa: E402

import numpy as np  # noqa: E402, F811


def _oracle_run_full_pipeline(
    children: list[list[int]],
    n_samples: int,
    alternate_carriers: "np.ndarray",
    observed_haplotypes: "np.ndarray",
    ancestral_alternate_index: "np.ndarray",
    word_size: int,
    dense_threshold: float,
) -> float:
    """Reference implementation chaining every earlier step."""
    if not isinstance(children, (list, tuple)) or len(children) < 2:
        raise ValueError("children must describe a nonempty graph")
    if not isinstance(n_samples, int) or isinstance(n_samples, bool) or not (1 <= n_samples < len(children)):
        raise ValueError("n_samples is invalid")
    if not isinstance(word_size, int) or isinstance(word_size, bool) or not (1 <= word_size <= 62):
        raise ValueError("word_size must be an integer from 1 through 62")
    if not isinstance(dense_threshold, (int, float)) or isinstance(dense_threshold, bool):
        raise ValueError("dense_threshold must be real")
    if not math.isfinite(float(dense_threshold)) or not (0.0 < float(dense_threshold) <= 1.0):
        raise ValueError("dense_threshold must lie in (0, 1]")
    alternate_carriers = np.asarray(alternate_carriers)
    observed_haplotypes = np.asarray(observed_haplotypes)
    ancestral_alternate_index = np.asarray(ancestral_alternate_index)
    if alternate_carriers.ndim != 3 or observed_haplotypes.ndim != 2 or ancestral_alternate_index.ndim != 1:
        raise ValueError("polarization arrays have incompatible dimensions")

    updates = _oracle_generate_polarization_updates(  # noqa: F821
        alternate_carriers, observed_haplotypes, ancestral_alternate_index
    )
    if updates.shape[1] != n_samples:
        raise ValueError("polarization sample count does not match n_samples")
    reach_encoding = _oracle_encode_adaptive_reaches(  # noqa: F821
        children, n_samples, dense_threshold
    )
    packed_memberships = _oracle_pack_batch_memberships(  # noqa: F821
        updates, word_size
    )
    state_words = _oracle_discover_shared_candidates(  # noqa: F821
        children, n_samples, packed_memberships, word_size, updates.shape[0]
    )
    attachment_plans = _oracle_plan_snapshot_attachments(  # noqa: F821
        children, n_samples, reach_encoding, state_words, updates, word_size
    )
    snapshot_nodes = len(children)
    snapshot_adjacency = np.zeros((snapshot_nodes, snapshot_nodes), dtype=np.int64)
    for parent, child_ids in enumerate(children):
        snapshot_adjacency[parent, child_ids] = 1
    audit_counts = _oracle_audit_independent_discovery(  # noqa: F821
        snapshot_adjacency, n_samples, updates
    )
    edited_adjacency, attachment_nodes = _oracle_apply_snapshot_batch(  # noqa: F821
        children, n_samples, attachment_plans
    )
    edited_reach = []
    for node in range(edited_adjacency.shape[0]):
        mask = 1 << node if node < n_samples else 0
        for child in np.flatnonzero(edited_adjacency[node]):
            mask |= edited_reach[int(child)]
        edited_reach.append(mask)
    for carriers, node in zip(updates, attachment_nodes):
        target = sum(1 << sample for sample in range(n_samples) if carriers[sample])
        reached = 0 if int(node) < 0 else edited_reach[int(node)]
        if reached != target:
            raise ValueError("the applied batch does not preserve exact carrier reachability")
    return _oracle_compute_batch_work_ratio(  # noqa: F821
        audit_counts, snapshot_nodes, word_size
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
children = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [0,1], [2,3], [4,5], [6,7], [8,9], [10,11], [12,13], [14,15], [16,17], [18,19], [15,16], [17,18], [19,20], [21,22], [22,23], [24,25]]
n_samples = 14
alternate_carriers = np.array([
    [[0,0,0,0,0,0,1,1,0,0,1,1,0,0], [0,0,0,0,0,0,0,0,1,1,0,0,0,0]],
    [[0,0,1,1,1,1,0,0,0,0,0,0,0,0], [1,1,0,0,0,0,0,0,0,0,0,0,0,0]],
    [[0,0,0,0,0,0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,1]],
    [[0,0,0,0,0,0,1,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,1,1,1,1,1,1,1]],
    [[0,0,0,0,0,0,0,0,0,0,0,0,0,1], [0,0,1,1,1,1,1,1,1,1,1,1,0,0]],
], dtype=int)
observed_haplotypes = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,0,0],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
], dtype=int)
ancestral_alternate_index = np.array([0,1,0,1,0], dtype=int)
word_size = 4
dense_threshold = 0.25
""",
            "call": "run_full_pipeline(children, n_samples, alternate_carriers, observed_haplotypes, ancestral_alternate_index, word_size, dense_threshold)",
            "gold_call": "_oracle_run_full_pipeline(children, n_samples, alternate_carriers, observed_haplotypes, ancestral_alternate_index, word_size, dense_threshold)",
        },
        {
            "setup": """import numpy as np
children = [[], [], [], [0,1], [3,2]]
n_samples = 3
alternate_carriers = np.array([[[1,0,0], [0,1,0]]], dtype=int)
observed_haplotypes = np.ones((1,3), dtype=int)
ancestral_alternate_index = np.array([0], dtype=int)
word_size = 2
dense_threshold = 0.5
""",
            "call": "run_full_pipeline(children, n_samples, alternate_carriers, observed_haplotypes, ancestral_alternate_index, word_size, dense_threshold)",
            "gold_call": "_oracle_run_full_pipeline(children, n_samples, alternate_carriers, observed_haplotypes, ancestral_alternate_index, word_size, dense_threshold)",
        },
        {
            "setup": """import numpy as np
children = [[], [], [0,1]]
n_samples = 2
alternate_carriers = np.array([[[1,0], [0,1]]], dtype=int)
observed_haplotypes = np.ones((1,2), dtype=int)
ancestral_alternate_index = np.array([0], dtype=int)
word_size = 0
dense_threshold = 0.5
def run_model():
    try:
        run_full_pipeline(children, n_samples, alternate_carriers, observed_haplotypes, ancestral_alternate_index, word_size, dense_threshold)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_run_full_pipeline(children, n_samples, alternate_carriers, observed_haplotypes, ancestral_alternate_index, word_size, dense_threshold)
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
