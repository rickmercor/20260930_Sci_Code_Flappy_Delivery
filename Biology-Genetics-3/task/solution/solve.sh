#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np  # noqa: E402, F811


def generate_polarization_updates(
    alternate_carriers: "np.ndarray",
    observed_haplotypes: "np.ndarray",
    ancestral_alternate_index: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation."""
    alternate_carriers = np.asarray(alternate_carriers)
    observed_haplotypes = np.asarray(observed_haplotypes)
    ancestral_alternate_index = np.asarray(ancestral_alternate_index)

    if alternate_carriers.ndim != 3:
        raise ValueError("alternate_carriers must be three dimensional")
    n_sites, n_alternates, n_samples = alternate_carriers.shape
    if n_sites < 1 or n_alternates < 2 or n_samples < 1:
        raise ValueError("at least one site, two alternates, and one sample are required")
    if observed_haplotypes.shape != (n_sites, n_samples):
        raise ValueError("observed_haplotypes has an incompatible shape")
    if ancestral_alternate_index.shape != (n_sites,):
        raise ValueError("ancestral_alternate_index has an incompatible shape")
    if not np.all((alternate_carriers == 0) | (alternate_carriers == 1)):
        raise ValueError("alternate_carriers must be binary")
    if not np.all((observed_haplotypes == 0) | (observed_haplotypes == 1)):
        raise ValueError("observed_haplotypes must be binary")
    if not np.all(np.isfinite(ancestral_alternate_index)):
        raise ValueError("ancestral_alternate_index must be finite")
    if not np.all(ancestral_alternate_index == np.floor(ancestral_alternate_index)):
        raise ValueError("ancestral_alternate_index must contain integers")

    alternate_carriers = alternate_carriers.astype(np.int64)
    observed_haplotypes = observed_haplotypes.astype(np.int64)
    ancestral_alternate_index = ancestral_alternate_index.astype(np.int64)
    if np.any(ancestral_alternate_index < 0) or np.any(ancestral_alternate_index >= n_alternates):
        raise ValueError("ancestral_alternate_index is out of range")
    if np.any(alternate_carriers > observed_haplotypes[:, None, :]):
        raise ValueError("alternate carriers must be observed")
    if np.any(alternate_carriers.sum(axis=1) > 1):
        raise ValueError("alternate carrier sets must be disjoint within a site")

    replacement_rows = []
    for site in range(n_sites):
        ancestral = int(ancestral_alternate_index[site])
        for alternate in range(n_alternates):
            if alternate != ancestral:
                replacement_rows.append(alternate_carriers[site, alternate].copy())
        occupied = alternate_carriers[site].max(axis=0)
        former_reference = observed_haplotypes[site] * (1 - occupied)
        replacement_rows.append(former_reference)
    return np.asarray(replacement_rows, dtype=np.int64)

import math  # noqa: E402

import numpy as np  # noqa: E402, F811


def encode_adaptive_reaches(
    children: list[list[int]], n_samples: int, dense_threshold: float
) -> "np.ndarray":
    """Reference implementation."""
    if not isinstance(n_samples, int) or isinstance(n_samples, bool) or not (1 <= n_samples <= 62):
        raise ValueError("n_samples must be an integer from 1 through 62")
    if not isinstance(dense_threshold, (int, float)) or isinstance(dense_threshold, bool):
        raise ValueError("dense_threshold must be real")
    dense_threshold = float(dense_threshold)
    if not math.isfinite(dense_threshold) or not (0.0 < dense_threshold <= 1.0):
        raise ValueError("dense_threshold must lie in (0, 1]")
    if not isinstance(children, (list, tuple)) or len(children) <= n_samples:
        raise ValueError("children must contain sample and internal nodes")

    reach_masks = np.zeros(len(children), dtype=np.int64)
    reach_sizes = np.zeros(len(children), dtype=np.int64)
    dense_flags = np.zeros(len(children), dtype=np.int64)
    for node, raw_children in enumerate(children):
        if not isinstance(raw_children, (list, tuple)):
            raise ValueError("every child row must be a sequence")
        if node < n_samples:
            if len(raw_children) != 0:
                raise ValueError("sample leaves cannot have children")
            reach_masks[node] = np.int64(1 << node)
        else:
            if len(raw_children) == 0:
                raise ValueError("internal nodes must have children")
            if any(not isinstance(child, int) or isinstance(child, bool) for child in raw_children):
                raise ValueError("child identifiers must be integers")
            if len(set(raw_children)) != len(raw_children):
                raise ValueError("child identifiers cannot repeat")
            if any(child < 0 or child >= node for child in raw_children):
                raise ValueError("children must precede their parent")
            mask = 0
            for child in raw_children:
                child_mask = int(reach_masks[child])
                if mask & child_mask:
                    raise ValueError("sibling descendant reaches must be disjoint")
                mask |= child_mask
            reach_masks[node] = np.int64(mask)
        reach_sizes[node] = int(int(reach_masks[node]).bit_count())
        dense_flags[node] = int(float(reach_sizes[node]) / n_samples >= dense_threshold)
    return np.column_stack((reach_masks, reach_sizes, dense_flags)).astype(np.int64)

import numpy as np  # noqa: E402, F811


def pack_batch_memberships(update_carriers: "np.ndarray", word_size: int) -> "np.ndarray":
    """Reference implementation."""
    update_carriers = np.asarray(update_carriers)
    if update_carriers.ndim != 2 or min(update_carriers.shape) < 1:
        raise ValueError("update_carriers must be a nonempty two-dimensional array")
    if not np.all((update_carriers == 0) | (update_carriers == 1)):
        raise ValueError("update_carriers must be binary")
    if not isinstance(word_size, int) or isinstance(word_size, bool) or not (1 <= word_size <= 62):
        raise ValueError("word_size must be an integer from 1 through 62")

    update_carriers = update_carriers.astype(np.int64)
    n_updates, n_samples = update_carriers.shape
    n_words = (n_updates + word_size - 1) // word_size
    packed = np.zeros((n_samples, n_words), dtype=np.int64)
    for update in range(n_updates):
        word = update // word_size
        bit = update % word_size
        packed[update_carriers[update] == 1, word] |= np.int64(1 << bit)
    return packed

import numpy as np  # noqa: E402, F811


def discover_shared_candidates(
    children: list[list[int]],
    n_samples: int,
    packed_leaf_memberships: "np.ndarray",
    word_size: int,
    n_updates: int,
) -> "np.ndarray":
    """Reference implementation."""
    if not isinstance(n_samples, int) or isinstance(n_samples, bool) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    if not isinstance(word_size, int) or isinstance(word_size, bool) or not (1 <= word_size <= 62):
        raise ValueError("word_size must be an integer from 1 through 62")
    if not isinstance(n_updates, int) or isinstance(n_updates, bool) or n_updates < 1:
        raise ValueError("n_updates must be a positive integer")
    if not isinstance(children, (list, tuple)) or len(children) <= n_samples:
        raise ValueError("children must contain sample and internal nodes")
    n_words = (n_updates + word_size - 1) // word_size
    packed_leaf_memberships = np.asarray(packed_leaf_memberships)
    if packed_leaf_memberships.shape != (n_samples, n_words):
        raise ValueError("packed_leaf_memberships has an incompatible shape")
    if not np.issubdtype(packed_leaf_memberships.dtype, np.integer):
        raise ValueError("packed_leaf_memberships must contain integers")
    if np.any(packed_leaf_memberships < 0):
        raise ValueError("packed_leaf_memberships cannot be negative")
    packed_leaf_memberships = packed_leaf_memberships.astype(np.int64)
    for word in range(n_words):
        used = min(word_size, n_updates - word * word_size)
        if np.any(packed_leaf_memberships[:, word] >> used):
            raise ValueError("packed_leaf_memberships contains out-of-range bits")

    states = np.zeros((len(children), n_words), dtype=np.int64)
    states[:n_samples] = packed_leaf_memberships
    for node, raw_children in enumerate(children):
        if not isinstance(raw_children, (list, tuple)):
            raise ValueError("every child row must be a sequence")
        if node < n_samples:
            if raw_children:
                raise ValueError("sample leaves cannot have children")
            continue
        if not raw_children:
            raise ValueError("internal nodes must have children")
        if any(not isinstance(child, int) or isinstance(child, bool) for child in raw_children):
            raise ValueError("child identifiers must be integers")
        if any(child < 0 or child >= node for child in raw_children):
            raise ValueError("children must precede their parent")
        if len(set(raw_children)) != len(raw_children):
            raise ValueError("child identifiers must be distinct within each row")
        state = states[raw_children[0]].copy()
        for child in raw_children[1:]:
            state &= states[child]
        states[node] = state
    return states

import numpy as np  # noqa: E402, F811


def plan_snapshot_attachments(
    children: list[list[int]],
    n_samples: int,
    reach_encoding: "np.ndarray",
    state_words: "np.ndarray",
    update_carriers: "np.ndarray",
    word_size: int,
) -> "np.ndarray":
    """Reference implementation."""
    if not isinstance(children, (list, tuple)) or not (1 <= len(children) <= 62):
        raise ValueError("children must describe from 1 through 62 snapshot nodes")
    if not isinstance(n_samples, int) or isinstance(n_samples, bool) or not (1 <= n_samples < len(children)):
        raise ValueError("n_samples is invalid")
    if not isinstance(word_size, int) or isinstance(word_size, bool) or not (1 <= word_size <= 62):
        raise ValueError("word_size must be an integer from 1 through 62")
    for node, raw_children in enumerate(children):
        if not isinstance(raw_children, (list, tuple)):
            raise ValueError("every child row must be a sequence")
        if node < n_samples and raw_children:
            raise ValueError("sample leaves cannot have children")
        if node >= n_samples and not raw_children:
            raise ValueError("internal nodes must have children")
        if any(not isinstance(child, int) or isinstance(child, bool) or child < 0 or child >= node for child in raw_children):
            raise ValueError("children must be integer identifiers preceding their parent")

    reach_encoding = np.asarray(reach_encoding)
    state_words = np.asarray(state_words)
    update_carriers = np.asarray(update_carriers)
    if reach_encoding.shape != (len(children), 3) or not np.issubdtype(reach_encoding.dtype, np.integer):
        raise ValueError("reach_encoding has an incompatible shape or type")
    if update_carriers.ndim != 2 or update_carriers.shape[1] != n_samples or update_carriers.shape[0] < 1:
        raise ValueError("update_carriers has an incompatible shape")
    if not np.all((update_carriers == 0) | (update_carriers == 1)):
        raise ValueError("update_carriers must be binary")
    n_updates = update_carriers.shape[0]
    n_words = (n_updates + word_size - 1) // word_size
    if state_words.shape != (len(children), n_words) or not np.issubdtype(state_words.dtype, np.integer):
        raise ValueError("state_words has an incompatible shape or type")
    if np.any(state_words < 0) or np.any(reach_encoding[:, :2] < 0):
        raise ValueError("reach and state encodings cannot be negative")
    if not np.all((reach_encoding[:, 2] == 0) | (reach_encoding[:, 2] == 1)):
        raise ValueError("adaptive storage flags must be binary")

    reach_encoding = reach_encoding.astype(np.int64)
    state_words = state_words.astype(np.int64)
    update_carriers = update_carriers.astype(np.int64)
    plans = np.full((n_updates, 2), (-1, 0), dtype=np.int64)
    for update in range(n_updates):
        target_mask = 0
        for sample in np.flatnonzero(update_carriers[update]):
            target_mask |= 1 << int(sample)
        word = update // word_size
        bit = update % word_size
        candidates = []
        for node in range(n_samples, len(children)):
            compatible = bool((int(state_words[node, word]) >> bit) & 1)
            reach_mask = int(reach_encoding[node, 0])
            if compatible != ((reach_mask & ~target_mask) == 0):
                raise ValueError("reach and compatibility data contradict update carriers")
            if compatible:
                candidates.append(node)

        if target_mask == 0:
            continue
        if target_mask.bit_count() == 1:
            plans[update, 0] = int(target_mask.bit_length() - 1)
            continue
        exact = [node for node in candidates if int(reach_encoding[node, 0]) == target_mask]
        if exact:
            plans[update, 0] = min(exact)
            continue

        covered = 0
        selected_nodes = 0
        ordered = sorted(candidates, key=lambda node: (-int(reach_encoding[node, 1]), node))
        for node in ordered:
            reach_mask = int(reach_encoding[node, 0])
            if covered & reach_mask == 0:
                covered |= reach_mask
                selected_nodes |= 1 << node
        uncovered = target_mask & ~covered
        selected_nodes |= uncovered
        plans[update, 1] = np.int64(selected_nodes)
    return plans

import numpy as np  # noqa: E402, F811


def apply_snapshot_batch(
    children: list[list[int]], n_samples: int, attachment_plans: "np.ndarray"
) -> "tuple[np.ndarray, np.ndarray]":
    """Reference implementation."""
    if not isinstance(children, (list, tuple)) or not (1 <= len(children) <= 62):
        raise ValueError("children must describe from 1 through 62 snapshot nodes")
    if not isinstance(n_samples, int) or isinstance(n_samples, bool) or not (1 <= n_samples < len(children)):
        raise ValueError("n_samples is invalid")
    normalized_children = []
    for node, raw_children in enumerate(children):
        if not isinstance(raw_children, (list, tuple)):
            raise ValueError("every child row must be a sequence")
        if node < n_samples and raw_children:
            raise ValueError("sample leaves cannot have children")
        if node >= n_samples and not raw_children:
            raise ValueError("internal nodes must have children")
        if any(not isinstance(child, int) or isinstance(child, bool) or child < 0 or child >= node for child in raw_children):
            raise ValueError("children must be integer identifiers preceding their parent")
        normalized_children.append(list(raw_children))

    attachment_plans = np.asarray(attachment_plans)
    if attachment_plans.ndim != 2 or attachment_plans.shape[0] < 1 or attachment_plans.shape[1] != 2:
        raise ValueError("attachment_plans must be a nonempty array with two columns")
    if not np.issubdtype(attachment_plans.dtype, np.integer):
        raise ValueError("attachment_plans must contain integers")
    attachment_plans = attachment_plans.astype(np.int64)
    snapshot_nodes = len(normalized_children)
    attachments = np.full(attachment_plans.shape[0], -1, dtype=np.int64)

    for update, (exact_raw, child_mask_raw) in enumerate(attachment_plans):
        exact = int(exact_raw)
        child_mask = int(child_mask_raw)
        if child_mask < 0:
            raise ValueError("child-node masks cannot be negative")
        if child_mask >> snapshot_nodes:
            raise ValueError("a child-node mask refers outside the discovery snapshot")
        if exact >= 0:
            if exact >= snapshot_nodes or child_mask != 0:
                raise ValueError("exact reuse must name one snapshot node and no children")
            attachments[update] = exact
            continue
        if exact != -1:
            raise ValueError("exact node identifiers must be -1 or a snapshot node")
        if child_mask == 0:
            continue
        child_ids = [node for node in range(snapshot_nodes) if (child_mask >> node) & 1]
        normalized_children.append(child_ids)
        attachments[update] = len(normalized_children) - 1

    adjacency = np.zeros((len(normalized_children), len(normalized_children)), dtype=np.int64)
    for parent, child_ids in enumerate(normalized_children):
        adjacency[parent, child_ids] = 1
    return adjacency, attachments

import numpy as np  # noqa: E402, F811


def audit_independent_discovery(
    snapshot_adjacency: "np.ndarray", n_samples: int, update_carriers: "np.ndarray"
) -> "np.ndarray":
    """Reference implementation."""
    snapshot_adjacency = np.asarray(snapshot_adjacency)
    if snapshot_adjacency.ndim != 2 or snapshot_adjacency.shape[0] < 1:
        raise ValueError("snapshot_adjacency must be a nonempty matrix")
    if snapshot_adjacency.shape[0] != snapshot_adjacency.shape[1] or snapshot_adjacency.shape[0] > 62:
        raise ValueError("snapshot_adjacency must be square with at most 62 nodes")
    if not np.all((snapshot_adjacency == 0) | (snapshot_adjacency == 1)):
        raise ValueError("snapshot_adjacency must be binary")
    n_nodes = snapshot_adjacency.shape[0]
    if not isinstance(n_samples, int) or isinstance(n_samples, bool) or not (1 <= n_samples < n_nodes):
        raise ValueError("n_samples is invalid")
    snapshot_adjacency = snapshot_adjacency.astype(np.int64)
    child_rows = []
    for node in range(n_nodes):
        child_ids = np.flatnonzero(snapshot_adjacency[node]).tolist()
        if node < n_samples and child_ids:
            raise ValueError("sample leaves cannot have children")
        if node >= n_samples and not child_ids:
            raise ValueError("internal nodes must have children")
        if any(child >= node for child in child_ids):
            raise ValueError("children must precede their parent")
        child_rows.append(child_ids)

    update_carriers = np.asarray(update_carriers)
    if update_carriers.ndim != 2 or update_carriers.shape[0] < 1 or update_carriers.shape[1] != n_samples:
        raise ValueError("update_carriers has an incompatible shape")
    if not np.all((update_carriers == 0) | (update_carriers == 1)):
        raise ValueError("update_carriers must be binary")
    update_carriers = update_carriers.astype(np.int64)

    audit = np.zeros((update_carriers.shape[0], 2), dtype=np.int64)
    for update, carrier_row in enumerate(update_carriers):
        compatible = np.zeros(n_nodes, dtype=bool)
        visited_mask = 0
        for sample in np.flatnonzero(carrier_row):
            sample = int(sample)
            compatible[sample] = True
            visited_mask |= 1 << sample
        candidate_count = 0
        for node in range(n_samples, n_nodes):
            child_ids = child_rows[node]
            if any(compatible[child] for child in child_ids):
                visited_mask |= 1 << node
                compatible[node] = all(compatible[child] for child in child_ids)
                if compatible[node]:
                    candidate_count += 1
        audit[update] = (np.int64(visited_mask), np.int64(candidate_count))
    return audit

import numpy as np  # noqa: E402, F811


def compute_batch_work_ratio(
    audit_counts: "np.ndarray", n_nodes: int, word_size: int
) -> float:
    """Reference implementation."""
    audit_counts = np.asarray(audit_counts)
    if audit_counts.ndim != 2 or audit_counts.shape[0] < 1 or audit_counts.shape[1] != 2:
        raise ValueError("audit_counts must be a nonempty array with two columns")
    if not np.issubdtype(audit_counts.dtype, np.integer):
        raise ValueError("audit_counts must contain integers")
    if not isinstance(n_nodes, int) or isinstance(n_nodes, bool) or not (1 <= n_nodes <= 62):
        raise ValueError("n_nodes must be an integer from 1 through 62")
    if not isinstance(word_size, int) or isinstance(word_size, bool) or not (1 <= word_size <= 62):
        raise ValueError("word_size must be an integer from 1 through 62")
    audit_counts = audit_counts.astype(np.int64)
    if np.any(audit_counts < 0):
        raise ValueError("visited masks and candidate counts cannot be negative")
    if np.any(audit_counts[:, 0] >> n_nodes):
        raise ValueError("a visited mask refers outside the graph")

    union_mask = 0
    independent_work = 0
    total_candidates = 0
    for visited_mask_raw, candidate_count_raw in audit_counts:
        visited_mask = int(visited_mask_raw)
        candidate_count = int(candidate_count_raw)
        union_mask |= visited_mask
        independent_work += visited_mask.bit_count() + candidate_count
        total_candidates += candidate_count
    if independent_work == 0:
        raise ValueError("independent work must be positive")
    n_updates = audit_counts.shape[0]
    packed_words = (n_updates + word_size - 1) // word_size
    batched_work = packed_words * union_mask.bit_count() + total_candidates
    return float(batched_work / independent_work)

import math  # noqa: E402

import numpy as np  # noqa: E402, F811


def run_full_pipeline(
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

    updates = generate_polarization_updates(  # noqa: F821
        alternate_carriers, observed_haplotypes, ancestral_alternate_index
    )
    if updates.shape[1] != n_samples:
        raise ValueError("polarization sample count does not match n_samples")
    reach_encoding = encode_adaptive_reaches(  # noqa: F821
        children, n_samples, dense_threshold
    )
    packed_memberships = pack_batch_memberships(  # noqa: F821
        updates, word_size
    )
    state_words = discover_shared_candidates(  # noqa: F821
        children, n_samples, packed_memberships, word_size, updates.shape[0]
    )
    attachment_plans = plan_snapshot_attachments(  # noqa: F821
        children, n_samples, reach_encoding, state_words, updates, word_size
    )
    snapshot_nodes = len(children)
    snapshot_adjacency = np.zeros((snapshot_nodes, snapshot_nodes), dtype=np.int64)
    for parent, child_ids in enumerate(children):
        snapshot_adjacency[parent, child_ids] = 1
    audit_counts = audit_independent_discovery(  # noqa: F821
        snapshot_adjacency, n_samples, updates
    )
    edited_adjacency, attachment_nodes = apply_snapshot_batch(  # noqa: F821
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
    return compute_batch_work_ratio(  # noqa: F821
        audit_counts, snapshot_nodes, word_size
    )
SCICODE_GOLD_EOF
