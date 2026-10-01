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


def polarize_carrier_sets(
    alternate_carriers: np.ndarray,
    observed_haplotypes: np.ndarray,
    ancestral_is_alternate: np.ndarray,
) -> np.ndarray:
    """Reference implementation."""
    alternate = np.asarray(alternate_carriers)
    observed = np.asarray(observed_haplotypes)
    ancestral = np.asarray(ancestral_is_alternate)
    if alternate.ndim != 2 or observed.shape != alternate.shape:
        raise ValueError("carrier matrices must be two dimensional with matching shapes")
    if ancestral.shape != (alternate.shape[0],):
        raise ValueError("ancestral_is_alternate must have one entry per site")
    if not np.all((alternate == 0) | (alternate == 1)):
        raise ValueError("alternate_carriers must be binary")
    if not np.all((observed == 0) | (observed == 1)):
        raise ValueError("observed_haplotypes must be binary")
    if not np.all((ancestral == 0) | (ancestral == 1)):
        raise ValueError("ancestral_is_alternate must be binary")
    if np.any(alternate > observed):
        raise ValueError("an alternate carrier must be observed")

    complemented = observed.astype(np.uint8) - alternate.astype(np.uint8)
    return np.where(ancestral[:, None] == 1, complemented, alternate).astype(np.uint8)

import numpy as np  # noqa: E402, F811


def pack_leaf_memberships(carrier_matrix: np.ndarray, word_size: int = 32) -> np.ndarray:
    """Reference implementation."""
    carriers = np.asarray(carrier_matrix)
    if carriers.ndim != 2 or carriers.shape[0] == 0 or carriers.shape[1] == 0:
        raise ValueError("carrier_matrix must be a nonempty two-dimensional array")
    if not np.all((carriers == 0) | (carriers == 1)):
        raise ValueError("carrier_matrix must be binary")
    if not isinstance(word_size, (int, np.integer)) or not 1 <= int(word_size) <= 62:
        raise ValueError("word_size must be an integer in [1, 62]")

    word_size = int(word_size)
    n_updates, n_samples = carriers.shape
    n_words = (n_updates + word_size - 1) // word_size
    packed = np.zeros((n_samples, n_words), dtype=np.int64)
    for update in range(n_updates):
        word = update // word_size
        offset = update % word_size
        packed[:, word] |= carriers[update].astype(np.int64) << offset
    return packed

def encode_adaptive_descendants(children: list, n_samples: int, dense_threshold: float) -> dict:
    """Reference implementation."""
    if not isinstance(children, list) or len(children) == 0:
        raise ValueError("children must be a nonempty list")
    if not isinstance(n_samples, int) or not 1 <= n_samples <= len(children):
        raise ValueError("n_samples must identify a nonempty leaf prefix")
    if not isinstance(dense_threshold, (int, float)) or not 0.0 <= float(dense_threshold) <= 1.0:
        raise ValueError("dense_threshold must be in [0, 1]")
    if any(list(children[s]) for s in range(n_samples)):
        raise ValueError("sample leaves must have no children")

    descendants = []
    dense_flags = []
    storage = []
    cardinalities = []
    for node, raw_children in enumerate(children):
        if not isinstance(raw_children, (list, tuple)):
            raise ValueError("each children entry must be a list or tuple")
        node_children = [int(child) for child in raw_children]
        if len(node_children) != len(set(node_children)):
            raise ValueError("a node cannot repeat a child")
        if any(child < 0 or child >= node for child in node_children):
            raise ValueError("every child identifier must precede its parent")
        if node < n_samples:
            reach = [node]
        else:
            if not node_children:
                raise ValueError("an internal node must have at least one child")
            seen = set()
            for child in node_children:
                child_reach = set(descendants[child])
                if seen.intersection(child_reach):
                    raise ValueError("child reaches must be disjoint")
                seen.update(child_reach)
            reach = sorted(seen)

        is_dense = int((len(reach) / n_samples) >= float(dense_threshold))
        if is_dense:
            n_words = (n_samples + 63) // 64
            payload = [0] * n_words
            for sample in reach:
                payload[sample // 64] |= 1 << (sample % 64)
        else:
            payload = list(reach)
        descendants.append(reach)
        dense_flags.append(is_dense)
        storage.append(payload)
        cardinalities.append(len(reach))

    return {
        "descendant_lists": descendants,
        "dense_flags": dense_flags,
        "storage": storage,
        "cardinalities": cardinalities,
    }

import numpy as np  # noqa: E402, F811


def discover_shared_candidates(
    children: list,
    n_samples: int,
    packed_leaf_memberships: np.ndarray,
    descendant_lists: list,
    n_updates: int,
    word_size: int = 32,
) -> dict:
    """Reference implementation."""
    if not isinstance(children, list) or len(children) == 0:
        raise ValueError("children must be a nonempty list")
    if not isinstance(n_samples, int) or not 1 <= n_samples <= len(children):
        raise ValueError("n_samples must identify the leaf prefix")
    if not isinstance(n_updates, int) or n_updates < 1:
        raise ValueError("n_updates must be positive")
    if not isinstance(word_size, int) or not 1 <= word_size <= 62:
        raise ValueError("word_size must be in [1, 62]")
    if len(descendant_lists) != len(children):
        raise ValueError("descendant_lists must have one entry per node")
    for node, node_children in enumerate(children):
        if node < n_samples and list(node_children):
            raise ValueError("sample leaves must have no children")
        if node >= n_samples and not list(node_children):
            raise ValueError("internal nodes must have children")
        if any(not isinstance(child, int) or child < 0 or child >= node for child in node_children):
            raise ValueError("every child must precede its parent")

    packed = np.asarray(packed_leaf_memberships)
    n_words = (n_updates + word_size - 1) // word_size
    if packed.shape != (n_samples, n_words):
        raise ValueError("packed_leaf_memberships has the wrong shape")
    if not np.issubdtype(packed.dtype, np.integer) or np.any(packed < 0):
        raise ValueError("packed memberships must be nonnegative integers")
    valid_masks = []
    for word in range(n_words):
        remaining = n_updates - word * word_size
        valid_masks.append((1 << min(word_size, remaining)) - 1)
    for word, mask in enumerate(valid_masks):
        if np.any(np.bitwise_and(packed[:, word].astype(np.int64), ~mask) != 0):
            raise ValueError("packed memberships set a bit outside the batch")

    parents = [[] for _ in children]
    for parent, node_children in enumerate(children):
        for child in node_children:
            parents[child].append(parent)
    for node_parents in parents:
        node_parents.sort()

    states = [[0] * n_words for _ in children]
    active = [False] * len(children)
    for sample in range(n_samples):
        states[sample] = [int(value) for value in packed[sample]]
        active[sample] = any(states[sample])

    candidates = [[] for _ in range(n_updates)]
    candidate_sizes = [[] for _ in range(n_updates)]
    visited = []
    for node in range(len(children)):
        if not active[node]:
            continue
        visited.append(node)
        if node >= n_samples:
            state = list(valid_masks)
            for child in children[node]:
                state = [left & right for left, right in zip(state, states[child])]
            states[node] = state
            if any(state):
                for update in range(n_updates):
                    word = update // word_size
                    offset = update % word_size
                    if state[word] & (1 << offset):
                        candidates[update].append(node)
                        candidate_sizes[update].append(len(descendant_lists[node]))
        if any(states[node]):
            for parent in parents[node]:
                active[parent] = True

    return {
        "node_states": states,
        "candidate_lists": candidates,
        "candidate_sizes": candidate_sizes,
        "visited_nodes": visited,
    }

import numpy as np  # noqa: E402, F811


def plan_greedy_attachments(
    candidate_lists: list,
    descendant_lists: list,
    target_carriers: np.ndarray,
    n_samples: int,
) -> dict:
    """Reference implementation."""
    targets = np.asarray(target_carriers)
    if targets.ndim != 2 or targets.shape[1] != n_samples or targets.shape[0] == 0:
        raise ValueError("target_carriers has the wrong shape")
    if not np.all((targets == 0) | (targets == 1)):
        raise ValueError("target_carriers must be binary")
    if len(candidate_lists) != targets.shape[0]:
        raise ValueError("candidate_lists must have one entry per update")
    if not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError("n_samples must be positive")

    exact_nodes = []
    selected_all = []
    uncovered_all = []
    target_lists = []
    creates_node = []
    for update, raw_candidates in enumerate(candidate_lists):
        target = set(np.flatnonzero(targets[update]).astype(int).tolist())
        if not target:
            raise ValueError("target carrier sets must be nonempty")
        candidates = sorted(set(int(node) for node in raw_candidates))
        if any(node < n_samples or node >= len(descendant_lists) for node in candidates):
            raise ValueError("candidate identifiers must name snapshot internal nodes")
        reaches = {node: set(int(sample) for sample in descendant_lists[node]) for node in candidates}
        if any(not reach.issubset(target) for reach in reaches.values()):
            raise ValueError("every candidate reach must be contained in its target")

        exact = [node for node in candidates if reaches[node] == target]
        target_lists.append(sorted(target))
        if exact:
            exact_nodes.append(min(exact))
            selected_all.append([])
            uncovered_all.append([])
            creates_node.append(0)
            continue

        covered = set()
        selected = []
        for node in sorted(candidates, key=lambda value: (-len(reaches[value]), value)):
            if covered.isdisjoint(reaches[node]):
                selected.append(node)
                covered.update(reaches[node])
        exact_nodes.append(-1)
        selected_all.append(selected)
        uncovered_all.append(sorted(target.difference(covered)))
        creates_node.append(1)

    return {
        "exact_nodes": exact_nodes,
        "selected_candidates": selected_all,
        "uncovered_samples": uncovered_all,
        "target_lists": target_lists,
        "creates_node": creates_node,
    }

def apply_snapshot_batch(children: list, attachment_plan: dict, n_samples: int) -> dict:
    """Reference implementation."""
    if not isinstance(children, list) or len(children) == 0:
        raise ValueError("children must be a nonempty list")
    if not isinstance(n_samples, int) or not 1 <= n_samples <= len(children):
        raise ValueError("n_samples must identify the leaf prefix")
    required = {"exact_nodes", "selected_candidates", "uncovered_samples", "creates_node"}
    if not isinstance(attachment_plan, dict) or not required.issubset(attachment_plan):
        raise ValueError("attachment_plan is missing required fields")
    lengths = [len(attachment_plan[key]) for key in required]
    if len(set(lengths)) != 1:
        raise ValueError("attachment plan fields must have matching lengths")

    snapshot_size = len(children)
    updated = [list(map(int, node_children)) for node_children in children]
    attachment_nodes = []
    new_node_ids = []
    for update in range(lengths[0]):
        exact = int(attachment_plan["exact_nodes"][update])
        selected = [int(node) for node in attachment_plan["selected_candidates"][update]]
        uncovered = [int(sample) for sample in attachment_plan["uncovered_samples"][update]]
        creates = int(attachment_plan["creates_node"][update])
        if creates not in (0, 1):
            raise ValueError("creates_node entries must be binary")
        if creates == 0:
            if exact < n_samples or exact >= snapshot_size or selected or uncovered:
                raise ValueError("an exact plan must reuse one snapshot internal node")
            attachment_nodes.append(exact)
            new_node_ids.append(-1)
            continue
        if exact != -1:
            raise ValueError("a new-node plan cannot also specify an exact node")
        if any(node < n_samples or node >= snapshot_size for node in selected):
            raise ValueError("selected candidates must come from the fixed snapshot")
        if any(sample < 0 or sample >= n_samples for sample in uncovered):
            raise ValueError("uncovered sample identifiers are invalid")
        node_children = selected + uncovered
        if not node_children or len(node_children) != len(set(node_children)):
            raise ValueError("a new node needs distinct snapshot children")
        new_node = len(updated)
        updated.append(node_children)
        attachment_nodes.append(new_node)
        new_node_ids.append(new_node)

    return {
        "children": updated,
        "attachment_nodes": attachment_nodes,
        "new_node_ids": new_node_ids,
        "edge_count": sum(len(node_children) for node_children in updated),
    }

import numpy as np  # noqa: E402, F811


def trace_independent_visits(children: list, n_samples: int, carrier_matrix: np.ndarray) -> dict:
    """Reference implementation."""
    if not isinstance(children, list) or len(children) == 0:
        raise ValueError("children must be a nonempty list")
    if not isinstance(n_samples, int) or not 1 <= n_samples <= len(children):
        raise ValueError("n_samples must identify the leaf prefix")
    for node, node_children in enumerate(children):
        if node < n_samples and list(node_children):
            raise ValueError("sample leaves must have no children")
        if node >= n_samples and not list(node_children):
            raise ValueError("internal nodes must have children")
        if any(not isinstance(child, int) or child < 0 or child >= node for child in node_children):
            raise ValueError("every child must precede its parent")

    carriers = np.asarray(carrier_matrix)
    if carriers.ndim != 2 or carriers.shape[1] != n_samples or carriers.shape[0] == 0:
        raise ValueError("carrier_matrix has the wrong shape")
    if not np.all((carriers == 0) | (carriers == 1)):
        raise ValueError("carrier_matrix must be binary")
    if np.any(np.sum(carriers, axis=1) == 0):
        raise ValueError("audit carrier sets must be nonempty")

    parents = [[] for _ in children]
    for parent, node_children in enumerate(children):
        for child in node_children:
            parents[child].append(parent)
    for node_parents in parents:
        node_parents.sort()

    visit_matrix = np.zeros((carriers.shape[0], len(children)), dtype=np.uint8)
    candidate_lists = []
    for update in range(carriers.shape[0]):
        compatible = [False] * len(children)
        active = [False] * len(children)
        for sample in range(n_samples):
            compatible[sample] = bool(carriers[update, sample])
            active[sample] = compatible[sample]
        candidates = []
        for node in range(len(children)):
            if not active[node]:
                continue
            visit_matrix[update, node] = 1
            if node >= n_samples:
                compatible[node] = all(compatible[child] for child in children[node])
                if compatible[node]:
                    candidates.append(node)
            if compatible[node]:
                for parent in parents[node]:
                    active[parent] = True
        candidate_lists.append(candidates)

    return {
        "visit_matrix": visit_matrix,
        "visited_counts": np.sum(visit_matrix, axis=1).astype(int).tolist(),
        "candidate_lists": candidate_lists,
    }

import numpy as np  # noqa: E402, F811


def compute_overlap_factor(visit_matrix: np.ndarray, word_size: int = 32) -> dict:
    """Reference implementation."""
    visits = np.asarray(visit_matrix)
    if visits.ndim != 2 or visits.shape[0] == 0 or visits.shape[1] == 0:
        raise ValueError("visit_matrix must be nonempty and two dimensional")
    if not np.all((visits == 0) | (visits == 1)):
        raise ValueError("visit_matrix must be binary")
    if np.any(np.sum(visits, axis=1) == 0):
        raise ValueError("each independent traversal must visit at least one node")
    if not isinstance(word_size, (int, np.integer)) or not 1 <= int(word_size) <= 62:
        raise ValueError("word_size must be an integer in [1, 62]")

    independent_visits = int(np.sum(visits))
    distinct_visits = int(np.count_nonzero(np.any(visits == 1, axis=0)))
    word_count = (visits.shape[0] + int(word_size) - 1) // int(word_size)
    overlap_factor = float(independent_visits / distinct_visits)
    return {
        "independent_visits": independent_visits,
        "distinct_visits": distinct_visits,
        "word_count": int(word_count),
        "overlap_factor": overlap_factor,
        "shared_advantage": int(overlap_factor > word_count),
    }

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
    """Reference implementation chaining every earlier step."""
    targets = polarize_carrier_sets(  # noqa: F821
        alternate_carriers, observed_haplotypes, ancestral_is_alternate
    )
    packed = pack_leaf_memberships(targets, word_size=word_size)  # noqa: F821
    encoding = encode_adaptive_descendants(  # noqa: F821
        children, n_samples, dense_threshold
    )
    discovery = discover_shared_candidates(  # noqa: F821
        children,
        n_samples,
        packed,
        encoding["descendant_lists"],
        int(targets.shape[0]),
        word_size,
    )
    plan = plan_greedy_attachments(  # noqa: F821
        discovery["candidate_lists"],
        encoding["descendant_lists"],
        targets,
        n_samples,
    )
    updated = apply_snapshot_batch(children, plan, n_samples)  # noqa: F821
    traces = trace_independent_visits(  # noqa: F821
        updated["children"], n_samples, audit_carriers
    )
    overlap = compute_overlap_factor(  # noqa: F821
        traces["visit_matrix"], word_size=word_size
    )
    return float(overlap["overlap_factor"])
SCICODE_GOLD_EOF
