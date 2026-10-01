"""
Reconcile compressed rooted gene-tree topologies to a rooted species tree by minimum nearest-neighbor interchanges and tabulate candidate-branch participation.

A rooted NNI replaces a local topology ((X, Y), Z) by either ((X, Z), Y) or ((Y, Z), X). The three clades around every move form its reconciliation triple. Direction evidence is the number of triples on a shortest topology path that contain each candidate branch, rather than the total number of moves alone.

Returns
-------
numpy.ndarray of integers with shape (number of gene-tree patterns, 4)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def reconcile_rooted_nni(
    species_tree: tuple,
    gene_trees: list,
    candidate_clade_masks: np.ndarray,
) -> np.ndarray:
    '''Find shortest rooted-NNI reconciliations and branch appearances.

    Parameters
    ----------
    species_tree : tuple
        Rooted binary nested tuple with integer leaves.
    gene_trees : list
        Rooted binary nested tuples on the species-tree leaves.
    candidate_clade_masks : np.ndarray
        Two species-tree clade bitmasks ordered test then focal.

    Returns
    -------
    reconciliation_summary : np.ndarray
        Columns [minimum_nni_distance, number_of_shortest_paths, test_clade_appearances, focal_clade_appearances].

    Raises
    ------
    ValueError
        If the species tree is not a rooted binary tree with three to eight unique nonnegative integer leaves, a gene tree does not contain the same leaves exactly once, candidate masks do not identify two distinct non-root species-tree clades, or a gene tree cannot be reconciled by rooted NNI.
    '''
    return reconciliation_summary

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from collections import deque
import numpy as np


def _oracle_reconcile_rooted_nni(
    species_tree: tuple,
    gene_trees: list,
    candidate_clade_masks: np.ndarray,
) -> np.ndarray:
    """Reference implementation."""

    def validate_tree(node):
        if isinstance(node, (int, np.integer)):
            leaf = int(node)
            if leaf < 0:
                raise ValueError("tree leaves must be nonnegative integers")
            return [leaf]
        if not isinstance(node, tuple) or len(node) != 2:
            raise ValueError("each tree must be a rooted binary nested tuple")
        return validate_tree(node[0]) + validate_tree(node[1])

    def canonicalize(node):
        if isinstance(node, (int, np.integer)):
            return int(node)
        left = canonicalize(node[0])
        right = canonicalize(node[1])
        if repr(left) <= repr(right):
            return (left, right)
        return (right, left)

    def clade_mask(node):
        if isinstance(node, int):
            return 1 << node
        return clade_mask(node[0]) | clade_mask(node[1])

    def collect_clades(node, result):
        result.add(clade_mask(node))
        if isinstance(node, tuple):
            collect_clades(node[0], result)
            collect_clades(node[1], result)

    def replace_subtree(root, path, replacement):
        if not path:
            return replacement
        side = path[0]
        if side == 0:
            rebuilt = (replace_subtree(root[0], path[1:], replacement), root[1])
        else:
            rebuilt = (root[0], replace_subtree(root[1], path[1:], replacement))
        return canonicalize(rebuilt)

    def rooted_nni_neighbors(tree):
        neighbors = {}

        def visit(node, path):
            if isinstance(node, int):
                return
            left, right = node
            for inner, outside in ((left, right), (right, left)):
                if isinstance(inner, tuple):
                    first, second = inner
                    for retained, displaced in ((first, second), (second, first)):
                        local = canonicalize(((retained, outside), displaced))
                        neighbor = replace_subtree(tree, path, local)
                        event = tuple(
                            sorted(
                                (
                                    clade_mask(retained),
                                    clade_mask(displaced),
                                    clade_mask(outside),
                                )
                            )
                        )
                        neighbors.setdefault(neighbor, event)
            visit(left, path + (0,))
            visit(right, path + (1,))

        visit(tree, ())
        return sorted(neighbors.items(), key=lambda item: repr(item[0]))

    species_leaves = validate_tree(species_tree)
    if len(species_leaves) < 3 or len(species_leaves) > 8:
        raise ValueError("the species tree must contain between three and eight leaves")
    if len(set(species_leaves)) != len(species_leaves):
        raise ValueError("species-tree leaves must be unique")
    if not isinstance(gene_trees, (list, tuple)) or len(gene_trees) == 0:
        raise ValueError("at least one gene tree is required")

    masks = np.asarray(candidate_clade_masks)
    if masks.shape != (2,) or not np.issubdtype(masks.dtype, np.integer):
        raise ValueError("candidate_clade_masks must contain two integers")
    test_mask, focal_mask = map(int, masks)
    if test_mask <= 0 or focal_mask <= 0 or test_mask == focal_mask:
        raise ValueError("candidate clade masks must be distinct and positive")

    species = canonicalize(species_tree)
    species_clades = set()
    collect_clades(species, species_clades)
    full_mask = clade_mask(species)
    if (
        test_mask not in species_clades
        or focal_mask not in species_clades
        or test_mask == full_mask
        or focal_mask == full_mask
    ):
        raise ValueError("candidate masks must identify non-root species-tree clades")

    expected_leaves = sorted(species_leaves)
    starts = []
    for gene_tree in gene_trees:
        gene_leaves = validate_tree(gene_tree)
        if sorted(gene_leaves) != expected_leaves or len(set(gene_leaves)) != len(gene_leaves):
            raise ValueError("every gene tree must contain each species-tree leaf once")
        starts.append(canonicalize(gene_tree))

    queue = deque([species])
    distance = {species: 0}
    path_count = {species: 1}
    remaining_starts = set(starts)
    while queue:
        current = queue.popleft()
        remaining_starts.discard(current)
        if not remaining_starts:
            break
        for neighbor, _ in rooted_nni_neighbors(current):
            next_distance = distance[current] + 1
            if neighbor not in distance:
                distance[neighbor] = next_distance
                path_count[neighbor] = path_count[current]
                queue.append(neighbor)
            elif distance[neighbor] == next_distance:
                path_count[neighbor] += path_count[current]
    if remaining_starts:
        raise ValueError("a gene tree cannot be reconciled by rooted NNI")

    output_rows = []
    for start in starts:
        current = start
        selected_events = []
        while current != species:
            options = [
                (repr(neighbor), neighbor, event)
                for neighbor, event in rooted_nni_neighbors(current)
                if distance.get(neighbor) == distance[current] - 1
            ]
            if not options:
                raise ValueError("a shortest rooted-NNI path could not be reconstructed")
            _, current, event = min(options, key=lambda option: option[0])
            selected_events.append(event)
        test_appearances = sum(test_mask in event for event in selected_events)
        focal_appearances = sum(focal_mask in event for event in selected_events)
        output_rows.append(
            [
                distance[start],
                path_count[start],
                test_appearances,
                focal_appearances,
            ]
        )
    return np.asarray(output_rows, dtype=int)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": '''species_tree = (((((0, 1), 2), 3), 4), ((5, 6), 7))
gene_trees = [
    ((((0, 1), 2), (((5, 6), 7), 3)), 4),
    (((0, 1), 2), ((((5, 6), 7), 3), 4)),
    ((((0, 1), 2), ((5, 6), 7)), (3, 4)),
    ((((0, 1), 2), (3, 4)), ((5, 6), 7)),
    (((((0, 1), 2), 4), ((5, 6), 7)), 3),
]
candidate_masks = np.array([224, 7], dtype=int)
expected = [
    [2, 1, 2, 1],
    [3, 4, 2, 2],
    [2, 1, 1, 2],
    [1, 1, 0, 1],
    [2, 1, 1, 1],
]
def check(value):
    result = value.tolist()
    if result != expected:
        raise AssertionError("unexpected prompt-instance reconciliation")
    return result
''',
            "call": "check(reconcile_rooted_nni(species_tree, gene_trees, candidate_masks))",
            "gold_call": "check(_oracle_reconcile_rooted_nni(species_tree, gene_trees, candidate_masks))",
        },
        {
            "setup": '''species_tree = ((((0, 1), 2), 3), ((4, 5), 6))
gene_trees = [((((4, 6), (5, 0)), 2), (1, 3))]
candidate_masks = np.array([7, 112], dtype=int)
expected = [[6, 5, 0, 2]]
def check(value):
    result = value.tolist()
    if result != expected:
        raise AssertionError("unexpected unordered multiple-path reconciliation")
    return result
''',
            "call": "check(reconcile_rooted_nni(species_tree, gene_trees, candidate_masks))",
            "gold_call": "check(_oracle_reconcile_rooted_nni(species_tree, gene_trees, candidate_masks))",
        },
        {
            "setup": '''species_tree = (((0, 1), 2), (3, 4))
gene_trees = [((((0, 2), 1), 3), 4)]
candidate_masks = np.array([7, 24], dtype=int)
expected = [[2, 2, 1, 0]]
def check(value):
    result = value.tolist()
    if result != expected:
        raise AssertionError("unexpected multiple-path reconciliation")
    return result
''',
            "call": "check(reconcile_rooted_nni(species_tree, gene_trees, candidate_masks))",
            "gold_call": "check(_oracle_reconcile_rooted_nni(species_tree, gene_trees, candidate_masks))",
        },
    ]
