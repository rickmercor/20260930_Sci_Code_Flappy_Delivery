#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np


def summarize_candidate_evidence(
    attachment_codes_by_candidate: list,
    branch_presence_by_candidate: list,
) -> np.ndarray:
    """Reference implementation."""
    if not isinstance(attachment_codes_by_candidate, (list, tuple)):
        raise ValueError("attachment codes must be a sequence")
    if not isinstance(branch_presence_by_candidate, (list, tuple)):
        raise ValueError("branch presence must be a sequence")
    if len(attachment_codes_by_candidate) == 0:
        raise ValueError("at least one candidate is required")
    if len(attachment_codes_by_candidate) != len(branch_presence_by_candidate):
        raise ValueError("candidate sequences must have equal length")

    rows = []
    for codes_value, presence_value in zip(
        attachment_codes_by_candidate,
        branch_presence_by_candidate,
    ):
        codes = np.asarray(codes_value)
        presence = np.asarray(presence_value)
        if codes.ndim != 1 or not np.issubdtype(codes.dtype, np.integer):
            raise ValueError("each attachment-code array must be one-dimensional integers")
        if presence.shape != (codes.size, 3) or not np.issubdtype(
            presence.dtype,
            np.integer,
        ):
            raise ValueError("each presence array must be an integer array of shape (n, 3)")
        if not np.all(np.isin(codes, (0, 1, 2))):
            raise ValueError("attachment codes must lie in {0, 1, 2}")
        if not np.all(np.isin(presence, (0, 1))):
            raise ValueError("branch-presence entries must be binary")

        raw_test = int(np.count_nonzero(codes == 1))
        raw_uncle = int(np.count_nonzero(codes == 2))
        availability = np.sum(presence, axis=0, dtype=int)
        joint_test_focal = int(np.count_nonzero((presence[:, 0] == 1) & (presence[:, 2] == 1)))
        rows.append(
            [
                raw_test,
                raw_uncle,
                int(availability[0]),
                int(availability[1]),
                int(availability[2]),
                joint_test_focal,
            ]
        )
    return np.asarray(rows, dtype=int)

import math
import numpy as np


def screen_attachment_candidates(
    candidate_summary: np.ndarray,
    z_cutoff: float = -1.96,
) -> np.ndarray:
    """Reference implementation."""
    summary = np.asarray(candidate_summary)
    if summary.ndim != 2 or summary.shape[1] != 6:
        raise ValueError("candidate_summary must have shape (c, 6)")
    if summary.shape[0] == 0 or not np.issubdtype(summary.dtype, np.number):
        raise ValueError("candidate_summary must contain numeric candidate rows")
    if not np.all(np.isfinite(summary)) or np.any(summary < 0):
        raise ValueError("candidate_summary entries must be finite and nonnegative")
    if not isinstance(z_cutoff, (int, float)) or not math.isfinite(float(z_cutoff)):
        raise ValueError("z_cutoff must be finite and numeric")

    rows = []
    for row in summary:
        raw_test, raw_uncle, avail_test, avail_uncle = map(float, row[:4])
        if avail_test <= 0.0 or avail_uncle <= 0.0:
            raise ValueError("test and uncle availability must be positive")
        if raw_test > avail_test or raw_uncle > avail_uncle:
            raise ValueError("an attachment count cannot exceed its branch availability")
        raw_denominator = math.sqrt(raw_test + raw_uncle)
        raw_z_score = (
            (raw_uncle - raw_test) / raw_denominator
            if raw_denominator > 0.0
            else 0.0
        )
        corrected_test = raw_test
        corrected_uncle = raw_uncle
        if avail_test > avail_uncle:
            corrected_test *= avail_uncle / avail_test
        elif avail_uncle > avail_test:
            corrected_uncle *= avail_test / avail_uncle
        corrected_denominator = math.sqrt(corrected_test + corrected_uncle)
        corrected_z_score = (
            (corrected_uncle - corrected_test) / corrected_denominator
            if corrected_denominator > 0.0
            else 0.0
        )
        supported = float(
            raw_z_score <= float(z_cutoff)
            and corrected_z_score <= float(z_cutoff)
        )
        rows.append(
            [
                corrected_test,
                corrected_uncle,
                raw_z_score,
                corrected_z_score,
                supported,
            ]
        )
    return np.asarray(rows, dtype=float)

from collections import deque
import numpy as np


def reconcile_rooted_nni(
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

import numpy as np


def aggregate_recipient_votes(
    reconciliation_summary: np.ndarray,
    multiplicities: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Reference implementation."""
    summary = np.asarray(reconciliation_summary)
    counts = np.asarray(multiplicities)
    if summary.ndim != 2 or summary.shape[1] != 4 or summary.shape[0] == 0:
        raise ValueError("reconciliation_summary must have shape (p, 4)")
    if not np.issubdtype(summary.dtype, np.integer) or np.any(summary < 0):
        raise ValueError("reconciliation_summary must contain nonnegative integers")
    if counts.shape != (summary.shape[0],) or not np.issubdtype(
        counts.dtype,
        np.integer,
    ):
        raise ValueError("multiplicities must be one integer per summary row")
    if np.any(counts <= 0):
        raise ValueError("multiplicities must be positive")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")

    rng = np.random.default_rng(int(seed))
    test_votes = 0
    focal_votes = 0
    for row, multiplicity in zip(summary, counts):
        test_appearances = int(row[2])
        focal_appearances = int(row[3])
        number = int(multiplicity)
        if test_appearances > focal_appearances:
            test_votes += number
        elif focal_appearances > test_appearances:
            focal_votes += number
        else:
            assignments = rng.integers(0, 2, size=number)
            test_votes += int(np.count_nonzero(assignments == 0))
            focal_votes += int(np.count_nonzero(assignments == 1))
    if test_votes > focal_votes:
        recipient_index = 0
    elif focal_votes > test_votes:
        recipient_index = 1
    else:
        recipient_index = int(rng.integers(0, 2))
    return np.asarray([test_votes, focal_votes, recipient_index], dtype=int)

import math
import numpy as np


def compute_three_descendant_configurations(
    first_interval: float,
    second_interval: float,
) -> np.ndarray:
    """Reference implementation."""
    for name, value in (
        ("first_interval", first_interval),
        ("second_interval", second_interval),
    ):
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite and numeric")
        if float(value) < 0.0:
            raise ValueError(f"{name} must be nonnegative")

    exp_first = math.exp(-float(first_interval))
    exp_second = math.exp(-float(second_interval))
    exp_three_second = math.exp(-3.0 * float(second_interval))
    one_specific_event = 0.5 * (exp_second - exp_three_second)
    two_events_given_first_pair = (
        1.0 / 3.0 - 0.5 * exp_second + exp_three_second / 6.0
    )
    probabilities = np.asarray(
        [
            exp_first * exp_three_second,
            (1.0 - exp_first) * exp_second + exp_first * one_specific_event,
            exp_first * one_specific_event,
            exp_first * one_specific_event,
            (1.0 - exp_first) * (1.0 - exp_second)
            + exp_first * two_events_given_first_pair,
            exp_first * two_events_given_first_pair,
            exp_first * two_events_given_first_pair,
        ],
        dtype=float,
    )
    probabilities[np.abs(probabilities) < 1e-15] = 0.0
    probabilities /= np.sum(probabilities)
    if not np.isclose(np.sum(probabilities), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("configuration probabilities failed normalization")
    return probabilities

import math
import numpy as np


def compute_parent_history_weights(
    configuration_probabilities: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Reference implementation."""
    probabilities = np.asarray(configuration_probabilities, dtype=float)
    if probabilities.shape != (7,) or not np.all(np.isfinite(probabilities)):
        raise ValueError("configuration_probabilities must contain seven finite values")
    if np.any(probabilities < 0.0) or not np.isclose(
        np.sum(probabilities),
        1.0,
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError("configuration probabilities must be nonnegative and sum to one")
    if not isinstance(gamma, (int, float)) or not math.isfinite(float(gamma)):
        raise ValueError("gamma must be finite and numeric")
    gamma_value = float(gamma)
    if gamma_value < 0.0 or gamma_value > 1.0:
        raise ValueError("gamma must lie in [0, 1]")

    lineage_counts = (3, 2, 2, 2, 1, 1, 1)
    weights = []
    for configuration_probability, lineage_count in zip(
        probabilities,
        lineage_counts,
    ):
        for subset_mask in range(1 << lineage_count):
            donor_count = subset_mask.bit_count()
            weights.append(
                float(configuration_probability)
                * gamma_value**donor_count
                * (1.0 - gamma_value) ** (lineage_count - donor_count)
            )
    result = np.asarray(weights, dtype=float)
    if not np.isclose(np.sum(result), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("parent-history weights failed normalization")
    return result

import numpy as np


def estimate_three_descendant_gamma(
    attachment_count: int,
    joint_availability: int,
    configuration_probabilities: np.ndarray,
    parent_attachment_probabilities: np.ndarray,
) -> float:
    """Reference implementation."""
    if not isinstance(attachment_count, (int, np.integer)) or attachment_count < 0:
        raise ValueError("attachment_count must be a nonnegative integer")
    if not isinstance(joint_availability, (int, np.integer)) or joint_availability <= 0:
        raise ValueError("joint_availability must be a positive integer")
    if attachment_count > joint_availability:
        raise ValueError("attachment_count cannot exceed joint_availability")

    configurations = np.asarray(configuration_probabilities, dtype=float)
    conditionals = np.asarray(parent_attachment_probabilities, dtype=float)
    if configurations.shape != (7,) or not np.all(np.isfinite(configurations)):
        raise ValueError("configuration_probabilities must contain seven finite values")
    if np.any(configurations < 0.0) or not np.isclose(
        np.sum(configurations),
        1.0,
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError("configuration probabilities must be nonnegative and sum to one")
    if conditionals.shape != (26,) or not np.all(np.isfinite(conditionals)):
        raise ValueError("parent_attachment_probabilities must contain 26 finite values")
    if np.any((conditionals < 0.0) | (conditionals > 1.0)):
        raise ValueError("conditional probabilities must lie in [0, 1]")

    gamma_polynomial = np.polynomial.Polynomial([0.0, 1.0])
    non_donor_polynomial = np.polynomial.Polynomial([1.0, -1.0])
    expected_polynomial = np.polynomial.Polynomial([0.0])
    lineage_counts = (3, 2, 2, 2, 1, 1, 1)
    conditional_index = 0
    for configuration_probability, lineage_count in zip(
        configurations,
        lineage_counts,
    ):
        for subset_mask in range(1 << lineage_count):
            donor_count = subset_mask.bit_count()
            subset_polynomial = (
                gamma_polynomial**donor_count
                * non_donor_polynomial ** (lineage_count - donor_count)
            )
            expected_polynomial += (
                float(configuration_probability)
                * float(conditionals[conditional_index])
                * subset_polynomial
            )
            conditional_index += 1

    observed_frequency = float(attachment_count) / float(joint_availability)
    equation = expected_polynomial - np.polynomial.Polynomial([observed_frequency])
    coefficients = np.trim_zeros(equation.coef, trim="b")
    if coefficients.size <= 1:
        raise ValueError("the attachment frequency does not identify gamma")
    roots = np.polynomial.Polynomial(coefficients).roots()
    admissible = []
    for root in roots:
        if abs(float(np.imag(root))) <= 1e-10:
            candidate = float(np.real(root))
            if -1e-10 <= candidate <= 1.0 + 1e-10:
                candidate = min(1.0, max(0.0, candidate))
                if abs(float(expected_polynomial(candidate)) - observed_frequency) <= 1e-8:
                    if not any(abs(candidate - prior) <= 1e-10 for prior in admissible):
                        admissible.append(candidate)
    if len(admissible) != 1:
        raise ValueError("the inputs must yield exactly one root in [0, 1]")
    return float(admissible[0])

import numpy as np


def run_full_pipeline(seed: int = 17) -> float:
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

    candidate_summary = summarize_candidate_evidence(
        attachment_codes_by_candidate,
        branch_presence_by_candidate,
    )
    screen = screen_attachment_candidates(candidate_summary, z_cutoff=-1.96)
    supported_candidates = np.flatnonzero(screen[:, 4] == 1.0)
    if supported_candidates.size != 1:
        raise ValueError("the configuration must yield exactly one supported candidate")
    candidate_index = int(supported_candidates[0])

    reconciliation_summary = reconcile_rooted_nni(
        species_tree,
        gene_tree_patterns_by_candidate[candidate_index],
        candidate_clade_masks[candidate_index],
    )
    vote_summary = aggregate_recipient_votes(
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

    configuration_probabilities = compute_three_descendant_configurations(
        first_interval,
        second_interval,
    )
    gamma_hat = estimate_three_descendant_gamma(
        int(candidate_summary[candidate_index, 0]),
        int(candidate_summary[candidate_index, 5]),
        configuration_probabilities,
        parent_attachment_probabilities,
    )
    parent_history_weights = compute_parent_history_weights(
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
SCICODE_GOLD_EOF
