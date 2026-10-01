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

def select_informative_radius(
    changes_by_radius: 'np.ndarray',
    radii: 'np.ndarray',
) -> 'np.ndarray':
    import numpy as np

    changes = np.asarray(changes_by_radius)
    radius_values = np.asarray(radii)

    if changes.ndim != 2 or changes.shape[0] == 0 or changes.shape[1] == 0:
        raise ValueError(
            "changes_by_radius must be a nonempty two-dimensional array"
        )
    if radius_values.ndim != 1 or radius_values.size != changes.shape[0]:
        raise ValueError("radii must have one entry per change vector")
    if (
        not np.issubdtype(changes.dtype, np.number)
        or not np.isrealobj(changes)
        or np.any(~np.isfinite(changes))
    ):
        raise ValueError("changes_by_radius must be finite and numeric")
    if (
        not np.issubdtype(radius_values.dtype, np.number)
        or not np.isrealobj(radius_values)
        or np.any(~np.isfinite(radius_values))
    ):
        raise ValueError("radii must be finite and numeric")
    if np.any(changes != np.rint(changes)):
        raise ValueError("moiety changes must be integers")
    if (
        np.any(radius_values != np.rint(radius_values))
        or np.any(radius_values <= 0)
    ):
        raise ValueError("radii must be positive integers")

    radius_values = radius_values.astype(int, copy=False)
    if np.any(np.diff(radius_values) <= 0):
        raise ValueError("radii must be strictly increasing")

    changes = changes.astype(int, copy=False)

    for radius, row in zip(radius_values, changes):
        if np.any(row != 0):
            return np.concatenate(
                (np.array([radius], dtype=int), row.copy())
            )

    raise ValueError(
        "no supplied radius resolves a nonzero transformation"
    )

import numpy as np


def rank_parsimonious_rule_sets(
    rule_matrix: 'np.ndarray',
    target_change: 'np.ndarray',
    max_steps: int,
    limit: int,
) -> 'np.ndarray':
    matrix = np.asarray(rule_matrix)
    target = np.asarray(target_change)

    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("rule_matrix must be a nonempty two-dimensional array")
    if target.ndim != 1 or target.size != matrix.shape[0]:
        raise ValueError("target_change has incompatible shape")
    if (
        not np.issubdtype(matrix.dtype, np.number)
        or not np.isrealobj(matrix)
        or np.any(~np.isfinite(matrix))
    ):
        raise ValueError("rule_matrix must be finite and numeric")
    if (
        not np.issubdtype(target.dtype, np.number)
        or not np.isrealobj(target)
        or np.any(~np.isfinite(target))
    ):
        raise ValueError("target_change must be finite and numeric")
    if np.any(matrix != np.rint(matrix)) or np.any(target != np.rint(target)):
        raise ValueError("rule changes and target must be integers")
    if not np.any(target != 0):
        raise ValueError("target_change must be nonzero")
    if (
        isinstance(max_steps, (bool, np.bool_))
        or not isinstance(max_steps, (int, np.integer))
        or int(max_steps) <= 0
    ):
        raise ValueError("max_steps must be a positive integer")
    if (
        isinstance(limit, (bool, np.bool_))
        or not isinstance(limit, (int, np.integer))
        or int(limit) <= 0
    ):
        raise ValueError("limit must be a positive integer")

    matrix = matrix.astype(int, copy=False)
    target = target.astype(int, copy=False)
    n_rules = matrix.shape[1]
    suffix_min = np.minimum.accumulate(matrix[:, ::-1], axis=1)[:, ::-1]
    suffix_max = np.maximum.accumulate(matrix[:, ::-1], axis=1)[:, ::-1]
    cuts = []
    selected = []

    def _admissible(residual, start, remaining):
        if remaining == 0:
            return not np.any(residual)
        if start >= n_rules:
            return False
        return bool(
            np.all(residual >= remaining * suffix_min[:, start])
            and np.all(residual <= remaining * suffix_max[:, start])
        )

    for n_steps in range(1, int(max_steps) + 1):
        labels = []

        def _search(start, remaining, residual):
            if len(selected) == int(limit):
                return True
            if remaining == 0:
                counts = np.bincount(labels, minlength=n_rules).astype(int)
                if any(
                    int(np.sum(counts[support])) > support.size - 1
                    for support in cuts
                ):
                    return False
                selected.append(
                    np.concatenate((np.array([n_steps], dtype=int), counts))
                )
                cuts.append(np.flatnonzero(counts))
                return len(selected) == int(limit)

            for rule in range(start, n_rules):
                next_residual = residual - matrix[:, rule]
                if not _admissible(next_residual, rule, remaining - 1):
                    continue
                labels.append(rule)
                done = _search(rule, remaining - 1, next_residual)
                labels.pop()
                if done:
                    return True
            return False

        if _admissible(target, 0, n_steps):
            _search(0, n_steps, target.copy())
        if len(selected) == int(limit):
            break

    if not selected:
        return np.empty((0, n_rules + 1), dtype=int)

    return np.vstack(selected)

import numpy as np


def order_rule_multiset(
    rule_counts: 'np.ndarray',
    rule_matrix: 'np.ndarray',
    initial_counts: 'np.ndarray',
    exempt_moieties: 'np.ndarray',
) -> 'np.ndarray':
    import numpy as np

    counts = np.asarray(rule_counts)
    matrix = np.asarray(rule_matrix)
    initial = np.asarray(initial_counts)
    exempt = np.asarray(exempt_moieties)

    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(
            "rule_matrix must be a nonempty two-dimensional array"
        )
    if counts.ndim != 1 or counts.size != matrix.shape[1]:
        raise ValueError("rule_counts has incompatible shape")
    if initial.ndim != 1 or initial.size != matrix.shape[0]:
        raise ValueError("initial_counts has incompatible shape")
    if (
        exempt.ndim != 1
        or exempt.size != matrix.shape[0]
        or exempt.dtype != np.bool_
    ):
        raise ValueError(
            "exempt_moieties must be a boolean vector"
        )

    for value, name in (
        (counts, "rule_counts"),
        (matrix, "rule_matrix"),
        (initial, "initial_counts"),
    ):
        if (
            not np.issubdtype(value.dtype, np.number)
            or not np.isrealobj(value)
            or np.any(~np.isfinite(value))
        ):
            raise ValueError(f"{name} must be finite and numeric")
        if np.any(value != np.rint(value)):
            raise ValueError(f"{name} must contain integers")

    counts = counts.astype(int, copy=True)
    matrix = matrix.astype(int, copy=False)
    initial = initial.astype(int, copy=True)

    if np.any(counts < 0) or int(np.sum(counts)) == 0:
        raise ValueError(
            "rule_counts must be nonnegative and select at least one rule"
        )
    if np.any(initial < 0):
        raise ValueError("initial_counts must be nonnegative")

    failed = set()

    def _search(remaining, current, prefix):
        if int(np.sum(remaining)) == 0:
            return prefix

        state = (
            tuple(int(x) for x in remaining),
            tuple(int(x) for x in current[~exempt]),
        )
        if state in failed:
            return None

        for rule in np.flatnonzero(remaining):
            next_counts = current + matrix[:, rule]
            if np.any(next_counts[~exempt] < 0):
                continue

            remaining[rule] -= 1
            ordered = _search(
                remaining,
                next_counts,
                prefix + (int(rule) + 1,),
            )
            remaining[rule] += 1

            if ordered is not None:
                return ordered

        failed.add(state)
        return None

    answer = _search(counts, initial, tuple())

    if answer is None:
        return np.empty(0, dtype=int)

    return np.asarray(answer, dtype=int)

import numpy as np


def generate_parsimonious_mechanisms(
    rule_matrix: 'np.ndarray',
    target_change: 'np.ndarray',
    initial_counts: 'np.ndarray',
    exempt_moieties: 'np.ndarray',
    max_steps: int,
    top_k: int,
    candidate_limit: int,
) -> 'np.ndarray':
    import numpy as np

    for value, name in (
        (max_steps, "max_steps"),
        (top_k, "top_k"),
        (candidate_limit, "candidate_limit"),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")

    ranked = rank_parsimonious_rule_sets(
        rule_matrix,
        target_change,
        int(max_steps),
        int(candidate_limit),
    )

    rows = []

    for candidate_rank, row in enumerate(ranked, start=1):
        n_steps = int(row[0])
        sequence = order_rule_multiset(
            row[1:],
            rule_matrix,
            initial_counts,
            exempt_moieties,
        )

        if sequence.size == 0:
            continue
        if sequence.size != n_steps or n_steps > int(max_steps):
            raise ValueError(
                "ordered mechanism length is inconsistent"
            )

        padded = np.zeros(int(max_steps), dtype=int)
        padded[:n_steps] = sequence

        rows.append(
            np.concatenate(
                (
                    np.array(
                        [candidate_rank, n_steps],
                        dtype=int,
                    ),
                    padded,
                )
            )
        )

        if len(rows) == int(top_k):
            break

    if not rows:
        return np.empty(
            (0, int(max_steps) + 2),
            dtype=int,
        )

    return np.vstack(rows)

import numpy as np


def build_arrow_environment_profiles(
    mechanisms: 'np.ndarray',
    rule_arrow_matrix: 'np.ndarray',
) -> 'np.ndarray':
    import numpy as np

    mechs = np.asarray(mechanisms)
    arrows = np.asarray(rule_arrow_matrix)

    if mechs.ndim != 2 or mechs.shape[1] < 3:
        raise ValueError(
            "mechanisms must be a two-dimensional padded table"
        )
    if (
        arrows.ndim != 2
        or arrows.shape[0] == 0
        or arrows.shape[1] == 0
    ):
        raise ValueError(
            "rule_arrow_matrix must be nonempty and two-dimensional"
        )

    for value, name in (
        (mechs, "mechanisms"),
        (arrows, "rule_arrow_matrix"),
    ):
        if (
            not np.issubdtype(value.dtype, np.number)
            or not np.isrealobj(value)
            or np.any(~np.isfinite(value))
        ):
            raise ValueError(f"{name} must be finite and numeric")

    if np.any(mechs != np.rint(mechs)):
        raise ValueError("mechanisms must contain integers")
    if np.any((arrows != 0) & (arrows != 1)):
        raise ValueError("rule_arrow_matrix must be binary")

    mechs = mechs.astype(int, copy=False)
    profiles = np.zeros(
        (mechs.shape[0], arrows.shape[1]),
        dtype=int,
    )
    capacity = mechs.shape[1] - 2

    for index, row in enumerate(mechs):
        n_steps = int(row[1])

        if n_steps <= 0 or n_steps > capacity:
            raise ValueError("mechanism step count is invalid")

        labels = row[2:2 + n_steps]

        if (
            np.any(labels < 1)
            or np.any(labels > arrows.shape[0])
        ):
            raise ValueError(
                "mechanism contains an out-of-range rule label"
            )
        if np.any(row[2 + n_steps:] != 0):
            raise ValueError(
                "mechanism padding must be zero"
            )

        profiles[index] = np.any(
            arrows[labels - 1] != 0,
            axis=0,
        ).astype(int)

    return profiles

import numpy as np


def maximum_archive_similarity(
    candidate_profiles: 'np.ndarray',
    reference_profiles: 'np.ndarray',
) -> 'np.ndarray':
    import numpy as np

    candidates = np.asarray(candidate_profiles)
    references = np.asarray(reference_profiles)

    if candidates.ndim != 2 or references.ndim != 2:
        raise ValueError(
            "profiles must be two-dimensional"
        )
    if (
        candidates.shape[1] == 0
        or references.shape[1] != candidates.shape[1]
    ):
        raise ValueError(
            "candidate and reference feature dimensions must agree"
        )

    for value, name in (
        (candidates, "candidate_profiles"),
        (references, "reference_profiles"),
    ):
        if (
            not np.issubdtype(value.dtype, np.number)
            or not np.isrealobj(value)
            or np.any(~np.isfinite(value))
        ):
            raise ValueError(f"{name} must be finite and numeric")

    candidates = candidates != 0
    references = references != 0

    valid_references = np.flatnonzero(
        np.any(references, axis=1)
    )
    if valid_references.size == 0:
        raise ValueError(
            "reference archive must contain a nonempty profile"
        )

    result_rows = []

    for candidate in candidates:
        best = None

        for reference_index in valid_references:
            reference = references[reference_index]
            intersection = int(
                np.count_nonzero(candidate & reference)
            )
            union = int(
                np.count_nonzero(candidate | reference)
            )
            score = float(intersection / union)

            row = (
                score,
                int(reference_index) + 1,
                intersection,
                union,
            )

            if best is None or score > best[0]:
                best = row

        result_rows.append(best)

    if not result_rows:
        return np.empty((0, 4), dtype=float)

    return np.asarray(result_rows, dtype=float)

import numpy as np


def rerank_mechanisms(
    mechanisms: 'np.ndarray',
    similarity_summary: 'np.ndarray',
) -> 'np.ndarray':
    import numpy as np

    mechs = np.asarray(mechanisms)
    summary = np.asarray(similarity_summary)

    if mechs.ndim != 2 or mechs.shape[1] < 3:
        raise ValueError(
            "mechanisms must be a padded two-dimensional table"
        )
    if (
        summary.ndim != 2
        or summary.shape != (mechs.shape[0], 4)
    ):
        raise ValueError(
            "similarity_summary must have four columns and matching rows"
        )
    if (
        not np.issubdtype(mechs.dtype, np.number)
        or not np.isrealobj(mechs)
        or np.any(~np.isfinite(mechs))
    ):
        raise ValueError(
            "mechanisms must be finite and numeric"
        )
    if (
        not np.issubdtype(summary.dtype, np.number)
        or not np.isrealobj(summary)
        or np.any(~np.isfinite(summary))
    ):
        raise ValueError(
            "similarity_summary must be finite and numeric"
        )
    if (
        np.any(summary[:, 0] < 0.0)
        or np.any(summary[:, 0] > 1.0)
    ):
        raise ValueError(
            "similarities must lie between zero and one"
        )

    order = np.lexsort(
        (
            np.arange(mechs.shape[0]),
            -summary[:, 0],
        )
    )

    rows = []

    for index in order:
        row = np.concatenate(
            (
                np.array(
                    [summary[index, 0], mechs[index, 0]],
                    dtype=float,
                ),
                summary[index, 1:4].astype(
                    float,
                    copy=False,
                ),
                mechs[index, 1:].astype(
                    float,
                    copy=False,
                ),
            )
        )
        rows.append(row)

    if not rows:
        return np.empty(
            (0, mechs.shape[1] + 4),
            dtype=float,
        )

    return np.vstack(rows)

import numpy as np


def adaptive_moiety_mechanism_similarity(
    changes_by_radius: 'np.ndarray',
    radii: 'np.ndarray',
    rule_matrix: 'np.ndarray',
    initial_counts: 'np.ndarray',
    exempt_moieties: 'np.ndarray',
    rule_arrow_matrix: 'np.ndarray',
    reference_profiles: 'np.ndarray',
    max_steps: int,
    top_k: int,
    candidate_limit: int,
) -> float:
    resolved = select_informative_radius(changes_by_radius, radii)
    target_change = resolved[1:]
    mechanisms = generate_parsimonious_mechanisms(
        rule_matrix,
        target_change,
        initial_counts,
        exempt_moieties,
        max_steps,
        top_k,
        candidate_limit,
    )
    if mechanisms.shape[0] == 0:
        raise ValueError("the supplied instance yields no feasible mechanism")

    profiles = build_arrow_environment_profiles(
        mechanisms,
        rule_arrow_matrix,
    )
    similarity = maximum_archive_similarity(
        profiles,
        reference_profiles,
    )
    ranked = rerank_mechanisms(
        mechanisms,
        similarity,
    )

    return float(100.0 * ranked[0, 0])


def _append_benchmark_rule(
    columns,
    n_moieties,
    consumed,
    produced,
    catalyst_delta=0,
):
    column = np.zeros(n_moieties, dtype=int)

    for index, amount in consumed.items():
        column[index] -= amount
    for index, amount in produced.items():
        column[index] += amount

    column[-1] += catalyst_delta
    columns.append(column)


def _benchmark_mechanism_fixture(
    permutation_seed,
    arrow_seed,
):
    n_layers = 5
    layer_width = 4
    source_index = 0
    layer_start = 1
    target_index = layer_start + n_layers * layer_width
    cycle_start = target_index + 1
    catalyst_index = cycle_start + 6
    n_moieties = catalyst_index + 1
    columns = []

    for offset in range(5):
        _append_benchmark_rule(
            columns,
            n_moieties,
            {cycle_start + offset: 1},
            {cycle_start + offset + 1: 1},
        )

    _append_benchmark_rule(
        columns,
        n_moieties,
        {cycle_start + 5: 1, source_index: 1},
        {cycle_start: 1, target_index: 1},
    )

    for node in range(layer_width):
        _append_benchmark_rule(
            columns,
            n_moieties,
            {source_index: 1},
            {layer_start + node: 1},
            -1,
        )

    for layer in range(n_layers - 1):
        left = layer_start + layer * layer_width
        right = left + layer_width

        for a in range(layer_width):
            for b in range(layer_width):
                _append_benchmark_rule(
                    columns,
                    n_moieties,
                    {left + a: 1},
                    {right + b: 1},
                )

    final_layer = layer_start + (n_layers - 1) * layer_width

    for node in range(layer_width):
        _append_benchmark_rule(
            columns,
            n_moieties,
            {final_layer + node: 1},
            {target_index: 1},
            1,
        )

    rule_matrix = np.stack(columns, axis=1)
    permutation_rng = np.random.default_rng(permutation_seed)
    rule_permutation = np.concatenate(
        (
            np.arange(6),
            permutation_rng.permutation(
                np.arange(6, rule_matrix.shape[1])
            ),
        )
    )
    rule_matrix = rule_matrix[:, rule_permutation]

    target = np.zeros(n_moieties, dtype=int)
    target[source_index], target[target_index] = -1, 1

    changes_by_radius = np.stack(
        (
            np.zeros_like(target),
            np.zeros_like(target),
            target,
            2 * target,
        )
    )
    radii = np.array([1, 2, 3, 4], dtype=int)

    initial_counts = np.zeros(n_moieties, dtype=int)
    initial_counts[source_index] = 1

    exempt_moieties = np.zeros(n_moieties, dtype=bool)
    exempt_moieties[catalyst_index] = True

    arrow_rng = np.random.default_rng(arrow_seed)
    rule_arrow_matrix = np.zeros(
        (rule_matrix.shape[1], 83),
        dtype=int,
    )

    for rule in range(rule_matrix.shape[1]):
        width = 4 + (3 * rule) % 5
        features = arrow_rng.choice(
            83,
            size=width,
            replace=False,
        )
        rule_arrow_matrix[rule, features] = 1

    reference_profiles = (
        arrow_rng.random((13, 83)) < 0.11
    ).astype(int)

    for reference in range(reference_profiles.shape[0]):
        reference_profiles[
            reference,
            (11 * reference + 5) % 83,
        ] = 1

    return (
        changes_by_radius,
        radii,
        rule_matrix,
        initial_counts,
        exempt_moieties,
        rule_arrow_matrix,
        reference_profiles,
        6,
        10,
        18,
    )
SCICODE_GOLD_EOF
