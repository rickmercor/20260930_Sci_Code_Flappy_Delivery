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


def compute_state_scalar(
    rg_matrix: np.ndarray,
    target_index: int,
) -> float:
    rg = np.asarray(rg_matrix, dtype=float)

    if (
        rg.ndim != 2
        or rg.shape[0] != rg.shape[1]
        or rg.shape[0] < 2
    ):
        raise ValueError(
            "rg_matrix must be square with at least two traits"
        )

    if not np.all(np.isfinite(rg)):
        raise ValueError(
            "rg_matrix must contain only finite values"
        )

    if not isinstance(target_index, (int, np.integer)):
        raise ValueError(
            "target_index must be an integer"
        )

    target = int(target_index)
    n_traits = rg.shape[0]

    if target < 0 or target >= n_traits:
        raise ValueError(
            "target_index is out of range"
        )

    if not np.allclose(
        rg,
        rg.T,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "rg_matrix must be symmetric"
        )

    if not np.allclose(
        np.diag(rg),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "rg_matrix must have a unit diagonal"
        )

    if np.any(np.abs(rg) > 1.0 + 1e-12):
        raise ValueError(
            "correlations must lie in [-1, 1]"
        )

    remaining = np.array(
        [
            idx
            for idx in range(n_traits)
            if idx != target
        ],
        dtype=int,
    )

    b = rg[remaining, target]
    A = rg[np.ix_(remaining, remaining)]

    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "remaining-trait matrix must be nonsingular"
        ) from exc

    scalar = float(b @ x)

    if not np.isfinite(scalar):
        raise ValueError(
            "state scalar must be finite"
        )

    return scalar

import numpy as np


def compute_replicate_vector(
    rg_matrix: np.ndarray,
    rg_replicates: np.ndarray,
    target_index: int,
) -> np.ndarray:
    rg = np.asarray(rg_matrix, dtype=float)
    replicates = np.asarray(
        rg_replicates,
        dtype=float,
    )

    if (
        rg.ndim != 2
        or rg.shape[0] != rg.shape[1]
        or rg.shape[0] < 2
    ):
        raise ValueError(
            "rg_matrix must be square with at least two traits"
        )

    if not np.all(np.isfinite(rg)):
        raise ValueError(
            "rg_matrix must contain only finite values"
        )

    if not isinstance(target_index, (int, np.integer)):
        raise ValueError(
            "target_index must be an integer"
        )

    target = int(target_index)
    n_traits = rg.shape[0]

    if target < 0 or target >= n_traits:
        raise ValueError(
            "target_index is out of range"
        )

    if not np.allclose(
        rg,
        rg.T,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "rg_matrix must be symmetric"
        )

    if not np.allclose(
        np.diag(rg),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "rg_matrix must have a unit diagonal"
        )

    if np.any(np.abs(rg) > 1.0 + 1e-12):
        raise ValueError(
            "correlations must lie in [-1, 1]"
        )

    expected_columns = (
        n_traits * (n_traits - 1) // 2
    )

    if (
        replicates.ndim != 2
        or replicates.shape[0] < 2
        or replicates.shape[1] != expected_columns
    ):
        raise ValueError(
            "rg_replicates has incompatible shape"
        )

    if not np.all(np.isfinite(replicates)):
        raise ValueError(
            "rg_replicates must contain only finite values"
        )

    if np.any(
        np.abs(replicates) > 1.0 + 1e-12
    ):
        raise ValueError(
            "replicate correlations must lie in [-1, 1]"
        )

    n_blocks = replicates.shape[0]

    packed_pairs = [
        (row, column)
        for column in range(n_traits - 1)
        for row in range(column + 1, n_traits)
    ]

    replicate_matrices = np.repeat(
        np.eye(
            n_traits,
            dtype=float,
        )[None, :, :],
        n_blocks,
        axis=0,
    )

    for packed_column, (
        row,
        column,
    ) in enumerate(packed_pairs):
        replicate_matrices[
            :,
            row,
            column,
        ] = replicates[
            :,
            packed_column,
        ]

        replicate_matrices[
            :,
            column,
            row,
        ] = replicates[
            :,
            packed_column,
        ]

    remaining = np.array(
        [
            idx
            for idx in range(n_traits)
            if idx != target
        ],
        dtype=int,
    )

    point_b = rg[
        remaining,
        target,
    ]

    point_A = rg[
        np.ix_(
            remaining,
            remaining,
        )
    ]

    try:
        point_x = np.linalg.solve(
            point_A,
            point_b,
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "remaining-trait matrix must be nonsingular"
        ) from exc

    point_scalar = float(
        point_b @ point_x
    )

    replicate_scalars = np.empty(
        n_blocks,
        dtype=float,
    )

    for block_index, matrix in enumerate(
        replicate_matrices
    ):
        replicate_b = matrix[
            remaining,
            target,
        ]

        replicate_A = matrix[
            np.ix_(
                remaining,
                remaining,
            )
        ]

        try:
            replicate_x = np.linalg.solve(
                replicate_A,
                replicate_b,
            )
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "replicate remaining-trait matrix "
                "must be nonsingular"
            ) from exc

        replicate_scalars[
            block_index
        ] = float(
            replicate_b @ replicate_x
        )

    scalar_pseudovalues = (
        n_blocks * point_scalar
        - (n_blocks - 1)
        * replicate_scalars
    )

    scalar_scale = float(
        np.std(
            scalar_pseudovalues,
            ddof=1,
        )
        / np.sqrt(n_blocks)
    )

    standardized_pairs = np.empty(
        remaining.size,
        dtype=float,
    )

    for output_index, trait_index in enumerate(
        remaining
    ):
        point_value = float(
            rg[
                trait_index,
                target,
            ]
        )

        replicate_values = (
            replicate_matrices[
                :,
                trait_index,
                target,
            ]
        )

        pair_pseudovalues = (
            n_blocks * point_value
            - (n_blocks - 1)
            * replicate_values
        )

        pair_scale = float(
            np.std(
                pair_pseudovalues,
                ddof=1,
            )
            / np.sqrt(n_blocks)
        )

        if (
            pair_scale <= 0.0
            or not np.isfinite(pair_scale)
        ):
            raise ValueError(
                "target-pair replicate scale "
                "must be positive"
            )

        standardized_pairs[
            output_index
        ] = (
            point_value / pair_scale
        )

    vector = np.concatenate(
        (
            np.array(
                [scalar_scale],
                dtype=float,
            ),
            standardized_pairs,
        )
    )

    if not np.all(np.isfinite(vector)):
        raise ValueError(
            "replicate vector must contain "
            "only finite values"
        )

    return vector

import numpy as np


def construct_replicate_state(
    rg_replicates: np.ndarray,
) -> np.ndarray:
    replicates = np.asarray(
        rg_replicates,
        dtype=float,
    )

    if (
        replicates.ndim != 2
        or replicates.shape[0] < 2
        or replicates.shape[1] < 1
    ):
        raise ValueError(
            "rg_replicates must have shape "
            "(n_blocks, n_coordinates)"
        )

    if not np.all(np.isfinite(replicates)):
        raise ValueError(
            "rg_replicates must contain only finite values"
        )

    if np.any(np.abs(replicates) > 1.0 + 1e-12):
        raise ValueError(
            "replicate correlations must lie in [-1, 1]"
        )

    n_blocks, n_coordinates = replicates.shape

    discriminant = 1 + 8 * n_coordinates
    root = int(round(np.sqrt(discriminant)))

    if (
        root * root != discriminant
        or (1 + root) % 2 != 0
    ):
        raise ValueError(
            "packed width must be a "
            "strict-lower-triangle size"
        )

    replicate_state = (
        n_blocks
        * np.atleast_2d(
            np.cov(
                replicates,
                rowvar=False,
                ddof=1,
            )
        )
    )

    replicate_state = 0.5 * (
        replicate_state
        + replicate_state.T
    )

    expected_shape = (
        n_coordinates,
        n_coordinates,
    )

    if replicate_state.shape != expected_shape:
        raise ValueError(
            "replicate state has an invalid shape"
        )

    if not np.all(np.isfinite(replicate_state)):
        raise ValueError(
            "replicate state must contain only finite values"
        )

    return replicate_state

import numpy as np


def generate_perturbed_states(
    rg_matrix: np.ndarray,
    replicate_state: np.ndarray,
    sample_count: int,
    proposal_limit: int,
    seed: int,
) -> np.ndarray:
    rg = np.asarray(rg_matrix, dtype=float)
    state = np.asarray(
        replicate_state,
        dtype=float,
    )

    if (
        rg.ndim != 2
        or rg.shape[0] != rg.shape[1]
        or rg.shape[0] < 2
    ):
        raise ValueError(
            "rg_matrix must be square with at least two traits"
        )

    if not np.all(np.isfinite(rg)):
        raise ValueError(
            "rg_matrix must contain only finite values"
        )

    if not np.allclose(
        rg,
        rg.T,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "rg_matrix must be symmetric"
        )

    if not np.allclose(
        np.diag(rg),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "rg_matrix must have a unit diagonal"
        )

    if np.any(np.abs(rg) > 1.0 + 1e-12):
        raise ValueError(
            "correlations must lie in [-1, 1]"
        )

    try:
        np.linalg.cholesky(rg)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "rg_matrix must be positive definite"
        ) from exc

    n_traits = rg.shape[0]
    n_coordinates = (
        n_traits * (n_traits - 1) // 2
    )

    if state.shape != (
        n_coordinates,
        n_coordinates,
    ):
        raise ValueError(
            "replicate_state has an incompatible shape"
        )

    if not np.all(np.isfinite(state)):
        raise ValueError(
            "replicate_state must contain only finite values"
        )

    if not np.allclose(
        state,
        state.T,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "replicate_state must be symmetric"
        )

    if (
        not isinstance(
            sample_count,
            (int, np.integer),
        )
        or isinstance(
            sample_count,
            (bool, np.bool_),
        )
        or int(sample_count) < 1
    ):
        raise ValueError(
            "sample_count must be a positive integer"
        )

    if (
        not isinstance(
            proposal_limit,
            (int, np.integer),
        )
        or isinstance(
            proposal_limit,
            (bool, np.bool_),
        )
        or int(proposal_limit) < 1
    ):
        raise ValueError(
            "proposal_limit must be a positive integer"
        )

    if (
        not isinstance(
            seed,
            (int, np.integer),
        )
        or isinstance(
            seed,
            (bool, np.bool_),
        )
        or int(seed) < 0
    ):
        raise ValueError(
            "seed must be a non-negative integer"
        )

    try:
        factor = np.linalg.cholesky(state)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "replicate_state must be positive definite"
        ) from exc

    packed_pairs = [
        (row, column)
        for column in range(n_traits - 1)
        for row in range(column + 1, n_traits)
    ]

    rng = np.random.default_rng(int(seed))

    result = np.empty(
        (
            int(sample_count),
            n_coordinates + 1,
        ),
        dtype=float,
    )

    for sample_index in range(
        int(sample_count)
    ):
        retained = False

        for proposal_count in range(
            1,
            int(proposal_limit) + 1,
        ):
            packed_error = (
                factor
                @ rng.standard_normal(
                    n_coordinates
                )
            )

            error_matrix = np.zeros(
                (n_traits, n_traits),
                dtype=float,
            )

            for value, (row, column) in zip(
                packed_error,
                packed_pairs,
            ):
                error_matrix[row, column] = value
                error_matrix[column, row] = value

            perturbed = rg + error_matrix

            try:
                np.linalg.cholesky(perturbed)
            except np.linalg.LinAlgError:
                continue

            result[
                sample_index,
                :-1,
            ] = packed_error

            result[
                sample_index,
                -1,
            ] = float(proposal_count)

            retained = True
            break

        if not retained:
            raise ValueError(
                "proposal limit exhausted before retention"
            )

    return result

import numpy as np


def construct_state_components(
    rg_matrix: np.ndarray,
    perturbed_states: np.ndarray,
    target_index: int,
) -> np.ndarray:
    rg = np.asarray(rg_matrix, dtype=float)
    states = np.asarray(
        perturbed_states,
        dtype=float,
    )

    if (
        rg.ndim != 2
        or rg.shape[0] != rg.shape[1]
        or rg.shape[0] < 2
    ):
        raise ValueError(
            "rg_matrix must be square with at least two traits"
        )

    if not np.all(np.isfinite(rg)):
        raise ValueError(
            "rg_matrix must contain only finite values"
        )

    if (
        not isinstance(
            target_index,
            (int, np.integer),
        )
        or isinstance(
            target_index,
            (bool, np.bool_),
        )
    ):
        raise ValueError(
            "target_index must be an integer"
        )

    target = int(target_index)
    n_traits = rg.shape[0]

    if target < 0 or target >= n_traits:
        raise ValueError(
            "target_index is out of range"
        )

    if not np.allclose(
        rg,
        rg.T,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "rg_matrix must be symmetric"
        )

    if not np.allclose(
        np.diag(rg),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "rg_matrix must have a unit diagonal"
        )

    if np.any(np.abs(rg) > 1.0 + 1e-12):
        raise ValueError(
            "correlations must lie in [-1, 1]"
        )

    try:
        np.linalg.cholesky(rg)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "rg_matrix must be positive definite"
        ) from exc

    n_coordinates = (
        n_traits * (n_traits - 1) // 2
    )

    if (
        states.ndim != 2
        or states.shape[0] < 1
        or states.shape[1]
        != n_coordinates + 1
    ):
        raise ValueError(
            "perturbed_states has an incompatible shape"
        )

    if not np.all(np.isfinite(states)):
        raise ValueError(
            "perturbed_states must contain only finite values"
        )

    proposal_counts = states[:, -1]

    if (
        np.any(proposal_counts < 1.0)
        or not np.allclose(
            proposal_counts,
            np.round(proposal_counts),
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise ValueError(
            "proposal counts must be positive integers"
        )

    packed_pairs = [
        (row, column)
        for column in range(n_traits - 1)
        for row in range(column + 1, n_traits)
    ]

    remaining = np.array(
        [
            index
            for index in range(n_traits)
            if index != target
        ],
        dtype=int,
    )

    point_vector = rg[
        remaining,
        target,
    ]

    components = np.empty(
        (states.shape[0], 3),
        dtype=float,
    )

    for state_index, state_row in enumerate(
        states
    ):
        error_matrix = np.zeros(
            (n_traits, n_traits),
            dtype=float,
        )

        for value, (row, column) in zip(
            state_row[:-1],
            packed_pairs,
        ):
            error_matrix[row, column] = value
            error_matrix[column, row] = value

        perturbed_matrix = rg + error_matrix

        try:
            np.linalg.cholesky(
                perturbed_matrix
            )
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "each perturbed state must be "
                "positive definite"
            ) from exc

        auxiliary_matrix = perturbed_matrix[
            np.ix_(remaining, remaining)
        ]

        error_vector = error_matrix[
            remaining,
            target,
        ]

        try:
            solved_point = np.linalg.solve(
                auxiliary_matrix,
                point_vector,
            )
            solved_error = np.linalg.solve(
                auxiliary_matrix,
                error_vector,
            )
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "each auxiliary state must be nonsingular"
            ) from exc

        components[
            state_index,
            0,
        ] = float(
            point_vector @ solved_point
        )

        components[
            state_index,
            1,
        ] = float(
            point_vector @ solved_error
            + error_vector @ solved_point
        )

        components[
            state_index,
            2,
        ] = float(
            error_vector @ solved_error
        )

    if not np.all(np.isfinite(components)):
        raise ValueError(
            "state components must be finite"
        )

    return components

import numpy as np


def solve_state_scale(
    initial_scalar: float,
    initial_standard_error: float,
    state_components: np.ndarray,
    tolerance: float,
    max_iterations: int,
) -> np.ndarray:
    if (
        not isinstance(
            initial_scalar,
            (int, float, np.integer, np.floating),
        )
        or isinstance(
            initial_scalar,
            (bool, np.bool_),
        )
    ):
        raise ValueError(
            "initial_scalar must be a real number"
        )

    scalar = float(initial_scalar)

    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(
            "initial_scalar must be finite and non-negative"
        )

    if (
        not isinstance(
            initial_standard_error,
            (int, float, np.integer, np.floating),
        )
        or isinstance(
            initial_standard_error,
            (bool, np.bool_),
        )
    ):
        raise ValueError(
            "initial_standard_error must be a real number"
        )

    standard_error = float(
        initial_standard_error
    )

    if (
        not np.isfinite(standard_error)
        or standard_error < 0.0
    ):
        raise ValueError(
            "initial_standard_error must be finite "
            "and non-negative"
        )

    components = np.asarray(
        state_components,
        dtype=float,
    )

    if (
        components.ndim != 2
        or components.shape[0] < 2
        or components.shape[1] != 3
    ):
        raise ValueError(
            "state_components must have shape "
            "(n_states, 3), with n_states >= 2"
        )

    if not np.all(np.isfinite(components)):
        raise ValueError(
            "state_components must contain only finite values"
        )

    if (
        not isinstance(
            tolerance,
            (int, float, np.integer, np.floating),
        )
        or isinstance(
            tolerance,
            (bool, np.bool_),
        )
    ):
        raise ValueError(
            "tolerance must be a real number"
        )

    absolute_tolerance = float(tolerance)

    if (
        not np.isfinite(absolute_tolerance)
        or absolute_tolerance <= 0.0
    ):
        raise ValueError(
            "tolerance must be finite and positive"
        )

    if (
        not isinstance(
            max_iterations,
            (int, np.integer),
        )
        or isinstance(
            max_iterations,
            (bool, np.bool_),
        )
        or int(max_iterations) < 1
    ):
        raise ValueError(
            "max_iterations must be a positive integer"
        )

    def state_values(
        scale: float,
    ) -> np.ndarray:
        return (
            components[:, 0] * scale * scale
            + components[:, 1] * scale
            + components[:, 2]
        )

    mean_at_zero = float(
        np.mean(state_values(0.0))
    )

    mean_at_one = float(
        np.mean(state_values(1.0))
    )

    if (
        scalar
        < min(
            mean_at_zero,
            mean_at_one,
        )
        - absolute_tolerance
        or scalar
        > max(
            mean_at_zero,
            mean_at_one,
        )
        + absolute_tolerance
    ):
        raise ValueError(
            "initial_scalar is not bracketed on [0, 1]"
        )

    lower = 0.0
    upper = 1.0
    solved_scale = None

    for _ in range(int(max_iterations)):
        midpoint = 0.5 * (
            lower + upper
        )

        midpoint_mean = float(
            np.mean(
                state_values(midpoint)
            )
        )

        if (
            abs(
                midpoint_mean
                - scalar
            )
            <= absolute_tolerance
        ):
            solved_scale = midpoint
            break

        if midpoint_mean > scalar:
            upper = midpoint
        else:
            lower = midpoint

    if solved_scale is None:
        raise ValueError(
            "scale search did not converge"
        )

    values_at_one = state_values(1.0)

    values_at_scale = state_values(
        solved_scale
    )

    standard_deviation_at_one = float(
        np.std(
            values_at_one,
            ddof=1,
        )
    )

    standard_deviation_at_scale = float(
        np.std(
            values_at_scale,
            ddof=1,
        )
    )

    if (
        not np.isfinite(
            standard_deviation_at_one
        )
        or standard_deviation_at_one <= 0.0
        or not np.isfinite(
            standard_deviation_at_scale
        )
    ):
        raise ValueError(
            "state standard deviations must be finite, "
            "with a positive value at scale 1"
        )

    adjusted_scalar = float(
        scalar
        * solved_scale
        * solved_scale
    )

    adjusted_standard_error = float(
        standard_error
        * standard_deviation_at_scale
        / standard_deviation_at_one
    )

    result = np.array(
        [
            solved_scale,
            adjusted_scalar,
            adjusted_standard_error,
        ],
        dtype=float,
    )

    if not np.all(np.isfinite(result)):
        raise ValueError(
            "state-scale result must be finite"
        )

    return result

import numpy as np


def select_state_mask(
    rg_matrix: np.ndarray,
    target_index: int,
    target_scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    rg = np.asarray(rg_matrix, dtype=float)
    scores = np.asarray(
        target_scores,
        dtype=float,
    )

    if (
        rg.ndim != 2
        or rg.shape[0] != rg.shape[1]
        or rg.shape[0] < 3
    ):
        raise ValueError(
            "rg_matrix must be square with at least three traits"
        )

    if not np.all(np.isfinite(rg)):
        raise ValueError(
            "rg_matrix must contain only finite values"
        )

    if not np.allclose(
        rg,
        rg.T,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "rg_matrix must be symmetric"
        )

    if not np.allclose(
        np.diag(rg),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "rg_matrix must have a unit diagonal"
        )

    if np.any(np.abs(rg) > 1.0 + 1e-12):
        raise ValueError(
            "correlations must lie in [-1, 1]"
        )

    try:
        np.linalg.cholesky(rg)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "rg_matrix must be positive definite"
        ) from exc

    if (
        not isinstance(
            target_index,
            (int, np.integer),
        )
        or isinstance(
            target_index,
            (bool, np.bool_),
        )
    ):
        raise ValueError(
            "target_index must be an integer"
        )

    target = int(target_index)
    n_traits = rg.shape[0]

    if target < 0 or target >= n_traits:
        raise ValueError(
            "target_index is out of range"
        )

    auxiliary = np.array(
        [
            index
            for index in range(n_traits)
            if index != target
        ],
        dtype=int,
    )

    if (
        scores.shape != (auxiliary.size,)
        or not np.all(np.isfinite(scores))
    ):
        raise ValueError(
            "target_scores must align with non-target traits"
        )

    if (
        not isinstance(
            threshold,
            (int, float, np.integer, np.floating),
        )
        or isinstance(
            threshold,
            (bool, np.bool_),
        )
    ):
        raise ValueError(
            "threshold must be a real number"
        )

    threshold_value = float(threshold)

    if (
        not np.isfinite(threshold_value)
        or threshold_value < 0.0
        or threshold_value >= 1.0
    ):
        raise ValueError(
            "threshold must lie in [0, 1)"
        )

    keep_mask = np.ones(
        n_traits,
        dtype=float,
    )

    for first_position in range(
        auxiliary.size - 1
    ):
        for second_position in range(
            first_position + 1,
            auxiliary.size,
        ):
            first_trait = int(
                auxiliary[
                    first_position
                ]
            )

            second_trait = int(
                auxiliary[
                    second_position
                ]
            )

            squared_correlation = float(
                rg[
                    first_trait,
                    second_trait,
                ]
                ** 2
            )

            if (
                squared_correlation
                > threshold_value
            ):
                first_score = float(
                    scores[
                        first_position
                    ]
                )

                second_score = float(
                    scores[
                        second_position
                    ]
                )

                if (
                    abs(
                        first_score
                        - second_score
                    )
                    <= 1e-12
                ):
                    raise ValueError(
                        "a qualifying pair has tied target scores"
                    )

                if first_score > second_score:
                    keep_mask[
                        second_trait
                    ] = 0.0
                else:
                    keep_mask[
                        first_trait
                    ] = 0.0

    keep_mask[target] = 1.0

    return keep_mask

def compute_terminal_scalar(
    rg_point: np.ndarray,
    rg_replicates: np.ndarray,
    seed: int = 123,
    sample_count: int = 200,
    proposal_limit: int = 200,
    tolerance: float = 1e-10,
    max_iterations: int = 80,
) -> float:
    current_point = np.asarray(
        rg_point,
        dtype=float,
    ).copy()

    current_replicates = np.asarray(
        rg_replicates,
        dtype=float,
    ).copy()

    if (
        current_point.ndim != 2
        or current_point.shape[0]
        != current_point.shape[1]
        or current_point.shape[0] < 2
    ):
        raise ValueError(
            "rg_point must be square with at least two traits"
        )

    expected_width = (
        current_point.shape[0]
        * (current_point.shape[0] - 1)
        // 2
    )

    if (
        current_replicates.ndim != 2
        or current_replicates.shape[0] < 2
        or current_replicates.shape[1]
        != expected_width
    ):
        raise ValueError(
            "rg_replicates has incompatible shape"
        )

    thresholds = np.array(
        [0.5, 0.4, 0.3, 0.2, 0.1],
        dtype=float,
    )

    threshold_index = 0

    def subset_packed_replicates(
        packed_replicates: np.ndarray,
        keep_mask: np.ndarray,
    ) -> np.ndarray:
        n_traits = int(keep_mask.size)

        packed_pairs = [
            (row, column)
            for column in range(n_traits - 1)
            for row in range(
                column + 1,
                n_traits,
            )
        ]

        matrices = np.repeat(
            np.eye(
                n_traits,
                dtype=float,
            )[None, :, :],
            packed_replicates.shape[0],
            axis=0,
        )

        for packed_index, (
            row,
            column,
        ) in enumerate(packed_pairs):
            matrices[
                :,
                row,
                column,
            ] = packed_replicates[
                :,
                packed_index,
            ]

            matrices[
                :,
                column,
                row,
            ] = packed_replicates[
                :,
                packed_index,
            ]

        keep_indices = np.flatnonzero(
            keep_mask > 0.5
        )

        if keep_indices.size < 2:
            raise ValueError(
                "a state transition must retain the target "
                "and at least one non-target trait"
            )

        matrices = matrices[
            :,
            keep_indices,
        ][
            :,
            :,
            keep_indices,
        ]

        retained_traits = int(
            keep_indices.size
        )

        retained_pairs = [
            (row, column)
            for column in range(
                retained_traits - 1
            )
            for row in range(
                column + 1,
                retained_traits,
            )
        ]

        return np.stack(
            [
                matrices[
                    :,
                    row,
                    column,
                ]
                for row, column
                in retained_pairs
            ],
            axis=1,
        )

    while True:
        initial_scalar = compute_state_scalar(
            current_point,
            0,
        )

        replicate_vector = compute_replicate_vector(
            current_point,
            current_replicates,
            0,
        )

        initial_scale = float(
            replicate_vector[0]
        )

        target_scores = np.asarray(
            replicate_vector[1:],
            dtype=float,
        )

        solved_state = None
        numerical_failure = False

        if initial_scale <= 0.5:
            try:
                replicate_state = (
                    construct_replicate_state(
                        current_replicates
                    )
                )

                perturbed_states = (
                    generate_perturbed_states(
                        current_point,
                        replicate_state,
                        int(sample_count),
                        int(proposal_limit),
                        int(seed),
                    )
                )

                state_components = (
                    construct_state_components(
                        current_point,
                        perturbed_states,
                        0,
                    )
                )

                solved_state = solve_state_scale(
                    initial_scalar,
                    initial_scale,
                    state_components,
                    float(tolerance),
                    int(max_iterations),
                )

            except ValueError:
                numerical_failure = True

        nonterminal = (
            initial_scale > 0.5
            or numerical_failure
            or solved_state is None
            or float(solved_state[0]) < 0.5
        )

        if not nonterminal:
            terminal_scalar = float(
                solved_state[1]
            )

            if not np.isfinite(
                terminal_scalar
            ):
                raise ValueError(
                    "terminal scalar must be finite"
                )

            return terminal_scalar

        if current_point.shape[0] < 3:
            raise ValueError(
                "nonterminal state has no non-target pair "
                "available for transition"
            )

        transitioned = False

        while threshold_index < thresholds.size:
            keep_mask = select_state_mask(
                current_point,
                0,
                target_scores,
                float(
                    thresholds[
                        threshold_index
                    ]
                ),
            )

            threshold_index += 1

            if (
                int(np.sum(keep_mask))
                < current_point.shape[0]
            ):
                retained = (
                    keep_mask > 0.5
                )

                current_point = current_point[
                    np.ix_(
                        retained,
                        retained,
                    )
                ]

                current_replicates = (
                    subset_packed_replicates(
                        current_replicates,
                        keep_mask,
                    )
                )

                transitioned = True
                break

        if not transitioned:
            raise ValueError(
                "nonterminal state cannot be changed "
                "by the remaining thresholds"
            )
SCICODE_GOLD_EOF
