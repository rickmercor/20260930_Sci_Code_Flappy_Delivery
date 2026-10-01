"""
Resolve the unique source-matched paired component state under aligned steady-state, fixed-value, and one-way signed constraints.

The supplied inputs define one constrained reaction system.

constraint_matrix contains the aligned steady-state linear constraints.

reaction_weights contains the strictly positive final reaction-level weights for this state.

fixed_indices and fixed_values specify exact signed net values.

one_way_indices identifies coordinates whose signed net values may not be negative.

The signed net value associated with reaction j is the first returned component minus the second returned component for that reaction.

Use the quantitative treatment identified from the task's matching recent primary source to determine the unique paired component state consistent with these inputs.

Return one row per reaction, preserving the input reaction order, with the two source-defined components in their source-defined order.

constraint_matrix must be a finite non-empty two-dimensional numerical array. reaction_weights must be a finite strictly positive one-dimensional numerical array aligned with its columns.

fixed_indices and one_way_indices must be one-dimensional integer arrays containing valid, non-duplicated reaction indices. fixed_values must be a finite one-dimensional numerical array aligned with fixed_indices.

Raise ValueError if the public input contract is violated or if no constrained source-matched state exists.

Returns
-------
2D NumPy float array of shape (n_reactions, 2) containing the source-matched paired component state in aligned reaction order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def resolve_source_component_state(
    constraint_matrix: np.ndarray,
    reaction_weights: np.ndarray,
    fixed_indices: np.ndarray,
    fixed_values: np.ndarray,
    one_way_indices: np.ndarray,
) -> np.ndarray:
    """
    Resolve the source-matched paired component state.

    Returns
    -------
    np.ndarray
        Float array of shape (n_reactions, 2), preserving reaction order.
    """
    return np.empty((0, 2), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_resolve_source_component_state(
    constraint_matrix: np.ndarray,
    reaction_weights: np.ndarray,
    fixed_indices: np.ndarray,
    fixed_values: np.ndarray,
    one_way_indices: np.ndarray,
) -> np.ndarray:
    import itertools
    import numpy as np

    matrix = np.asarray(constraint_matrix, dtype=float)
    weights = np.asarray(reaction_weights, dtype=float)
    raw_fixed = np.asarray(fixed_indices)
    values = np.asarray(fixed_values, dtype=float)
    raw_one_way = np.asarray(one_way_indices)

    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(
            "constraint_matrix must be a non-empty two-dimensional array"
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            "constraint_matrix must contain only finite values"
        )

    n_reactions = matrix.shape[1]

    if weights.ndim != 1 or weights.size != n_reactions:
        raise ValueError(
            "reaction_weights must align with constraint_matrix columns"
        )

    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError(
            "reaction_weights must be finite and strictly positive"
        )

    def parse_indices(
        raw: np.ndarray,
        name: str,
    ) -> np.ndarray:
        if raw.ndim != 1:
            raise ValueError(
                f"{name} must be one-dimensional"
            )

        if not np.issubdtype(
            raw.dtype,
            np.integer,
        ):
            raise ValueError(
                f"{name} must contain integers"
            )

        out = raw.astype(
            int,
            copy=False,
        )

        if (
            np.any(out < 0)
            or np.any(out >= n_reactions)
        ):
            raise ValueError(
                f"{name} contains an out-of-range index"
            )

        if np.unique(out).size != out.size:
            raise ValueError(
                f"{name} must not contain duplicates"
            )

        return out

    fixed = parse_indices(
        raw_fixed,
        "fixed_indices",
    )

    one_way = parse_indices(
        raw_one_way,
        "one_way_indices",
    )

    if (
        values.ndim != 1
        or values.size != fixed.size
    ):
        raise ValueError(
            "fixed_values must align with fixed_indices"
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "fixed_values must contain only finite values"
        )

    rows = [
        matrix
    ]

    rhs = [
        np.zeros(
            matrix.shape[0],
            dtype=float,
        )
    ]

    if fixed.size:
        rows.append(
            np.eye(
                n_reactions,
                dtype=float,
            )[fixed]
        )

        rhs.append(
            values
        )

    base_matrix = np.vstack(
        rows
    )

    base_values = np.concatenate(
        rhs
    )

    def null_basis(
        a: np.ndarray,
    ) -> np.ndarray:
        _, singular_values, vh = np.linalg.svd(
            a,
            full_matrices=True,
        )

        if singular_values.size == 0:
            return np.eye(
                a.shape[1],
                dtype=float,
            )

        tolerance = (
            max(a.shape)
            * np.finfo(float).eps
            * singular_values[0]
        )

        rank = int(
            np.sum(
                singular_values
                > tolerance
            )
        )

        return vh[
            rank:
        ].T.copy()

    def paired_state(
        signed_net: np.ndarray,
    ):
        scaled = (
            np.e
            * signed_net
            / (
                2.0
                * weights
            )
        )

        mu = np.arcsinh(
            scaled
        )

        first = (
            weights
            / np.e
        ) * np.exp(
            mu
        )

        second = (
            weights
            / np.e
        ) * np.exp(
            -mu
        )

        return (
            mu,
            first,
            second,
        )

    def objective(
        signed_net: np.ndarray,
    ) -> float:
        (
            _,
            first,
            second,
        ) = paired_state(
            signed_net
        )

        return float(
            np.sum(
                first
                * (
                    np.log(first)
                    - np.log(weights)
                )
                + second
                * (
                    np.log(second)
                    - np.log(weights)
                )
            )
        )

    def solve_equalities(
        a: np.ndarray,
        b: np.ndarray,
    ):
        particular = np.linalg.lstsq(
            a,
            b,
            rcond=None,
        )[0]

        if (
            np.max(
                np.abs(
                    a
                    @ particular
                    - b
                )
            )
            > 1.0e-10
        ):
            return None

        basis = null_basis(
            a
        )

        if basis.shape[1] == 0:
            return particular

        coordinates = np.zeros(
            basis.shape[1],
            dtype=float,
        )

        for _ in range(200):
            signed_net = (
                particular
                + basis
                @ coordinates
            )

            scaled = (
                np.e
                * signed_net
                / (
                    2.0
                    * weights
                )
            )

            mu = np.arcsinh(
                scaled
            )

            gradient = (
                basis.T
                @ mu
            )

            gradient_norm = float(
                np.max(
                    np.abs(
                        gradient
                    )
                )
            )

            if gradient_norm <= 1.0e-12:
                return signed_net

            curvature = (
                (
                    np.e
                    / (
                        2.0
                        * weights
                    )
                )
                / np.sqrt(
                    1.0
                    + scaled
                    * scaled
                )
            )

            hessian = (
                basis.T
                @ (
                    curvature[:, None]
                    * basis
                )
            )

            try:
                step = np.linalg.solve(
                    hessian,
                    gradient,
                )
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(
                    hessian,
                    gradient,
                    rcond=None,
                )[0]

            step_norm = float(
                np.max(
                    np.abs(
                        step
                    )
                )
            )

            if step_norm <= 1.0e-10:
                return signed_net

            current_value = objective(
                signed_net
            )

            directional_derivative = float(
                gradient
                @ step
            )

            objective_roundoff = (
                32.0
                * np.finfo(float).eps
                * max(
                    1.0,
                    abs(current_value),
                )
            )

            accepted = False
            scale = 1.0

            for _ in range(60):
                candidate_coordinates = (
                    coordinates
                    - scale
                    * step
                )

                candidate_net = (
                    particular
                    + basis
                    @ candidate_coordinates
                )

                candidate_value = objective(
                    candidate_net
                )

                candidate_scaled = (
                    np.e
                    * candidate_net
                    / (
                        2.0
                        * weights
                    )
                )

                candidate_gradient = (
                    basis.T
                    @ np.arcsinh(
                        candidate_scaled
                    )
                )

                candidate_gradient_norm = float(
                    np.max(
                        np.abs(
                            candidate_gradient
                        )
                    )
                )

                armijo_ok = (
                    np.isfinite(
                        candidate_value
                    )
                    and candidate_value
                    <= (
                        current_value
                        - 1.0e-4
                        * scale
                        * directional_derivative
                    )
                )

                roundoff_progress_ok = (
                    np.isfinite(
                        candidate_value
                    )
                    and candidate_value
                    <= (
                        current_value
                        + objective_roundoff
                    )
                    and candidate_gradient_norm
                    < gradient_norm
                )

                if (
                    armijo_ok
                    or roundoff_progress_ok
                ):
                    coordinates = (
                        candidate_coordinates
                    )

                    accepted = True
                    break

                scale *= 0.5

            if not accepted:
                if (
                    gradient_norm <= 1.0e-10
                    or step_norm <= 1.0e-10
                ):
                    return signed_net

                return None

        signed_net = (
            particular
            + basis
            @ coordinates
        )

        final_scaled = (
            np.e
            * signed_net
            / (
                2.0
                * weights
            )
        )

        final_gradient = (
            basis.T
            @ np.arcsinh(
                final_scaled
            )
        )

        if (
            np.max(
                np.abs(
                    final_gradient
                )
            )
            <= 1.0e-10
        ):
            return signed_net

        return None

    best_value = None
    best_net = None

    for subset_size in range(
        one_way.size + 1
    ):
        for active_subset in itertools.combinations(
            one_way.tolist(),
            subset_size,
        ):
            rows = [
                base_matrix
            ]

            rhs = [
                base_values
            ]

            if active_subset:
                active = np.asarray(
                    active_subset,
                    dtype=int,
                )

                rows.append(
                    np.eye(
                        n_reactions,
                        dtype=float,
                    )[active]
                )

                rhs.append(
                    np.zeros(
                        active.size,
                        dtype=float,
                    )
                )

            equality_matrix = np.vstack(
                rows
            )

            equality_values = np.concatenate(
                rhs
            )

            signed_net = solve_equalities(
                equality_matrix,
                equality_values,
            )

            if signed_net is None:
                continue

            if (
                np.max(
                    np.abs(
                        base_matrix
                        @ signed_net
                        - base_values
                    )
                )
                > 5.0e-9
            ):
                continue

            if (
                one_way.size
                and np.any(
                    signed_net[
                        one_way
                    ]
                    < -5.0e-9
                )
            ):
                continue

            value = objective(
                signed_net
            )

            if (
                best_value is None
                or value < best_value
            ):
                best_value = value
                best_net = signed_net

    if best_net is None:
        raise ValueError(
            "no constrained source-matched state exists"
        )

    (
        _,
        first_component,
        second_component,
    ) = paired_state(
        best_net
    )

    result = np.column_stack(
        (
            first_component,
            second_component,
        )
    )

    if (
        not np.all(
            np.isfinite(result)
        )
        or np.any(
            result <= 0.0
        )
    ):
        raise ValueError(
            "resolved source-matched component state must be finite and strictly positive"
        )

    return result.astype(
        float,
        copy=False,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for resolve_source_component_state."""
    return [
        {
            "setup": """import numpy as np

constraint_matrix = np.array(
    [
        [1.0, -1.0, -1.0],
    ],
    dtype=float,
)

reaction_weights = np.array(
    [2.0, 3.0, 1.0],
    dtype=float,
)

fixed_indices = np.array(
    [0],
    dtype=int,
)

fixed_values = np.array(
    [5.0],
    dtype=float,
)

one_way_indices = np.array(
    [],
    dtype=int,
)
""",
            "call": "resolve_source_component_state(constraint_matrix.copy(), reaction_weights.copy(), fixed_indices.copy(), fixed_values.copy(), one_way_indices.copy())",
            "gold_call": "_oracle_resolve_source_component_state(constraint_matrix.copy(), reaction_weights.copy(), fixed_indices.copy(), fixed_values.copy(), one_way_indices.copy())",
        },
        {
            "setup": """import numpy as np

constraint_matrix = np.array(
    [
        [ 1.0, 1.0, 1.0, -1.0, 0.0],
        [-1.0, 0.0, 1.0, -1.0, 1.0],
    ],
    dtype=float,
)

reaction_weights = np.array(
    [
        2.7666427900,
        0.4197447900,
        3.8080718700,
        3.2982829700,
        0.4690827600,
    ],
    dtype=float,
)

fixed_indices = np.array(
    [0],
    dtype=int,
)

fixed_values = np.array(
    [5.0],
    dtype=float,
)

one_way_indices = np.array(
    [0, 2, 3],
    dtype=int,
)
""",
            "call": "resolve_source_component_state(constraint_matrix.copy(), reaction_weights.copy(), fixed_indices.copy(), fixed_values.copy(), one_way_indices.copy())",
            "gold_call": "_oracle_resolve_source_component_state(constraint_matrix.copy(), reaction_weights.copy(), fixed_indices.copy(), fixed_values.copy(), one_way_indices.copy())",
        },
        {
            "setup": """import numpy as np

constraint_matrix = np.array(
    [
        [ 1.0, 1.0, 1.0, -1.0, 0.0],
        [-1.0, 0.0, 1.0, -1.0, 1.0],
    ],
    dtype=float,
)

reaction_weights = np.array(
    [
        0.5090449200,
        2.2603373600,
        3.7917407800,
        2.1090312900,
        1.0760028300,
    ],
    dtype=float,
)

fixed_indices = np.array(
    [0],
    dtype=int,
)

fixed_values = np.array(
    [5.0],
    dtype=float,
)

one_way_indices = np.array(
    [0, 2, 3],
    dtype=int,
)
""",
            "call": "resolve_source_component_state(constraint_matrix.copy(), reaction_weights.copy(), fixed_indices.copy(), fixed_values.copy(), one_way_indices.copy())",
            "gold_call": "_oracle_resolve_source_component_state(constraint_matrix.copy(), reaction_weights.copy(), fixed_indices.copy(), fixed_values.copy(), one_way_indices.copy())",
        },
        {
            "setup": """import numpy as np

constraint_matrix = np.array(
    [
        [ 1.0, 1.0, 1.0, -1.0, 0.0],
        [-1.0, 0.0, 1.0, -1.0, 1.0],
    ],
    dtype=float,
)

reaction_weights = np.array(
    [
        2.7666427900,
        0.4197447900,
        3.8080718700,
        3.2982829700,
        0.4690827600,
    ],
    dtype=float,
)

fixed_indices = np.array(
    [0],
    dtype=int,
)

fixed_values = np.array(
    [5.0],
    dtype=float,
)

one_way_indices = np.array(
    [0, 2, 3],
    dtype=int,
)

permutation = np.array(
    [3, 0, 4, 2, 1],
    dtype=int,
)

new_index = np.empty(
    permutation.size,
    dtype=int,
)

new_index[permutation] = np.arange(
    permutation.size,
    dtype=int,
)

constraint_matrix = constraint_matrix[
    :,
    permutation,
]

reaction_weights = reaction_weights[
    permutation
]

fixed_indices = new_index[
    fixed_indices
]

one_way_indices = new_index[
    one_way_indices
]
""",
            "call": "resolve_source_component_state(constraint_matrix.copy(), reaction_weights.copy(), fixed_indices.copy(), fixed_values.copy(), one_way_indices.copy())",
            "gold_call": "_oracle_resolve_source_component_state(constraint_matrix.copy(), reaction_weights.copy(), fixed_indices.copy(), fixed_values.copy(), one_way_indices.copy())",
        },
    ]
