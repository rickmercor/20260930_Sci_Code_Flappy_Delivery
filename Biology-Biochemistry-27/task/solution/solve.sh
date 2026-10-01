#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def compute_steady_state_residuals(
    stoichiometric_matrix: np.ndarray,
    net_fluxes: np.ndarray,
) -> np.ndarray:
    import numpy as np

    S = np.asarray(stoichiometric_matrix, dtype=float)
    V = np.asarray(net_fluxes, dtype=float)

    if S.ndim != 2 or S.shape[0] < 1 or S.shape[1] < 1:
        raise ValueError(
            "stoichiometric_matrix must be a non-empty two-dimensional array"
        )

    if V.ndim != 2 or V.shape[0] < 1 or V.shape[1] < 1:
        raise ValueError(
            "net_fluxes must be a non-empty two-dimensional array"
        )

    if V.shape[1] != S.shape[1]:
        raise ValueError(
            "stoichiometric_matrix and net_fluxes must have the same reaction dimension"
        )

    if not np.all(np.isfinite(S)):
        raise ValueError(
            "stoichiometric_matrix must contain only finite values"
        )

    if not np.all(np.isfinite(V)):
        raise ValueError(
            "net_fluxes must contain only finite values"
        )

    residuals = V @ S.T

    if not np.all(np.isfinite(residuals)):
        raise ValueError("computed residuals must be finite")

    return residuals.astype(float, copy=False)

def compute_paired_component_differences(
    component_pairs: np.ndarray,
) -> np.ndarray:
    import numpy as np

    components = np.asarray(
        component_pairs,
        dtype=float,
    )

    if (
        components.ndim != 3
        or components.shape[0] < 1
        or components.shape[1] < 1
        or components.shape[2] != 2
    ):
        raise ValueError(
            "component_pairs must be a non-empty three-dimensional array with final dimension 2"
        )

    if not np.all(
        np.isfinite(components)
    ):
        raise ValueError(
            "component_pairs must contain only finite values"
        )

    result = (
        components[..., 0]
        - components[..., 1]
    )

    if not np.all(
        np.isfinite(result)
    ):
        raise ValueError(
            "computed signed differences must be finite"
        )

    return result.astype(
        float,
        copy=False,
    )

def compute_activity_mask(
    signed_values: np.ndarray,
    active_threshold: float,
) -> np.ndarray:
    import numpy as np

    values = np.asarray(
        signed_values,
        dtype=float,
    )

    if (
        values.ndim != 2
        or values.shape[0] < 1
        or values.shape[1] < 1
    ):
        raise ValueError(
            "signed_values must be a non-empty two-dimensional array"
        )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "signed_values must contain only finite values"
        )

    threshold = float(
        active_threshold
    )

    if (
        not np.isfinite(threshold)
        or threshold < 0.0
    ):
        raise ValueError(
            "active_threshold must be finite and non-negative"
        )

    return (
        np.abs(values)
        > threshold
    )

def evaluate_signed_constraint_violations(
    signed_values: np.ndarray,
    constraint_matrix: np.ndarray,
    fixed_indices: np.ndarray,
    fixed_values: np.ndarray,
    one_way_indices: np.ndarray,
) -> np.ndarray:
    import numpy as np

    values = np.asarray(
        signed_values,
        dtype=float,
    )

    matrix = np.asarray(
        constraint_matrix,
        dtype=float,
    )

    raw_fixed = np.asarray(
        fixed_indices
    )

    fixed_values = np.asarray(
        fixed_values,
        dtype=float,
    )

    raw_one_way = np.asarray(
        one_way_indices
    )

    if (
        values.ndim != 2
        or values.shape[0] < 1
        or values.shape[1] < 1
    ):
        raise ValueError(
            "signed_values must be a non-empty two-dimensional array"
        )

    if (
        matrix.ndim != 2
        or matrix.shape[0] < 1
        or matrix.shape[1] != values.shape[1]
    ):
        raise ValueError(
            "constraint_matrix must be non-empty and align with signed_values columns"
        )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "signed_values must contain only finite values"
        )

    if not np.all(
        np.isfinite(matrix)
    ):
        raise ValueError(
            "constraint_matrix must contain only finite values"
        )

    n_variables = values.shape[1]

    def _parse_indices(
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

        indices = raw.astype(
            int,
            copy=False,
        )

        if (
            np.any(indices < 0)
            or np.any(indices >= n_variables)
        ):
            raise ValueError(
                f"{name} contains an out-of-range index"
            )

        if (
            np.unique(indices).size
            != indices.size
        ):
            raise ValueError(
                f"{name} must not contain duplicates"
            )

        return indices

    fixed = _parse_indices(
        raw_fixed,
        "fixed_indices",
    )

    one_way = _parse_indices(
        raw_one_way,
        "one_way_indices",
    )

    if (
        fixed_values.ndim != 1
        or fixed_values.size != fixed.size
    ):
        raise ValueError(
            "fixed_values must align with fixed_indices"
        )

    if not np.all(
        np.isfinite(fixed_values)
    ):
        raise ValueError(
            "fixed_values must contain only finite values"
        )

    linear_residual = (
        values @ matrix.T
    )

    maximum_linear = np.max(
        np.abs(linear_residual),
        axis=1,
    )

    maximum_fixed = np.zeros(
        values.shape[0],
        dtype=float,
    )

    if fixed.size:
        maximum_fixed = np.max(
            np.abs(
                values[:, fixed]
                - fixed_values.reshape(
                    1,
                    -1,
                )
            ),
            axis=1,
        )

    maximum_one_way = np.zeros(
        values.shape[0],
        dtype=float,
    )

    if one_way.size:
        maximum_one_way = np.max(
            np.maximum(
                -values[:, one_way],
                0.0,
            ),
            axis=1,
        )

    result = np.column_stack(
        (
            maximum_linear,
            maximum_fixed,
            maximum_one_way,
        )
    )

    return result.astype(
        float,
        copy=False,
    )

def resolve_source_component_state(
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

def compute_scaled_log_ratios(
    component_pairs: np.ndarray,
    scale: float,
) -> np.ndarray:
    import numpy as np

    components = np.asarray(
        component_pairs,
        dtype=float,
    )

    if (
        components.ndim != 3
        or components.shape[0] < 1
        or components.shape[1] < 1
        or components.shape[2] != 2
    ):
        raise ValueError(
            "component_pairs must be a non-empty three-dimensional array with final dimension 2"
        )

    if not np.all(
        np.isfinite(components)
    ):
        raise ValueError(
            "component_pairs must contain only finite values"
        )

    if np.any(
        components <= 0.0
    ):
        raise ValueError(
            "component_pairs must contain only strictly positive values"
        )

    value_scale = float(
        scale
    )

    if (
        not np.isfinite(value_scale)
        or value_scale <= 0.0
    ):
        raise ValueError(
            "scale must be finite and strictly positive"
        )

    result = (
        value_scale
        * (
            np.log(
                components[..., 1]
            )
            - np.log(
                components[..., 0]
            )
        )
    )

    if not np.all(
        np.isfinite(result)
    ):
        raise ValueError(
            "computed scaled log ratios must be finite"
        )

    return result.astype(
        float,
        copy=False,
    )

def classify_active_interval_states(
    validation_values: np.ndarray,
    active_mask: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    import numpy as np

    values = np.asarray(
        validation_values,
        dtype=float,
    )

    mask = np.asarray(
        active_mask
    )

    lower = np.asarray(
        lower_bounds,
        dtype=float,
    )

    upper = np.asarray(
        upper_bounds,
        dtype=float,
    )

    if (
        values.ndim != 2
        or values.shape[0] < 1
        or values.shape[1] < 1
    ):
        raise ValueError(
            "validation_values must be a non-empty two-dimensional array"
        )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "validation_values must contain only finite values"
        )

    if mask.dtype != np.bool_:
        raise ValueError(
            "active_mask must contain Boolean values"
        )

    if mask.shape != values.shape:
        raise ValueError(
            "active_mask must have the same shape as validation_values"
        )

    n_validation = values.shape[1]

    if (
        lower.ndim != 1
        or lower.size != n_validation
    ):
        raise ValueError(
            "lower_bounds must align with validation columns"
        )

    if (
        upper.ndim != 1
        or upper.size != n_validation
    ):
        raise ValueError(
            "upper_bounds must align with validation columns"
        )

    if (
        not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
    ):
        raise ValueError(
            "bounds must contain only finite values"
        )

    if np.any(
        lower > upper
    ):
        raise ValueError(
            "lower_bounds must not exceed upper_bounds"
        )

    within = (
        (values >= lower.reshape(1, -1))
        & (values <= upper.reshape(1, -1))
    )

    result = np.all(
        (~mask) | within,
        axis=1,
    )

    return result.astype(
        bool,
        copy=False,
    )

def resolve_condition_panel_scalar(
    constraint_matrix: np.ndarray,
    condition_weights: np.ndarray,
    fixed_indices: np.ndarray,
    fixed_values: np.ndarray,
    one_way_indices: np.ndarray,
    validation_indices: np.ndarray,
    validation_lower_bounds: np.ndarray,
    validation_upper_bounds: np.ndarray,
    condition_ids: np.ndarray,
    target_index: int,
    temperature: float,
    gas_constant: float,
    active_threshold: float,
) -> float:
    import numpy as np

    matrix = np.asarray(
        constraint_matrix,
        dtype=float,
    )

    weights = np.asarray(
        condition_weights,
        dtype=float,
    )

    raw_fixed = np.asarray(
        fixed_indices
    )

    fixed_values = np.asarray(
        fixed_values,
        dtype=float,
    )

    raw_one_way = np.asarray(
        one_way_indices
    )

    raw_validation = np.asarray(
        validation_indices
    )

    lower_bounds = np.asarray(
        validation_lower_bounds,
        dtype=float,
    )

    upper_bounds = np.asarray(
        validation_upper_bounds,
        dtype=float,
    )

    raw_ids = np.asarray(
        condition_ids
    )

    if (
        matrix.ndim != 2
        or matrix.shape[0] < 1
        or matrix.shape[1] < 1
    ):
        raise ValueError(
            "constraint_matrix must be a non-empty two-dimensional array"
        )

    if not np.all(
        np.isfinite(matrix)
    ):
        raise ValueError(
            "constraint_matrix must contain only finite values"
        )

    n_reactions = matrix.shape[1]

    if (
        weights.ndim != 2
        or weights.shape[0] < 1
        or weights.shape[1] != n_reactions
    ):
        raise ValueError(
            "condition_weights must align with constraint_matrix columns"
        )

    if (
        not np.all(
            np.isfinite(weights)
        )
        or np.any(
            weights <= 0.0
        )
    ):
        raise ValueError(
            "condition_weights must be finite and strictly positive"
        )

    def _parse_indices(
        raw: np.ndarray,
        name: str,
        require_nonempty: bool = False,
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

        indices = raw.astype(
            int,
            copy=False,
        )

        if (
            require_nonempty
            and indices.size < 1
        ):
            raise ValueError(
                f"{name} must be non-empty"
            )

        if (
            np.any(indices < 0)
            or np.any(
                indices >= n_reactions
            )
        ):
            raise ValueError(
                f"{name} contains an out-of-range index"
            )

        if (
            np.unique(indices).size
            != indices.size
        ):
            raise ValueError(
                f"{name} must not contain duplicates"
            )

        return indices

    fixed = _parse_indices(
        raw_fixed,
        "fixed_indices",
    )

    one_way = _parse_indices(
        raw_one_way,
        "one_way_indices",
    )

    validation = _parse_indices(
        raw_validation,
        "validation_indices",
        require_nonempty=True,
    )

    if (
        fixed_values.ndim != 1
        or fixed_values.size != fixed.size
    ):
        raise ValueError(
            "fixed_values must align with fixed_indices"
        )

    if not np.all(
        np.isfinite(fixed_values)
    ):
        raise ValueError(
            "fixed_values must contain only finite values"
        )

    if (
        lower_bounds.ndim != 1
        or upper_bounds.ndim != 1
        or lower_bounds.size != validation.size
        or upper_bounds.size != validation.size
    ):
        raise ValueError(
            "validation bounds must align with validation_indices"
        )

    if (
        not np.all(
            np.isfinite(lower_bounds)
        )
        or not np.all(
            np.isfinite(upper_bounds)
        )
    ):
        raise ValueError(
            "validation bounds must contain only finite values"
        )

    if np.any(
        lower_bounds > upper_bounds
    ):
        raise ValueError(
            "validation lower bounds must not exceed upper bounds"
        )

    if (
        raw_ids.ndim != 1
        or raw_ids.size != weights.shape[0]
    ):
        raise ValueError(
            "condition_ids must align with condition_weights rows"
        )

    if not np.issubdtype(
        raw_ids.dtype,
        np.integer,
    ):
        raise ValueError(
            "condition_ids must contain integers"
        )

    ids = raw_ids.astype(
        int,
        copy=False,
    )

    if (
        np.unique(ids).size
        != ids.size
    ):
        raise ValueError(
            "condition_ids must be unique"
        )

    if (
        isinstance(
            target_index,
            (bool, np.bool_),
        )
        or not isinstance(
            target_index,
            (int, np.integer),
        )
    ):
        raise ValueError(
            "target_index must be an integer"
        )

    target = int(
        target_index
    )

    if (
        target < 0
        or target >= n_reactions
    ):
        raise ValueError(
            "target_index is out of range"
        )

    temperature_value = float(
        temperature
    )

    gas_constant_value = float(
        gas_constant
    )

    threshold = float(
        active_threshold
    )

    if (
        not np.isfinite(
            temperature_value
        )
        or temperature_value <= 0.0
    ):
        raise ValueError(
            "temperature must be finite and strictly positive"
        )

    if (
        not np.isfinite(
            gas_constant_value
        )
        or gas_constant_value <= 0.0
    ):
        raise ValueError(
            "gas_constant must be finite and strictly positive"
        )

    if (
        not np.isfinite(
            threshold
        )
        or threshold < 0.0
    ):
        raise ValueError(
            "active_threshold must be finite and non-negative"
        )

    paired_states = []

    for reaction_weights in weights:
        paired_state = (
            resolve_source_component_state(
                matrix,
                reaction_weights,
                fixed,
                fixed_values,
                one_way,
            )
        )

        paired_states.append(
            paired_state
        )

    component_panel = np.stack(
        paired_states,
        axis=0,
    )

    signed_values = (
        compute_paired_component_differences(
            component_panel
        )
    )

    steady_state_residuals = (
        compute_steady_state_residuals(
            matrix,
            signed_values,
        )
    )

    if (
        np.max(
            np.abs(
                steady_state_residuals
            )
        )
        > 1.0e-7
    ):
        raise ValueError(
            "resolved source state violates steady-state constraints"
        )

    constraint_violations = (
        evaluate_signed_constraint_violations(
            signed_values,
            matrix,
            fixed,
            fixed_values,
            one_way,
        )
    )

    if (
        np.max(
            constraint_violations
        )
        > 1.0e-7
    ):
        raise ValueError(
            "resolved source state violates supplied constraints"
        )

    activity_mask = (
        compute_activity_mask(
            signed_values,
            threshold,
        )
    )

    scaled_log_ratios = (
        compute_scaled_log_ratios(
            component_panel,
            gas_constant_value
            * temperature_value,
        )
    )

    validation_values = (
        scaled_log_ratios[
            :,
            validation,
        ]
    )

    validation_activity = (
        activity_mask[
            :,
            validation,
        ]
    )

    admissible = (
        classify_active_interval_states(
            validation_values,
            validation_activity,
            lower_bounds,
            upper_bounds,
        )
    )

    target_values = (
        signed_values[
            :,
            target,
        ]
    )

    eligible = (
        admissible
        & (
            target_values > 0.0
        )
    )

    if not np.any(
        eligible
    ):
        raise ValueError(
            "no condition satisfies the terminal selection requirements"
        )

    maximum = np.max(
        target_values[
            eligible
        ]
    )

    tied_indices = np.flatnonzero(
        eligible
        & (
            target_values
            == maximum
        )
    )

    selected_index = tied_indices[
        np.argmin(
            ids[
                tied_indices
            ]
        )
    ]

    result = float(
        target_values[
            selected_index
        ]
    )

    if not np.isfinite(
        result
    ):
        raise ValueError(
            "selected target value must be finite"
        )

    return result
SCICODE_GOLD_EOF
