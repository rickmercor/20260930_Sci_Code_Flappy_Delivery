"""
Resolve the source-matched state for each condition, apply active physiological validation, and return the largest positive target value from the surviving conditions.

The inputs describe a panel of aligned constrained reaction systems.



Each row of condition_weights supplies the strictly positive final reaction-level weights for one condition.



For every condition:



- obtain its source-matched paired component state using the preceding public operations;

- recover the aligned signed net values;

- verify consistency with the supplied steady-state, fixed-value, and one-way constraints;

- identify active coordinates using active_threshold;

- obtain the aligned scaled logarithmic ratios using gas_constant × temperature;

- apply the supplied inclusive validation intervals only at validation_indices that are active.



A resolved source state is numerically acceptable only when its maximum steady-state, fixed-value, and one-way constraint violation does not exceed 1.0 × 10^-7.



A condition is eligible for terminal ranking only when it satisfies every applicable active validation interval and its target signed value is strictly positive.



Among eligible conditions, return the largest target signed value.



If multiple eligible conditions have exactly equal largest target values, the condition with the lowest integer condition ID is selected.



The function must genuinely compose the public operations from Items 1-7. Do not substitute a separate source-specific derivation or stored condition-specific result inside this function.



constraint_matrix must be a finite non-empty two-dimensional numerical array.



condition_weights must be a finite strictly positive two-dimensional numerical array with one reaction column per constraint_matrix column and at least one condition row.

fixed_indices, one_way_indices, and validation_indices must be one-dimensional integer arrays containing valid, non-duplicated reaction indices. validation_indices must contain at least one index.

fixed_values must be finite and aligned with fixed_indices.

validation_lower_bounds and validation_upper_bounds must be finite one-dimensional arrays aligned with validation_indices, with lower bounds not exceeding upper bounds.

condition_ids must be a one-dimensional array of unique integers aligned with the condition rows.

target_index must be a valid integer reaction index.

temperature and gas_constant must be finite and strictly positive.

active_threshold must be finite and non-negative.

Raise ValueError if the public input contract is violated, if a condition has no valid source-matched state, or if no condition remains eligible for terminal ranking.

Returns
-------
Single finite float containing the largest positive target signed value among conditions that satisfy all applicable active validation requirements.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

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
    """
    Resolve a panel of constrained source-matched states and return
    the selected positive target scalar.

    Returns
    -------
    float
        Selected positive target signed value.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_resolve_condition_panel_scalar(
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
            _oracle_resolve_source_component_state(
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
        _oracle_compute_paired_component_differences(
            component_panel
        )
    )

    steady_state_residuals = (
        _oracle_compute_steady_state_residuals(
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
        _oracle_evaluate_signed_constraint_violations(
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
        _oracle_compute_activity_mask(
            signed_values,
            threshold,
        )
    )

    scaled_log_ratios = (
        _oracle_compute_scaled_log_ratios(
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
        _oracle_classify_active_interval_states(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for resolve_condition_panel_scalar."""
    return [
        {
            "setup": """import numpy as np

constraint_matrix = np.array(
    [
        [1.0, -1.0, -1.0],
    ],
    dtype=float,
)

condition_weights = np.array(
    [
        [2.0, 3.0, 1.0],
        [2.0, 1.0, 3.0],
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
    [0],
    dtype=int,
)

validation_indices = np.array(
    [1],
    dtype=int,
)

validation_lower_bounds = np.array(
    [-10.0],
    dtype=float,
)

validation_upper_bounds = np.array(
    [10.0],
    dtype=float,
)

condition_ids = np.array(
    [11, 4],
    dtype=int,
)

target_index = 1
temperature = 300.0
gas_constant = 8.314462618e-3
active_threshold = 1.0e-9
""",
            "call": "resolve_condition_panel_scalar(constraint_matrix, condition_weights, fixed_indices, fixed_values, one_way_indices, validation_indices, validation_lower_bounds, validation_upper_bounds, condition_ids, target_index, temperature, gas_constant, active_threshold)",
            "gold_call": "_oracle_resolve_condition_panel_scalar(constraint_matrix, condition_weights, fixed_indices, fixed_values, one_way_indices, validation_indices, validation_lower_bounds, validation_upper_bounds, condition_ids, target_index, temperature, gas_constant, active_threshold)",
        },
        {
            "setup": """import numpy as np

constraint_matrix = np.array(
    [
        [1.0, -1.0, -1.0],
    ],
    dtype=float,
)

condition_weights = np.array(
    [
        [2.0, 1.0, 3.0],
        [2.0, 3.0, 1.0],
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
    [0],
    dtype=int,
)

validation_indices = np.array(
    [1],
    dtype=int,
)

validation_lower_bounds = np.array(
    [-10.0],
    dtype=float,
)

validation_upper_bounds = np.array(
    [10.0],
    dtype=float,
)

condition_ids = np.array(
    [4, 11],
    dtype=int,
)

target_index = 1
temperature = 300.0
gas_constant = 8.314462618e-3
active_threshold = 1.0e-9
""",
            "call": "resolve_condition_panel_scalar(constraint_matrix, condition_weights, fixed_indices, fixed_values, one_way_indices, validation_indices, validation_lower_bounds, validation_upper_bounds, condition_ids, target_index, temperature, gas_constant, active_threshold)",
            "gold_call": "_oracle_resolve_condition_panel_scalar(constraint_matrix, condition_weights, fixed_indices, fixed_values, one_way_indices, validation_indices, validation_lower_bounds, validation_upper_bounds, condition_ids, target_index, temperature, gas_constant, active_threshold)",
        },
        {
            "setup": """import numpy as np

constraint_matrix = np.array(
    [
        [1.0, -1.0, -1.0],
    ],
    dtype=float,
)

condition_weights = np.array(
    [
        [2.0, 3.0, 1.0],
        [2.0, 1.0, 3.0],
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
    [0],
    dtype=int,
)

validation_indices = np.array(
    [1],
    dtype=int,
)

validation_lower_bounds = np.array(
    [-10.0],
    dtype=float,
)

validation_upper_bounds = np.array(
    [10.0],
    dtype=float,
)

condition_ids = np.array(
    [11, 4],
    dtype=int,
)

target_index = 1
temperature = 300.0
gas_constant = 8.314462618e-3
active_threshold = 1.0e-9

permutation = np.array(
    [2, 0, 1],
    dtype=int,
)

new_index = np.empty(
    permutation.size,
    dtype=int,
)

new_index[
    permutation
] = np.arange(
    permutation.size,
    dtype=int,
)

constraint_matrix = constraint_matrix[
    :,
    permutation,
]

condition_weights = condition_weights[
    :,
    permutation,
]

fixed_indices = new_index[
    fixed_indices
]

one_way_indices = new_index[
    one_way_indices
]

validation_indices = new_index[
    validation_indices
]

target_index = int(
    new_index[
        target_index
    ]
)
""",
            "call": "resolve_condition_panel_scalar(constraint_matrix, condition_weights, fixed_indices, fixed_values, one_way_indices, validation_indices, validation_lower_bounds, validation_upper_bounds, condition_ids, target_index, temperature, gas_constant, active_threshold)",
            "gold_call": "_oracle_resolve_condition_panel_scalar(constraint_matrix, condition_weights, fixed_indices, fixed_values, one_way_indices, validation_indices, validation_lower_bounds, validation_upper_bounds, condition_ids, target_index, temperature, gas_constant, active_threshold)",
        },
        {
            "setup": """import numpy as np

constraint_matrix = np.array(
    [
        [1.0, -1.0, -1.0],
    ],
    dtype=float,
)

condition_weights = np.array(
    [
        [2.0, 3.0, 1.0],
        [2.0, 1.0, 3.0],
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
    [0],
    dtype=int,
)

validation_indices = np.array(
    [1],
    dtype=int,
)

validation_lower_bounds = np.array(
    [100.0],
    dtype=float,
)

validation_upper_bounds = np.array(
    [101.0],
    dtype=float,
)

condition_ids = np.array(
    [11, 4],
    dtype=int,
)

target_index = 1
temperature = 300.0
gas_constant = 8.314462618e-3
active_threshold = 0.0

def candidate_wrapper():
    try:
        resolve_condition_panel_scalar(
            constraint_matrix,
            condition_weights,
            fixed_indices,
            fixed_values,
            one_way_indices,
            validation_indices,
            validation_lower_bounds,
            validation_upper_bounds,
            condition_ids,
            target_index,
            temperature,
            gas_constant,
            active_threshold,
        )
    except ValueError:
        return 1.0

    return 0.0

def gold_wrapper():
    try:
        _oracle_resolve_condition_panel_scalar(
            constraint_matrix,
            condition_weights,
            fixed_indices,
            fixed_values,
            one_way_indices,
            validation_indices,
            validation_lower_bounds,
            validation_upper_bounds,
            condition_ids,
            target_index,
            temperature,
            gas_constant,
            active_threshold,
        )
    except ValueError:
        return 1.0

    return 0.0
""",
            "call": "candidate_wrapper()",
            "gold_call": "gold_wrapper()",
        },
        {
            "setup": """import numpy as np

constraint_matrix = np.array(
    [
        [1.0, -1.0, -1.0],
    ],
    dtype=float,
)

condition_weights = np.array(
    [
        [2.0, 3.0, 1.0],
        [2.0, 1.0, 3.0],
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
    [0],
    dtype=int,
)

validation_indices = np.array(
    [1],
    dtype=int,
)

validation_lower_bounds = np.array(
    [100.0],
    dtype=float,
)

validation_upper_bounds = np.array(
    [101.0],
    dtype=float,
)

condition_ids = np.array(
    [11, 4],
    dtype=int,
)

target_index = 1
temperature = 300.0
gas_constant = 8.314462618e-3

# Every possible signed value in this fixture has magnitude below 10,
# so the validation coordinate is inactive even though its interval is
# deliberately incompatible with its finite transformed value.
active_threshold = 10.0
""",
            "call": "resolve_condition_panel_scalar(constraint_matrix, condition_weights, fixed_indices, fixed_values, one_way_indices, validation_indices, validation_lower_bounds, validation_upper_bounds, condition_ids, target_index, temperature, gas_constant, active_threshold)",
            "gold_call": "_oracle_resolve_condition_panel_scalar(constraint_matrix, condition_weights, fixed_indices, fixed_values, one_way_indices, validation_indices, validation_lower_bounds, validation_upper_bounds, condition_ids, target_index, temperature, gas_constant, active_threshold)",
        },
    ]
