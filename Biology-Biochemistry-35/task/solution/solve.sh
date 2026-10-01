#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def compute_indexed_differences(
    profile_values,
    difference_pairs,
):
    """Reference implementation for compute_indexed_differences."""
    import numpy as np

    values = np.asarray(
        profile_values,
        dtype=float,
    )

    pairs = np.asarray(
        difference_pairs,
    )

    if values.ndim != 2:
        raise ValueError(
            "profile_values must be a two-dimensional array"
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "profile_values must contain only finite values"
        )

    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(
            "difference_pairs must have shape (m, 2)"
        )

    if not np.issubdtype(
        pairs.dtype,
        np.integer,
    ):
        raise ValueError(
            "difference_pairs must contain integer indices"
        )

    if (
        np.any(pairs < 0)
        or np.any(pairs >= values.shape[1])
    ):
        raise ValueError(
            "difference-pair indices are out of bounds"
        )

    return (
        values[:, pairs[:, 0]]
        - values[:, pairs[:, 1]]
    ).astype(
        float,
        copy=False,
    )

def transform_directed_differences(
    directed_differences: np.ndarray,
    thermal_scale: float,
) -> np.ndarray:
    """Reference implementation for transform_directed_differences."""
    import numpy as np

    differences = np.asarray(
        directed_differences,
        dtype=float,
    )

    if (
        differences.ndim != 2
        or differences.shape[0] < 1
        or differences.shape[1] < 1
    ):
        raise ValueError(
            "directed_differences must be a non-empty "
            "two-dimensional array"
        )

    if not np.all(
        np.isfinite(differences)
    ):
        raise ValueError(
            "directed_differences must contain only finite values"
        )

    if isinstance(
        thermal_scale,
        (bool, np.bool_),
    ):
        raise ValueError(
            "thermal_scale must be a real numerical scalar"
        )

    try:
        scale = float(
            thermal_scale
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "thermal_scale must be a real numerical scalar"
        ) from exc

    if (
        not np.isfinite(scale)
        or scale <= 0.0
    ):
        raise ValueError(
            "thermal_scale must be finite and strictly positive"
        )

    with np.errstate(
        over="ignore",
        under="ignore",
        invalid="ignore",
    ):
        factors = np.exp(
            -differences / scale
        )

    if (
        not np.all(
            np.isfinite(factors)
        )
        or np.any(
            factors <= 0.0
        )
    ):
        raise ValueError(
            "transformed factors must be finite and "
            "strictly positive"
        )

    return factors.astype(
        float,
        copy=False,
    )

def evaluate_factor_observables(
    rate_factors,
):
    """Reference implementation for evaluate_factor_observables."""
    import numpy as np

    values = np.asarray(
        rate_factors,
        dtype=float,
    )

    if (
        values.ndim != 2
        or values.shape[0] < 1
        or values.shape[1] != 5
    ):
        raise ValueError(
            "rate_factors must have shape (n, 5) with n >= 1"
        )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "rate_factors must contain only finite values"
        )

    if np.any(
        values <= 0.0
    ):
        raise ValueError(
            "rate_factors must be strictly positive"
        )

    scale = np.max(
        values,
        axis=1,
    )

    scaled = (
        values
        / scale[:, np.newaxis]
    )

    common = (
        scaled[:, 2]
        + scaled[:, 3]
        + scaled[:, 4]
    )

    turnover = (
        scale
        * scaled[:, 2]
        * scaled[:, 4]
        / common
    )

    michaelis = (
        scaled[:, 2] * scaled[:, 4]
        + scaled[:, 1] * scaled[:, 3]
        + scaled[:, 1] * scaled[:, 4]
    ) / (
        scaled[:, 0]
        * common
    )

    efficiency = (
        turnover
        / michaelis
    )

    observables = np.column_stack(
        (
            turnover,
            michaelis,
            efficiency,
        )
    )

    if (
        not np.all(
            np.isfinite(observables)
        )
        or np.any(
            observables <= 0.0
        )
    ):
        raise ValueError(
            "all returned observables must be finite and positive"
        )

    return observables.astype(
        float,
        copy=False,
    )

def compute_pair_departure_factors(
    profile_values: np.ndarray,
    pair_rows: np.ndarray,
    member_pairs: np.ndarray,
    anchor_index: int,
) -> np.ndarray:
    """Reference implementation for compute_pair_departure_factors."""
    import numpy as np

    values = np.asarray(
        profile_values,
        dtype=float,
    )

    pair_rows_raw = np.asarray(
        pair_rows,
    )

    member_pairs_raw = np.asarray(
        member_pairs,
    )

    if (
        values.ndim != 1
        or values.size < 1
    ):
        raise ValueError(
            "profile_values must be a non-empty one-dimensional array"
        )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "profile_values must contain only finite values"
        )

    if np.any(
        values <= 0.0
    ):
        raise ValueError(
            "profile_values must be strictly positive"
        )

    if (
        pair_rows_raw.ndim != 1
        or pair_rows_raw.size < 1
        or not np.issubdtype(
            pair_rows_raw.dtype,
            np.integer,
        )
    ):
        raise ValueError(
            "pair_rows must be a non-empty integer array"
        )

    if (
        member_pairs_raw.ndim != 2
        or member_pairs_raw.shape[1] != 2
        or member_pairs_raw.shape[0] != pair_rows_raw.size
        or not np.issubdtype(
            member_pairs_raw.dtype,
            np.integer,
        )
    ):
        raise ValueError(
            "member_pairs must be an integer array of shape (p, 2)"
        )

    if isinstance(
        anchor_index,
        (bool, np.bool_),
    ) or not isinstance(
        anchor_index,
        (int, np.integer),
    ):
        raise ValueError(
            "anchor_index must be an integer"
        )

    anchor = int(
        anchor_index
    )

    pair_indices = pair_rows_raw.astype(
        int,
        copy=False,
    )

    members = member_pairs_raw.astype(
        int,
        copy=False,
    )

    n_profiles = values.size

    if (
        anchor < 0
        or anchor >= n_profiles
    ):
        raise ValueError(
            "anchor_index is out of range"
        )

    if (
        np.any(pair_indices < 0)
        or np.any(pair_indices >= n_profiles)
        or np.any(members < 0)
        or np.any(members >= n_profiles)
    ):
        raise ValueError(
            "pair_rows or member_pairs contains an out-of-range index"
        )

    log_values = np.log(
        values
    )

    log_factors = (
        log_values[pair_indices]
        + log_values[anchor]
        - log_values[members[:, 0]]
        - log_values[members[:, 1]]
    )

    with np.errstate(
        over="ignore",
        under="ignore",
        invalid="ignore",
    ):
        departure_factors = np.exp(
            log_factors
        )

    if (
        not np.all(
            np.isfinite(departure_factors)
        )
        or np.any(
            departure_factors <= 0.0
        )
    ):
        raise ValueError(
            "departure factors must be finite and strictly positive"
        )

    return departure_factors.astype(
        float,
        copy=False,
    )

def derive_pair_factor_states(
    factor_states: np.ndarray,
    member_pairs: np.ndarray,
    anchor_index: int,
) -> np.ndarray:
    """Reference implementation for derive_pair_factor_states."""
    import numpy as np

    states = np.asarray(
        factor_states,
        dtype=float,
    )

    member_pairs_raw = np.asarray(
        member_pairs,
    )

    if (
        states.ndim != 2
        or states.shape[0] < 1
        or states.shape[1] != 5
    ):
        raise ValueError(
            "factor_states must have shape (n, 5) with n >= 1"
        )

    if not np.all(
        np.isfinite(states)
    ):
        raise ValueError(
            "factor_states must contain only finite values"
        )

    if np.any(
        states <= 0.0
    ):
        raise ValueError(
            "factor_states must be strictly positive"
        )

    if (
        member_pairs_raw.ndim != 2
        or member_pairs_raw.shape[0] < 1
        or member_pairs_raw.shape[1] != 2
        or not np.issubdtype(
            member_pairs_raw.dtype,
            np.integer,
        )
    ):
        raise ValueError(
            "member_pairs must be a non-empty integer array "
            "of shape (p, 2)"
        )

    if isinstance(
        anchor_index,
        (bool, np.bool_),
    ) or not isinstance(
        anchor_index,
        (int, np.integer),
    ):
        raise ValueError(
            "anchor_index must be an integer"
        )

    anchor = int(
        anchor_index
    )

    members = member_pairs_raw.astype(
        int,
        copy=False,
    )

    n_profiles = states.shape[0]

    if (
        anchor < 0
        or anchor >= n_profiles
    ):
        raise ValueError(
            "anchor_index is out of range"
        )

    if (
        np.any(members < 0)
        or np.any(members >= n_profiles)
    ):
        raise ValueError(
            "member_pairs contains an out-of-range index"
        )

    log_states = np.log(
        states
    )

    log_pair_states = (
        log_states[members[:, 0]]
        + log_states[members[:, 1]]
        - log_states[anchor]
    )

    with np.errstate(
        over="ignore",
        under="ignore",
        invalid="ignore",
    ):
        pair_factor_states = np.exp(
            log_pair_states
        )

    if (
        not np.all(
            np.isfinite(pair_factor_states)
        )
        or np.any(
            pair_factor_states <= 0.0
        )
    ):
        raise ValueError(
            "derived pair factor states must be finite "
            "and strictly positive"
        )

    return pair_factor_states.astype(
        float,
        copy=False,
    )

def compute_aligned_ratio_scores(
    numerator_values: np.ndarray,
    denominator_values: np.ndarray,
) -> np.ndarray:
    """Reference implementation for compute_aligned_ratio_scores."""
    import numpy as np

    numerators = np.asarray(
        numerator_values,
        dtype=float,
    )

    denominators = np.asarray(
        denominator_values,
        dtype=float,
    )

    if (
        numerators.ndim != 1
        or numerators.size < 1
    ):
        raise ValueError(
            "numerator_values must be a non-empty "
            "one-dimensional array"
        )

    if denominators.shape != numerators.shape:
        raise ValueError(
            "denominator_values must have the same shape "
            "as numerator_values"
        )

    if not np.all(
        np.isfinite(numerators)
    ):
        raise ValueError(
            "numerator_values must contain only finite values"
        )

    if not np.all(
        np.isfinite(denominators)
    ):
        raise ValueError(
            "denominator_values must contain only finite values"
        )

    if np.any(
        numerators <= 0.0
    ):
        raise ValueError(
            "numerator_values must be strictly positive"
        )

    if np.any(
        denominators <= 0.0
    ):
        raise ValueError(
            "denominator_values must be strictly positive"
        )

    log_scores = (
        np.log(numerators)
        - np.log(denominators)
    )

    with np.errstate(
        over="ignore",
        under="ignore",
        invalid="ignore",
    ):
        ratio_scores = np.exp(
            log_scores
        )

    if (
        not np.all(
            np.isfinite(ratio_scores)
        )
        or np.any(
            ratio_scores <= 0.0
        )
    ):
        raise ValueError(
            "ratio scores must be finite and strictly positive"
        )

    return ratio_scores.astype(
        float,
        copy=False,
    )

def select_bounded_maximum(
    candidate_ids: np.ndarray,
    eligibility_values: np.ndarray,
    priority_values: np.ndarray,
    lower_bound: float,
    upper_bound: float,
) -> np.ndarray:
    """Reference implementation for select_bounded_maximum."""
    import numpy as np

    ids_raw = np.asarray(
        candidate_ids,
    )

    eligibility = np.asarray(
        eligibility_values,
        dtype=float,
    )

    priority = np.asarray(
        priority_values,
        dtype=float,
    )

    if (
        ids_raw.ndim != 1
        or ids_raw.size < 1
        or not np.issubdtype(
            ids_raw.dtype,
            np.integer,
        )
    ):
        raise ValueError(
            "candidate_ids must be a non-empty "
            "one-dimensional integer array"
        )

    ids = ids_raw.astype(
        int,
        copy=False,
    )

    if np.unique(
        ids
    ).size != ids.size:
        raise ValueError(
            "candidate_ids must be unique"
        )

    if eligibility.shape != ids.shape:
        raise ValueError(
            "eligibility_values must align with candidate_ids"
        )

    if priority.shape != ids.shape:
        raise ValueError(
            "priority_values must align with candidate_ids"
        )

    if not np.all(
        np.isfinite(eligibility)
    ):
        raise ValueError(
            "eligibility_values must contain only finite values"
        )

    if not np.all(
        np.isfinite(priority)
    ):
        raise ValueError(
            "priority_values must contain only finite values"
        )

    if np.any(
        eligibility <= 0.0
    ):
        raise ValueError(
            "eligibility_values must be strictly positive"
        )

    if np.any(
        priority <= 0.0
    ):
        raise ValueError(
            "priority_values must be strictly positive"
        )

    if isinstance(
        lower_bound,
        (bool, np.bool_),
    ) or isinstance(
        upper_bound,
        (bool, np.bool_),
    ):
        raise ValueError(
            "bounds must be real numerical scalars"
        )

    try:
        lower = float(
            lower_bound
        )

        upper = float(
            upper_bound
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "bounds must be real numerical scalars"
        ) from exc

    if (
        not np.isfinite(lower)
        or not np.isfinite(upper)
        or lower <= 0.0
        or upper <= 0.0
        or lower > upper
    ):
        raise ValueError(
            "bounds must be finite, positive, and ordered"
        )

    eligible = (
        (eligibility >= lower)
        & (eligibility <= upper)
    )

    if not np.any(
        eligible
    ):
        raise ValueError(
            "no candidate satisfies the supplied bounds"
        )

    largest_priority = np.max(
        priority[eligible]
    )

    tied = (
        eligible
        & (priority == largest_priority)
    )

    selected_id = int(
        np.min(
            ids[tied]
        )
    )

    selected_priority = float(
        priority[
            np.flatnonzero(
                ids == selected_id
            )[0]
        ]
    )

    selection = np.array(
        [
            float(selected_id),
            selected_priority,
        ],
        dtype=float,
    )

    return selection

def resolve_panel_scalar(
    profile_values: np.ndarray,
    auxiliary_values: np.ndarray,
    pair_rows: np.ndarray,
    member_pairs: np.ndarray,
    candidate_ids: np.ndarray,
    anchor_index: int,
    temperature: float,
    gas_constant: float,
) -> float:
    """Reference implementation for resolve_panel_scalar."""
    import numpy as np

    profiles = np.asarray(
        profile_values,
        dtype=float,
    )

    auxiliary = np.asarray(
        auxiliary_values,
        dtype=float,
    )

    pair_rows_raw = np.asarray(
        pair_rows,
    )

    member_pairs_raw = np.asarray(
        member_pairs,
    )

    candidate_ids_raw = np.asarray(
        candidate_ids,
    )

    if (
        profiles.ndim != 2
        or profiles.shape[0] < 1
        or profiles.shape[1] != 6
    ):
        raise ValueError(
            "profile_values must have shape (n, 6) with n >= 1"
        )

    if not np.all(
        np.isfinite(profiles)
    ):
        raise ValueError(
            "profile_values must contain only finite values"
        )

    if auxiliary.shape != (
        profiles.shape[0],
    ):
        raise ValueError(
            "auxiliary_values must align with profile rows"
        )

    if (
        not np.all(
            np.isfinite(auxiliary)
        )
        or np.any(
            auxiliary <= 0.0
        )
    ):
        raise ValueError(
            "auxiliary_values must be finite and strictly positive"
        )

    if (
        pair_rows_raw.ndim != 1
        or pair_rows_raw.size < 1
        or not np.issubdtype(
            pair_rows_raw.dtype,
            np.integer,
        )
    ):
        raise ValueError(
            "pair_rows must be a non-empty integer array"
        )

    if (
        member_pairs_raw.ndim != 2
        or member_pairs_raw.shape
        != (
            pair_rows_raw.size,
            2,
        )
        or not np.issubdtype(
            member_pairs_raw.dtype,
            np.integer,
        )
    ):
        raise ValueError(
            "member_pairs must be an integer array of shape (p, 2)"
        )

    if (
        candidate_ids_raw.ndim != 1
        or candidate_ids_raw.shape
        != pair_rows_raw.shape
        or not np.issubdtype(
            candidate_ids_raw.dtype,
            np.integer,
        )
    ):
        raise ValueError(
            "candidate_ids must be an integer array aligned "
            "with pair_rows"
        )

    if isinstance(
        anchor_index,
        (bool, np.bool_),
    ) or not isinstance(
        anchor_index,
        (int, np.integer),
    ):
        raise ValueError(
            "anchor_index must be an integer"
        )

    if isinstance(
        temperature,
        (bool, np.bool_),
    ) or isinstance(
        gas_constant,
        (bool, np.bool_),
    ):
        raise ValueError(
            "temperature and gas_constant must be real scalars"
        )

    try:
        temperature_value = float(
            temperature
        )

        gas_constant_value = float(
            gas_constant
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "temperature and gas_constant must be real scalars"
        ) from exc

    if (
        not np.isfinite(
            temperature_value
        )
        or temperature_value <= 0.0
        or not np.isfinite(
            gas_constant_value
        )
        or gas_constant_value <= 0.0
    ):
        raise ValueError(
            "temperature and gas_constant must be finite "
            "and strictly positive"
        )

    pair_indices = pair_rows_raw.astype(
        int,
        copy=False,
    )

    members = member_pairs_raw.astype(
        int,
        copy=False,
    )

    ids = candidate_ids_raw.astype(
        int,
        copy=False,
    )

    anchor = int(
        anchor_index
    )

    n_profiles = profiles.shape[0]

    if (
        anchor < 0
        or anchor >= n_profiles
        or np.any(
            pair_indices < 0
        )
        or np.any(
            pair_indices >= n_profiles
        )
        or np.any(
            members < 0
        )
        or np.any(
            members >= n_profiles
        )
    ):
        raise ValueError(
            "one or more supplied profile indices are out of range"
        )

    if np.unique(
        ids
    ).size != ids.size:
        raise ValueError(
            "candidate_ids must be unique"
        )

    difference_pairs = np.array(
        [
            [1, 0],
            [1, 2],
            [3, 2],
            [3, 4],
            [5, 4],
        ],
        dtype=int,
    )

    directed_differences = compute_indexed_differences(
        profiles,
        difference_pairs,
    )

    thermal_scale = (
        temperature_value
        * gas_constant_value
    )

    factor_states = transform_directed_differences(
        directed_differences,
        thermal_scale,
    )

    profile_observables = evaluate_factor_observables(
        factor_states,
    )

    profile_efficiencies = (
        profile_observables[:, 2]
    )

    eligibility_values = compute_pair_departure_factors(
        profile_efficiencies,
        pair_indices,
        members,
        anchor,
    )

    derived_pair_states = derive_pair_factor_states(
        factor_states,
        members,
        anchor,
    )

    derived_pair_observables = evaluate_factor_observables(
        derived_pair_states,
    )

    priority_values = compute_aligned_ratio_scores(
        profile_efficiencies[
            pair_indices
        ],
        derived_pair_observables[:, 2],
    )

    selection = select_bounded_maximum(
        ids,
        eligibility_values,
        priority_values,
        1.0 / 1.5,
        1.5,
    )

    selected_scalar = float(
        selection[1]
    )

    if (
        not np.isfinite(
            selected_scalar
        )
        or selected_scalar <= 0.0
    ):
        raise ValueError(
            "selected scalar must be finite and strictly positive"
        )

    return selected_scalar
SCICODE_GOLD_EOF
