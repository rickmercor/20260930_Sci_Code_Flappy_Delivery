#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def compute_reaction_driving_forces(concentration_states_mM, stoichiometric_columns, standard_gibbs_kj, temperature, gas_constant):
    import numpy as np
    concentrations = np.asarray(concentration_states_mM, dtype=float)
    stoichiometry = np.asarray(stoichiometric_columns, dtype=float)
    standard = np.asarray(standard_gibbs_kj, dtype=float)
    if concentrations.ndim != 2 or min(concentrations.shape) < 1:
        raise ValueError('concentration_states_mM must be a non-empty two-dimensional array')
    if not np.all(np.isfinite(concentrations)) or np.any(concentrations <= 0.0):
        raise ValueError('concentration_states_mM must contain finite strictly positive values')
    if stoichiometry.ndim != 2 or stoichiometry.shape[0] != concentrations.shape[1] or stoichiometry.shape[1] < 1 or (not np.all(np.isfinite(stoichiometry))):
        raise ValueError('stoichiometric_columns must be finite and aligned with metabolite columns')
    if standard.ndim != 1 or standard.size != stoichiometry.shape[1] or (not np.all(np.isfinite(standard))):
        raise ValueError('standard_gibbs_kj must be finite and aligned with reaction columns')
    try:
        t = float(temperature)
        r = float(gas_constant)
    except (TypeError, ValueError) as exc:
        raise ValueError('temperature and gas_constant must be real scalars') from exc
    if not np.isfinite(t) or t <= 0.0 or (not np.isfinite(r)) or (r <= 0.0):
        raise ValueError('temperature and gas_constant must be finite and strictly positive')
    log_molar = np.log(concentrations * 0.001)
    delta_g = standard[None, :] + r * t * (log_molar @ stoichiometry)
    return -delta_g

def compute_direct_binding_saturation_factors(
    concentration_states_mM,
    substrate_indices,
    substrate_km_mM,
    substrate_fixed_terms,
    product_indices,
    product_km_mM,
    product_fixed_terms,
):
    import numpy as np

    concentrations = np.asarray(concentration_states_mM, dtype=float)
    raw_sidx = np.asarray(substrate_indices)
    skm = np.asarray(substrate_km_mM, dtype=float)
    sfixed = np.asarray(substrate_fixed_terms, dtype=float)
    raw_pidx = np.asarray(product_indices)
    pkm = np.asarray(product_km_mM, dtype=float)
    pfixed = np.asarray(product_fixed_terms, dtype=float)

    if concentrations.ndim != 2 or min(concentrations.shape) < 1:
        raise ValueError(
            'concentration_states_mM must be a non-empty two-dimensional array'
        )

    if not np.all(np.isfinite(concentrations)) or np.any(concentrations <= 0.0):
        raise ValueError(
            'concentration_states_mM must contain finite strictly positive values'
        )

    if (
        raw_sidx.ndim != 1
        or raw_sidx.size < 1
        or not np.issubdtype(raw_sidx.dtype, np.integer)
    ):
        raise ValueError(
            'substrate_indices must be a non-empty one-dimensional integer array'
        )

    n_reactions = raw_sidx.size

    arrays = (
        skm,
        sfixed,
        raw_pidx,
        pkm,
        pfixed,
    )

    if any(
        arr.ndim != 1 or arr.size != n_reactions
        for arr in arrays
    ):
        raise ValueError(
            'all reaction-level arrays must be one-dimensional and aligned'
        )

    if not np.issubdtype(raw_pidx.dtype, np.integer):
        raise ValueError(
            'product_indices must be an integer array'
        )

    sidx = raw_sidx.astype(int, copy=False)
    pidx = raw_pidx.astype(int, copy=False)

    n_metabolites = concentrations.shape[1]

    result = np.empty(
        (concentrations.shape[0], n_reactions),
        dtype=float,
    )

    for reaction in range(n_reactions):
        if sidx[reaction] == -1:
            if (
                not np.isnan(skm[reaction])
                or not np.isfinite(sfixed[reaction])
                or sfixed[reaction] <= 0.0
            ):
                raise ValueError(
                    'a fixed substrate term requires index -1, NaN KM, and a finite positive fixed term'
                )

            sbar = np.full(
                concentrations.shape[0],
                sfixed[reaction],
                dtype=float,
            )

        else:
            if (
                sidx[reaction] < 0
                or sidx[reaction] >= n_metabolites
                or not np.isfinite(skm[reaction])
                or skm[reaction] <= 0.0
                or not np.isnan(sfixed[reaction])
            ):
                raise ValueError(
                    'an internal substrate requires a valid index, positive KM, and NaN fixed term'
                )

            sbar = (
                concentrations[:, sidx[reaction]]
                / skm[reaction]
            )

        if pidx[reaction] == -1:
            if (
                not np.isnan(pkm[reaction])
                or not np.isfinite(pfixed[reaction])
                or pfixed[reaction] < 0.0
            ):
                raise ValueError(
                    'a fixed product term requires index -1, NaN KM, and a finite non-negative fixed term'
                )

            pbar = np.full(
                concentrations.shape[0],
                pfixed[reaction],
                dtype=float,
            )

        else:
            if (
                pidx[reaction] < 0
                or pidx[reaction] >= n_metabolites
                or not np.isfinite(pkm[reaction])
                or pkm[reaction] <= 0.0
                or not np.isnan(pfixed[reaction])
            ):
                raise ValueError(
                    'an internal product requires a valid index, positive KM, and NaN fixed term'
                )

            pbar = (
                concentrations[:, pidx[reaction]]
                / pkm[reaction]
            )

        result[:, reaction] = (
            sbar
            / (1.0 + sbar + pbar)
        )

    return result

def compute_thermodynamic_efficiency_factors(driving_forces_kj, temperature, gas_constant):
    import numpy as np
    forces = np.asarray(driving_forces_kj, dtype=float)
    if forces.ndim != 2 or min(forces.shape) < 1 or (not np.all(np.isfinite(forces))):
        raise ValueError('driving_forces_kj must be a finite non-empty two-dimensional array')
    try:
        t = float(temperature)
        r = float(gas_constant)
    except (TypeError, ValueError) as exc:
        raise ValueError('temperature and gas_constant must be real scalars') from exc
    if not np.isfinite(t) or t <= 0.0 or (not np.isfinite(r)) or (r <= 0.0):
        raise ValueError('temperature and gas_constant must be finite and strictly positive')
    return -np.expm1(-forces / (r * t))

def encode_pattern_group_states(pattern_mask, coupling_group_ids):
    import numpy as np
    pattern = np.asarray(pattern_mask, dtype=float)
    raw_groups = np.asarray(coupling_group_ids)
    if pattern.ndim != 2 or min(pattern.shape) < 1:
        raise ValueError('pattern_mask must be a non-empty two-dimensional array')
    if not np.all(np.isfinite(pattern)) or not np.all((pattern == 0.0) | (pattern == 1.0)):
        raise ValueError('pattern_mask must contain only numerical 0.0 and 1.0 values')
    if raw_groups.ndim != 1 or raw_groups.size != pattern.shape[1] or (not np.issubdtype(raw_groups.dtype, np.integer)):
        raise ValueError('coupling_group_ids must be an integer array aligned with reaction columns')
    groups = raw_groups.astype(int, copy=False)
    if np.any(groups <= 0):
        raise ValueError('coupling_group_ids must contain positive integers')
    unique_groups = np.unique(groups)
    result = np.empty((pattern.shape[0], unique_groups.size + 1), dtype=float)
    valid = np.ones(pattern.shape[0], dtype=float)
    for output_col, group in enumerate(unique_groups):
        positions = np.flatnonzero(groups == group)
        group_values = pattern[:, positions]
        uniform = np.all(group_values == group_values[:, [0]], axis=1)
        result[:, output_col] = np.where(uniform, group_values[:, 0], -1.0)
        valid[~uniform] = 0.0
    result[:, -1] = valid
    return result

def select_source_screen_records(candidate_ids, candidate_parent_ids, parent_ids, relaxed_objective_values, candidate_group_summary, parent_group_summary, objective_tolerance):
    import numpy as np
    raw_candidate_ids = np.asarray(candidate_ids)
    raw_candidate_parent_ids = np.asarray(candidate_parent_ids)
    raw_parent_ids = np.asarray(parent_ids)
    relaxed = np.asarray(relaxed_objective_values, dtype=float)
    candidate_summary = np.asarray(candidate_group_summary, dtype=float)
    parent_summary = np.asarray(parent_group_summary, dtype=float)
    if raw_candidate_ids.ndim != 1 or raw_candidate_ids.size < 1 or (not np.issubdtype(raw_candidate_ids.dtype, np.integer)):
        raise ValueError('candidate_ids must be a non-empty one-dimensional integer array')
    candidate_id_values = raw_candidate_ids.astype(int, copy=False)
    if np.any(candidate_id_values <= 0) or np.unique(candidate_id_values).size != candidate_id_values.size:
        raise ValueError('candidate_ids must contain unique positive integers')
    if raw_candidate_parent_ids.ndim != 1 or raw_candidate_parent_ids.size != candidate_id_values.size or (not np.issubdtype(raw_candidate_parent_ids.dtype, np.integer)):
        raise ValueError('candidate_parent_ids must be an integer array aligned with candidates')
    candidate_parent_values = raw_candidate_parent_ids.astype(int, copy=False)
    if raw_parent_ids.ndim != 1 or raw_parent_ids.size < 1 or (not np.issubdtype(raw_parent_ids.dtype, np.integer)):
        raise ValueError('parent_ids must be a non-empty integer array')
    parent_id_values = raw_parent_ids.astype(int, copy=False)
    if np.any(parent_id_values <= 0) or np.unique(parent_id_values).size != parent_id_values.size:
        raise ValueError('parent_ids must contain unique positive integers')
    if not np.all(np.isin(candidate_parent_values, parent_id_values)):
        raise ValueError('every candidate parent ID must occur in parent_ids')
    if relaxed.ndim != 1 or relaxed.size != candidate_id_values.size or (not np.all(np.isfinite(relaxed))):
        raise ValueError('relaxed_objective_values must be finite and aligned with candidates')
    if candidate_summary.ndim != 2 or candidate_summary.shape[0] != candidate_id_values.size or candidate_summary.shape[1] < 2:
        raise ValueError('candidate_group_summary must align with candidates and contain a validity column')
    if parent_summary.shape != (parent_id_values.size, candidate_summary.shape[1]):
        raise ValueError('parent_group_summary must align with parent IDs and candidate group columns')
    if not np.all(np.isfinite(candidate_summary)) or not np.all(np.isfinite(parent_summary)):
        raise ValueError('group summaries must contain only finite numerical values')
    candidate_bits = candidate_summary[:, :-1]
    candidate_valid = candidate_summary[:, -1]
    parent_bits = parent_summary[:, :-1]
    parent_valid = parent_summary[:, -1]
    if not np.all((candidate_bits == -1.0) | (candidate_bits == 0.0) | (candidate_bits == 1.0)):
        raise ValueError('candidate group codes must be -1.0, 0.0, or 1.0')
    if not np.all((parent_bits == -1.0) | (parent_bits == 0.0) | (parent_bits == 1.0)):
        raise ValueError('parent group codes must be -1.0, 0.0, or 1.0')
    if not np.all((candidate_valid == 0.0) | (candidate_valid == 1.0)):
        raise ValueError('candidate validity codes must be 0.0 or 1.0')
    if not np.all(parent_valid == 1.0) or np.any(parent_bits < 0.0):
        raise ValueError('every parent pattern must have a valid coupled-group representation')
    try:
        tolerance = float(objective_tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError('objective_tolerance must be a real scalar') from exc
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError('objective_tolerance must be finite and non-negative')
    result = np.zeros(candidate_id_values.size, dtype=float)
    for parent_position, parent in enumerate(parent_id_values):
        positions = np.flatnonzero(candidate_parent_values == parent)
        if positions.size < 1:
            raise ValueError('every parent must have at least one candidate row')
        valid_positions = positions[(candidate_valid[positions] == 1.0) & np.all(candidate_bits[positions] <= parent_bits[parent_position][None, :], axis=1)]
        if valid_positions.size < 1:
            raise ValueError('a parent has no source-admissible candidate pattern')
        best_objective = np.max(relaxed[valid_positions])
        optimal_positions = valid_positions[np.abs(relaxed[valid_positions] - best_objective) <= tolerance]
        counts = np.sum(candidate_bits[optimal_positions], axis=1)
        for target_count in (np.max(counts), np.min(counts)):
            tied = optimal_positions[counts == target_count]
            chosen = tied[np.argmin(candidate_id_values[tied])]
            result[chosen] = 1.0
    return result

def classify_pattern_state_eligibility(candidate_pattern_mask, driving_forces_kj, minimum_driving_force):
    import numpy as np
    pattern = np.asarray(candidate_pattern_mask, dtype=float)
    forces = np.asarray(driving_forces_kj, dtype=float)
    if pattern.ndim != 2 or min(pattern.shape) < 1:
        raise ValueError('candidate_pattern_mask must be a non-empty two-dimensional array')
    if not np.all(np.isfinite(pattern)) or not np.all((pattern == 0.0) | (pattern == 1.0)):
        raise ValueError('candidate_pattern_mask must contain only numerical 0.0 and 1.0 values')
    if forces.ndim != 2 or forces.shape[0] < 1 or forces.shape[1] != pattern.shape[1]:
        raise ValueError('driving_forces_kj must align with state rows and pattern reaction columns')
    if not np.all(np.isfinite(forces)):
        raise ValueError('driving_forces_kj must contain only finite values')
    try:
        floor = float(minimum_driving_force)
    except (TypeError, ValueError) as exc:
        raise ValueError('minimum_driving_force must be a real scalar') from exc
    if not np.isfinite(floor):
        raise ValueError('minimum_driving_force must be finite')
    meets = forces >= floor
    eligible = np.all((pattern[:, None, :] == 0.0) | meets[None, :, :], axis=2)
    return eligible.astype(float)

def compute_candidate_state_metrics(
    candidate_fluxes,
    candidate_pattern_mask,
    turnover_forward_h,
    molecular_weights_g_per_mmol,
    saturation_factors,
    thermodynamic_factors,
    pattern_state_eligibility,
    concentration_states_mM,
    reference_profile_mM,
    enzyme_pool_limit,
    objective_coefficients,
):
    import numpy as np
 
    fluxes = np.asarray(candidate_fluxes, dtype=float)
    pattern = np.asarray(candidate_pattern_mask, dtype=float)
    turnover = np.asarray(turnover_forward_h, dtype=float)
    weights = np.asarray(molecular_weights_g_per_mmol, dtype=float)
    kappa = np.asarray(saturation_factors, dtype=float)
    gamma = np.asarray(thermodynamic_factors, dtype=float)
    eligibility = np.asarray(pattern_state_eligibility, dtype=float)
    concentrations = np.asarray(concentration_states_mM, dtype=float)
    reference = np.asarray(reference_profile_mM, dtype=float)
    objective = np.asarray(objective_coefficients, dtype=float)
 
    if fluxes.ndim != 2 or min(fluxes.shape) < 1:
        raise ValueError("candidate_fluxes must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(fluxes)) or np.any(fluxes < 0.0):
        raise ValueError("candidate_fluxes must contain finite non-negative values")
    n_candidates, n_reactions = fluxes.shape
 
    if pattern.shape != fluxes.shape or not np.all(np.isfinite(pattern)):
        raise ValueError("candidate_pattern_mask must align with candidate_fluxes")
    if not np.all((pattern == 0.0) | (pattern == 1.0)):
        raise ValueError("candidate_pattern_mask must contain only 0.0 and 1.0")
    if np.any((fluxes > 0.0) & (pattern == 0.0)):
        raise ValueError("positive continuous flux requires a row-level binary state of 1.0")
 
    if turnover.shape != (n_reactions,) or not np.all(np.isfinite(turnover)) or np.any(turnover <= 0.0):
        raise ValueError("turnover_forward_h must be finite, positive, and reaction-aligned")
    if weights.shape != (n_reactions,) or not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("molecular_weights_g_per_mmol must be finite, positive, and reaction-aligned")
    if kappa.ndim != 2 or kappa.shape[0] < 1 or kappa.shape[1] != n_reactions:
        raise ValueError("saturation_factors must align with state and reaction dimensions")
    if not np.all(np.isfinite(kappa)) or np.any(kappa <= 0.0) or np.any(kappa > 1.0):
        raise ValueError("saturation_factors must be finite and lie in (0, 1]")
    if gamma.shape != kappa.shape or not np.all(np.isfinite(gamma)):
        raise ValueError("thermodynamic_factors must be finite and aligned with saturation_factors")
 
    n_states = kappa.shape[0]
    if eligibility.shape != (n_candidates, n_states):
        raise ValueError("pattern_state_eligibility must align with candidates and states")
    if not np.all(np.isfinite(eligibility)) or not np.all((eligibility == 0.0) | (eligibility == 1.0)):
        raise ValueError("pattern_state_eligibility must contain only 0.0 and 1.0")
    if concentrations.ndim != 2 or concentrations.shape[0] != n_states:
        raise ValueError("concentration_states_mM must align with state rows")
    if not np.all(np.isfinite(concentrations)) or np.any(concentrations <= 0.0):
        raise ValueError("concentration_states_mM must contain finite positive values")
    if reference.shape != (concentrations.shape[1],) or not np.all(np.isfinite(reference)) or np.any(reference <= 0.0):
        raise ValueError("reference_profile_mM must be finite, positive, and metabolite-aligned")
    if objective.shape != (n_reactions,) or not np.all(np.isfinite(objective)):
        raise ValueError("objective_coefficients must be finite and reaction-aligned")
 
    try:
        pool_limit = float(enzyme_pool_limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("enzyme_pool_limit must be a real scalar") from exc
    if not np.isfinite(pool_limit) or pool_limit <= 0.0:
        raise ValueError("enzyme_pool_limit must be finite and positive")
 
    profile_error = np.sum(np.abs(concentrations - reference[None, :]), axis=1)
    result = np.zeros((n_candidates, n_states, 4 + n_reactions), dtype=float)
 
    for candidate in range(n_candidates):
        active = fluxes[candidate] > 0.0
        q_value = float(fluxes[candidate] @ objective)
        if not np.isfinite(q_value):
            raise ValueError("candidate objective values must be finite")
        for state in range(n_states):
            result[candidate, state, 0] = q_value
            result[candidate, state, 2] = profile_error[state]
            if eligibility[candidate, state] == 0.0:
                result[candidate, state, 1] = np.inf
                continue
            denominator = turnover * kappa[state] * gamma[state]
            if np.any(denominator[active] <= 0.0):
                raise ValueError("eligible positive-flux reactions require positive effective efficiencies")
            contributions = np.zeros(n_reactions, dtype=float)
            contributions[active] = weights[active] * fluxes[candidate, active] / denominator[active]
            demand = float(np.sum(contributions, dtype=float))
            result[candidate, state, 1] = demand
            result[candidate, state, 3] = float(demand <= pool_limit)
            result[candidate, state, 4:] = contributions
 
    return result

def filter_source_postprocessing_points(
    candidate_state_metrics,
    source_selection_codes,
    objective_retention_fraction,
):
    import numpy as np
 
    metrics = np.asarray(candidate_state_metrics, dtype=float)
    codes = np.asarray(source_selection_codes, dtype=float)
 
    if metrics.ndim != 3 or min(metrics.shape[:2]) < 1 or metrics.shape[2] < 4:
        raise ValueError("candidate_state_metrics must have shape (candidates, states, channels)")
    if codes.shape != (metrics.shape[0],) or not np.all(np.isfinite(codes)):
        raise ValueError("source_selection_codes must be a finite candidate-aligned vector")
    if not np.all((codes == 0.0) | (codes == 1.0)):
        raise ValueError("source_selection_codes must contain only 0.0 and 1.0")
 
    objective = metrics[:, :, 0]
    feasible = metrics[:, :, 3]
    if not np.all(np.isfinite(objective)):
        raise ValueError("objective channel must contain finite values")
    if not np.all(np.isfinite(feasible)) or not np.all((feasible == 0.0) | (feasible == 1.0)):
        raise ValueError("feasible-code channel must contain only 0.0 and 1.0")
 
    try:
        fraction = float(objective_retention_fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError("objective_retention_fraction must be a real scalar") from exc
    if not np.isfinite(fraction) or fraction <= 0.0 or fraction > 1.0:
        raise ValueError("objective_retention_fraction must lie in (0, 1]")
 
    eligible = (codes[:, None] == 1.0) & (feasible == 1.0)
    if not np.any(eligible):
        raise ValueError("no source-selected nonlinear-feasible point exists")
    maximum = float(np.max(objective[eligible]))
    if maximum <= 0.0:
        raise ValueError("the maximal eligible objective must be positive")
    threshold = fraction * maximum
    return (eligible & (objective >= threshold)).astype(float)

def select_source_terminal_point(
    candidate_ids,
    state_ids,
    candidate_state_metrics,
    near_optimal_codes,
    profile_tolerance,
    demand_tolerance,
):
    import numpy as np
 
    raw_candidate_ids = np.asarray(candidate_ids)
    raw_state_ids = np.asarray(state_ids)
    metrics = np.asarray(candidate_state_metrics, dtype=float)
    codes = np.asarray(near_optimal_codes, dtype=float)
 
    if raw_candidate_ids.ndim != 1 or raw_candidate_ids.size < 1 or not np.issubdtype(raw_candidate_ids.dtype, np.integer):
        raise ValueError("candidate_ids must be a non-empty integer vector")
    candidate_values = raw_candidate_ids.astype(int, copy=False)
    if np.any(candidate_values <= 0) or np.unique(candidate_values).size != candidate_values.size:
        raise ValueError("candidate_ids must contain unique positive integers")
    if raw_state_ids.ndim != 1 or raw_state_ids.size < 1 or not np.issubdtype(raw_state_ids.dtype, np.integer):
        raise ValueError("state_ids must be a non-empty integer vector")
    state_values = raw_state_ids.astype(int, copy=False)
    if np.any(state_values <= 0) or np.unique(state_values).size != state_values.size:
        raise ValueError("state_ids must contain unique positive integers")
 
    if metrics.ndim != 3 or metrics.shape[0] != candidate_values.size or metrics.shape[1] != state_values.size or metrics.shape[2] < 4:
        raise ValueError("candidate_state_metrics must align with candidates and states")
    if codes.shape != metrics.shape[:2] or not np.all(np.isfinite(codes)):
        raise ValueError("near_optimal_codes must be a finite aligned array")
    if not np.all((codes == 0.0) | (codes == 1.0)):
        raise ValueError("near_optimal_codes must contain only 0.0 and 1.0")
 
    profile = metrics[:, :, 2]
    demand = metrics[:, :, 1]
    feasible = metrics[:, :, 3]
    if np.any(np.isnan(profile)) or np.any(profile < 0.0):
        raise ValueError("profile-deviation channel must be non-negative and not NaN")
    if np.any(np.isnan(demand)) or np.any(demand < 0.0):
        raise ValueError("demand channel must be non-negative or positive infinity")
    if not np.all(np.isfinite(feasible)) or not np.all((feasible == 0.0) | (feasible == 1.0)):
        raise ValueError("feasible-code channel is invalid")
    if np.any((codes == 1.0) & ((feasible != 1.0) | (~np.isfinite(demand)))):
        raise ValueError("every near-optimal point must be feasible with finite demand")
 
    try:
        profile_tol = float(profile_tolerance)
        demand_tol = float(demand_tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError("tolerances must be real scalars") from exc
    if not np.isfinite(profile_tol) or profile_tol < 0.0 or not np.isfinite(demand_tol) or demand_tol < 0.0:
        raise ValueError("tolerances must be finite and non-negative")
 
    allowed = codes == 1.0
    if not np.any(allowed):
        raise ValueError("no near-optimal point exists")
    best_profile = float(np.min(profile[allowed]))
    profile_ties = allowed & (np.abs(profile - best_profile) <= profile_tol)
    best_demand = float(np.min(demand[profile_ties]))
    demand_ties = profile_ties & (np.abs(demand - best_demand) <= demand_tol)
    positions = np.argwhere(demand_ties)
    order = np.lexsort((state_values[positions[:, 1]], candidate_values[positions[:, 0]]))
    candidate_position, state_position = positions[order[0]]
    return np.array(
        [float(candidate_values[candidate_position]), float(state_values[state_position])],
        dtype=float,
    )

def resolve_auxiliary_enzyme_fraction(
    concentration_states_mM,
    candidate_fluxes,
    candidate_ids,
    candidate_parent_ids,
    parent_ids,
    parent_pattern_mask,
    candidate_pattern_mask,
    coupling_group_ids,
    relaxed_objective_values,
    state_ids,
    stoichiometric_columns,
    standard_gibbs_kj,
    substrate_indices,
    substrate_km_mM,
    substrate_fixed_terms,
    product_indices,
    product_km_mM,
    product_fixed_terms,
    turnover_forward_h,
    molecular_weights_g_per_mmol,
    temperature,
    gas_constant,
    minimum_driving_force,
    enzyme_pool_limit,
    objective_coefficients,
    reference_profile_mM,
    objective_retention_fraction,
    branch_reaction_indices,
    relaxed_objective_tolerance,
    profile_tolerance,
    demand_tolerance,
):
    import numpy as np
 
    driving_forces = compute_reaction_driving_forces(
        concentration_states_mM,
        stoichiometric_columns,
        standard_gibbs_kj,
        temperature,
        gas_constant,
    )
    saturation_factors = compute_direct_binding_saturation_factors(
        concentration_states_mM,
        substrate_indices,
        substrate_km_mM,
        substrate_fixed_terms,
        product_indices,
        product_km_mM,
        product_fixed_terms,
    )
    thermodynamic_factors = compute_thermodynamic_efficiency_factors(
        driving_forces,
        temperature,
        gas_constant,
    )
    candidate_group_summary = encode_pattern_group_states(
        candidate_pattern_mask,
        coupling_group_ids,
    )
    parent_group_summary = encode_pattern_group_states(
        parent_pattern_mask,
        coupling_group_ids,
    )
    source_selection_codes = select_source_screen_records(
        candidate_ids,
        candidate_parent_ids,
        parent_ids,
        relaxed_objective_values,
        candidate_group_summary,
        parent_group_summary,
        relaxed_objective_tolerance,
    )
    pattern_state_eligibility = classify_pattern_state_eligibility(
        candidate_pattern_mask,
        driving_forces,
        minimum_driving_force,
    )
    metrics = compute_candidate_state_metrics(
        candidate_fluxes,
        candidate_pattern_mask,
        turnover_forward_h,
        molecular_weights_g_per_mmol,
        saturation_factors,
        thermodynamic_factors,
        pattern_state_eligibility,
        concentration_states_mM,
        reference_profile_mM,
        enzyme_pool_limit,
        objective_coefficients,
    )
    near_optimal_codes = filter_source_postprocessing_points(
        metrics,
        source_selection_codes,
        objective_retention_fraction,
    )
    selected = select_source_terminal_point(
        candidate_ids,
        state_ids,
        metrics,
        near_optimal_codes,
        profile_tolerance,
        demand_tolerance,
    )
 
    raw_candidate_ids = np.asarray(candidate_ids)
    raw_state_ids = np.asarray(state_ids)
    raw_branch_indices = np.asarray(branch_reaction_indices)
    fluxes = np.asarray(candidate_fluxes, dtype=float)
 
    if raw_candidate_ids.ndim != 1 or not np.issubdtype(raw_candidate_ids.dtype, np.integer):
        raise ValueError("candidate_ids must be a one-dimensional integer array")
    if raw_state_ids.ndim != 1 or not np.issubdtype(raw_state_ids.dtype, np.integer):
        raise ValueError("state_ids must be a one-dimensional integer array")
    if raw_branch_indices.ndim != 1 or raw_branch_indices.size < 1 or not np.issubdtype(raw_branch_indices.dtype, np.integer):
        raise ValueError("branch_reaction_indices must be a non-empty integer array")
 
    candidate_values = raw_candidate_ids.astype(int, copy=False)
    state_values = raw_state_ids.astype(int, copy=False)
    branch_indices = raw_branch_indices.astype(int, copy=False)
    if np.any(branch_indices < 0) or np.any(branch_indices >= fluxes.shape[1]) or np.unique(branch_indices).size != branch_indices.size:
        raise ValueError("branch_reaction_indices must contain unique valid reaction indices")
 
    selected_candidate_id = int(selected[0])
    selected_state_id = int(selected[1])
    candidate_rows = np.flatnonzero(candidate_values == selected_candidate_id)
    state_rows = np.flatnonzero(state_values == selected_state_id)
    if candidate_rows.size != 1 or state_rows.size != 1:
        raise ValueError("selected candidate and state IDs must each map to one row")
 
    contributions = metrics[int(candidate_rows[0]), int(state_rows[0]), 4:]
    if not np.all(np.isfinite(contributions)) or np.any(contributions < 0.0):
        raise ValueError("selected reaction enzyme-mass contributions must be finite and non-negative")
    denominator = float(np.sum(contributions, dtype=float))
    numerator = float(np.sum(contributions[branch_indices], dtype=float))
    if not np.isfinite(denominator) or denominator <= 0.0 or not np.isfinite(numerator):
        raise ValueError("selected enzyme-mass fraction is not finite")
    fraction = numerator / denominator
    if not np.isfinite(fraction):
        raise ValueError("final fraction must be finite")
    return float(fraction)
SCICODE_GOLD_EOF
