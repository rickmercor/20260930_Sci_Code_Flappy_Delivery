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

def simulate_communication_dataset(n_donors: int, n_instruments: int, n_covariates: int,
                                           instrument_strength: float, covariate_effect: float,
                                           receptor_effect: float, confounder_loading: float,
                                           ligand_effect: float, interaction_effect: float,
                                           seed: int) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    def _check_int(name, value, low):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if int(value) < low:
            raise ValueError(f"{name} must be at least {low}")
        return int(value)

    def _check_real(name, value):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
        return float(value)

    n_instrument = _check_int("n_instruments", n_instruments, 1)
    n_covariate = _check_int("n_covariates", n_covariates, 1)
    smallest = 2 * n_instrument + n_covariate + 5
    n_row = _check_int("n_donors", n_donors, smallest)
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    pi_value = _check_real("instrument_strength", instrument_strength)
    alpha_value = _check_real("covariate_effect", covariate_effect)
    beta_receptor = _check_real("receptor_effect", receptor_effect)
    lambda_value = _check_real("confounder_loading", confounder_loading)
    beta_ligand = _check_real("ligand_effect", ligand_effect)
    beta_interaction = _check_real("interaction_effect", interaction_effect)

    rng = np.random.default_rng(int(seed))
    ligand_instruments = rng.standard_normal((n_row, n_instrument))
    receptor_instruments = rng.standard_normal((n_row, n_instrument))
    covariates = rng.standard_normal((n_row, n_covariate))
    confounder = rng.standard_normal(n_row)
    ligand_residual = rng.standard_normal(n_row)
    receptor_residual = rng.standard_normal(n_row)
    pathway_residual = rng.standard_normal(n_row)

    instrument_vector = np.full(n_instrument, pi_value)
    covariate_vector = np.full(n_covariate, alpha_value)
    covariate_signal = covariates @ covariate_vector

    # The two exposure equations of the structural model.
    ligand = (ligand_instruments @ instrument_vector + covariate_signal
              + lambda_value * confounder + ligand_residual)
    receptor = (receptor_instruments @ instrument_vector + covariate_signal
                + lambda_value * confounder + receptor_residual)

    # The outcome equation, with the receptor modulated ligand effect entering
    # as the elementwise product of the two uncentred exposures.
    pathway = (beta_ligand * ligand + beta_receptor * receptor
               + beta_interaction * ligand * receptor + covariate_signal
               + lambda_value * confounder + pathway_residual)

    data = np.column_stack([ligand_instruments, receptor_instruments, covariates,
                            ligand, receptor, pathway])
    return np.asarray(data - data.mean(axis=0), dtype=float)

import numpy as np 

def compute_naive_communication_score(data: np.ndarray, n_instruments: int,
                                              n_covariates: int) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np
    from scipy.stats import f as f_distribution

    def _check_int(name, value, low):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if int(value) < low:
            raise ValueError(f"{name} must be at least {low}")
        return int(value)

    n_instrument = _check_int("n_instruments", n_instruments, 1)
    n_covariate = _check_int("n_covariates", n_covariates, 1)
    table = np.asarray(data, dtype=float)
    if table.ndim != 2:
        raise ValueError("data must be a two dimensional array")
    if table.shape[1] != 2 * n_instrument + n_covariate + 3:
        raise ValueError("data has the wrong number of columns for these counts")
    if not np.all(np.isfinite(table)):
        raise ValueError("data must be finite")

    n_row = table.shape[0]
    offset = 2 * n_instrument + n_covariate
    covariates = table[:, offset - n_covariate:offset]
    ligand = table[:, offset]
    receptor = table[:, offset + 1]
    pathway = table[:, offset + 2]

    design = np.column_stack([np.ones(n_row), ligand, receptor,
                              ligand * receptor, covariates])
    n_parameter = design.shape[1]
    residual_degrees = n_row - n_parameter
    if residual_degrees < 1:
        raise ValueError("the cohort leaves no residual degrees of freedom")

    gram = design.T @ design
    if not np.isfinite(np.linalg.cond(gram)) or np.linalg.cond(gram) > 1.0e12:
        raise ValueError("the associational design is numerically singular")
    gram_inverse = np.linalg.inv(gram)
    coefficients = gram_inverse @ (design.T @ pathway)
    residual = pathway - design @ coefficients
    residual_variance = float(residual @ residual) / residual_degrees
    coefficient_covariance = residual_variance * gram_inverse

    # The contrast picks the ligand coefficient and the ligand by receptor
    # coefficient, the two terms that carry communication under this model.
    contrast = np.zeros((2, n_parameter))
    contrast[0, 1] = 1.0
    contrast[1, 3] = 1.0
    contrast_effect = contrast @ coefficients
    contrast_covariance = contrast @ coefficient_covariance @ contrast.T
    if np.linalg.cond(contrast_covariance) > 1.0e12:
        raise ValueError("the contrast covariance is numerically singular")
    statistic = float(contrast_effect @ np.linalg.solve(contrast_covariance,
                                                        contrast_effect)) / 2.0
    if not np.isfinite(statistic) or statistic <= 0.0:
        raise ValueError("the joint F statistic is not a positive finite number")

    # Both tails are taken in logarithmic form so that the odds survive a tail
    # probability far below the smallest representable double.
    log_upper = float(f_distribution.logsf(statistic, 2, residual_degrees))
    log_lower = float(f_distribution.logcdf(statistic, 2, residual_degrees))
    log10_odds = (log_lower - log_upper) / float(np.log(10.0))
    if not np.isfinite(log10_odds):
        raise ValueError("the associational communication odds are not finite")

    return np.asarray([statistic, log10_odds, coefficients[1], coefficients[3]],
                      dtype=float)

def initialise_gibbs_state(data: np.ndarray, n_instruments: int, n_covariates: int,
                                   a_rho: float, b_rho: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    def _check_int(name, value, low):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if int(value) < low:
            raise ValueError(f"{name} must be at least {low}")
        return int(value)

    def _check_positive(name, value):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number")
        if not (np.isfinite(float(value)) and float(value) > 0.0):
            raise ValueError(f"{name} must be a positive finite number")
        return float(value)

    n_instrument = _check_int("n_instruments", n_instruments, 1)
    n_covariate = _check_int("n_covariates", n_covariates, 1)
    shape_a = _check_positive("a_rho", a_rho)
    shape_b = _check_positive("b_rho", b_rho)
    table = np.asarray(data, dtype=float)
    if table.ndim != 2:
        raise ValueError("data must be a two dimensional array")
    if table.shape[1] != 2 * n_instrument + n_covariate + 3:
        raise ValueError("data has the wrong number of columns for these counts")
    if not np.all(np.isfinite(table)):
        raise ValueError("data must be finite")

    n_row = table.shape[0]
    ligand_instruments = table[:, :n_instrument]
    receptor_instruments = table[:, n_instrument:2 * n_instrument]
    offset = 2 * n_instrument + n_covariate
    covariates = table[:, offset - n_covariate:offset]
    ligand = table[:, offset]
    receptor = table[:, offset + 1]
    pathway = table[:, offset + 2]

    exposure_columns = n_instrument + n_covariate
    if n_row - exposure_columns < 1 or n_row - (4 + n_covariate) < 1:
        raise ValueError("the cohort leaves no residual degrees of freedom")

    def _least_squares(design, response):
        coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
        residual = response - design @ coefficients
        degrees = max(design.shape[0] - design.shape[1], 1)
        variance = max(float(residual @ residual) / degrees, 1.0e-8)
        return coefficients, design @ coefficients, variance

    ligand_design = np.column_stack([ligand_instruments, covariates])
    ligand_coefficients, ligand_fit, ligand_variance = _least_squares(ligand_design, ligand)
    receptor_design = np.column_stack([receptor_instruments, covariates])
    receptor_coefficients, receptor_fit, receptor_variance = _least_squares(
        receptor_design, receptor)

    # The outcome fit uses the instrument based conditional means, and its
    # product column is built from those same fitted values.
    outcome_design = np.column_stack([np.ones(n_row), ligand_fit, receptor_fit,
                                      ligand_fit * receptor_fit, covariates])
    outcome_coefficients, _, outcome_variance = _least_squares(outcome_design, pathway)

    state = np.concatenate([
        ligand_coefficients[:n_instrument],
        ligand_coefficients[n_instrument:],
        [ligand_variance],
        receptor_coefficients[:n_instrument],
        receptor_coefficients[n_instrument:],
        [receptor_variance],
        [outcome_coefficients[0]],
        outcome_coefficients[4:],
        [outcome_coefficients[2]],
        [outcome_variance],
        [outcome_coefficients[1]],
        [outcome_coefficients[3]],
        [1.0],
        [shape_a / (shape_a + shape_b)],
    ])
    return np.asarray(state, dtype=float)

import numpy as np 

def update_exposure_equations(data: np.ndarray, n_instruments: int, n_covariates: int,
                                      state: np.ndarray, g_prior: float, a_sigma: float,
                                      b_sigma: float, ridge: float,
                                      rng: np.random.Generator) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    def _check_int(name, value, low):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if int(value) < low:
            raise ValueError(f"{name} must be at least {low}")
        return int(value)

    def _check_positive(name, value):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number")
        if not (np.isfinite(float(value)) and float(value) > 0.0):
            raise ValueError(f"{name} must be a positive finite number")
        return float(value)

    n_instrument = _check_int("n_instruments", n_instruments, 1)
    n_covariate = _check_int("n_covariates", n_covariates, 1)
    g_value = _check_positive("g_prior", g_prior)
    shape_prior = _check_positive("a_sigma", a_sigma)
    scale_prior = _check_positive("b_sigma", b_sigma)
    if isinstance(ridge, bool) or not isinstance(ridge, (int, float, np.integer, np.floating)):
        raise ValueError("ridge must be a real number")
    if not (np.isfinite(float(ridge)) and float(ridge) >= 0.0):
        raise ValueError("ridge must be a non negative finite number")
    ridge_value = float(ridge)
    table = np.asarray(data, dtype=float)
    if table.ndim != 2 or table.shape[1] != 2 * n_instrument + n_covariate + 3:
        raise ValueError("data has the wrong shape for these counts")
    if not np.all(np.isfinite(table)):
        raise ValueError("data must be finite")
    work = np.asarray(state, dtype=float).ravel().copy()
    if work.size != 2 * n_instrument + 3 * n_covariate + 9:
        raise ValueError("state has the wrong length for these counts")
    if not np.all(np.isfinite(work)):
        raise ValueError("state must be finite")

    n_row = table.shape[0]
    ligand_instruments = table[:, :n_instrument]
    receptor_instruments = table[:, n_instrument:2 * n_instrument]
    offset = 2 * n_instrument + n_covariate
    covariates = table[:, offset - n_covariate:offset]
    ligand = table[:, offset]
    receptor = table[:, offset + 1]
    pathway = table[:, offset + 2]

    ligand_instrument_slice = slice(0, n_instrument)
    ligand_covariate_slice = slice(n_instrument, n_instrument + n_covariate)
    ligand_variance_index = n_instrument + n_covariate
    receptor_instrument_slice = slice(ligand_variance_index + 1,
                                      ligand_variance_index + 1 + n_instrument)
    receptor_covariate_slice = slice(receptor_instrument_slice.stop,
                                     receptor_instrument_slice.stop + n_covariate)
    receptor_variance_index = receptor_covariate_slice.stop
    intercept_index = receptor_variance_index + 1
    outcome_covariate_slice = slice(intercept_index + 1, intercept_index + 1 + n_covariate)
    receptor_effect_index = outcome_covariate_slice.stop
    outcome_variance_index = receptor_effect_index + 1
    ligand_effect_index = outcome_variance_index + 1
    interaction_index = ligand_effect_index + 1
    for index in (ligand_variance_index, receptor_variance_index, outcome_variance_index):
        if work[index] <= 0.0:
            raise ValueError("state carries a residual variance that is not positive")
    pathway_variance = float(work[outcome_variance_index])

    def _draw(design, modulation, exposure_residual, outcome_residual,
              exposure_variance, normal):
        # An exposure block is informed by its own equation and, through the
        # donor specific weight, by the outcome equation as well.
        scaled = design * modulation[:, None]
        precision = ((1.0 + 1.0 / g_value) * (design.T @ design) / exposure_variance
                     + (scaled.T @ scaled) / pathway_variance
                     + ridge_value * np.eye(design.shape[1]))
        try:
            covariance = np.linalg.inv(precision)
            factor = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as error:
            raise ValueError("an exposure precision matrix is not positive definite") from error
        mean = covariance @ (design.T @ exposure_residual / exposure_variance
                             + scaled.T @ outcome_residual / pathway_variance)
        return mean + factor @ normal

    def _equation(instruments, exposure, modulation, base, instrument_slice,
                  covariate_slice, variance_index):
        exposure_variance = float(work[variance_index])
        covariate_fit = covariates @ work[covariate_slice]
        instrument_effect = _draw(instruments, modulation, exposure - covariate_fit,
                                  base - covariate_fit * modulation, exposure_variance,
                                  rng.standard_normal(n_instrument))
        instrument_fit = instruments @ instrument_effect
        covariate_effect = _draw(covariates, modulation, exposure - instrument_fit,
                                 base - instrument_fit * modulation, exposure_variance,
                                 rng.standard_normal(n_covariate))
        residual = exposure - instrument_fit - covariates @ covariate_effect
        shape = shape_prior + 0.5 * (n_row + n_instrument + n_covariate)
        scale = scale_prior + 0.5 * (
            float(residual @ residual)
            + float(instrument_effect @ (instruments.T @ instruments)
                    @ instrument_effect) / g_value
            + float(covariate_effect @ (covariates.T @ covariates)
                    @ covariate_effect) / g_value)
        variance = scale / float(rng.standard_gamma(shape))
        work[instrument_slice] = instrument_effect
        work[covariate_slice] = covariate_effect
        work[variance_index] = variance
        return np.concatenate([instrument_effect, covariate_effect, [variance]])

    def _representation(instruments, instrument_slice, covariate_slice):
        return instruments @ work[instrument_slice] + covariates @ work[covariate_slice]

    # The ligand equation is drawn against the representation the state
    # currently implies.
    receptor_representation = _representation(receptor_instruments,
                                              receptor_instrument_slice,
                                              receptor_covariate_slice)
    unexplained = (pathway - work[intercept_index]
                   - work[receptor_effect_index] * receptor_representation
                   - covariates @ work[outcome_covariate_slice])
    drawn_ligand = _equation(
        ligand_instruments, ligand,
        work[ligand_effect_index] + work[interaction_index] * receptor_representation,
        unexplained, ligand_instrument_slice, ligand_covariate_slice,
        ligand_variance_index)

    # The receptor equation conditions on the ligand equation as it now
    # stands, so its half of the representation is rebuilt rather than reused.
    ligand_representation = _representation(ligand_instruments,
                                            ligand_instrument_slice,
                                            ligand_covariate_slice)
    unexplained = (pathway - work[intercept_index]
                   - work[ligand_effect_index] * ligand_representation
                   - covariates @ work[outcome_covariate_slice])
    drawn_receptor = _equation(
        receptor_instruments, receptor,
        work[receptor_effect_index] + work[interaction_index] * ligand_representation,
        unexplained, receptor_instrument_slice, receptor_covariate_slice,
        receptor_variance_index)

    return np.concatenate([drawn_ligand, drawn_receptor]).astype(float)

import numpy as np
def update_outcome_equation(data: np.ndarray, n_instruments: int, n_covariates: int,
                                    state: np.ndarray, g_prior: float, nu_spike: float,
                                    a_sigma: float, b_sigma: float, ridge: float,
                                    rng: np.random.Generator) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    def _check_int(name, value, low):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if int(value) < low:
            raise ValueError(f"{name} must be at least {low}")
        return int(value)

    def _check_positive(name, value):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number")
        if not (np.isfinite(float(value)) and float(value) > 0.0):
            raise ValueError(f"{name} must be a positive finite number")
        return float(value)

    n_instrument = _check_int("n_instruments", n_instruments, 1)
    n_covariate = _check_int("n_covariates", n_covariates, 1)
    g_value = _check_positive("g_prior", g_prior)
    spike = _check_positive("nu_spike", nu_spike)
    if spike > 1.0:
        raise ValueError("nu_spike must not exceed one")
    shape_prior = _check_positive("a_sigma", a_sigma)
    scale_prior = _check_positive("b_sigma", b_sigma)
    if isinstance(ridge, bool) or not isinstance(ridge, (int, float, np.integer, np.floating)):
        raise ValueError("ridge must be a real number")
    if not (np.isfinite(float(ridge)) and float(ridge) >= 0.0):
        raise ValueError("ridge must be a non negative finite number")
    ridge_value = float(ridge)
    table = np.asarray(data, dtype=float)
    if table.ndim != 2 or table.shape[1] != 2 * n_instrument + n_covariate + 3:
        raise ValueError("data has the wrong shape for these counts")
    if not np.all(np.isfinite(table)):
        raise ValueError("data must be finite")
    parameters = np.asarray(state, dtype=float).ravel()
    if parameters.size != 2 * n_instrument + 3 * n_covariate + 9:
        raise ValueError("state has the wrong length for these counts")
    if not np.all(np.isfinite(parameters)):
        raise ValueError("state must be finite")

    n_row = table.shape[0]
    offset = 2 * n_instrument + n_covariate
    covariates = table[:, offset - n_covariate:offset]
    pathway = table[:, offset + 2]
    cursor = n_instrument + n_covariate
    ligand_representation = (table[:, :n_instrument] @ parameters[0:n_instrument]
                             + covariates @ parameters[n_instrument:cursor])
    receptor_representation = (
        table[:, n_instrument:2 * n_instrument] @ parameters[cursor + 1:cursor + 1 + n_instrument]
        + covariates @ parameters[cursor + 1 + n_instrument:2 * cursor + 1])
    intercept_index = 2 * n_instrument + 2 * n_covariate + 2
    outcome_covariate_slice = slice(intercept_index + 1, intercept_index + 1 + n_covariate)
    receptor_effect_index = outcome_covariate_slice.stop
    outcome_variance = float(parameters[receptor_effect_index + 1])
    if outcome_variance <= 0.0:
        raise ValueError("state carries an outcome variance that is not positive")
    inclusion = float(parameters[receptor_effect_index + 4])
    if inclusion not in (0.0, 1.0):
        raise ValueError("state carries an inclusion indicator that is neither zero nor one")
    communication = parameters[receptor_effect_index + 2:receptor_effect_index + 4]
    current_covariate = parameters[outcome_covariate_slice]
    current_receptor = float(parameters[receptor_effect_index])

    # The column that carries receptor modulated ligand action is built from
    # the same representation as the two main effect columns.
    communication_design = np.column_stack(
        [ligand_representation, ligand_representation * receptor_representation])
    communication_fit = communication_design @ communication
    draws = rng.standard_normal(n_covariate + 2)

    denominator = n_row + 1.0 / n_row
    left = (pathway - communication_fit - current_receptor * receptor_representation
            - covariates @ current_covariate)
    intercept = (float(left.sum()) / denominator
                 + np.sqrt(outcome_variance / denominator) * draws[0])

    shrinkage = g_value / (1.0 + g_value)
    gram = covariates.T @ covariates + ridge_value * np.eye(n_covariate)
    try:
        gram_inverse = np.linalg.inv(gram)
        factor = np.linalg.cholesky(shrinkage * outcome_variance * gram_inverse)
    except np.linalg.LinAlgError as error:
        raise ValueError("the covariate covariance is not positive definite") from error
    left = pathway - intercept - communication_fit - current_receptor * receptor_representation
    covariate_new = (shrinkage * (gram_inverse @ (covariates.T @ left))
                     + factor @ draws[1:1 + n_covariate])

    receptor_sum_squares = float(receptor_representation @ receptor_representation) + ridge_value
    if receptor_sum_squares <= 0.0:
        raise ValueError("the receptor representation carries no variation")
    left = pathway - intercept - communication_fit - covariates @ covariate_new
    receptor_new = (shrinkage * float(receptor_representation @ left) / receptor_sum_squares
                    + np.sqrt(shrinkage * outcome_variance / receptor_sum_squares)
                    * draws[1 + n_covariate])

    # Every prior written in units of this variance contributes to its scale,
    # the communication prior through whichever of its two components is live.
    spike_factor = 1.0 if inclusion >= 0.5 else spike
    communication_gram = (communication_design.T @ communication_design
                          + ridge_value * np.eye(2))
    residual = (pathway - intercept - communication_fit
                - receptor_new * receptor_representation - covariates @ covariate_new)
    shape = shape_prior + 0.5 * (n_row + n_covariate + 4)
    scale = scale_prior + 0.5 * (
        float(residual @ residual)
        + intercept * intercept / n_row
        + float(communication @ communication_gram @ communication) / (g_value * spike_factor)
        + receptor_new * receptor_new * float(receptor_representation
                                              @ receptor_representation) / g_value
        + float(covariate_new @ (covariates.T @ covariates) @ covariate_new) / g_value)
    variance = scale / float(rng.standard_gamma(shape))

    return np.concatenate([[intercept], covariate_new, [receptor_new],
                           [variance]]).astype(float)

import numpy as np
def update_communication_state(data: np.ndarray, n_instruments: int, n_covariates: int,
                                       state: np.ndarray, g_prior: float, nu_spike: float,
                                       a_rho: float, b_rho: float, ridge: float,
                                       rng: np.random.Generator) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    def _check_int(name, value, low):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if int(value) < low:
            raise ValueError(f"{name} must be at least {low}")
        return int(value)

    def _check_positive(name, value):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number")
        if not (np.isfinite(float(value)) and float(value) > 0.0):
            raise ValueError(f"{name} must be a positive finite number")
        return float(value)

    n_instrument = _check_int("n_instruments", n_instruments, 1)
    n_covariate = _check_int("n_covariates", n_covariates, 1)
    g_value = _check_positive("g_prior", g_prior)
    spike = _check_positive("nu_spike", nu_spike)
    if spike > 1.0:
        raise ValueError("nu_spike must not exceed one")
    shape_a = _check_positive("a_rho", a_rho)
    shape_b = _check_positive("b_rho", b_rho)
    if isinstance(ridge, bool) or not isinstance(ridge, (int, float, np.integer, np.floating)):
        raise ValueError("ridge must be a real number")
    if not (np.isfinite(float(ridge)) and float(ridge) >= 0.0):
        raise ValueError("ridge must be a non negative finite number")
    ridge_value = float(ridge)
    table = np.asarray(data, dtype=float)
    if table.ndim != 2 or table.shape[1] != 2 * n_instrument + n_covariate + 3:
        raise ValueError("data has the wrong shape for these counts")
    if not np.all(np.isfinite(table)):
        raise ValueError("data must be finite")
    parameters = np.asarray(state, dtype=float).ravel()
    if parameters.size != 2 * n_instrument + 3 * n_covariate + 9:
        raise ValueError("state has the wrong length for these counts")
    if not np.all(np.isfinite(parameters)):
        raise ValueError("state must be finite")

    offset = 2 * n_instrument + n_covariate
    covariates = table[:, offset - n_covariate:offset]
    pathway = table[:, offset + 2]
    cursor = n_instrument + n_covariate
    ligand_representation = (table[:, :n_instrument] @ parameters[0:n_instrument]
                             + covariates @ parameters[n_instrument:cursor])
    receptor_representation = (
        table[:, n_instrument:2 * n_instrument] @ parameters[cursor + 1:cursor + 1 + n_instrument]
        + covariates @ parameters[cursor + 1 + n_instrument:2 * cursor + 1])
    intercept = float(parameters[2 * n_instrument + 2 * n_covariate + 2])
    outcome_covariate_slice = slice(2 * n_instrument + 2 * n_covariate + 3,
                                    2 * n_instrument + 3 * n_covariate + 3)
    receptor_effect = float(parameters[outcome_covariate_slice.stop])
    outcome_variance = float(parameters[outcome_covariate_slice.stop + 1])
    if outcome_variance <= 0.0:
        raise ValueError("state carries an outcome variance that is not positive")
    inclusion = float(parameters[outcome_covariate_slice.stop + 4])
    if inclusion not in (0.0, 1.0):
        raise ValueError("state carries an inclusion indicator that is neither zero nor one")
    prior_probability = float(parameters[outcome_covariate_slice.stop + 5])
    if not 0.0 < prior_probability < 1.0:
        raise ValueError("state carries an inclusion probability outside the unit interval")

    # The two columns the communication claim is carried by, both built from
    # the instrument based representation rather than from observed expression.
    communication_design = np.column_stack(
        [ligand_representation, ligand_representation * receptor_representation])
    gram = communication_design.T @ communication_design + ridge_value * np.eye(2)

    # Under the spike the effective g value collapses by four orders of
    # magnitude, which is what pins the drawn vector near zero.
    effective_g = g_value * (1.0 if inclusion >= 0.5 else spike)
    shrinkage = effective_g / (1.0 + effective_g)
    left = (pathway - intercept - receptor_effect * receptor_representation
            - covariates @ parameters[outcome_covariate_slice])
    try:
        gram_inverse = np.linalg.inv(gram)
        factor = np.linalg.cholesky(shrinkage * outcome_variance * gram_inverse)
    except np.linalg.LinAlgError as error:
        raise ValueError("the communication covariance is not positive definite") from error
    communication = (shrinkage * (gram_inverse @ (communication_design.T @ left))
                     + factor @ rng.standard_normal(2))

    # The determinant of a two dimensional spike covariance contributes the
    # reciprocal of the spike factor, which is the penalty the slab must
    # overcome, and the comparison is normalised on the log scale because the
    # two exponents differ by orders of magnitude.
    quadratic = float(communication @ gram @ communication)
    log_slab = (-0.5 * quadratic / (g_value * outcome_variance)
                + np.log(prior_probability))
    log_spike = (-0.5 * quadratic / (g_value * spike * outcome_variance)
                 + np.log1p(-prior_probability) - np.log(spike))
    largest = max(log_slab, log_spike)
    slab_weight = np.exp(log_slab - largest)
    spike_weight = np.exp(log_spike - largest)
    conditional = float(slab_weight / (slab_weight + spike_weight))

    indicator = 1.0 if float(rng.random()) < conditional else 0.0
    refreshed = float(rng.beta(shape_a + indicator, shape_b + 1.0 - indicator))
    return np.asarray([communication[0], communication[1], indicator, conditional,
                       refreshed], dtype=float)

import numpy as np
def summarise_communication_evidence(naive_summary: np.ndarray,
                                             posterior_inclusion_probability: float,
                                             posterior_ligand_effect: float,
                                             posterior_interaction_effect: float,
                                             ligand_sd: float, receptor_sd: float,
                                             pathway_sd: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    def _check_positive(name, value):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number")
        if not (np.isfinite(float(value)) and float(value) > 0.0):
            raise ValueError(f"{name} must be a positive finite number")
        return float(value)

    def _check_real(name, value):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
        return float(value)

    summary = np.asarray(naive_summary, dtype=float).ravel()
    if summary.size != 4:
        raise ValueError("naive_summary must hold exactly four floats")
    if not np.all(np.isfinite(summary)):
        raise ValueError("naive_summary must be finite")
    if isinstance(posterior_inclusion_probability, bool) or not isinstance(
            posterior_inclusion_probability, (int, float, np.integer, np.floating)):
        raise ValueError("posterior_inclusion_probability must be a real number")
    inclusion = float(posterior_inclusion_probability)
    if not (np.isfinite(inclusion) and 0.0 < inclusion < 1.0):
        raise ValueError("posterior_inclusion_probability must lie strictly inside "
                         "the unit interval")
    ligand_effect = _check_real("posterior_ligand_effect", posterior_ligand_effect)
    interaction_effect = _check_real("posterior_interaction_effect",
                                     posterior_interaction_effect)
    ligand_spread = _check_positive("ligand_sd", ligand_sd)
    receptor_spread = _check_positive("receptor_sd", receptor_sd)
    pathway_spread = _check_positive("pathway_sd", pathway_sd)

    # Both verdicts are read as odds, which keeps the comparison finite where a
    # score on the unit interval would saturate.
    naive_log_odds = float(summary[1])
    causal_log_odds = float(np.log10(inclusion / (1.0 - inclusion)))
    gap = naive_log_odds - causal_log_odds

    # The main effect carries one exposure and the interaction carries two, so
    # they are standardised by a different number of exposure spreads.
    standardised_main = ligand_effect * ligand_spread / pathway_spread
    standardised_interaction = (interaction_effect * ligand_spread * receptor_spread
                                / pathway_spread)
    return np.asarray([gap, naive_log_odds, causal_log_odds, standardised_main,
                       standardised_interaction], dtype=float)

# ORACLE SOLUTION

import numpy as np
def run_causal_communication_pipeline(n_donors: int = 600, n_instruments: int = 4,
                                              n_covariates: int = 3,
                                              instrument_strength: float = 0.45,
                                              covariate_effect: float = 0.25,
                                              receptor_effect: float = 0.5,
                                              confounder_loading: float = 0.7,
                                              ligand_effect: float = 0.0,
                                              interaction_effect: float = 0.0,
                                              data_seed: int = 20260826,
                                              n_iterations: int = 20000, burn_in: int = 2000,
                                              thin: int = 5, nu_spike: float = 1e-4,
                                              a_sigma: float = 3.0, b_sigma: float = 2.0,
                                              a_rho: float = 3.0, b_rho: float = 1.0,
                                              ridge: float = 1e-6,
                                              sampler_seed: int = 2026) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    if isinstance(n_iterations, bool) or not isinstance(n_iterations, (int, np.integer)):
        raise ValueError("n_iterations must be an integer")
    if isinstance(burn_in, bool) or not isinstance(burn_in, (int, np.integer)):
        raise ValueError("burn_in must be an integer")
    if isinstance(thin, bool) or not isinstance(thin, (int, np.integer)):
        raise ValueError("thin must be an integer")
    if int(burn_in) < 0:
        raise ValueError("burn_in must not be negative")
    if int(thin) < 1:
        raise ValueError("thin must be at least one")
    if int(n_iterations) <= int(burn_in):
        raise ValueError("n_iterations must exceed burn_in")
    sweeps = int(n_iterations)
    discarded = int(burn_in)
    spacing = int(thin)

    # -- Sub-problem 01: the cohort every later step reads.
    data = simulate_communication_dataset(n_donors, n_instruments, n_covariates,
                                                  instrument_strength, covariate_effect,
                                                  receptor_effect, confounder_loading,
                                                  ligand_effect, interaction_effect, data_seed)

    # -- Sub-problem 02: the associational verdict on the same cohort.
    naive_summary = compute_naive_communication_score(data, n_instruments, n_covariates)

    n_instrument = int(n_instruments)
    n_covariate = int(n_covariates)
    offset = 2 * n_instrument + n_covariate
    ligand = data[:, offset]
    receptor = data[:, offset + 1]
    pathway = data[:, offset + 2]

    # -- Index of every block inside the flat parameter state.
    exposure_block = slice(0, 2 * (n_instrument + n_covariate + 1))
    intercept_index = exposure_block.stop
    outcome_covariate_slice = slice(intercept_index + 1, intercept_index + 1 + n_covariate)
    receptor_effect_index = outcome_covariate_slice.stop
    outcome_variance_index = receptor_effect_index + 1
    ligand_effect_index = outcome_variance_index + 1
    interaction_index = ligand_effect_index + 1
    inclusion_index = interaction_index + 1
    inclusion_probability_index = inclusion_index + 1

    # -- Sub-problem 03: the starting point of the sampler.
    state = initialise_gibbs_state(data, n_instruments, n_covariates, a_rho, b_rho)
    g_prior = float(min(int(n_donors), 100))
    rng = np.random.default_rng(int(sampler_seed))
    inclusion_trace = []
    ligand_trace = []
    interaction_trace = []

    for sweep in range(sweeps):
        # -- Sub-problem 04: both exposure equations.
        exposures = update_exposure_equations(data, n_instruments, n_covariates, state,
                                                      g_prior, a_sigma, b_sigma, ridge, rng)
        state[exposure_block] = exposures

        # -- Sub-problem 05: the four outcome parameters that carry no selection.
        outcome = update_outcome_equation(data, n_instruments, n_covariates, state,
                                                  g_prior, nu_spike, a_sigma, b_sigma,
                                                  ridge, rng)
        state[intercept_index] = outcome[0]
        state[outcome_covariate_slice] = outcome[1:1 + n_covariate]
        state[receptor_effect_index] = outcome[1 + n_covariate]
        state[outcome_variance_index] = outcome[2 + n_covariate]

        # -- Sub-problem 06: the communication effect vector and its selection.
        communication = update_communication_state(data, n_instruments, n_covariates,
                                                           state, g_prior, nu_spike, a_rho,
                                                           b_rho, ridge, rng)
        state[ligand_effect_index] = communication[0]
        state[interaction_index] = communication[1]
        state[inclusion_index] = communication[2]
        state[inclusion_probability_index] = communication[4]

        if sweep >= discarded and (sweep - discarded) % spacing == 0:
            inclusion_trace.append(float(state[inclusion_index]))
            ligand_trace.append(float(state[ligand_effect_index]))
            interaction_trace.append(float(state[interaction_index]))

    posterior_inclusion = float(np.mean(inclusion_trace))
    if not 0.0 < posterior_inclusion < 1.0:
        raise ValueError("the retained sweeps leave the communication odds undefined")

    # -- Sub-problem 07: both verdicts on one odds scale.
    evidence = summarise_communication_evidence(naive_summary, posterior_inclusion,
                                                        float(np.mean(ligand_trace)),
                                                        float(np.mean(interaction_trace)),
                                                        float(np.std(ligand)),
                                                        float(np.std(receptor)),
                                                        float(np.std(pathway)))
    return float(evidence[0])
SCICODE_GOLD_EOF
