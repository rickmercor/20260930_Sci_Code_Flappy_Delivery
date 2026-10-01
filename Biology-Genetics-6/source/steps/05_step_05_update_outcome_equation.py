"""
Draw the four parameters of the outcome equation that carry no selection decision, the intercept, the outcome covariate coefficients, the receptor main effect and the residual variance of pathway activity, in that order and from the flat parameter state.

The outcome equation of this model is not the regression of pathway activity on the observed exposures, and the difference is the whole of the causal argument. It is a regression on a representation of the exposures built from the instruments and the covariates alone, and that substitution is what removes the latent donor factor, since the representation is a function of variables the factor is assumed independent of. The substitution has one consequence that is easy to miss and expensive to get wrong: it does not commute with the product that carries receptor modulated ligand action, and the discrepancy it leaves behind does not vary across donors. An equation that had no intercept would have nowhere to put that discrepancy and would be forced to push it into the very coefficient the analysis exists to report, so the intercept is mandatory here even though every observed column has been centred and an intercept would otherwise be redundant.




Three of the four parameters drawn here are ordinary linear model parameters and carry no selection, and their conditionals differ from textbook ones only through their priors. The covariate coefficients and the receptor main effect carry Zellner g priors, normal with mean zero and covariance g times the outcome residual variance times the inverse cross product matrix of their own design, which is what keeps them invariant to the scale of the measurements and makes any of this comparable across ligand receptor pathway triplets whose genes are quantified on unrelated scales. The intercept is the exception and is given a deliberately diffuse prior, normal with mean zero and variance the number of donors times the outcome residual variance, since it is absorbing an artefact of the substitution rather than estimating a biological quantity and the data should be left to determine it.




The residual variance is drawn last and is not a free standing scale parameter. Every prior in the outcome equation, the diffuse one on the intercept, those on the covariate coefficients and on the receptor main effect, and the spike and slab prior on the communication effect vector, is written with a covariance proportional to that same variance. That is exactly what makes them scale invariant, and it is also what makes the conditional distribution of the variance draw on all of them at once rather than on the residuals alone. Treating the variance as though only the residuals informed it is not a simplification of this model; it is a different model, and the selection behaviour it produces differs too, because the point at which the spike scale enters the variance is the point at which selection and scale are tied together.

Returns
-------
np.ndarray of n_covariates + 3 floats: the drawn intercept, the drawn outcome covariate coefficients, the drawn receptor main effect and the drawn residual variance of pathway activity.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def update_outcome_equation(data: np.ndarray, n_instruments: int, n_covariates: int,
                            state: np.ndarray, g_prior: float, nu_spike: float,
                            a_sigma: float, b_sigma: float, ridge: float,
                            rng: np.random.Generator) -> np.ndarray:
    '''Draw the intercept, the covariate coefficients, the receptor main effect and the outcome variance.

    The four conditionals are visited in the order in which they are returned,
    and each one conditions on the values the earlier ones have just taken
    rather than on the values they held on entry. The communication effect
    vector and the inclusion indicator are read from state and are not
    modified, and neither is any entry of the two exposure blocks. The spike
    and slab scale factor is one when the inclusion indicator held in state is
    at least one half and nu_spike otherwise. Every matrix inverted along the
    way carries ridge added to its diagonal beforehand.

    Randomness comes from rng, which draws one vector of n_covariates + 2
    standard normals and then one standard gamma variate. The first entry of
    the normal vector drives the intercept, the next n_covariates entries
    drive the covariate block through the lower Cholesky factor of its
    conditional covariance, and the last entry drives the receptor main
    effect. The residual variance is the scale of its inverse gamma
    conditional divided by the standard gamma variate.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (n_donors, 2 * n_instruments + n_covariates + 3) laid
        out as returned by the cohort simulation step.
    n_instruments : int
        Number of cis-eQTL instruments for each exposure, n_instruments >= 1.
    n_covariates : int
        Number of observed donor covariates, n_covariates >= 1.
    state : np.ndarray
        Array of 2 * n_instruments + 3 * n_covariates + 9 floats laid out as
        returned by the state initialisation step.
    g_prior : float
        Zellner g value shared by the outcome priors, strictly positive.
    nu_spike : float
        Scale factor of the spike component, 0 < nu_spike <= 1.
    a_sigma : float
        Shape of the inverse gamma prior on the variance, a_sigma > 0.
    b_sigma : float
        Scale of the inverse gamma prior on the variance, b_sigma > 0.
    ridge : float
        Non negative diagonal regularisation added before any inversion.
    rng : np.random.Generator
        Generator supplying the draws, in the order described above.

    Returns
    -------
    outcome_block : np.ndarray
        Array of n_covariates + 3 floats holding the drawn intercept, the
        drawn outcome covariate coefficients, the drawn receptor main effect
        and the drawn residual variance of pathway activity, in that order.

    Raises
    ------
    ValueError
        If data is not a two dimensional finite array with the column count
        implied by n_instruments and n_covariates, if either count is below
        one, if state is not a finite one dimensional array of the implied
        length, if the outcome residual variance held in state is not strictly
        positive, if the inclusion indicator held in state is neither zero nor
        one, if g_prior, a_sigma or b_sigma is not a positive finite number,
        if nu_spike is outside the half open interval from zero to one, if
        ridge is negative, or if a matrix that has to be inverted is not
        positive definite.
    '''
    return outcome_block  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_update_outcome_equation(data: np.ndarray, n_instruments: int, n_covariates: int,
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

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np
def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the inclusion indicator on, so the communication prior is on its slab scale ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(71)
n_donors, n_instruments, n_covariates = 90, 4, 3
instruments = rng.standard_normal((n_donors, 2 * n_instruments))
covariates = rng.standard_normal((n_donors, n_covariates))
confounder = rng.standard_normal(n_donors)
ligand = instruments[:, :n_instruments] @ np.full(n_instruments, 0.45) + covariates @ np.full(n_covariates, 0.25) + 0.7 * confounder + rng.standard_normal(n_donors)
receptor = instruments[:, n_instruments:] @ np.full(n_instruments, 0.45) + covariates @ np.full(n_covariates, 0.25) + 0.7 * confounder + rng.standard_normal(n_donors)
pathway = 0.5 * receptor + covariates @ np.full(n_covariates, 0.25) + 0.7 * confounder + rng.standard_normal(n_donors)
data = np.column_stack([instruments, covariates, ligand, receptor, pathway])
data = data - data.mean(axis=0)
state = np.full(2 * n_instruments + 3 * n_covariates + 9, 0.3)
state[n_instruments + n_covariates] = 1.1
state[2 * n_instruments + 2 * n_covariates + 1] = 0.9
state[2 * n_instruments + 3 * n_covariates + 4] = 1.5
state[2 * n_instruments + 3 * n_covariates + 7] = 1.0
state[2 * n_instruments + 3 * n_covariates + 8] = 0.75
def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(update_outcome_equation(data, n_instruments, n_covariates, state, 100.0, 1e-4, 3.0, 2.0, 1e-6, np.random.default_rng(9)))",
            "gold_call": "digest(_oracle_update_outcome_equation(data, n_instruments, n_covariates, state, 100.0, 1e-4, 3.0, 2.0, 1e-6, np.random.default_rng(9)))",
        },
        # --- Valid: the inclusion indicator off, so the spike scale governs the variance ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(79)
n_donors, n_instruments, n_covariates = 70, 3, 2
instruments = rng.standard_normal((n_donors, 2 * n_instruments))
covariates = rng.standard_normal((n_donors, n_covariates))
ligand = instruments[:, :n_instruments] @ np.full(n_instruments, 0.5) + rng.standard_normal(n_donors)
receptor = instruments[:, n_instruments:] @ np.full(n_instruments, 0.5) + rng.standard_normal(n_donors)
pathway = 0.5 * receptor + covariates @ np.full(n_covariates, 0.3) + rng.standard_normal(n_donors)
data = np.column_stack([instruments, covariates, ligand, receptor, pathway])
data = data - data.mean(axis=0)
state = np.linspace(-0.2, 0.7, 2 * n_instruments + 3 * n_covariates + 9)
state[n_instruments + n_covariates] = 1.0
state[2 * n_instruments + 2 * n_covariates + 1] = 1.0
state[2 * n_instruments + 3 * n_covariates + 4] = 1.2
state[2 * n_instruments + 3 * n_covariates + 5] = 0.02
state[2 * n_instruments + 3 * n_covariates + 6] = 0.01
state[2 * n_instruments + 3 * n_covariates + 7] = 0.0
state[2 * n_instruments + 3 * n_covariates + 8] = 0.4
def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(update_outcome_equation(data, n_instruments, n_covariates, state, 100.0, 1e-4, 3.0, 2.0, 1e-6, np.random.default_rng(4)))",
            "gold_call": "digest(_oracle_update_outcome_equation(data, n_instruments, n_covariates, state, 100.0, 1e-4, 3.0, 2.0, 1e-6, np.random.default_rng(4)))",
        },
        # --- Boundary: a communication effect vector of exactly zero ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(83)
n_donors, n_instruments, n_covariates = 55, 2, 2
instruments = rng.standard_normal((n_donors, 2 * n_instruments))
covariates = rng.standard_normal((n_donors, n_covariates))
ligand = instruments[:, :n_instruments] @ np.full(n_instruments, 0.5) + rng.standard_normal(n_donors)
receptor = instruments[:, n_instruments:] @ np.full(n_instruments, 0.5) + rng.standard_normal(n_donors)
pathway = 0.5 * receptor + rng.standard_normal(n_donors)
data = np.column_stack([instruments, covariates, ligand, receptor, pathway])
data = data - data.mean(axis=0)
state = np.full(2 * n_instruments + 3 * n_covariates + 9, 0.25)
state[n_instruments + n_covariates] = 1.0
state[2 * n_instruments + 2 * n_covariates + 1] = 1.0
state[2 * n_instruments + 3 * n_covariates + 4] = 1.0
state[2 * n_instruments + 3 * n_covariates + 5] = 0.0
state[2 * n_instruments + 3 * n_covariates + 6] = 0.0
state[2 * n_instruments + 3 * n_covariates + 7] = 1.0
state[2 * n_instruments + 3 * n_covariates + 8] = 0.75
def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(update_outcome_equation(data, n_instruments, n_covariates, state, 100.0, 1e-4, 3.0, 2.0, 1e-6, np.random.default_rng(15)))",
            "gold_call": "digest(_oracle_update_outcome_equation(data, n_instruments, n_covariates, state, 100.0, 1e-4, 3.0, 2.0, 1e-6, np.random.default_rng(15)))",
        },
        # --- Edge: a small cohort, one covariate and a g value at the cohort size ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(89)
n_donors, n_instruments, n_covariates = 16, 1, 1
instruments = rng.standard_normal((n_donors, 2 * n_instruments))
covariates = rng.standard_normal((n_donors, n_covariates))
ligand = instruments[:, :n_instruments] @ np.full(n_instruments, 0.5) + rng.standard_normal(n_donors)
receptor = instruments[:, n_instruments:] @ np.full(n_instruments, 0.5) + rng.standard_normal(n_donors)
pathway = 0.5 * receptor + rng.standard_normal(n_donors)
data = np.column_stack([instruments, covariates, ligand, receptor, pathway])
data = data - data.mean(axis=0)
state = np.linspace(-0.4, 0.5, 2 * n_instruments + 3 * n_covariates + 9)
state[n_instruments + n_covariates] = 0.8
state[2 * n_instruments + 2 * n_covariates + 1] = 1.3
state[2 * n_instruments + 3 * n_covariates + 4] = 0.9
state[2 * n_instruments + 3 * n_covariates + 7] = 1.0
state[2 * n_instruments + 3 * n_covariates + 8] = 0.5
def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(update_outcome_equation(data, n_instruments, n_covariates, state, 16.0, 1e-4, 3.0, 2.0, 1e-6, np.random.default_rng(1)))",
            "gold_call": "digest(_oracle_update_outcome_equation(data, n_instruments, n_covariates, state, 16.0, 1e-4, 3.0, 2.0, 1e-6, np.random.default_rng(1)))",
        },
        # --- Invalid: a state whose inclusion indicator is neither zero nor one ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(1)
data = rng.standard_normal((40, 2 * 2 + 2 + 3))
state = np.full(2 * 2 + 3 * 2 + 9, 0.3)
state[2 + 2] = 1.0
state[2 * 2 + 2 * 2 + 1] = 1.0
state[2 * 2 + 3 * 2 + 4] = 1.0
state[2 * 2 + 3 * 2 + 7] = 0.5
state[2 * 2 + 3 * 2 + 8] = 0.75
def run_model():
    try:
        update_outcome_equation(data, 2, 2, state, 100.0, 1e-4, 3.0, 2.0, 1e-6, np.random.default_rng(1))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_update_outcome_equation(data, 2, 2, state, 100.0, 1e-4, 3.0, 2.0, 1e-6, np.random.default_rng(1))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a spike scale factor above one ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(1)
data = rng.standard_normal((40, 2 * 2 + 2 + 3))
state = np.full(2 * 2 + 3 * 2 + 9, 0.3)
state[2 + 2] = 1.0
state[2 * 2 + 2 * 2 + 1] = 1.0
state[2 * 2 + 3 * 2 + 4] = 1.0
state[2 * 2 + 3 * 2 + 7] = 1.0
state[2 * 2 + 3 * 2 + 8] = 0.75
def run_model():
    try:
        update_outcome_equation(data, 2, 2, state, 100.0, 4.0, 3.0, 2.0, 1e-6, np.random.default_rng(1))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_update_outcome_equation(data, 2, 2, state, 100.0, 4.0, 3.0, 2.0, 1e-6, np.random.default_rng(1))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
