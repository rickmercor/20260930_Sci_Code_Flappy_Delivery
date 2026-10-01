"""
Draw both exposure equations of one sampler sweep from the flat parameter state, refreshing the instrument coefficients, the covariate coefficients and the residual variance of the ligand equation and then those of the receptor equation.

An instrumental variable analysis that fits its exposure equations once and then treats the fitted values as though they were observed data is a two stage procedure, and its second stage inherits none of the uncertainty of its first. A Bayesian formulation fits the two jointly instead, which changes what the exposure coefficients are informed by. They are informed twice: once by how well they explain the exposure they belong to, and once by how well the representation they generate explains pathway activity, since the outcome equation of this model is written in terms of those representations rather than in terms of the observed exposures. Discarding the second contribution is what a two stage procedure does, and it is the standard criticism of using such a procedure as though its first stage were exact.




The consequence for a Gibbs sweep is that neither exposure equation can be updated in isolation, and that the weight with which an exposure enters the outcome is not a constant. It depends on the outcome coefficients currently held and on the other exposure, so it has to be recomputed from the state every time the state moves. A sweep that drew both exposure blocks against a representation computed once on entry would not be a Gibbs sweep for this model at all; it would be a valid sweep for a different model, it would converge perfectly happily, and it would converge to the wrong posterior.




The exposure equations carry no intercept, because every column of the cohort has been centred. Each coefficient block of an exposure equation carries a Zellner g prior, normal with mean zero and covariance g times that equation's residual variance times the inverse cross product matrix of the block's own design, and each residual variance carries an inverse gamma prior with the shape and scale supplied to this step. Writing coefficient priors in units of the residual variance is what keeps the specification invariant to the scale on which expression happens to be measured, and it is also why a residual variance here is informed by the coefficients as well as by the residuals. What remains is bookkeeping, but bookkeeping of a kind that fails silently: a conditional distribution that omits one contribution is still a proper distribution, and a sampler built from it still mixes.

Returns
-------
np.ndarray of 2 * (n_instruments + n_covariates + 1) floats: the drawn instrument coefficients, covariate coefficients and residual variance of the ligand equation, followed by those of the receptor equation.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def update_exposure_equations(data: np.ndarray, n_instruments: int, n_covariates: int,
                              state: np.ndarray, g_prior: float, a_sigma: float,
                              b_sigma: float, ridge: float,
                              rng: np.random.Generator) -> np.ndarray:
    '''Draw the ligand exposure equation and then the receptor exposure equation.

    The two exposure equations are visited in the order the state packs them,
    the ligand equation first and the receptor equation second, and the
    receptor equation conditions on the ligand equation as it has just been
    drawn rather than as it stood on entry. Within an equation the instrument
    coefficients are drawn first, then the covariate coefficients conditioning
    on those instrument coefficients, then the residual variance conditioning
    on both. No entry of the outcome block of the state is modified. Every
    matrix inverted along the way carries ridge added to its diagonal
    beforehand.

    Randomness comes from rng, which for each equation in turn draws one
    vector of n_instruments standard normals, then one vector of n_covariates
    standard normals, then one standard gamma variate. Each coefficient block
    is its conditional mean plus the lower Cholesky factor of its conditional
    covariance times its standard normal vector, and each residual variance is
    the scale of its inverse gamma conditional divided by its standard gamma
    variate.

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
        Zellner g value shared by every coefficient prior, strictly positive.
    a_sigma : float
        Shape of the inverse gamma prior on a residual variance, a_sigma > 0.
    b_sigma : float
        Scale of the inverse gamma prior on a residual variance, b_sigma > 0.
    ridge : float
        Non negative diagonal regularisation added before any inversion.
    rng : np.random.Generator
        Generator supplying the draws, in the order described above.

    Returns
    -------
    exposures : np.ndarray
        Array of 2 * (n_instruments + n_covariates + 1) floats holding the
        drawn ligand instrument coefficients, ligand covariate coefficients
        and ligand residual variance, followed by the same three quantities of
        the receptor equation, in that order.

    Raises
    ------
    ValueError
        If data is not a two dimensional finite array with the column count
        implied by n_instruments and n_covariates, if either count is below
        one, if state is not a finite one dimensional array of the implied
        length, if any residual variance held in state is not strictly
        positive, if g_prior, a_sigma or b_sigma is not a positive finite
        number, if ridge is negative, or if a matrix that has to be inverted
        is not positive definite.
    '''
    return exposures  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np 

def _oracle_update_exposure_equations(data: np.ndarray, n_instruments: int, n_covariates: int,
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

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np 

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: a confounded cohort at the benchmark shape, state away from zero ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(13)
n_donors, n_instruments, n_covariates = 80, 4, 3
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
            "call": "digest(update_exposure_equations(data, n_instruments, n_covariates, state, 100.0, 3.0, 2.0, 1e-6, np.random.default_rng(7)))",
            "gold_call": "digest(_oracle_update_exposure_equations(data, n_instruments, n_covariates, state, 100.0, 3.0, 2.0, 1e-6, np.random.default_rng(7)))",
        },
        # --- Valid: a state carrying a large interaction, so the two equations are strongly coupled ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(29)
n_donors, n_instruments, n_covariates = 120, 2, 2
instruments = rng.standard_normal((n_donors, 2 * n_instruments))
covariates = rng.standard_normal((n_donors, n_covariates))
ligand = instruments[:, :n_instruments] @ np.full(n_instruments, 0.6) + rng.standard_normal(n_donors)
receptor = instruments[:, n_instruments:] @ np.full(n_instruments, 0.6) + rng.standard_normal(n_donors)
pathway = 0.5 * receptor + 0.4 * ligand * receptor + rng.standard_normal(n_donors)
data = np.column_stack([instruments, covariates, ligand, receptor, pathway])
data = data - data.mean(axis=0)
state = np.linspace(-0.4, 0.8, 2 * n_instruments + 3 * n_covariates + 9)
state[n_instruments + n_covariates] = 1.3
state[2 * n_instruments + 2 * n_covariates + 1] = 1.2
state[2 * n_instruments + 3 * n_covariates + 4] = 0.8
state[2 * n_instruments + 3 * n_covariates + 6] = 0.45
state[2 * n_instruments + 3 * n_covariates + 7] = 1.0
state[2 * n_instruments + 3 * n_covariates + 8] = 0.6
def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(update_exposure_equations(data, n_instruments, n_covariates, state, 100.0, 3.0, 2.0, 1e-6, np.random.default_rng(21)))",
            "gold_call": "digest(_oracle_update_exposure_equations(data, n_instruments, n_covariates, state, 100.0, 3.0, 2.0, 1e-6, np.random.default_rng(21)))",
        },
        # --- Boundary: an outcome block that is identically zero ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(37)
n_donors, n_instruments, n_covariates = 60, 3, 2
instruments = rng.standard_normal((n_donors, 2 * n_instruments))
covariates = rng.standard_normal((n_donors, n_covariates))
ligand = instruments[:, :n_instruments] @ np.full(n_instruments, 0.5) + rng.standard_normal(n_donors)
receptor = instruments[:, n_instruments:] @ np.full(n_instruments, 0.5) + rng.standard_normal(n_donors)
pathway = rng.standard_normal(n_donors)
data = np.column_stack([instruments, covariates, ligand, receptor, pathway])
data = data - data.mean(axis=0)
state = np.zeros(2 * n_instruments + 3 * n_covariates + 9)
state[:n_instruments] = 0.5
state[n_instruments + n_covariates + 1:2 * n_instruments + n_covariates + 1] = 0.5
state[n_instruments + n_covariates] = 1.0
state[2 * n_instruments + 2 * n_covariates + 1] = 1.0
state[2 * n_instruments + 3 * n_covariates + 4] = 1.0
state[2 * n_instruments + 3 * n_covariates + 8] = 0.75
def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(update_exposure_equations(data, n_instruments, n_covariates, state, 100.0, 3.0, 2.0, 1e-6, np.random.default_rng(2)))",
            "gold_call": "digest(_oracle_update_exposure_equations(data, n_instruments, n_covariates, state, 100.0, 3.0, 2.0, 1e-6, np.random.default_rng(2)))",
        },
        # --- Edge: one instrument, one covariate, a small cohort and a small g value ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(43)
n_donors, n_instruments, n_covariates = 18, 1, 1
instruments = rng.standard_normal((n_donors, 2 * n_instruments))
covariates = rng.standard_normal((n_donors, n_covariates))
ligand = instruments[:, :n_instruments] @ np.full(n_instruments, 0.5) + rng.standard_normal(n_donors)
receptor = instruments[:, n_instruments:] @ np.full(n_instruments, 0.5) + rng.standard_normal(n_donors)
pathway = 0.5 * receptor + rng.standard_normal(n_donors)
data = np.column_stack([instruments, covariates, ligand, receptor, pathway])
data = data - data.mean(axis=0)
state = np.linspace(-0.3, 0.6, 2 * n_instruments + 3 * n_covariates + 9)
state[n_instruments + n_covariates] = 0.9
state[2 * n_instruments + 2 * n_covariates + 1] = 1.4
state[2 * n_instruments + 3 * n_covariates + 4] = 1.1
state[2 * n_instruments + 3 * n_covariates + 7] = 1.0
state[2 * n_instruments + 3 * n_covariates + 8] = 0.5
def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(update_exposure_equations(data, n_instruments, n_covariates, state, 0.5, 3.0, 2.0, 1e-6, np.random.default_rng(5)))",
            "gold_call": "digest(_oracle_update_exposure_equations(data, n_instruments, n_covariates, state, 0.5, 3.0, 2.0, 1e-6, np.random.default_rng(5)))",
        },
        # --- Invalid: a state vector of the wrong length ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(1)
data = rng.standard_normal((40, 2 * 2 + 2 + 3))
state = np.ones(11)
def run_model():
    try:
        update_exposure_equations(data, 2, 2, state, 100.0, 3.0, 2.0, 1e-6, np.random.default_rng(1))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_update_exposure_equations(data, 2, 2, state, 100.0, 3.0, 2.0, 1e-6, np.random.default_rng(1))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a state whose ligand residual variance sits at zero ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(1)
data = rng.standard_normal((40, 2 * 2 + 2 + 3))
state = np.full(2 * 2 + 3 * 2 + 9, 0.4)
state[2 + 2] = 0.0
state[2 * 2 + 2 * 2 + 1] = 1.0
state[2 * 2 + 3 * 2 + 4] = 1.0
state[2 * 2 + 3 * 2 + 8] = 0.75
def run_model():
    try:
        update_exposure_equations(data, 2, 2, state, 100.0, 3.0, 2.0, 1e-6, np.random.default_rng(1))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_update_exposure_equations(data, 2, 2, state, 100.0, 3.0, 2.0, 1e-6, np.random.default_rng(1))
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
