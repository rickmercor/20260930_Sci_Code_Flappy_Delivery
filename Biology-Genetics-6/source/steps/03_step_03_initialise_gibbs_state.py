"""
Build the starting parameter vector of the sampler from three least squares fits, packing the two exposure blocks, the outcome nuisance block, the communication effect vector and the selection parameters into one flat array.

Every parameter of the causal model has to be given a starting value before the sampler can take its first sweep, and the choice is not free of consequence. Two of the conditional distributions in this model divide by quantities formed from the current representation of the exposures. If the exposure coefficients start at zero that representation is identically zero, the conditional variances of the receptor main effect and of the communication effect vector are then controlled entirely by the ridge term added for numerical safety, and the first draw is enormous, so the chain needs a long excursion before it is anywhere near the posterior. Starting instead from the fit each equation of the model would have if the confounder were absent avoids that pathology and puts the chain in the bulk of the posterior from the first sweep.




The starting point therefore has to mirror the model rather than the data. Each equation of the structural model is fitted on its own by ordinary least squares, in the representation the sampler itself will work in, and each fit supplies its own coefficients together with a starting residual variance taken from its residual sum of squares over its residual degrees of freedom. Fitting the outcome equation in some other representation would not be wrong in any absolute sense, since an initialisation cannot change a stationary distribution, but it would place the chain somewhere the conditionals do not expect and the burn in would have to pay for it.




The two selection parameters start at their least committal values. The inclusion indicator starts at one so that the communication effect is drawn from the unshrunk component on the first sweep and the chain is free to move either way, and the inclusion probability starts at the mean of its beta prior. Nothing about this initialisation biases the stationary distribution, which depends only on the conditionals, but it does decide how quickly the chain reaches it.

Returns
-------
np.ndarray of 2 * n_instruments + 3 * n_covariates + 9 floats: the packed starting value of every parameter of the sampler.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def initialise_gibbs_state(data: np.ndarray, n_instruments: int, n_covariates: int,
                           a_rho: float, b_rho: float) -> np.ndarray:
    '''Build the starting parameter vector of the sampler by least squares.

    Ligand expression is regressed on the ligand instruments and the
    covariates, receptor expression on the receptor instruments and the
    covariates, and pathway activity on an intercept, the two fitted exposure
    values, their elementwise product and the covariates. Residual variances
    are the residual sums of squares divided by the residual degrees of
    freedom of their own fit, floored at 1e-8. The inclusion indicator starts
    at one and the inclusion probability at a_rho / (a_rho + b_rho).

    The returned vector has length 2 * n_instruments + 3 * n_covariates + 9
    and holds, in order, the ligand instrument coefficients, the ligand
    covariate coefficients, the ligand residual variance, the receptor
    instrument coefficients, the receptor covariate coefficients, the receptor
    residual variance, the outcome intercept, the outcome covariate
    coefficients, the receptor main effect, the outcome residual variance, the
    ligand main effect, the ligand by receptor interaction effect, the
    inclusion indicator and the inclusion probability.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (n_donors, 2 * n_instruments + n_covariates + 3) laid
        out as returned by the cohort simulation step.
    n_instruments : int
        Number of cis-eQTL instruments for each exposure, n_instruments >= 1.
    n_covariates : int
        Number of observed donor covariates, n_covariates >= 1.
    a_rho : float
        First shape of the beta prior on the inclusion probability, a_rho > 0.
    b_rho : float
        Second shape of the beta prior on the inclusion probability, b_rho > 0.

    Returns
    -------
    state : np.ndarray
        Array of 2 * n_instruments + 3 * n_covariates + 9 floats holding the
        starting value of every parameter in the order given above.

    Raises
    ------
    ValueError
        If data is not a two dimensional finite array with the column count
        implied by n_instruments and n_covariates, if either count is below
        one, if either beta shape is not a positive finite number, or if the
        cohort leaves no residual degrees of freedom for one of the three
        fits.
    '''
    return state  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_initialise_gibbs_state(data: np.ndarray, n_instruments: int, n_covariates: int,
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

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np 


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: confounded null cohort at the benchmark shape ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(31)
n_donors, n_instruments, n_covariates = 180, 4, 3
instruments = rng.standard_normal((n_donors, 2 * n_instruments))
covariates = rng.standard_normal((n_donors, n_covariates))
confounder = rng.standard_normal(n_donors)
ligand = instruments[:, :n_instruments] @ np.full(n_instruments, 0.45) + 0.7 * confounder + rng.standard_normal(n_donors)
receptor = instruments[:, n_instruments:] @ np.full(n_instruments, 0.45) + 0.7 * confounder + rng.standard_normal(n_donors)
pathway = 0.5 * receptor + covariates @ np.full(n_covariates, 0.25) + 0.7 * confounder + rng.standard_normal(n_donors)
data = np.column_stack([instruments, covariates, ligand, receptor, pathway])
data = data - data.mean(axis=0)
def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(initialise_gibbs_state(data, n_instruments, n_covariates, 3.0, 1.0))",
            "gold_call": "digest(_oracle_initialise_gibbs_state(data, n_instruments, n_covariates, 3.0, 1.0))",
        },
        # --- Valid: cohort with communication and receptor modulation present ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(6)
n_donors, n_instruments, n_covariates = 120, 3, 2
instruments = rng.standard_normal((n_donors, 2 * n_instruments))
covariates = rng.standard_normal((n_donors, n_covariates))
confounder = rng.standard_normal(n_donors)
ligand = instruments[:, :n_instruments] @ np.full(n_instruments, 0.5) + 0.7 * confounder + rng.standard_normal(n_donors)
receptor = instruments[:, n_instruments:] @ np.full(n_instruments, 0.5) + 0.7 * confounder + rng.standard_normal(n_donors)
pathway = 0.3 * ligand + 0.5 * receptor + 0.3 * ligand * receptor + covariates @ np.full(n_covariates, 0.3) + 0.7 * confounder + rng.standard_normal(n_donors)
data = np.column_stack([instruments, covariates, ligand, receptor, pathway])
data = data - data.mean(axis=0)
def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(initialise_gibbs_state(data, n_instruments, n_covariates, 3.0, 1.0))",
            "gold_call": "digest(_oracle_initialise_gibbs_state(data, n_instruments, n_covariates, 3.0, 1.0))",
        },
        # --- Boundary: a symmetric beta prior, so the inclusion probability starts at one half ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(19)
n_donors, n_instruments, n_covariates = 70, 1, 1
instruments = rng.standard_normal((n_donors, 2 * n_instruments))
covariates = rng.standard_normal((n_donors, n_covariates))
ligand = instruments[:, :n_instruments] @ np.full(n_instruments, 0.5) + rng.standard_normal(n_donors)
receptor = instruments[:, n_instruments:] @ np.full(n_instruments, 0.5) + rng.standard_normal(n_donors)
pathway = 0.5 * receptor + covariates @ np.full(n_covariates, 0.3) + rng.standard_normal(n_donors)
data = np.column_stack([instruments, covariates, ligand, receptor, pathway])
data = data - data.mean(axis=0)
def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(initialise_gibbs_state(data, n_instruments, n_covariates, 1.0, 1.0))",
            "gold_call": "digest(_oracle_initialise_gibbs_state(data, n_instruments, n_covariates, 1.0, 1.0))",
        },
        # --- Edge: a cohort barely large enough for the outcome fit ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(23)
n_donors, n_instruments, n_covariates = 14, 1, 1
instruments = rng.standard_normal((n_donors, 2 * n_instruments))
covariates = rng.standard_normal((n_donors, n_covariates))
ligand = instruments[:, :n_instruments] @ np.full(n_instruments, 0.5) + rng.standard_normal(n_donors)
receptor = instruments[:, n_instruments:] @ np.full(n_instruments, 0.5) + rng.standard_normal(n_donors)
pathway = 0.5 * receptor + covariates @ np.full(n_covariates, 0.3) + rng.standard_normal(n_donors)
data = np.column_stack([instruments, covariates, ligand, receptor, pathway])
data = data - data.mean(axis=0)
def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(initialise_gibbs_state(data, n_instruments, n_covariates, 3.0, 1.0))",
            "gold_call": "digest(_oracle_initialise_gibbs_state(data, n_instruments, n_covariates, 3.0, 1.0))",
        },
        # --- Invalid: a beta shape at zero ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(1)
data = rng.standard_normal((40, 8))
data = data - data.mean(axis=0)
def run_model():
    try:
        initialise_gibbs_state(data, 2, 1, 0.0, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_initialise_gibbs_state(data, 2, 1, 0.0, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a column count that does not match the declared counts ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(1)
data = rng.standard_normal((40, 12))
data = data - data.mean(axis=0)
def run_model():
    try:
        initialise_gibbs_state(data, 2, 1, 3.0, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_initialise_gibbs_state(data, 2, 1, 3.0, 1.0)
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
